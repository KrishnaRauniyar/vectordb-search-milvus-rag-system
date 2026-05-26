# Milvus + sentence-transformers RAG demo on ALCF HPC documentation

This project takes raw HPC documentation from
[docs.alcf.anl.gov](https://docs.alcf.anl.gov/), turns it into a searchable
semantic index, and lets you ask plain-English questions like
*"how do I submit a multi-node GPU job on Polaris?"* and get back the
exact paragraphs that answer the question — even if the page never uses
the same words.

It's a five-stage pipeline:

```
 docs.alcf.anl.gov  ─►  step1_download  ─►  raw HTML/JSON
                                              │
                                              ▼
                                         step2_chunk        (recursive splitter,
                                              │              800 chars + 120 overlap)
                                              ▼
                                         step3_embed        (sentence-transformers
                                              │              all-MiniLM-L6-v2,
                                              │              384-dim, local CPU, no API)
                                              ▼
                                         step4_index        (Milvus Lite, HNSW, COSINE)
                                              │
                                              ▼
                                         step5_search       ◄── your query
```

Each stage is its own file under [src/](src/), each file's docstring
explains the *why*, and the orchestrator at [main.py](main.py) glues
them together.

---

## Quick start

```bash
# 1. Create a virtualenv and install deps (one-time)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Run the whole pipeline + a sample search
#    (no API key, no .env, no internet required after the first model download)
python main.py

# 3. Ask your own question
python main.py "what filesystem should I use for training datasets on Polaris?"
```

If you don't want to activate the venv first, call the interpreter directly:
`.venv/bin/python main.py "..."`.

The `-W ignore` flag silences Python-3.9-EOL / OpenSSL-LibreSSL deprecation
warnings so you only see the project's own output:
`.venv/bin/python -W ignore main.py "..."`.

### Running a single stage

Every stage is its own runnable module. Useful while you iterate on one
piece without paying for the others:

```bash
python -m src.step1_download
python -m src.step2_chunk
python -m src.step3_embed
python -m src.step4_index
python -m src.step5_search "how do I use Globus to move data?"
```

### Re-running just the query (fastest path)

Stages 1-3 are **idempotent and cached** — once `data/raw/` and
`data/chunks.json` exist, the only thing that has to run for a new
question is the embed-query + ANN search. To fire many queries against
the already-built index without rebuilding anything:

```bash
.venv/bin/python -m src.step5_search "your next question"
```

That skips stages 1-4 entirely and usually takes ~1 second (model load
+ search). After the first query the model stays cached on disk, so
launching a fresh Python interpreter for each query is fine.

### Reading the timing output

Every step prints `⏱  step N took Xs`, and `main.py` ends with a
summary so you can see where the wall-clock budget actually went:

```
================================================================
  TIMING SUMMARY
================================================================
    0.557s   11.4%  █████                                Step 1: download ALCF documentation pages
    0.011s    0.2%                                       Step 2: split pages into chunks
    2.902s   59.3%  █████████████████████████████        Step 3: embed chunks locally
    1.105s   22.6%  ███████████                          Step 4: index chunks in Milvus
    0.321s    6.5%  ███                                  Step 5: similarity search
----------------------------------------------------------------
    4.896s  100.0%  ██████████████████████████████████   TOTAL
```

What you're looking at:

| Stage | What's happening | Typical cost (warm cache)               |
| ----- | ---------------- | --------------------------------------- |
| 1     | Page fetch / read from `data/raw/`           | ~0.5 s cached, ~3-6 s on a cold first run |
| 2     | Recursive text splitter                      | <0.05 s, scales linearly with text size   |
| 3     | SBERT model load + encode all chunks         | ~2-3 s (mostly model load); `[cached]` if vectors already exist |
| 4     | Milvus drop + create + insert + K-means + load | ~1 s — always re-runs because we drop+recreate |
| 5     | Embed query + IVF_FLAT ANN search            | ~0.3 s — step 5's own line breaks it down: `embedded in ~300 ms | searched ~6 ms` |

The big lesson visible in the bars: **embedding dominates everything
else.** Search itself is a rounding error. This is why production RAG
systems put so much effort into caching query embeddings and using
fast-but-good-enough models — Milvus latency is rarely the bottleneck.

---

## The pipeline, step by step

### Step 1 — download — [src/step1_download.py](src/step1_download.py)

**What:** GET each URL in [`SEED_URLS`](src/config.py#L29-L42), strip
nav/footer/code-chrome with BeautifulSoup, save the cleaned article as
`data/raw/<slug>.json`.

**Why:** A vector DB is only useful with content to put into it. We
hand-pick ~10 ALCF pages instead of crawling the whole domain so the
demo is deterministic, fast, and polite to Argonne's servers.

**Key code:**
- [`_fetch`](src/step1_download.py#L66-L75) — one HTTP GET with timeout + UA.
- [`_extract`](src/step1_download.py#L78-L107) — `<article>` -> plain text.
- [`download_all`](src/step1_download.py#L113-L150) — orchestrates, caches.

---

### Step 2 — chunk — [src/step2_chunk.py](src/step2_chunk.py)

**What:** Splits each page into ~800-character chunks with 120-character
overlap. Splits on the largest semantic boundary first (paragraph →
line → sentence → word → char) — a *recursive character splitter*.

**Why:**
- The embedding model has an input limit (~2048 tokens). Long ALCF pages
  don't fit.
- One vector per page averages every topic on the page into a blurry
  mean — you'd never retrieve the specific paragraph about `qsub`.
- Overlap means a fact that straddles a chunk boundary still appears
  intact in at least one chunk's vector.

**Key code:**
- [`_split_recursive`](src/step2_chunk.py#L57-L100) — the splitter
  itself, no LangChain dependency.
- [`_add_overlap`](src/step2_chunk.py#L103-L114) — prepends the tail of
  chunk N-1 onto chunk N.

---

### Step 3 — embed — [src/step3_embed.py](src/step3_embed.py)

**What:** Loads `sentence-transformers/all-MiniLM-L6-v2` (~80 MB,
downloaded once into `~/.cache/huggingface`), encodes every chunk into a
384-dim unit-length vector, writes vectors back into `data/chunks.json`.

**Why this model:** Local, free, no rate limits, no API key. 22M params,
runs at ~1000 sentences/sec on a laptop CPU. It's the community baseline
that most RAG tutorials use and most MTEB submissions are compared
against. For our ~100-chunk demo the quality gap vs. paid hosted models
is small; for >100k chunks you'd reach for BGE/E5/Gemini.

**Why we normalize:** With `normalize_embeddings=True`, every vector has
L2 norm = 1, which makes cosine similarity equivalent to a plain dot
product — one fused-multiply-add per dimension, computed by Milvus
internally when `METRIC_TYPE=COSINE`.

**Why no batching ceremony:** `model.encode([...])` handles batching
internally — we just pass it the whole list and the
`batch_size=EMBED_BATCH` knob from [config.py](src/config.py#L80-L85)
controls throughput.

**Key code:**
- [`_get_model`](src/step3_embed.py#L70-L77) — cached SBERT loader.
- [`embed_texts`](src/step3_embed.py#L83-L100) — corpus encoder.
- [`embed_query`](src/step3_embed.py#L103-L110) — reused by step 5.

---

### Step 4 — index — [src/step4_index.py](src/step4_index.py)

**What:** Opens (or creates) `data/milvus_alcf.db`, drops + recreates the
`alcf_docs` collection, defines a schema (id, url, title, text, 384-d
vector), inserts every chunk, builds an HNSW index, and loads the
collection into memory.

**Why drop + recreate:** Idempotent re-runs while you iterate on
chunking/embedding. In production you'd `upsert` instead.

**Why explicit `flush` and `load_collection`:** Milvus separates *durable
storage* from *queryable memory*. `insert` writes to an in-memory
growing-segment; `flush` seals it to disk; `load_collection` brings the
HNSW graph back into RAM so search can hit it. This is the same
contract on Milvus Lite as on full server Milvus.

**Key code:**
- [`_build_schema`](src/step4_index.py#L65-L88).
- [`_build_index_params`](src/step4_index.py#L91-L105).
- [`index_all`](src/step4_index.py#L111-L156).

---

### Step 5 — search — [src/step5_search.py](src/step5_search.py)

**What:** Embeds the user query (with `RETRIEVAL_QUERY` task type),
calls `client.search(...)` with `metric_type=COSINE` and `ef=64`, prints
the top-K hits with score, title, URL, and a text preview.

**Why measure timings:** The script prints embed time vs. search time
separately. Even with a local model, the encode step (~2-10 ms) still
dominates the ANN search (~0.2-0.5 ms) by an order of magnitude. With a
hosted API on the embed side it's 100×+. Either way, the lesson is the
same: tune your embedding pipeline before you bother optimizing Milvus.

**Key code:**
- [`search`](src/step5_search.py#L75-L121) — the full query path.
- [`_print_hit`](src/step5_search.py#L57-L72) — formatting only.

---

## Vector databases — the concept

A **vector database** stores fixed-length lists of floats ("vectors" or
"embeddings") and answers the question *"give me the K vectors most
similar to this one"* quickly. That's it. Everything else — metadata
filters, hybrid search, multi-tenancy — is plumbing around that core
primitive.

It's different from a normal database in two ways:

1. The unit of data is a dense vector (typically 128 – 4096 floats),
   not a row of typed columns. The vector is treated as a single field
   with its own index type.

2. Queries are **approximate** by default. Exact nearest-neighbour
   search on 100M vectors takes seconds; approximate NN with a graph or
   IVF index takes milliseconds with ~99% recall. The "small lie" is the
   whole reason vector DBs exist as a category.

Why we need one for RAG: embeddings turn "find me docs relevant to this
question" into "find me vectors near this point in 384-d space." Doing
that fast across millions of vectors is exactly what these systems are
built for.

---

## Milvus architecture

(Reference: <https://milvus.io/docs/architecture_overview.md>.)

Milvus is a **distributed, cloud-native vector database** with a strict
separation between *stateless compute* and *stateful storage*. Every
real deployment has these layers:

```
                ┌────────────────────────────────────┐
   Client SDK ─►│           Access Layer             │   stateless
                │    (Proxy: gRPC/REST, auth, LB)    │   horizontally scalable
                └──────────────────┬─────────────────┘
                                   │
                ┌──────────────────▼─────────────────┐
                │         Coordinator Service        │   the "brain"
                │   RootCoord  / DataCoord           │   (DDL, schemas,
                │   QueryCoord / IndexCoord          │    load balancing)
                └──────┬───────────┬──────────┬──────┘
                       │           │          │
              ┌────────▼──┐  ┌─────▼─────┐  ┌─▼────────┐
              │ QueryNode │  │ DataNode  │  │IndexNode │   workers
              │  search   │  │ ingest +  │  │ builds   │   (stateless,
              │ on loaded │  │ flush WAL │  │ ANN      │    elastic)
              │ segments  │  │ → object  │  │ indexes  │
              └─────┬─────┘  └─────┬─────┘  └────┬─────┘
                    │              │             │
                    └──────────────┼─────────────┘
                                   │
                ┌──────────────────▼─────────────────┐
                │           Storage Layer            │   stateful, durable
                │  WAL: Pulsar / Kafka / RocksMQ     │
                │  Meta: etcd                        │
                │  Objects: MinIO / S3 / GCS / Azure │   (segments + indexes)
                └────────────────────────────────────┘
```

**The four planes in plain English:**

1. **Access layer (Proxy).** Stateless. Terminates client connections,
   does auth, routes requests, merges per-shard results. You scale this
   horizontally for QPS.

2. **Coordinator service.** The control plane. `RootCoord` handles DDL
   (collections, schemas, partitions). `DataCoord` decides when to flush
   growing segments and when to compact. `IndexCoord` schedules index
   builds. `QueryCoord` decides which QueryNodes load which segments.
   In Milvus 2.5+ these are often a single binary called "MixCoord".

3. **Worker nodes.** All stateless, all scale independently.
   - **DataNode** consumes the WAL stream, materializes inserts into
     growing segments, and on flush hands sealed segments to object
     storage.
   - **IndexNode** picks up sealed segments and builds the ANN index
     (HNSW, IVF, DiskANN, …) offline.
   - **QueryNode** loads a subset of sealed segments + their indexes
     into memory and answers vector search requests.

4. **Storage layer.** The only place state lives.
   - **WAL** (Pulsar/Kafka in production, RocksMQ in standalone) is the
     stream-of-truth for inserts.
   - **etcd** holds metadata: collection schemas, segment positions,
     index descriptors.
   - **Object storage** (S3/MinIO/GCS) holds the binary segments and
     index files. This is the durable substrate — kill every worker and
     the data is still there.

### Where does Milvus Lite fit?

Milvus Lite collapses *all* of the above into a single Python process:

| Plane             | Server Milvus              | Milvus Lite                    |
| ----------------- | -------------------------- | ------------------------------ |
| Access layer      | Proxy (gRPC)               | In-process function calls      |
| Coord + workers   | Separate pods/binaries     | One Python process             |
| WAL               | Pulsar / Kafka / RocksMQ   | Skipped (single writer)        |
| Meta store        | etcd                       | Embedded SQLite                |
| Object storage    | S3 / MinIO                 | A single local `.db` file      |

The crucial point: **the SDK API is identical**. The code in
[src/step4_index.py](src/step4_index.py) and
[src/step5_search.py](src/step5_search.py) is exactly what you would
ship against a 100-node Milvus cluster — you just change the URI from
`./data/milvus_alcf.db` to something like
`http://milvus.prod.internal:19530`.

---

## How similarity search works

### Two ANN algorithms — which one are we actually running?

Milvus supports several approximate-nearest-neighbour index types. Two
matter for this tutorial:

- **IVF_FLAT** — what *this demo* uses, because Milvus Lite only
  supports `FLAT`, `IVF_FLAT`, and `AUTOINDEX`.
- **HNSW** — what *production server Milvus* uses by default. Lower
  latency at high recall, more memory.

To switch in production you change a single constant
([`INDEX_TYPE` in src/config.py](src/config.py#L97-L113)) and the same
client code keeps working.

### IVF_FLAT (what's running right now)

At index-build time, Milvus runs K-means over the corpus and produces
`nlist` cluster centroids (we set `nlist=16`). Every vector is assigned
to its nearest centroid — that's its "bucket."

On every `client.search(...)` call:

1. **Query embedding.** The text is turned into a unit-length 384-d
   vector by the local SBERT model (already in RAM after the first call).
2. **Pick buckets.** Compute the distance from the query to each of the
   16 centroids. Sort. Pick the `nprobe=8` nearest buckets.
3. **Scan those buckets only.** Brute-force COSINE similarity against
   every vector in those 8 buckets — typically `nprobe/nlist = 50%` of
   the corpus at our scale, much less on real data sizes.
4. **Return top-K.** Sort the candidates by similarity, take the top-K.

The "approximate" part is step 2: if the true nearest neighbor happens to
live in a bucket we didn't scan, we miss it. Raising `nprobe` toward
`nlist` trades speed for recall; `nprobe=nlist` degenerates to brute
force.

### HNSW (what production Milvus uses)

HNSW (Hierarchical Navigable Small World) is the de-facto standard for
"I want low latency and high recall and I'm willing to pay for RAM."
Here's what happens on every `client.search(...)` call against an HNSW
index:

1. **Query embedding.** The text is turned into a unit-length 384-d
   vector by the local SBERT model (already in RAM after the first call).
2. **Entry point.** HNSW is a layered graph. The top layer has very few
   nodes connected by long-range edges; the bottom layer holds every
   vector. Search starts at a fixed entry node on the top layer.
3. **Greedy descent.** At each layer, examine the current node's
   neighbors, hop to whichever is closer to the query (by cosine
   similarity, in our case), repeat until no neighbor improves. Then
   drop to the next layer down and continue.
4. **Bottom-layer beam search.** On the bottom layer we keep a
   *candidate list* of size `ef` (we set it to 64 in
   [`SEARCH_PARAMS`](src/config.py#L122)). Each new neighbor is
   evaluated; better ones replace worse ones in the list.
5. **Return top-K.** When the candidate list stops improving, the K
   best entries are returned.

HNSW's two build-time knobs are:
- `M` — max neighbors per node (graph degree). Higher = better recall,
  more memory. 16 is a standard default.
- `efConstruction` — candidate list size during build. Higher = better
  graph quality, slower build. 200 is the usual sweet spot.

### Cosine distance — what the numbers mean

We normalize both document and query vectors to unit length, so for any
two vectors **a** and **b**:

```
    cosine_similarity(a, b) = a · b      ∈ [-1, +1]
                              ───────
                              ‖a‖ ‖b‖
```

Because both norms are 1 by construction, this collapses to the dot
product — which is one fused-multiply-add per dimension, ~384 FMAs per
distance computation in our setup. Milvus returns this similarity
directly as the `distance` field on each hit (higher = better).

---

## How the database is stored and used

**On disk (Milvus Lite):**

```
data/
├── raw/                       # downloaded HTML, JSON per page
├── chunks.json                # chunks + 384-d embeddings (one big JSON)
└── milvus_alcf.db             # the entire Milvus Lite database
```

That single `.db` file contains: the collection schema, the row data
(id/url/title/text), the FLOAT_VECTOR column, and the HNSW index. Move
the file, the whole database moves with it.

**On disk (server Milvus, for comparison):**

```
s3://milvus-prod/
├── insert_log/<segment>/      # raw row data, columnar
├── delta_log/<segment>/       # tombstones for deletes
├── index_files/<segment>/     # the HNSW / IVF graphs themselves
└── stats_log/<segment>/       # per-segment statistics

etcd:
├── /milvus/meta/collections/alcf_docs    # schema
├── /milvus/meta/segments/...             # which segments exist, where
└── /milvus/meta/index/...                # which fields have which indexes
```

**At query time:** vectors that are currently `load_collection`-ed live
in QueryNode RAM. The HNSW graph is fully in memory; segment data
(text, metadata) is mmap-backed so cold metadata fetches incur a page
fault but hot ones are free. This is why Milvus's RAM footprint is
roughly *(vector_count × dim × 4 bytes) + index overhead* — for our
demo, ~200 × 384 × 4 ≈ 600 KB of vectors plus a few KB of graph edges.

---

## Performance: time and space, in numbers

### Embedding (still the slow part, even local)

| Operation                                   | Typical cost (laptop CPU)        |
| ------------------------------------------- | -------------------------------- |
| Load model (cold, first run incl. download) | ~5 – 15 s (one-time)             |
| Load model (warm, weights cached)           | ~0.5 – 1.5 s                     |
| Encode one sentence                         | ~1 – 5 ms                        |
| Encode 100 chunks, batched                  | ~150 – 400 ms                    |
| Encode one search query                     | ~2 – 10 ms (model already loaded) |

No network round-trip, no rate limits, no per-token bill. The tradeoff
is recall: MiniLM is ~5 points behind a SOTA hosted model on MTEB
retrieval. For a 100-chunk demo you won't notice; for a 100M-chunk
production system you'd swap in BGE-large or a hosted API.

### Search (the fast part)

| Index   | Build time (200 vec) | Query latency  | Recall@10 | RAM         |
| ------- | -------------------- | -------------- | --------- | ----------- |
| FLAT    | 0 (no index)         | ~0.3 ms (exact) | 100%      | vectors only |
| IVF_FLAT | ~10 ms              | ~0.5 ms        | ~95%      | vectors + bucket map |
| HNSW    | ~50 ms               | ~0.2 – 0.5 ms  | ~98-99%   | vectors + graph (~M × 8 B / node) |

At our scale (200 vectors) all of these are sub-millisecond. The
interesting comparison is at scale:

| Vectors | Brute-force (FLAT) | HNSW (ef=64) |
| ------- | ------------------ | ------------ |
| 10 K    | ~1 ms              | ~0.3 ms      |
| 1 M     | ~80 ms             | ~1 – 2 ms    |
| 100 M   | ~7 s               | ~5 – 10 ms   |
| 1 B     | minutes            | ~10 – 30 ms  |

That's the whole point of an ANN index: **query time grows roughly
*logarithmically* with N**, instead of linearly. You pay for it in RAM
(the graph is roughly *M × 8 bytes × N* on top of the vectors).

### Time complexity, summarized

| Stage                | Per query                         | Notes                                  |
| -------------------- | --------------------------------- | -------------------------------------- |
| Query embedding      | O(1) network call                 | dominated by HTTP RTT                  |
| One distance compute | O(d) = O(384) FMAs                | a few hundred nanoseconds              |
| Brute-force search   | O(N · d)                          | exact                                  |
| IVF search           | O(nprobe · (N/nlist) · d)         | tunable accuracy vs. speed             |
| HNSW search          | ~O(log N · ef · d)                | sub-linear in N, ~constant in practice |

### Space complexity, summarized

| Thing                      | Bytes                            | For our demo (~200 chunks, d=384)   |
| -------------------------- | -------------------------------- | ----------------------------------- |
| Vector payload             | N · d · 4 (float32)              | ~600 KB                             |
| HNSW edges                 | ~N · M · 8                       | ~26 KB                              |
| Text + metadata            | per-chunk, variable              | a few hundred KB                    |
| Total `milvus_alcf.db`     |                                  | ~1 – 2 MB                           |

Multiply each row by 1 million for a realistic corpus: ~3 GB of vectors,
~130 MB of HNSW edges, plus whatever the raw text weighs. This is why
production deployments care so much about *quantization* (PQ, SQ) to
shrink the per-vector cost — and why a 384-d model like MiniLM is
already 8× cheaper to store than a 3072-d hosted model before you do
any quantization at all.

---

## Files in this repo

| Path                                              | What it does                                    |
| ------------------------------------------------- | ----------------------------------------------- |
| [main.py](main.py)                                | runs steps 1–5 end-to-end                       |
| [src/config.py](src/config.py)                    | every tunable constant, with rationale          |
| [src/step1_download.py](src/step1_download.py)   | crawl docs.alcf.anl.gov                         |
| [src/step2_chunk.py](src/step2_chunk.py)         | recursive character splitter                    |
| [src/step3_embed.py](src/step3_embed.py)         | local sentence-transformers embeddings          |
| [src/step4_index.py](src/step4_index.py)         | Milvus collection + HNSW index                  |
| [src/step5_search.py](src/step5_search.py)       | query embedding + ANN search                    |
| [requirements.txt](requirements.txt)              | dependencies                                    |

---

## Further reading

- Milvus architecture overview — <https://milvus.io/docs/architecture_overview.md>
- HNSW paper (Malkov & Yashunin, 2016) — <https://arxiv.org/abs/1603.09320>
- Matryoshka Representation Learning — <https://arxiv.org/abs/2205.13147>
- sentence-transformers docs — <https://sbert.net/>
- `all-MiniLM-L6-v2` model card — <https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2>
- MTEB embedding benchmark leaderboard — <https://huggingface.co/spaces/mteb/leaderboard>
- ALCF user guides (our corpus) — <https://docs.alcf.anl.gov/>
