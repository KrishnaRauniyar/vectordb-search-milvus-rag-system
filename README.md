# Milvus + sentence-transformers + Qwen RAG on ALCF HPC documentation

This project takes raw HPC documentation from
[docs.alcf.anl.gov](https://docs.alcf.anl.gov/), turns it into a searchable
semantic index, and lets you ask plain-English questions like
*"how do I submit a multi-node GPU job on Polaris?"* — then a local
instruction-tuned LLM (Qwen2.5-1.5B-Instruct) reads the retrieved
passages and writes a grounded answer with citations.

That last part is **Retrieval-Augmented Generation (RAG)**. Steps 1-5
build and query the vector index; step 6 hands the top-K chunks to the
LLM as context so it can answer in natural language without
hallucinating. No API keys, no rate limits, no internet after the first
model download — everything runs locally.

It's a six-stage pipeline:

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
                                         step4_index        (Milvus Lite, IVF_FLAT, COSINE)
                                              │
                                              ▼
                                         step5_search       ◄── your query
                                              │
                                              ▼
                                         step6_rag          (Qwen2.5-1.5B-Instruct,
                                              │              local LLM, CPU/MPS/CUDA)
                                              ▼
                                         grounded answer + sources
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

# 2. Run the whole pipeline + a sample question
#    First run downloads ~80 MB of MiniLM weights and ~3 GB of Qwen2.5-1.5B
#    weights into ~/.cache/huggingface. Subsequent runs reuse the cache.
#    No API key, no .env, no internet required after that.
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
python -m src.step6_rag    "how do I use Globus to move data?"
```

### Re-running just the query (fastest path)

Stages 1-4 are **idempotent and cached** — once `data/raw/`,
`data/chunks.json`, and `data/milvus_alcf.db` exist, the only thing
that has to run for a new question is the embed-query + ANN search
(+ LLM generation in RAG mode). To fire many queries against the
already-built index without rebuilding anything:

```bash
# Retrieval only — fast (~1 s including model load + search).
.venv/bin/python -m src.step5_search "your next question"

# Retrieval + RAG answer — slower (~5-30 s, depending on device).
.venv/bin/python -m src.step6_rag    "your next question"
```

After the first invocation both model weights stay cached on disk, so
launching a fresh Python interpreter for each query is fine.

### Reading the timing output

Every step prints `⏱  step N took Xs`, and `main.py` ends with a
summary so you can see where the wall-clock budget actually went:

```
================================================================
  TIMING SUMMARY
================================================================
    0.664s    3.4%  █                                    Step 1: download ALCF documentation pages
    0.011s    0.1%                                       Step 2: split pages into chunks
    1.755s    9.1%  ████                                 Step 3: embed chunks locally
    1.088s    5.6%  ██                                   Step 4: index chunks in Milvus
   15.836s   81.8%  ████████████████████████████████████ Step 6: RAG: retrieve top-K (step 5) + answer with local LLM
----------------------------------------------------------------
   19.354s  100.0%  ███████████████████████████████████  TOTAL
```

What you're looking at:

| Stage | What's happening | Typical cost (warm cache)               |
| ----- | ---------------- | --------------------------------------- |
| 1     | Page fetch / read from `data/raw/`           | ~0.5 s cached, ~3-6 s on a cold first run |
| 2     | Recursive text splitter                      | <0.05 s, scales linearly with text size   |
| 3     | SBERT model load + encode all chunks         | ~2-3 s (mostly model load); `[cached]` if vectors already exist |
| 4     | Milvus drop + create + insert + K-means + load | ~1 s — always re-runs because we drop+recreate |
| 5     | Embed query + IVF_FLAT ANN search (folded into step 6) | ~0.3 s — `embedded in ~300 ms | searched ~6 ms` |
| 6     | LLM weight load + token generation           | ~3 s load (warm cache) + 5-30 s/answer (MPS); 30-60 s/answer (CPU); 1-3 s/answer (CUDA) |

The big lesson visible in the bars: **generation dominates retrieval.**
The LLM forward passes take 10x-100x longer than the ANN search. This
is why production RAG systems care about (a) caching query embeddings,
(b) using fast-but-good-enough embedders, and (c) using *smaller*
LLMs whenever the retrieved context is strong enough to compensate for
less parametric knowledge. Milvus latency is rarely the bottleneck.

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

**What:** Embeds the user query with the *same* MiniLM model used at
ingest, calls `client.search(...)` with `metric_type=COSINE` and
`nprobe=8`, prints the top-K hits with score, title, URL, and a text
preview. Returns the hits so step 6 can use them.

**Why measure timings:** The script prints embed time vs. search time
separately. Even with a local model, the encode step (~80 ms) still
dominates the ANN search (~5 ms) by more than an order of magnitude.
With a hosted API on the embed side it's 100×+. Either way, the lesson
is the same: tune your embedding pipeline before you bother optimizing
Milvus.

**Key code:**
- [`search`](src/step5_search.py#L88-L135) — the full query path.
- [`_print_hit`](src/step5_search.py#L67-L82) — formatting only.

---

### Step 6 — RAG — [src/step6_rag.py](src/step6_rag.py)

**What:** Takes the top-K hits from step 5, builds a chat-formatted
prompt (system rules + numbered context + question), feeds it to a
local instruction-tuned LLM (`Qwen/Qwen2.5-1.5B-Instruct`), and prints
a grounded natural-language answer plus a list of source URLs.

**Why this exists at all:** Steps 1-5 give you *retrieved passages*,
not an *answer*. Most users don't want to read 5 chunks; they want one
paragraph. An LLM is the right tool for "read these passages and write
the answer." But asking an LLM to answer from its parametric memory
gets you confident hallucinations on anything outside its training
cutoff or training distribution. RAG fixes both: hand the LLM the
relevant text first, then ask the question.

**Why Qwen2.5-1.5B-Instruct:**
- 1.5 B parameters, ~3 GB on disk in float32, ~1.5 GB in RAM at bf16.
- Apache-2.0 — free for any use, no HF auth wall.
- One of the most downloaded small instruct models on Hugging Face.
- Strong at "answer from the provided context" tasks, which is exactly
  what RAG needs — the model doesn't have to know HPC trivia; it just
  has to read the passages we give it. Bigger isn't always better for
  RAG: a 7B model would generate slightly cleaner prose but cost 4×
  more memory and 3-5× more latency for marginal quality gains on this
  workload. (To try a bigger one, change `LLM_MODEL` in
  [src/config.py](src/config.py#L149).)

**Why a chat template, not a hand-built prompt string:** Qwen was
fine-tuned with specific role tokens (`<|im_start|>system`,
`<|im_end|>`, etc.). The tokenizer's `apply_chat_template` writes them
correctly; doing it by hand is the #1 source of garbled RAG outputs.

**Why these generation knobs:**
- `temperature=0.2` — low randomness; we want the model to stick to
  what the retrieved chunks say, not creatively riff.
- `max_new_tokens=350` — caps answer length so latency is bounded and
  the model doesn't ramble or repeat itself.
- `device` auto-picks CUDA > MPS > CPU; we use bfloat16 on
  GPU/MPS (halves memory, no quality loss) and float32 on CPU (bf16 on
  CPU is currently *slower* in PyTorch — scalar paths aren't optimised).

**Key code:**
- [`rag_answer`](src/step6_rag.py#L218-L255) — the full retrieve→prompt→generate→present pipeline.
- [`_build_messages`](src/step6_rag.py#L168-L195) — the prompt template.
- [`_generate`](src/step6_rag.py#L201-L215) — tokenize → `model.generate` → decode.
- [`_load_llm`](src/step6_rag.py#L121-L150) — `lru_cache`'d weight loader.
- [`_pick_device`](src/step6_rag.py#L103-L116) — CUDA > MPS > CPU autodetect.

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

## Retrieval-Augmented Generation — what it is and why it works

### The problem RAG solves

An LLM trained today has two failure modes when you ask it about
something specific (your company's docs, a new HPC cluster, last
week's news):

1. **It doesn't know.** The fact wasn't in its training data, full
   stop. Bigger models help, but no amount of scale closes the gap on
   private or freshly updated information.
2. **It pretends it does.** When asked an unknown question, an LLM's
   "I don't know" probability is often lower than its "make up
   something plausible" probability. This is the famous *hallucination*
   problem.

RAG attacks both with the same trick: **don't ask the model to
remember — give it the answer in the prompt and ask it to read.**

### The three-step RAG loop

```
   user question
        │
        ▼
   ┌──────────────────┐
   │ 1. RETRIEVE      │   embed query  →  Milvus top-K  →  list of chunks
   └────────┬─────────┘
            │
            ▼
   ┌──────────────────┐
   │ 2. AUGMENT       │   build prompt: system rules + numbered chunks + question
   └────────┬─────────┘
            │
            ▼
   ┌──────────────────┐
   │ 3. GENERATE      │   LLM reads the augmented prompt → writes answer
   └──────────────────┘
```

In this project:
- **Retrieve** is steps 1-5: chunks are pre-embedded, the query is
  embedded at request time, Milvus returns the top-K chunks by cosine
  similarity.
- **Augment** is the prompt builder in
  [src/step6_rag.py](src/step6_rag.py): the retrieved chunks become
  numbered context blocks inside a chat-template message.
- **Generate** is Qwen2.5-1.5B-Instruct producing tokens
  autoregressively until it hits `max_new_tokens` or an end-of-turn
  token.

### What the prompt actually looks like

```
<|im_start|>system
You are a helpful assistant for the Argonne Leadership Computing Facility
(ALCF) user documentation. Answer the user's question using ONLY the
numbered context passages provided. If the answer is not in the context,
say you don't know rather than guessing. Be concise and, when helpful,
cite the passage number like [2].
<|im_end|>
<|im_start|>user
Context:
[1] <text of top-1 chunk>

[2] <text of top-2 chunk>

[3] <text of top-3 chunk>

[4] <text of top-4 chunk>

[5] <text of top-5 chunk>

Question: how do I submit a multi-node GPU job on Polaris?
Answer:
<|im_end|>
<|im_start|>assistant
```

Every piece is load-bearing:
- **System message** sets the contract ("answer from context only,
  say so if unknown"). Drops hallucination rates a lot.
- **Numbered chunks** let the model cite "[2]" instead of quoting
  whole paragraphs back at you.
- **`Answer:`** at the end nudges the model to start answering
  immediately rather than restating the question.
- **`apply_chat_template`** wraps the role markers (`<|im_start|>`,
  `<|im_end|>`) — these are *exactly* the tokens Qwen was fine-tuned
  on. Writing them by hand and getting them slightly wrong is the #1
  reason RAG answers come out garbled.

### Why generation looks like that (a 30-second LLM primer)

Causal language models like Qwen are autoregressive: given a sequence
of tokens, they predict the probability distribution over the *next*
token. Generation is a loop:

```
prompt_tokens = tokenize(prompt)
while not done and len(generated) < max_new_tokens:
    logits   = model.forward(prompt_tokens + generated)   # [vocab_size]
    probs    = softmax(logits[-1] / temperature)
    next_tok = sample(probs, top_p=0.9)                    # or argmax if temp=0
    generated.append(next_tok)
    if next_tok == eos_token: break
return decode(generated)
```

Each forward pass is one full transformer evaluation — for a 1.5 B
model that's ~3 GFLOPs per token at fp32, or ~1.5 GFLOPs at bf16. On
Apple Silicon MPS that's ~10-15 tokens/second; on a modern CUDA GPU
~80-150 tokens/second; on a laptop CPU ~3-6 tokens/second. Our 350-token
cap therefore costs us ~20-30 seconds on MPS, ~3-5 seconds on CUDA,
~60-120 seconds on CPU.

**Why temperature 0.2:** at `temperature=0` generation is greedy
(argmax), which is *too* deterministic and tends to get stuck in
repetition loops on small models. At `temperature=1.0` it's at the
training distribution — fine for creative writing, bad for "answer
from this exact text." A small temperature (0.1-0.3) is the standard
RAG sweet spot.

### Where RAG can still go wrong

A surprising amount of "the LLM gave me a bad answer" turns out to be
*retrieval* failing, not the LLM. Useful debugging order when an
answer looks wrong:

1. **Look at the retrieved chunks** (step 5 prints them). Is the right
   passage in the top-K at all? If not, the embedder or the chunker is
   the problem — not the LLM.
2. **Look at the scores.** If the best score is below ~0.45 the
   retrieved chunks are weak matches; the LLM was asked to answer from
   irrelevant text and is filling in the gaps.
3. **Look at the prompt itself.** Did the chunks get truncated? Are
   the chat role tokens correct?
4. *Then* look at the LLM. Try `temperature=0`, try a bigger model,
   try a different system prompt.

The order matters: garbage in, garbage out. A 70B model can't rescue
the answer if the retrieval handed it the wrong page.

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

### LLM generation (the slowest step in the whole pipeline)

| Operation                                  | Typical cost (Qwen2.5-1.5B-Instruct)        |
| ------------------------------------------ | ------------------------------------------- |
| First-time weight download                 | ~3 GB into `~/.cache/huggingface`           |
| Cold model load (weights cached on disk)   | ~2 – 5 s on MPS/CUDA, ~5 – 10 s on CPU      |
| Token throughput, CUDA                     | ~80 – 150 tok/s                             |
| Token throughput, Apple Silicon MPS        | ~10 – 15 tok/s                              |
| Token throughput, laptop CPU               | ~3 – 6 tok/s                                |
| Full answer (≤ 350 new tokens), CUDA       | ~1 – 3 s                                    |
| Full answer (≤ 350 new tokens), MPS        | ~5 – 30 s                                   |
| Full answer (≤ 350 new tokens), CPU        | ~30 – 120 s                                 |
| Peak RAM at bf16 on GPU/MPS                | ~1.5 – 2 GB                                 |
| Peak RAM at fp32 on CPU                    | ~6 GB                                       |

Two consequences worth internalizing:

1. **Generation is the new bottleneck.** Once you switch from
   retrieval-only to RAG, the LLM forward passes dominate everything
   else by 10×-100×. Optimizing Milvus past "good enough" is wasted
   effort until you've done something about generation latency.
2. **Smaller LLMs win for RAG.** A 1.5B model with strong retrieval
   often beats a 7B model with the same retrieval, because the answer
   is mostly *in the prompt* — the LLM is paraphrasing, not recalling.
   Spend your latency budget on better embeddings and bigger top-K
   before reaching for a bigger LLM.

---

## Files in this repo

| Path                                              | What it does                                    |
| ------------------------------------------------- | ----------------------------------------------- |
| [main.py](main.py)                                | runs steps 1–6 end-to-end                       |
| [src/config.py](src/config.py)                    | every tunable constant, with rationale          |
| [src/step1_download.py](src/step1_download.py)   | crawl docs.alcf.anl.gov                         |
| [src/step2_chunk.py](src/step2_chunk.py)         | recursive character splitter                    |
| [src/step3_embed.py](src/step3_embed.py)         | local sentence-transformers embeddings          |
| [src/step4_index.py](src/step4_index.py)         | Milvus collection + IVF_FLAT index              |
| [src/step5_search.py](src/step5_search.py)       | query embedding + ANN search                    |
| [src/step6_rag.py](src/step6_rag.py)             | local LLM (Qwen2.5-1.5B-Instruct) RAG answer    |
| [requirements.txt](requirements.txt)              | dependencies                                    |

---

## Further reading

- Milvus architecture overview — <https://milvus.io/docs/architecture_overview.md>
- Milvus's own "build a RAG" tutorial (the design this project follows) — <https://milvus.io/docs/build-rag-with-milvus.md>
- HNSW paper (Malkov & Yashunin, 2016) — <https://arxiv.org/abs/1603.09320>
- Matryoshka Representation Learning — <https://arxiv.org/abs/2205.13147>
- sentence-transformers docs — <https://sbert.net/>
- `all-MiniLM-L6-v2` model card — <https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2>
- `Qwen/Qwen2.5-1.5B-Instruct` model card — <https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct>
- Qwen2.5 technical report — <https://arxiv.org/abs/2412.15115>
- Original RAG paper (Lewis et al., 2020) — <https://arxiv.org/abs/2005.11401>
- MTEB embedding benchmark leaderboard — <https://huggingface.co/spaces/mteb/leaderboard>
- ALCF user guides (our corpus) — <https://docs.alcf.anl.gov/>
