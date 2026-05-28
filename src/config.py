"""
Project-wide configuration.

Everything that another module might want to tweak lives here so we don't
have magic numbers / paths scattered across the pipeline. Each constant has
a short note explaining *why* its current value was chosen — when you come
back to this file in six months you should be able to retune it without
re-reading the whole codebase.
"""

# Needed for the `str | None` style annotations on Python 3.9.
from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# Resolve once, relative to this file, so the project works no matter what
# directory you invoke `python main.py` from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR     = PROJECT_ROOT / "data"
RAW_DIR      = DATA_DIR / "raw"           # downloaded HTML pages live here
CHUNKS_PATH  = DATA_DIR / "chunks.json"   # output of step 2
MILVUS_DB    = DATA_DIR / "milvus_alcf.db"  # Milvus Lite's single-file store

# ---------------------------------------------------------------------------
# Step 1: crawler
# ---------------------------------------------------------------------------
# ALCF hosts user documentation as a MkDocs site. We seed the crawl with a
# handful of representative pages instead of recursively spidering the whole
# domain, because:
#   (a) it keeps the demo deterministic and fast,
#   (b) it avoids hammering Argonne's servers,
#   (c) it gives us enough breadth (Polaris, Aurora, software, data) to
#       produce interesting similarity-search results.
# Add or remove URLs here to change what gets ingested.
SEED_URLS = [
    "https://docs.alcf.anl.gov/polaris/getting-started/",
    "https://docs.alcf.anl.gov/polaris/hardware-overview/machine-overview/",
    "https://docs.alcf.anl.gov/polaris/running-jobs/",
    "https://docs.alcf.anl.gov/polaris/compiling-and-linking/compiling-and-linking-overview/",
    "https://docs.alcf.anl.gov/aurora/getting-started-on-aurora/",
    "https://docs.alcf.anl.gov/aurora/hardware-overview/machine-overview/",
    "https://docs.alcf.anl.gov/aurora/running-jobs-aurora/",
    "https://docs.alcf.anl.gov/data-management/filesystem-and-storage/data-storage/",
    "https://docs.alcf.anl.gov/data-management/data-transfer/using-globus/",
    "https://docs.alcf.anl.gov/services/jupyter-hub/",
]

# Polite crawler defaults. ALCF is a small ops team — be a good citizen.
HTTP_TIMEOUT   = 20          # seconds before we give up on a slow page
CRAWL_DELAY_S  = 0.5         # sleep between requests so we don't burst
USER_AGENT     = "milvus-alcf-tutorial/1.0 (educational; contact: student)"

# ---------------------------------------------------------------------------
# Step 2: chunking
# ---------------------------------------------------------------------------
# Why 800 chars with 120-char overlap?
#   - all-MiniLM-L6-v2 truncates input at 256 tokens (~1000 chars of English
#     prose). 800 stays just under the cliff so every chunk gets embedded
#     in full instead of having its tail silently dropped.
#   - Overlap stitches ideas that straddle a chunk boundary — without it,
#     "submit a job with qsub" might land in chunk N while the example
#     command lands in chunk N+1, and neither alone answers the query.
#   - Smaller chunks (~200 chars) make retrieval more precise but explode the
#     index size and lose context. Larger chunks (~2000) hurt precision —
#     the relevant sentence gets diluted by surrounding noise.
CHUNK_SIZE      = 800
CHUNK_OVERLAP   = 120

# ---------------------------------------------------------------------------
# Step 3: embedding
# ---------------------------------------------------------------------------
# `all-MiniLM-L6-v2` is the SBERT community's most-downloaded model. It's
# small (22M params, ~80 MB), runs on CPU at ~1000 sentences/sec, and
# produces 384-dim vectors that are L2-normalized by default. Quality is
# good enough for a learning project — it ranks well on MTEB for its size
# and is the de facto baseline most RAG tutorials compare against.
#
# Why local and not Gemini / OpenAI:
#   - No API key, no $0.0/token bill, no rate-limit games. The whole
#     pipeline runs offline after the first model download.
#   - For ~100-1000 chunk corpora the quality gap vs. a paid model is
#     small; for >100k chunks you'd reach for BGE / E5 / Gemini.
EMBED_MODEL  = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIM    = 384
# Internal batch size handed to model.encode(). Larger = better CPU
# throughput up to the model's onnx/torch saturation point; 32 is a safe
# default for laptops.
EMBED_BATCH  = 32

# ---------------------------------------------------------------------------
# Step 4: Milvus collection / index
# ---------------------------------------------------------------------------
COLLECTION_NAME = "alcf_docs"

# Distance metric. We call SBERT with `normalize_embeddings=True` so every
# vector has L2 norm = 1, which makes cosine similarity and inner product
# numerically equivalent. We pick COSINE because it's the intuition
# everyone has ("how similar are these two vectors, ignoring magnitude").
METRIC_TYPE = "COSINE"

# Index type. Options Milvus supports include:
#   FLAT     — brute force, exact, O(N*d) per query. Fine for <10k vectors.
#   IVF_FLAT — clusters vectors into nlist buckets via K-means; queries
#              only scan the nprobe buckets nearest to the query vector.
#              Sub-linear query time, easy to tune.
#   HNSW     — graph-based, the de-facto production standard ANN. Even
#              faster than IVF at high recall, more memory.
#
# Important: **Milvus Lite (the file-backed mode we use) only supports
# FLAT, IVF_FLAT, and AUTOINDEX.** HNSW is a server-Milvus-only feature
# at the moment. We use IVF_FLAT here so the demo runs in Lite mode while
# still showing a real, tunable ANN index (not just brute force).
# To move to server Milvus later, change this single constant to "HNSW"
# and update INDEX_PARAMS / SEARCH_PARAMS — nothing else needs to change.
INDEX_TYPE = "IVF_FLAT"
# IVF_FLAT knobs:
#   nlist = number of K-means clusters built over the corpus at index time.
#           Rule of thumb: ~sqrt(N) for small corpora, 4*sqrt(N) for big.
#           For our ~100 chunks, nlist=16 means each bucket holds ~6 vectors.
INDEX_PARAMS = {"nlist": 16}

# Search knob:
#   nprobe = how many of the nlist buckets to scan per query. Higher =
#            better recall, slower. nprobe=nlist degenerates to brute force.
SEARCH_PARAMS = {"nprobe": 8}

# ---------------------------------------------------------------------------
# Step 5: search
# ---------------------------------------------------------------------------
# How many neighbors to return. 5 is a reasonable default for a chat-style
# RAG demo — enough variety, not so many that the user drowns in noise.
TOP_K = 5

# ---------------------------------------------------------------------------
# Step 6: RAG generation
# ---------------------------------------------------------------------------
# Model choice: Qwen/Qwen2.5-1.5B-Instruct
#   - 1.5B parameters, ~3 GB on disk
#   - Apache 2.0 license, free for any use
#   - Currently one of the most-downloaded small instruct LLMs on Hugging
#     Face. Strong at "answer from context" tasks, which is exactly what
#     RAG needs (it doesn't have to know facts — we hand it the facts).
#   - Runs on CPU in ~30-60s per answer, on Apple Silicon MPS in ~5-15s,
#     on a CUDA GPU in ~1-3s. We auto-detect the best device below.
#
# To try a different model, change LLM_MODEL to any other instruct model
# on HF (e.g. "Qwen/Qwen2.5-0.5B-Instruct" for faster but lower quality,
# or "microsoft/Phi-3-mini-4k-instruct" for stronger but slower).
LLM_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

# Max tokens the model is allowed to generate per answer. 350 ~ 250-300
# English words, plenty for a focused HPC-docs Q&A. Cap exists because:
#   - longer answers = linearly slower generation,
#   - the model would otherwise sometimes ramble or repeat itself.
LLM_MAX_NEW_TOKENS = 350

# Sampling temperature.
#   0.0 - 0.3  → deterministic, factual; good for RAG.
#   0.7 - 1.0  → creative; bad for "answer from context" use cases.
# We pick 0.2 because we want the model to stick close to what the
# retrieved chunks say, not invent.
LLM_TEMPERATURE = 0.2

# Device selection. None ⇒ auto-detect (CUDA > MPS > CPU). Override to
# force a specific device — e.g. "cpu" for reproducibility tests.
LLM_DEVICE: str | None = None
