"""
Step 3 — turn each text chunk into a 384-dim vector with a local
SBERT model (`sentence-transformers/all-MiniLM-L6-v2`).

WHAT AN EMBEDDING IS
--------------------
An *embedding* is a fixed-length list of floats that represents the meaning
of a piece of text. The trick is that texts with similar meaning end up
near each other in this 384-dimensional space — even when they share no
words.

    "How do I submit a job on Polaris?"  ->  [0.012, -0.084, ..., 0.041]
    "Polaris PBS qsub example"           ->  [0.014, -0.079, ..., 0.038]
    "List of cafeterias at Argonne"      ->  [-0.21,  0.31,  ..., -0.17]

The first two are talking about the same thing in different words, so the
cosine angle between their vectors is small. The third is about something
else entirely, so it points in a wildly different direction. Vector search
is built on exactly this property.

WHICH MODEL & WHY
-----------------
Model:  sentence-transformers/all-MiniLM-L6-v2
Output: 384 dims, L2-normalized
Cost:   $0. Runs on CPU. No API key, no rate limits.

Why this model:
  - Small (22M params, ~80 MB on disk). First run downloads it from
    Hugging Face; subsequent runs use the local cache.
  - Fast: ~1000 sentences/sec on a single laptop CPU core.
  - Trained with contrastive loss on 1B+ sentence pairs, so it produces
    sensible retrieval embeddings out of the box.
  - The community baseline: most RAG tutorials and MTEB submissions
    compare against this exact model.

For larger corpora or higher recall, drop in any other
sentence-transformers model (`BAAI/bge-small-en-v1.5`,
`intfloat/e5-base-v2`, ...) — just update `EMBED_MODEL` and `EMBED_DIM`
in [config.py](config.py); the rest of this file is model-agnostic.

WHY WE NORMALIZE
----------------
`SentenceTransformer.encode(..., normalize_embeddings=True)` divides each
vector by its L2 norm so it has unit length. This makes cosine similarity
equivalent to a plain dot product — one fused-multiply-add per dimension
— which is what Milvus computes internally when METRIC_TYPE=COSINE.

OUTPUT
------
We update `data/chunks.json` in place so each chunk record gains an
`embedding` field. Keeping text + vector in the same file means step 4
only has to read one thing to load Milvus.

HOW TO RUN
----------
    python -m src.step3_embed
"""

from __future__ import annotations

import json
from functools import lru_cache

from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from src.config import (
    CHUNKS_PATH,
    EMBED_BATCH,
    EMBED_DIM,
    EMBED_MODEL,
)


# ---------------------------------------------------------------------------
# Model loader (cached so we don't reload the ~80MB weights twice in one run)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    """Load (and cache) the SBERT model.

    First call downloads from Hugging Face into ~/.cache/huggingface and
    takes a few seconds. Subsequent calls are instant.
    """
    print(f"  loading {EMBED_MODEL} (first run downloads ~80MB)...")
    return SentenceTransformer(EMBED_MODEL)


# ---------------------------------------------------------------------------
# Public helpers — used by both ingest (this file) and search (step 5)
# ---------------------------------------------------------------------------
def embed_texts(texts: list[str], show_progress: bool = False) -> list[list[float]]:
    """Embed an arbitrary list of strings. Returns a list of unit-length lists.

    We hand the whole list to `model.encode` and let it manage internal
    batching — that's what the library is designed for. The return type
    is plain Python lists (not numpy arrays) so we can json-serialize
    them straight into chunks.json.
    """
    model = _get_model()
    arr = model.encode(
        texts,
        batch_size=EMBED_BATCH,
        normalize_embeddings=True,     # see "WHY WE NORMALIZE" in the docstring
        convert_to_numpy=True,
        show_progress_bar=show_progress,
    )
    return arr.tolist()


def embed_query(text: str) -> list[float]:
    """Embed a single search query.

    Public helper so step 5 (search) can reuse the same model without
    re-implementing the encode/normalize dance. The model and weights are
    cached via `_get_model`, so this is fast after the first call.
    """
    return embed_texts([text])[0]


# ---------------------------------------------------------------------------
# Public entry point — embed every chunk produced by step 2
# ---------------------------------------------------------------------------
def embed_all() -> list[dict]:
    if not CHUNKS_PATH.exists():
        raise RuntimeError(
            f"{CHUNKS_PATH} not found. Run step 2 (chunk) first."
        )

    chunks: list[dict] = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))

    # Skip the embedding step entirely if every chunk already carries a
    # vector of the right dimension. Lets you re-run main.py while
    # iterating on Milvus / search code without re-encoding 100s of chunks.
    if (
        chunks
        and isinstance(chunks[0].get("embedding"), list)
        and len(chunks[0]["embedding"]) == EMBED_DIM
    ):
        print(f"  [cached] {len(chunks)} chunk(s) already have {EMBED_DIM}-d embeddings")
        return chunks

    print(f"  embedding {len(chunks)} chunk(s) with {EMBED_MODEL} (dim={EMBED_DIM})")

    # sentence-transformers handles batching internally, but we wrap with
    # tqdm by passing show_progress=True so the user sees movement on
    # corpora that take a few seconds to encode.
    vectors = embed_texts([c["text"] for c in chunks], show_progress=True)

    for chunk, vec in zip(chunks, vectors):
        chunk["embedding"] = vec

    CHUNKS_PATH.write_text(
        json.dumps(chunks, ensure_ascii=False),  # no indent: file would balloon to MBs
        encoding="utf-8",
    )
    print(f"  wrote {len(chunks)} embedded chunk(s) -> {CHUNKS_PATH}")
    return chunks


if __name__ == "__main__":
    embed_all()
