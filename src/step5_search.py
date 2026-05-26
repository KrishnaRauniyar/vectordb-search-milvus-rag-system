"""
Step 5 — embed a user query and ask Milvus for the top-K nearest chunks.

WHAT "SIMILARITY SEARCH" ACTUALLY DOES
--------------------------------------
1. The query string is embedded with the *same* model used at ingest
   (`all-MiniLM-L6-v2`) so the query vector and the chunk vectors live in
   the same 384-dim space. Using a different model on either side would
   make distances meaningless.
2. That single 768-dim vector is handed to Milvus along with the metric
   (COSINE) and the index search-time knob (`ef`).
3. Milvus walks the HNSW graph: start at an entry node, greedily hop to
   whichever neighbor is closer to the query, keep a candidate list of
   size `ef`, stop when no neighbor improves it. Cost is ~O(log N * ef)
   distance computations instead of O(N).
4. The K best candidates are returned, each with:
       - `id`        the chunk's primary key
       - `distance`  the similarity score (1.0 = identical, -1.0 = opposite)
       - `entity`    the metadata fields we requested (text, url, title)

COSINE vs OTHER METRICS
-----------------------
- COSINE   — angle between vectors. Magnitude-invariant. Our pick.
- IP       — inner product. Identical to COSINE *when both vectors are
             L2-normalized* (which ours are). Slightly cheaper to compute
             but conceptually less intuitive.
- L2       — straight-line Euclidean distance. Sensitive to vector
             magnitude. Wrong for sentence embeddings without
             normalization.

INTERPRETING THE SCORES
-----------------------
With COSINE on normalized vectors, Milvus returns *similarity* in [-1, 1]
where 1 means "same direction" (most similar). A typical good match for
ALCF queries lands around 0.55-0.75; below ~0.35 you're scraping the
bottom of the barrel and probably need a query rewrite.

HOW TO RUN
----------
    python -m src.step5_search "how do I request 4 GPUs on Polaris?"

Or, if no query is given, a small default question is used so you can
just run the script and see something happen.
"""

from __future__ import annotations

import sys
import textwrap
import time

from pymilvus import MilvusClient

from src.config import (
    COLLECTION_NAME,
    METRIC_TYPE,
    MILVUS_DB,
    SEARCH_PARAMS,
    TOP_K,
)
from src.step3_embed import embed_query


# ---------------------------------------------------------------------------
# Pretty printer — keeps the demo output readable
# ---------------------------------------------------------------------------
def _print_hit(rank: int, hit: dict) -> None:
    entity = hit.get("entity", {})
    # COSINE in Milvus is a similarity (higher = better), not a distance,
    # but the field is still called `distance` in the response — call it
    # "score" in the output so the user isn't confused.
    score = hit.get("distance", 0.0)

    title = entity.get("title", "")
    url   = entity.get("url", "")
    text  = entity.get("text", "")

    print(f"\n  #{rank}  score={score:+.4f}  [{title}]")
    print(f"        {url}")
    # Show the first ~280 chars so the console doesn't drown.
    preview = textwrap.shorten(text.replace("\n", " "), width=280, placeholder=" ...")
    print(f"        {preview}")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def search(query: str, top_k: int = TOP_K) -> list[dict]:
    """Embed `query`, hit Milvus, print + return the hits."""
    if not MILVUS_DB.exists():
        raise RuntimeError(
            f"{MILVUS_DB} not found. Run step 4 (index) first."
        )

    print(f"\n  query: {query!r}")

    # Time the embed and the search separately so it's obvious where the
    # wall-clock budget actually goes. (Spoiler: in a real RAG system,
    # the network round-trip to the embedding API dominates.)
    t0 = time.perf_counter()
    query_vector = embed_query(query)
    t_embed = time.perf_counter() - t0

    client = MilvusClient(uri=str(MILVUS_DB))
    # On Milvus Lite the file is opened fresh per process, so make sure
    # the collection is loaded into memory before we search it.
    client.load_collection(COLLECTION_NAME)

    t0 = time.perf_counter()
    results = client.search(
        collection_name=COLLECTION_NAME,
        data=[query_vector],          # search expects a *list* of queries
        limit=top_k,
        search_params={
            "metric_type": METRIC_TYPE,
            "params": SEARCH_PARAMS,  # {"ef": 64} for HNSW
        },
        # Tell Milvus which metadata fields to return alongside id+distance.
        # Without this we'd only get back IDs and have to re-join elsewhere.
        output_fields=["text", "url", "title"],
    )
    t_search = time.perf_counter() - t0

    # `results` is a list-of-lists (one inner list per query vector).
    hits = results[0]

    print(
        f"  embedded in {t_embed * 1000:6.1f} ms  |"
        f"  searched {t_search * 1000:6.1f} ms"
        f"  |  {len(hits)} hit(s)"
    )
    for rank, hit in enumerate(hits, start=1):
        _print_hit(rank, hit)

    return hits


if __name__ == "__main__":
    # Pull the query off the CLI; fall back to a sensible default so
    # `python -m src.step5_search` works with zero arguments.
    if len(sys.argv) > 1:
        user_query = " ".join(sys.argv[1:])
    else:
        user_query = "How do I submit a multi-node GPU job on Polaris?"
    search(user_query)
