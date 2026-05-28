"""
End-to-end orchestrator.

Runs steps 1 -> 6 in order:
    1. download  ALCF doc pages   ->  data/raw/*.json
    2. chunk     into ~800-char windows  ->  data/chunks.json
    3. embed     each chunk with SBERT (MiniLM)  ->  data/chunks.json (with vectors)
    4. index     into Milvus Lite        ->  data/milvus_alcf.db
    5. search    a sample query          ->  prints top-K hits
    6. RAG       retrieve + ask local LLM (Qwen2.5-1.5B-Instruct) ->
                 prints a grounded answer with sources

Each step is idempotent on its own (cached download files, cached
embeddings, dropped+recreated Milvus collection), so re-running this
script while iterating on one stage is cheap.

USAGE
    python main.py                       # runs the full pipeline + demo query
    python main.py "your question here"  # full pipeline, custom query
"""

from __future__ import annotations

import sys
import time
from contextlib import contextmanager

from src.step1_download import download_all
from src.step2_chunk import chunk_all
from src.step3_embed import embed_all
from src.step4_index import index_all
from src.step6_rag import rag_answer


# Per-step elapsed times collected here, summarized at the end.
_timings: list[tuple[str, float]] = []


@contextmanager
def _step(n: int, title: str):
    """Print a banner, run the step, record + print the elapsed time."""
    print(f"\n=== Step {n}: {title} " + "=" * max(1, 60 - len(title)))
    t0 = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - t0
        _timings.append((f"Step {n}: {title}", elapsed))
        print(f"  ⏱  step {n} took {elapsed:.3f}s")


def main(query: str) -> None:
    total_t0 = time.perf_counter()

    with _step(1, "download ALCF documentation pages"):
        download_all()

    with _step(2, "split pages into chunks"):
        chunk_all()

    with _step(3, "embed chunks locally (sentence-transformers, no API)"):
        embed_all()

    with _step(4, "index chunks in Milvus"):
        index_all()

    # Step 5 (similarity search) is exercised *inside* step 6's rag_answer().
    # We fold them into a single banner so the timing summary lines up with
    # what the user actually waits for: "retrieve + answer".
    with _step(6, "RAG: retrieve top-K (step 5) + answer with local LLM"):
        rag_answer(query)

    # Final summary table so you can see at a glance where the wall-clock
    # budget actually went.
    total = time.perf_counter() - total_t0
    print("\n" + "=" * 64)
    print("  TIMING SUMMARY")
    print("=" * 64)
    for label, t in _timings:
        pct = 100 * t / total if total > 0 else 0
        bar = "█" * int(pct / 2)  # 1 block per 2 %
        print(f"  {t:7.3f}s  {pct:5.1f}%  {bar:<50s}  {label}")
    print("-" * 64)
    print(f"  {total:7.3f}s  100.0%  {'█' * 50}  TOTAL")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        user_query = " ".join(sys.argv[1:])
    else:
        user_query = "How do I submit a multi-node GPU job on Polaris?"
    main(user_query)
