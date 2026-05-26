"""
Step 2 — split the cleaned pages into chunks.

WHY WE CHUNK
------------
Two hard constraints push us to split documents into smaller pieces:

1. Model input limit. `all-MiniLM-L6-v2` truncates inputs at 256 tokens
   (~1000 characters of English prose). A long ALCF page like "Running
   Jobs" is tens of KB — if we embedded the whole page, everything past
   the first ~1000 characters would be silently dropped.

2. Retrieval granularity. Even if a giant document fit, one vector for
   the whole page is a *blurry average* of every topic on it. A query
   like "how do I request 4 GPUs on Polaris" should hit the specific
   paragraph about `-l select=…:ngpus=4`, not a vector that smears that
   detail across an entire 5000-word page.

So we split each page into ~800-character windows that overlap by
~120 characters. Overlap matters: without it, a sentence that explains
"qsub" at the end of one chunk and gives the example at the start of the
next chunk would never appear together in any single vector.

CHUNKING STRATEGY
-----------------
We use a *recursive character splitter*: try to split on the largest
semantic boundary first (blank line = paragraph), then sentence, then
word, then any character. This way chunks land on natural breakpoints
when possible and we only fall through to ugly mid-word splits as a last
resort. It's the same algorithm LangChain's `RecursiveCharacterTextSplitter`
uses, written from scratch here so you can see how it works.

OUTPUT
------
Writes `data/chunks.json` — a flat list of records:
    {"id": "polaris__running-jobs__0007",
     "url": "...", "title": "...", "text": "..."}
Each record will become one row in the Milvus collection.

HOW TO RUN
----------
    python -m src.step2_chunk
"""

from __future__ import annotations

import json
from pathlib import Path

from src.config import CHUNK_OVERLAP, CHUNK_SIZE, CHUNKS_PATH, RAW_DIR


# ---------------------------------------------------------------------------
# Splitter
# ---------------------------------------------------------------------------
# Ordered list of "where would a human prefer to break this text?":
#   paragraph > line > sentence > clause > word > char.
# We walk down this list until we find a separator that produces pieces
# small enough to fit in CHUNK_SIZE.
_SEPARATORS: list[str] = ["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""]


def _split_recursive(text: str, size: int, separators: list[str]) -> list[str]:
    """Split `text` into pieces of at most `size` characters.

    Tries each separator in order. The first one that lets us produce pieces
    of acceptable size wins; pieces still too large are split again with the
    *remaining* (finer) separators.
    """
    if len(text) <= size:
        return [text]

    for i, sep in enumerate(separators):
        if sep == "":
            # Last-resort: just hard-cut every `size` characters.
            return [text[j : j + size] for j in range(0, len(text), size)]

        # Splitting on `sep` drops it from the output; re-attach it (except
        # for the trailing piece) so we preserve the original punctuation /
        # whitespace.
        parts = text.split(sep)
        parts = [p + sep for p in parts[:-1]] + [parts[-1]]

        # If the chosen separator didn't actually help (e.g. no "\n\n" in
        # the page), move on to the next finer one.
        if len(parts) == 1:
            continue

        # Greedy pack: glue consecutive parts together until adding the next
        # one would blow `size`. Any part that is itself too big gets
        # recursively re-split with the remaining separators.
        packed: list[str] = []
        buf = ""
        for part in parts:
            if len(part) > size:
                if buf:
                    packed.append(buf)
                    buf = ""
                packed.extend(_split_recursive(part, size, separators[i + 1 :]))
                continue
            if len(buf) + len(part) <= size:
                buf += part
            else:
                if buf:
                    packed.append(buf)
                buf = part
        if buf:
            packed.append(buf)
        return packed

    return [text]  # unreachable, the "" separator always matches


def _add_overlap(chunks: list[str], overlap: int) -> list[str]:
    """Prepend the last `overlap` chars of chunk N-1 to chunk N.

    This is what allows a fact that straddles two chunks to still appear
    intact in *at least one* embedded vector.
    """
    if overlap <= 0 or len(chunks) <= 1:
        return chunks
    out = [chunks[0]]
    for prev, curr in zip(chunks, chunks[1:]):
        tail = prev[-overlap:]
        out.append(tail + curr)
    return out


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def chunk_all() -> list[dict]:
    """Read every page JSON in `data/raw/`, chunk it, write chunks.json."""
    pages = sorted(RAW_DIR.glob("*.json"))
    if not pages:
        raise RuntimeError(
            f"No pages found in {RAW_DIR}. Run step 1 (download) first."
        )

    all_chunks: list[dict] = []
    for page_path in pages:
        page = json.loads(page_path.read_text(encoding="utf-8"))
        pieces = _split_recursive(page["text"], CHUNK_SIZE, _SEPARATORS)
        pieces = _add_overlap(pieces, CHUNK_OVERLAP)

        # `id` is the primary key in Milvus. It's stable as long as the
        # source pages don't change, which makes re-ingest debuggable.
        stem = page_path.stem
        for idx, piece in enumerate(pieces):
            all_chunks.append(
                {
                    "id": f"{stem}__{idx:04d}",
                    "url": page["url"],
                    "title": page["title"],
                    "text": piece.strip(),
                }
            )

    CHUNKS_PATH.parent.mkdir(parents=True, exist_ok=True)
    CHUNKS_PATH.write_text(
        json.dumps(all_chunks, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    avg = sum(len(c["text"]) for c in all_chunks) / max(len(all_chunks), 1)
    print(
        f"  produced {len(all_chunks)} chunk(s) from {len(pages)} page(s)"
        f"  (avg {avg:.0f} chars/chunk) -> {CHUNKS_PATH}"
    )
    return all_chunks


if __name__ == "__main__":
    chunk_all()
