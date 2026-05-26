"""
Step 1 — download ALCF user-documentation pages.

WHY THIS STEP EXISTS
--------------------
A vector database is only useful if you have something to put *into* it.
For this tutorial that "something" is the ALCF (Argonne Leadership
Computing Facility) user docs — the guides that describe how to run jobs
on Polaris and Aurora, transfer data with Globus, etc.

We could ingest any text corpus; HPC docs are a fun choice because:
  - the vocabulary is technical and full of acronyms (PBS, qsub, NVMe),
    which makes lexical (keyword) search struggle and showcases the value
    of semantic search,
  - the pages are short and well-structured (MkDocs site).

WHAT THIS SCRIPT DOES
---------------------
1. Iterates over the seed URLs declared in `config.SEED_URLS`.
2. Politely (with a User-Agent + sleep) downloads each page.
3. Uses BeautifulSoup to strip nav, footer, and code-block chrome so we
   keep only the human-readable article body.
4. Writes one JSON file per page to `data/raw/` with the cleaned text,
   the title, and the source URL — those three fields become the
   per-vector metadata in Milvus later.

HOW TO RUN
----------
    python -m src.step1_download
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

from src.config import (
    CRAWL_DELAY_S,
    HTTP_TIMEOUT,
    RAW_DIR,
    SEED_URLS,
    USER_AGENT,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _slugify(url: str) -> str:
    """Turn a URL into a filesystem-safe file stem.

    docs.alcf.anl.gov/polaris/running-jobs/  ->  polaris__running-jobs
    Keeping the path in the filename makes the raw/ directory self-describing
    when you `ls` it later.
    """
    path = url.replace("https://docs.alcf.anl.gov/", "").strip("/")
    return re.sub(r"[^a-zA-Z0-9]+", "__", path) or "index"


def _fetch(url: str) -> str:
    """GET one page, return raw HTML.

    Raises on non-2xx so a broken seed URL is loud, not silent — we'd rather
    fix the URL list than ship a half-empty index.
    """
    response = requests.get(
        url,
        headers={"User-Agent": USER_AGENT},
        timeout=HTTP_TIMEOUT,
    )
    response.raise_for_status()
    return response.text


def _extract(html: str) -> tuple[str, str]:
    """Pull (title, body_text) out of MkDocs-rendered HTML.

    MkDocs Material (which ALCF uses) wraps the article in
    <article class="md-content__inner">. We grab that and drop nav-y bits
    like the "edit on github" link and any anchor symbols (¶).
    """
    soup = BeautifulSoup(html, "lxml")

    # Title: prefer the H1 in the article; fall back to <title>.
    title_tag = soup.find("h1") or soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else "Untitled"

    # Body: the MkDocs article container if present, else the whole page.
    article = soup.find("article") or soup.body or soup

    # Remove things that add noise to embeddings:
    #   - nav/header/footer chrome,
    #   - script/style blocks,
    #   - the little ¶ anchor links MkDocs injects next to every heading.
    for selector in ["nav", "header", "footer", "script", "style", ".headerlink"]:
        for tag in article.select(selector):
            tag.decompose()

    text = article.get_text(separator="\n", strip=True)

    # Collapse runs of blank lines so chunking later doesn't waste tokens
    # on whitespace.
    text = re.sub(r"\n{3,}", "\n\n", text)
    return title, text


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def download_all() -> list[Path]:
    """Download every seed URL, return the list of files written.

    Idempotent: if `data/raw/<slug>.json` already exists we skip the fetch.
    That way re-running `main.py` while iterating on chunking/embedding
    code doesn't re-hit the network or burn through anyone's patience.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for url in SEED_URLS:
        out_path = RAW_DIR / f"{_slugify(url)}.json"
        if out_path.exists():
            print(f"  [cached] {url}")
            written.append(out_path)
            continue

        print(f"  [fetch ] {url}")
        try:
            html = _fetch(url)
            title, body = _extract(html)
        except Exception as exc:  # noqa: BLE001 — surface and continue
            # A single broken page shouldn't kill the whole pipeline.
            print(f"    skipped ({exc!s})")
            continue

        out_path.write_text(
            json.dumps(
                {"url": url, "title": title, "text": body},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        written.append(out_path)

        # Be polite — Argonne's docs server is not a CDN.
        time.sleep(CRAWL_DELAY_S)

    print(f"  downloaded {len(written)} page(s) -> {RAW_DIR}")
    return written


if __name__ == "__main__":
    download_all()
