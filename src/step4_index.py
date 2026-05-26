"""
Step 4 — load chunks + embeddings into Milvus and build an HNSW index.

WHY A VECTOR DATABASE
---------------------
We *could* just keep all our embeddings in a Python list and compute
cosine similarity against every one of them on each query. For 200 ALCF
chunks that's fine. For 200 million Wikipedia chunks that's a 30-second
query.

A vector database does three things a list can't:

  1. Persistence. The index lives on disk, survives restarts, can be
     queried by other processes.
  2. Approximate Nearest Neighbour (ANN) indexes — HNSW, IVF, DiskANN —
     that turn an O(N) scan into ~O(log N) graph walk, trading a tiny bit
     of recall for huge latency wins at scale.
  3. Hybrid querying: filter by metadata ("only docs about Polaris") and
     vector-search at the same time, in one pass.

WHY MILVUS, AND WHY "LITE"
--------------------------
Milvus is the most-deployed open-source vector DB (CNCF graduated). Its
production form is a distributed system: a Proxy receiving requests, a
Query Coord scheduling them, Data/Index/Query nodes doing the work,
Pulsar/Kafka as the WAL, Object Storage (S3/MinIO) for segments, etcd
for metadata. That's a lot to spin up for a tutorial.

Milvus Lite packs all of those roles into a single Python process and
swaps the distributed object store for a local SQLite-ish file
(`data/milvus_alcf.db`). The *API is identical to server Milvus*, so the
code you write here is exactly what you'd ship — only the connection URI
changes (`milvus://...` -> `./data/milvus_alcf.db`).

WHAT THIS SCRIPT DOES
---------------------
1. Opens (or creates) the Milvus Lite database file.
2. Drops + recreates the `alcf_docs` collection so re-runs are clean.
3. Defines the schema:
       id        VARCHAR  (primary key — stable, derived from chunk index)
       url       VARCHAR  (metadata)
       title     VARCHAR  (metadata)
       text      VARCHAR  (so we can return the chunk text, not just an id)
       embedding FLOAT_VECTOR[768]   (the thing we actually search)
4. Inserts all chunks.
5. Builds an IVF_FLAT index on the `embedding` field with COSINE distance.
   (Milvus Lite only supports FLAT/IVF_FLAT/AUTOINDEX. On server Milvus
   you'd typically switch this to HNSW — see config.py for how.)
6. Loads the collection into memory (Milvus, even Lite, requires an
   explicit "load" before search — segments live on disk by default).

HOW TO RUN
----------
    python -m src.step4_index
"""

from __future__ import annotations

import json

from pymilvus import DataType, MilvusClient

from src.config import (
    CHUNKS_PATH,
    COLLECTION_NAME,
    EMBED_DIM,
    INDEX_PARAMS,
    INDEX_TYPE,
    METRIC_TYPE,
    MILVUS_DB,
)


# ---------------------------------------------------------------------------
# Schema construction
# ---------------------------------------------------------------------------
def _build_schema(client: MilvusClient):
    """Define the columns of the alcf_docs collection.

    Milvus is *schema-first*, like a SQL DB. You declare field names,
    types, and max lengths up front and the server validates every insert
    against the schema. This is why a vector DB feels less like
    Elasticsearch (schema-less) and more like Postgres (with a special
    FLOAT_VECTOR type).
    """
    schema = client.create_schema(
        auto_id=False,          # we supply our own IDs (see step 2)
        enable_dynamic_field=False,
    )
    # Primary key — VARCHAR not INT, because our IDs are slugs.
    schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=128)
    # Metadata fields. `max_length` is required for VARCHAR; pick generously.
    schema.add_field("url",   DataType.VARCHAR, max_length=512)
    schema.add_field("title", DataType.VARCHAR, max_length=512)
    # 4096 covers our 800-char chunks (+ overlap) with room to spare.
    schema.add_field("text",  DataType.VARCHAR, max_length=4096)
    # The vector field — this is the only one that gets a vector index.
    schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=EMBED_DIM)
    return schema


def _build_index_params(client: MilvusClient):
    """Describe how to index the `embedding` field.

    The choice of index_type + metric_type + params is the single biggest
    knob on retrieval quality vs. speed vs. memory. See config.py for the
    rationale behind picking HNSW with COSINE distance.
    """
    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="embedding",
        index_type=INDEX_TYPE,    # IVF_FLAT in Lite mode, HNSW in server
        metric_type=METRIC_TYPE,  # COSINE
        params=INDEX_PARAMS,      # IVF_FLAT: {"nlist": 16}
    )
    return index_params


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def index_all() -> MilvusClient:
    """Wipe + rebuild the collection from data/chunks.json. Returns the client."""
    if not CHUNKS_PATH.exists():
        raise RuntimeError(
            f"{CHUNKS_PATH} not found. Run step 3 (embed) first."
        )

    chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    if not chunks or "embedding" not in chunks[0]:
        raise RuntimeError("Chunks have no `embedding` field. Re-run step 3.")

    # Passing a file path opens Milvus *Lite* (embedded mode). Passing
    # something like "http://localhost:19530" would talk to a server.
    MILVUS_DB.parent.mkdir(parents=True, exist_ok=True)
    client = MilvusClient(uri=str(MILVUS_DB))

    # Recreate the collection from scratch so re-running is idempotent.
    # In production you'd `upsert` instead — for a tutorial, drop+recreate
    # makes the demo's behavior obvious.
    if client.has_collection(COLLECTION_NAME):
        client.drop_collection(COLLECTION_NAME)

    client.create_collection(
        collection_name=COLLECTION_NAME,
        schema=_build_schema(client),
        index_params=_build_index_params(client),
    )

    # Insert. Milvus accepts a list of dicts; field names must match the
    # schema. We send everything in one call because our corpus is tiny;
    # for millions of vectors you'd chunk this into batches of 10k-50k.
    #
    # Behind the scenes Milvus appends the rows to an in-memory growing
    # segment, persists them, and (on the high-level MilvusClient) seals
    # the segment when it crosses a size threshold. The low-level
    # Collection API exposes an explicit `flush()`; the modern
    # MilvusClient handles it for you.
    client.insert(collection_name=COLLECTION_NAME, data=chunks)

    # Even with an index defined at create_collection time, you must
    # explicitly `load` the collection before searching it. Milvus keeps
    # cold collections off the heap; `load` is the moment it warms the
    # ANN index into RAM.
    client.load_collection(COLLECTION_NAME)

    stats = client.get_collection_stats(COLLECTION_NAME)
    print(
        f"  collection '{COLLECTION_NAME}' ready"
        f"  ({stats.get('row_count', len(chunks))} rows, dim={EMBED_DIM},"
        f" index={INDEX_TYPE}/{METRIC_TYPE})"
    )
    return client


if __name__ == "__main__":
    index_all()
