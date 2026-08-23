import json
import math
from contextlib import contextmanager

import psycopg2

from app.config import settings
from app.logger import get_logger

log = get_logger("db")

# Set to True during init_db() when pgvector is available on the PostgreSQL server.
# When False, embeddings are stored as JSON text and similarity is computed in Python.
_pg_has_vector: bool = False


def _strip_nul(value: str | None) -> str | None:
    if value is None:
        return None
    return value.replace("\x00", "")


# ── connection helpers ──────────────────────────────────────────────────────

@contextmanager
def get_conn():
    conn = psycopg2.connect(
        host=settings.db_host,
        port=settings.db_port,
        user=settings.db_user,
        password=settings.db_password,
        dbname=settings.db_name,
        sslmode="disable",
        options=f"-c search_path={settings.db_schema},public",
    )
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ── init ────────────────────────────────────────────────────────────────────

def _init_pg():
    global _pg_has_vector
    dim = settings.gemini_embedding_dim if settings.llm_provider == "gemini" else settings.openai_embedding_dim
    log.info("Using embedding dimension: %d (provider=%s)", dim, settings.llm_provider)
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(f"CREATE SCHEMA IF NOT EXISTS {settings.db_schema};")

            # Check if pgvector is installed on this PostgreSQL server before trying to use it
            cur.execute("SELECT COUNT(*) FROM pg_available_extensions WHERE name = 'vector';")
            _pg_has_vector = cur.fetchone()[0] > 0

            if _pg_has_vector:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector SCHEMA public;")
                log.info("pgvector extension available — using native vector similarity")
                embedding_col = f"vector({dim})"
            else:
                log.warning(
                    "pgvector is NOT installed on this PostgreSQL server. "
                    "Embeddings will be stored as JSON text and similarity will be "
                    "computed in Python. To enable native vector search, install "
                    "pgvector on the server (https://github.com/pgvector/pgvector)."
                )
                embedding_col = "TEXT"

            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {settings.db_schema}.document_chunks (
                    id          SERIAL PRIMARY KEY,
                    filename    TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    content     TEXT NOT NULL,
                    embedding   {embedding_col},
                    created_at  TIMESTAMPTZ DEFAULT NOW()
                );
            """)


def init_db():
    log.info("Initialising DB — host=%s db=%s schema=%s", settings.db_host, settings.db_name, settings.db_schema)
    try:
        _init_pg()
        log.info("DB initialisation complete")
    except Exception as exc:
        log.error("DB initialisation error: %s", exc, exc_info=True)
        raise


# ── insert ──────────────────────────────────────────────────────────────────

def _insert_chunks_pg(filename: str, chunks: list[dict]):
    safe_filename = _strip_nul(filename) or ""
    with get_conn() as conn:
        with conn.cursor() as cur:
            if _pg_has_vector:
                from psycopg2.extras import execute_values
                # Use json.dumps() for consistent vector formatting across pgvector versions
                rows = [
                    (
                        safe_filename,
                        c["chunk_index"],
                        _strip_nul(c.get("content")) or "",
                        json.dumps(c["embedding"]),
                    )
                    for c in chunks
                ]
                execute_values(
                    cur,
                    f"""
                    INSERT INTO {settings.db_schema}.document_chunks
                        (filename, chunk_index, content, embedding)
                    VALUES %s
                    """,
                    rows,
                    template="(%s, %s, %s, %s::vector)",
                )
            else:
                # pgvector not available — store embedding as JSON text
                for c in chunks:
                    emb_json = json.dumps(c["embedding"]) if c.get("embedding") is not None else None
                    cur.execute(
                        f"INSERT INTO {settings.db_schema}.document_chunks "
                        "(filename, chunk_index, content, embedding) VALUES (%s, %s, %s, %s)",
                        (safe_filename, c["chunk_index"], _strip_nul(c.get("content")) or "", emb_json),
                    )


def insert_chunks(filename: str, chunks: list[dict]):
    log.info("Inserting %d chunks for file '%s'", len(chunks), filename)
    try:
        _insert_chunks_pg(filename, chunks)
        log.info("Successfully inserted %d chunks for '%s'", len(chunks), filename)
    except Exception as exc:
        log.error("Failed to insert chunks for '%s': %s", filename, exc, exc_info=True)
        raise


# ── similarity search ────────────────────────────────────────────────────────

def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def _similarity_search_pg(query_embedding: list[float], top_k: int) -> list[dict]:
    if _pg_has_vector:
        # Use json.dumps() for consistent vector formatting across pgvector versions
        vec = json.dumps(query_embedding)
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT id, filename, chunk_index, content,
                           1 - (embedding <=> %s::vector) AS score
                    FROM {settings.db_schema}.document_chunks
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s;
                    """,
                    (vec, vec, top_k),
                )
                rows = cur.fetchall()
        return [
            {"id": r[0], "filename": r[1], "chunk_index": r[2], "content": r[3], "score": float(r[4])}
            for r in rows
        ]
    else:
        # pgvector not available — load all embeddings and rank in Python
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"SELECT id, filename, chunk_index, content, embedding "
                    f"FROM {settings.db_schema}.document_chunks WHERE embedding IS NOT NULL"
                )
                rows = cur.fetchall()
        scored = []
        for row in rows:
            try:
                emb = json.loads(row[4])
                score = _cosine_similarity(query_embedding, emb)
                scored.append({"id": row[0], "filename": row[1], "chunk_index": row[2], "content": row[3], "score": score})
            except Exception:
                pass
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]


def similarity_search(query_embedding: list[float], top_k: int = 5) -> list[dict]:
    log.info("Running similarity search — top_k=%d", top_k)
    try:
        results = _similarity_search_pg(query_embedding, top_k)
        log.info("Similarity search returned %d results", len(results))
        return results
    except Exception as exc:
        log.error("Similarity search failed: %s", exc, exc_info=True)
        raise