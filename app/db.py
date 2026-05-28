import psycopg2
from psycopg2.extras import execute_values

from app.config import settings
from app.logger import get_logger

log = get_logger("db")


def get_conn():
    return psycopg2.connect(
        host=settings.db_host,
        port=settings.db_port,
        user=settings.db_user,
        password=settings.db_password,
        dbname=settings.db_name,
        sslmode="disable",
        options=f"-c search_path={settings.db_schema},public",
    )


def init_db():
    log.info("Initialising DB — host=%s db=%s schema=%s", settings.db_host, settings.db_name, settings.db_schema)
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(f"CREATE SCHEMA IF NOT EXISTS {settings.db_schema};")
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector SCHEMA public;")
                dim = settings.gemini_embedding_dim if settings.llm_provider == "gemini" else settings.openai_embedding_dim
                log.info("Using embedding dimension: %d (provider=%s)", dim, settings.llm_provider)
                cur.execute(f"""
                    CREATE TABLE IF NOT EXISTS {settings.db_schema}.document_chunks (
                        id          SERIAL PRIMARY KEY,
                        filename    TEXT NOT NULL,
                        chunk_index INTEGER NOT NULL,
                        content     TEXT NOT NULL,
                        embedding   vector({dim}),
                        created_at  TIMESTAMPTZ DEFAULT NOW()
                    );
                """)
            conn.commit()
        log.info("DB initialisation complete")
    except Exception as exc:
        log.error("DB initialisation error: %s", exc, exc_info=True)
        raise


def insert_chunks(filename: str, chunks: list[dict]):
    log.info("Inserting %d chunks for file '%s'", len(chunks), filename)
    rows = [(filename, c["chunk_index"], c["content"], c["embedding"]) for c in chunks]
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
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
            conn.commit()
        log.info("Successfully inserted %d chunks for '%s'", len(chunks), filename)
    except Exception as exc:
        log.error("Failed to insert chunks for '%s': %s", filename, exc, exc_info=True)
        raise


def similarity_search(query_embedding: list[float], top_k: int = 5) -> list[dict]:
    log.info("Running similarity search — top_k=%d", top_k)
    vec = str(query_embedding)
    try:
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
        log.info("Similarity search returned %d results", len(rows))
        return [
            {"id": r[0], "filename": r[1], "chunk_index": r[2], "content": r[3], "score": float(r[4])}
            for r in rows
        ]
    except Exception as exc:
        log.error("Similarity search failed: %s", exc, exc_info=True)
        raise
