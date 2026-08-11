import sqlite3
import numpy as np
from contextlib import contextmanager

DB_PATH = "rag.db"


def init_db(db_path: str = DB_PATH):
    conn = sqlite3.connect(db_path)

    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
            chunk_index INTEGER NOT NULL,
            text TEXT NOT NULL,
            embedding BLOB NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_chunks_document_id
        ON chunks(document_id);
        """
    )

    conn.commit()
    conn.close()


@contextmanager
def get_conn(db_path: str = DB_PATH):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")

    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def insert_document(conn, filename: str) -> int:
    cur = conn.execute(
        "INSERT INTO documents (filename) VALUES (?)",
        (filename,)
    )

    return cur.lastrowid


def insert_chunks(
    conn,
    document_id: int,
    chunks: list[str],
    embeddings: list[list[float]]
):
    rows = [
        (
            document_id,
            i,
            text,
            np.array(emb, dtype=np.float32).tobytes()
        )
        for i, (text, emb) in enumerate(zip(chunks, embeddings))
    ]

    conn.executemany(
        """
        INSERT INTO chunks (
            document_id,
            chunk_index,
            text,
            embedding
        )
        VALUES (?, ?, ?, ?)
        """,
        rows,
    )


def get_all_chunks(conn):
    

    rows = conn.execute(
        """
        SELECT
            c.id,
            c.document_id,
            c.chunk_index,
            c.text,
            c.embedding,
            d.filename
        FROM chunks c
        INNER JOIN documents d
            ON c.document_id = d.id
        ORDER BY c.document_id, c.chunk_index
        """
    ).fetchall()

    return [
        {
            "id": r["id"],
            "document_id": r["document_id"],
            "chunk_index": r["chunk_index"],
            "filename": r["filename"],
            "text": r["text"],
            "embedding": np.frombuffer(
                r["embedding"],
                dtype=np.float32
            ),
        }
        for r in rows
    ]


def list_documents(conn):

    rows = conn.execute(
        """
        SELECT
            d.id,
            d.filename,
            d.uploaded_at,
            COUNT(c.id) AS chunk_count
        FROM documents d
        LEFT JOIN chunks c
            ON c.document_id = d.id
        GROUP BY d.id
        ORDER BY d.uploaded_at DESC
        """
    ).fetchall()

    return [dict(r) for r in rows]


def delete_document(conn, document_id: int):
    conn.execute(
        "DELETE FROM documents WHERE id = ?",
        (document_id,)
    )