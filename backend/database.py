"""
database.py
-----------
SQLite metadata store for the Jewellery Visual Search system.
Manages image records: filenames, paths, upload timestamps,
vector IDs, and embedding status.
"""

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

DB_PATH = Path(__file__).parent / "jewellery.db"


# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------

def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

def create_tables() -> None:
    conn = get_connection()
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS images (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                filename         TEXT    NOT NULL UNIQUE,
                image_path       TEXT    NOT NULL,
                uploaded_at      TEXT    NOT NULL DEFAULT (datetime('now', 'utc')),
                vector_id        INTEGER,
                embedding_status TEXT    NOT NULL DEFAULT 'pending'
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_images_vector_id   ON images(vector_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_images_filename     ON images(filename)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_images_uploaded_at  ON images(uploaded_at)"
        )
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Write operations
# ---------------------------------------------------------------------------

def insert_image(
    filename: str,
    image_path: str,
    vector_id: Optional[int] = None,
    embedding_status: str = "pending",
) -> int:
    """Insert a new image record and return its auto-assigned ID."""
    conn = get_connection()
    try:
        cursor = conn.execute(
            """
            INSERT INTO images (filename, image_path, uploaded_at, vector_id, embedding_status)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                filename,
                image_path,
                datetime.utcnow().isoformat(timespec="seconds"),
                vector_id,
                embedding_status,
            ),
        )
        conn.commit()
        return cursor.lastrowid
    finally:
        conn.close()


def update_vector_id(
    image_id: int, vector_id: int, status: str = "indexed"
) -> None:
    """Link a SQLite image record to its Qdrant point ID."""
    conn = get_connection()
    try:
        conn.execute(
            "UPDATE images SET vector_id = ?, embedding_status = ? WHERE id = ?",
            (vector_id, status, image_id),
        )
        conn.commit()
    finally:
        conn.close()


def delete_image_by_id(image_id: int) -> Optional[Dict]:
    """Delete a record and return the deleted row, or None if not found."""
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT * FROM images WHERE id = ?", (image_id,)
        ).fetchone()
        if row is None:
            return None
        conn.execute("DELETE FROM images WHERE id = ?", (image_id,))
        conn.commit()
        return dict(row)
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Read operations
# ---------------------------------------------------------------------------

def get_image_by_id(image_id: int) -> Optional[Dict]:
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT * FROM images WHERE id = ?", (image_id,)
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def get_images_by_ids(image_ids: List[int]) -> Dict[int, Dict]:
    """Fetch multiple rows at once; returns a dict keyed by ID."""
    if not image_ids:
        return {}
    conn = get_connection()
    try:
        placeholders = ",".join("?" * len(image_ids))
        rows = conn.execute(
            f"SELECT * FROM images WHERE id IN ({placeholders})", image_ids
        ).fetchall()
        return {row["id"]: dict(row) for row in rows}
    finally:
        conn.close()


def get_all_images(limit: int = 100, offset: int = 0) -> List[Dict]:
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT * FROM images ORDER BY uploaded_at DESC LIMIT ? OFFSET ?",
            (limit, offset),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def count_images() -> int:
    conn = get_connection()
    try:
        return conn.execute("SELECT COUNT(*) FROM images").fetchone()[0]
    finally:
        conn.close()


def image_exists(filename: str) -> bool:
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT 1 FROM images WHERE filename = ?", (filename,)
        ).fetchone()
        return row is not None
    finally:
        conn.close()
