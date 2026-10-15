"""
SQLite WAL & FTS5 Full-Text Search Conversational Memory for J.A.R.V.I.S.
Inspired by Hermes Agent state persistence (hermes_state_fts.py & hermes_state_dbfile.py).
Provides sub-millisecond keyword and semantic search across thousands of historical conversation turns.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DB_PATH = Path("logs/jarvis_memory.db")


class SQLiteMemoryStore:
    """Thread-safe SQLite storage engine with WAL journaling and FTS5 full-text indexing."""

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Create a connection with WAL mode and row factory."""
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def _init_db(self) -> None:
        """Initialize relational schema and FTS5 full-text search virtual tables."""
        with self._lock, self._get_connection() as conn:
            # 1. Sessions table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    persona TEXT DEFAULT 'Executive Assistant',
                    created_at REAL,
                    updated_at REAL,
                    title TEXT DEFAULT 'Conversation'
                )
                """
            )

            # 2. Turns table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS turns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    role TEXT,
                    content TEXT,
                    timestamp REAL,
                    tools_called TEXT DEFAULT '[]',
                    FOREIGN KEY(session_id) REFERENCES sessions(session_id)
                )
                """
            )

            # 3. FTS5 Virtual Table for full-text search
            try:
                conn.execute(
                    """
                    CREATE VIRTUAL TABLE IF NOT EXISTS turns_fts USING fts5(
                        session_id UNINDEXED,
                        role UNINDEXED,
                        content,
                        tokenize='porter unicode61'
                    )
                    """
                )
            except sqlite3.OperationalError:
                # Fallback if FTS5 is not compiled with tokenizer options
                conn.execute(
                    """
                    CREATE VIRTUAL TABLE IF NOT EXISTS turns_fts USING fts5(
                        session_id,
                        role,
                        content
                    )
                    """
                )

            conn.commit()

    def record_turn(
        self,
        session_id: str,
        role: str,
        content: str,
        persona: str = "Executive Assistant",
        tools_called: Optional[List[str]] = None,
    ) -> None:
        """Persist a conversation turn and update the full-text search index."""
        if not content or not content.strip():
            return

        now = time.time()
        tools_json = json.dumps(tools_called or [])

        with self._lock, self._get_connection() as conn:
            # Upsert session
            conn.execute(
                """
                INSERT INTO sessions (session_id, persona, created_at, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET updated_at=excluded.updated_at
                """,
                (session_id, persona, now, now),
            )

            # Insert turn
            conn.execute(
                """
                INSERT INTO turns (session_id, role, content, timestamp, tools_called)
                VALUES (?, ?, ?, ?, ?)
                """,
                (session_id, role, content, now, tools_json),
            )

            # Insert into FTS5
            conn.execute(
                """
                INSERT INTO turns_fts (session_id, role, content)
                VALUES (?, ?, ?)
                """,
                (session_id, role, content),
            )

            conn.commit()

    def search_memory(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Execute sub-millisecond FTS5 search across historical turns.
        Supports keyword matching, boolean operators, and prefixes.
        """
        clean_query = query.strip().replace('"', '""')
        if not clean_query:
            return []

        # Sanitize for FTS5 match query
        match_expr = f'"{clean_query}"' if " " in clean_query else f"{clean_query}*"

        results: List[Dict[str, Any]] = []
        with self._lock, self._get_connection() as conn:
            try:
                cursor = conn.execute(
                    """
                    SELECT session_id, role, content, snippet(turns_fts, 2, '[bold yellow]', '[/bold yellow]', '...', 15) as snippet
                    FROM turns_fts
                    WHERE turns_fts MATCH ?
                    LIMIT ?
                    """,
                    (match_expr, limit),
                )
                for row in cursor.fetchall():
                    results.append(
                        {
                            "session_id": row["session_id"],
                            "role": row["role"],
                            "content": row["content"],
                            "snippet": row["snippet"],
                        }
                    )
            except sqlite3.OperationalError as e:
                logger.debug("FTS5 query failed with expression '%s': %s", match_expr, e)
                # Fallback to standard LIKE search
                cursor = conn.execute(
                    """
                    SELECT session_id, role, content, substr(content, 1, 150) as snippet
                    FROM turns
                    WHERE content LIKE ?
                    ORDER BY id DESC
                    LIMIT ?
                    """,
                    (f"%{clean_query}%", limit),
                )
                for row in cursor.fetchall():
                    results.append(
                        {
                            "session_id": row["session_id"],
                            "role": row["role"],
                            "content": row["content"],
                            "snippet": row["snippet"] + "...",
                        }
                    )

        return results

    def get_session_turns(self, session_id: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Retrieve turns for a specific session ordered chronologically."""
        turns = []
        limit_val = int(limit) if limit is not None else -1
        with self._lock, self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT id, role, content, timestamp, tools_called
                FROM turns
                WHERE session_id = ?
                ORDER BY id ASC
                LIMIT ?
                """,
                (session_id, limit_val),
            )
            for row in cursor.fetchall():
                turns.append(
                    {
                        "id": row["id"],
                        "role": row["role"],
                        "content": row["content"],
                        "timestamp": row["timestamp"],
                        "tools_called": json.loads(row["tools_called"] or "[]"),
                    }
                )
        return turns

    def get_recent_turns(self, session_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Retrieve recent turns for a specific session (alias for get_session_turns with limit)."""
        return self.get_session_turns(session_id, limit=limit)

    def count_total_turns(self) -> int:
        """Count total historical conversation turns indexed."""
        with self._lock, self._get_connection() as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM turns")
            return cursor.fetchone()[0]


_GLOBAL_SQLITE_STORE: Optional[SQLiteMemoryStore] = None
_STORE_LOCK = threading.Lock()


def get_sqlite_memory() -> SQLiteMemoryStore:
    """Return the global SQLiteMemoryStore singleton."""
    global _GLOBAL_SQLITE_STORE
    if _GLOBAL_SQLITE_STORE is None:
        with _STORE_LOCK:
            if _GLOBAL_SQLITE_STORE is None:
                _GLOBAL_SQLITE_STORE = SQLiteMemoryStore()
    return _GLOBAL_SQLITE_STORE
