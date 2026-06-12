import sqlite3
from datetime import datetime
from pathlib import Path
from typing import List, Dict
import config
from utils.logger import get_logger

logger = get_logger(__name__)

Path(config.SQLITE_DB_PATH).parent.mkdir(parents=True, exist_ok=True)

def _get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(config.SQLITE_DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def initialise_db() -> None:
    conn = _get_connection()
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS conversation_history (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id  TEXT     NOT NULL,
                role        TEXT     NOT NULL CHECK(role IN ('user', 'assistant')),
                content     TEXT     NOT NULL,
                timestamp   DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_session_timestamp
            ON conversation_history (session_id, timestamp)
        """)
        conn.commit()
        logger.info("SQLite DB initialised (conversation_history table ready).")
    finally:
        conn.close()

def save_turn(session_id: str, role: str, content: str) -> None:
    conn = _get_connection()
    try:
        conn.execute(
            "INSERT INTO conversation_history (session_id, role, content, timestamp) "
            "VALUES (?, ?, ?, ?)",
            (session_id, role, content, datetime.utcnow().isoformat()),
        )
        conn.commit()
        logger.debug(f"Saved [{role}] turn for session '{session_id}'.")
    finally:
        conn.close()

def load_history(session_id: str, limit: int = 10) -> List[Dict[str, str]]:
    conn = _get_connection()
    try:
        cursor = conn.execute(
            """
            SELECT role, content FROM conversation_history
            WHERE session_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (session_id, limit),
        )
        rows = cursor.fetchall()
        history = [{"role": row["role"], "content": row["content"]} for row in reversed(rows)]
        logger.info(f"Loaded {len(history)} turns for session '{session_id}'.")
        return history
    finally:
        conn.close()

def get_session_summary(session_id: str) -> str:
    turns = load_history(session_id, limit=10)
    if not turns:
        return "No previous conversation history."

    lines = [f"{t['role'].capitalize()}: {t['content']}" for t in turns]
    return "\n".join(lines)

def count_turns(session_id: str) -> int:
    conn = _get_connection()
    try:
        cursor = conn.execute(
            "SELECT COUNT(*) FROM conversation_history WHERE session_id = ?",
            (session_id,),
        )
        return cursor.fetchone()[0]
    finally:
        conn.close()

def clear_session(session_id: str) -> None:
    conn = _get_connection()
    try:
        conn.execute(
            "DELETE FROM conversation_history WHERE session_id = ?",
            (session_id,),
        )
        conn.commit()
        logger.info(f"Cleared history for session '{session_id}'.")
    finally:
        conn.close()

initialise_db()
