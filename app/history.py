import sqlite3
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage

from app.config import Config

_HISTORY_PATH = Path(Config.DATA_DIR) / "chat_history.db"


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(_HISTORY_PATH))
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def ensure_history() -> None:
    conn = _conn()
    conn.execute(
        """CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )"""
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_messages_session_id ON messages(session_id)"
    )
    conn.commit()
    conn.close()


def load_history(session_id: str, limit: int = 10) -> list:
    conn = _conn()
    rows = conn.execute(
        "SELECT role, content FROM messages WHERE session_id = ? ORDER BY id DESC LIMIT ?",
        (session_id, limit),
    ).fetchall()
    conn.close()
    rows.reverse()
    msgs = []
    for role, content in rows:
        if role == "user":
            msgs.append(HumanMessage(content=content))
        elif role == "assistant":
            msgs.append(AIMessage(content=content))
    return msgs


def append_history(session_id: str, role: str, content: str) -> None:
    conn = _conn()
    conn.execute(
        "INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)",
        (session_id, role, content),
    )
    conn.commit()
    conn.close()


def list_sessions() -> list[dict]:
    conn = _conn()
    rows = conn.execute(
        """SELECT session_id,
                  COUNT(*) AS message_count,
                  MAX(created_at) AS last_active
           FROM messages
           GROUP BY session_id
           ORDER BY last_active DESC"""
    ).fetchall()
    conn.close()
    return [
        {
            "session_id": r[0],
            "message_count": r[1],
            "last_active": r[2],
        }
        for r in rows
    ]


def get_session_messages(session_id: str) -> list[dict]:
    conn = _conn()
    rows = conn.execute(
        "SELECT id, role, content, created_at FROM messages WHERE session_id = ? ORDER BY id ASC",
        (session_id,),
    ).fetchall()
    conn.close()
    return [
        {"id": r[0], "role": r[1], "content": r[2], "created_at": r[3]}
        for r in rows
    ]


def delete_session(session_id: str) -> int:
    conn = _conn()
    cur = conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
    conn.commit()
    deleted = cur.rowcount
    conn.close()
    return deleted
