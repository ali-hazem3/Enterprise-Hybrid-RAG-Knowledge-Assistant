import sqlite3
from datetime import datetime
from pathlib import Path


MEMORY_DB_PATH = Path(
    "storage/app.db"
)


def initialize_memory():
    MEMORY_DB_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    connection = sqlite3.connect(
        MEMORY_DB_PATH
    )

    cursor = connection.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            intent TEXT,
            created_at TEXT NOT NULL
        )
        """
    )

    cursor.execute(
        """
        PRAGMA table_info(messages)
        """
    )

    existing_columns = [
        row[1]
        for row in cursor.fetchall()
    ]

    if "intent" not in existing_columns:
        cursor.execute(
            """
            ALTER TABLE messages
            ADD COLUMN intent TEXT
            """
        )

    connection.commit()
    connection.close()


def save_message(
    session_id: str,
    role: str,
    content: str,
    intent: str | None = None,
):
    initialize_memory()

    connection = sqlite3.connect(
        MEMORY_DB_PATH
    )

    cursor = connection.cursor()

    cursor.execute(
        """
        INSERT INTO messages (
            session_id,
            role,
            content,
            intent,
            created_at
        )
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            session_id,
            role,
            content,
            intent,
            datetime.now().isoformat(),
        ),
    )

    connection.commit()
    connection.close()


def get_conversation_history(
    session_id: str,
    limit: int = 10,
):
    initialize_memory()

    connection = sqlite3.connect(
        MEMORY_DB_PATH
    )

    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT
            role,
            content
        FROM messages
        WHERE session_id = ?
        ORDER BY id DESC
        LIMIT ?
        """,
        (
            session_id,
            limit,
        ),
    )

    rows = cursor.fetchall()

    connection.close()

    rows.reverse()

    return [
        {
            "role": role,
            "content": content,
        }
        for role, content in rows
    ]


def get_all_messages(
    session_id: str,
):
    initialize_memory()

    connection = sqlite3.connect(
        MEMORY_DB_PATH
    )

    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT
            id,
            session_id,
            role,
            content,
            intent,
            created_at
        FROM messages
        WHERE session_id = ?
        ORDER BY id ASC
        """,
        (session_id,),
    )

    rows = cursor.fetchall()

    connection.close()

    return [
        {
            "id": row[0],
            "session_id": row[1],
            "role": row[2],
            "content": row[3],
            "intent": row[4],
            "created_at": row[5],
        }
        for row in rows
    ]