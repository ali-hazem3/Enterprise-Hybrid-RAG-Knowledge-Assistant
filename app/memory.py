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
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            original_user_question TEXT NOT NULL,
            contextualized_question TEXT NOT NULL,
            selected_route TEXT NOT NULL,
            assistant_response TEXT NOT NULL,
            timestamp TEXT NOT NULL
        )
        """
    )

    connection.commit()
    connection.close()


def save_interaction(
    session_id: str,
    original_user_question: str,
    contextualized_question: str,
    selected_route: str,
    assistant_response: str,
):
    initialize_memory()

    connection = sqlite3.connect(
        MEMORY_DB_PATH
    )

    cursor = connection.cursor()

    cursor.execute(
        """
        INSERT INTO conversations (
            session_id,
            original_user_question,
            contextualized_question,
            selected_route,
            assistant_response,
            timestamp
        )
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            session_id,
            original_user_question,
            contextualized_question,
            selected_route,
            assistant_response,
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
            original_user_question,
            contextualized_question,
            assistant_response
        FROM conversations
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

    history = []

    for row in rows:
        original_question = row[0]
        contextualized_question = row[1]
        assistant_response = row[2]

        history.append(
            {
                "role": "user",
                "content": original_question,
                "contextualized_question": contextualized_question,
            }
        )

        history.append(
            {
                "role": "assistant",
                "content": assistant_response,
            }
        )

    return history


def get_all_interactions(
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
            original_user_question,
            contextualized_question,
            selected_route,
            assistant_response,
            timestamp
        FROM conversations
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
            "original_user_question": row[2],
            "contextualized_question": row[3],
            "selected_route": row[4],
            "assistant_response": row[5],
            "timestamp": row[6],
        }
        for row in rows
    ]