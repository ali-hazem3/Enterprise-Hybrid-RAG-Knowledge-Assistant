from pathlib import Path
import hashlib
import sqlite3
from datetime import datetime

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.rag import embeddings


VECTOR_STORE_PATH = Path("storage/vector_store")
REGISTRY_DB_PATH = Path("storage/app.db")


def load_text_file(file_path: str):
    path = Path(file_path)

    text = path.read_text(encoding="utf-8")

    document = Document(
        page_content=text,
        metadata={
            "source": path.name,
        },
    )

    return [document]


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
    )

    return splitter.split_documents(documents)


def initialize_registry():
    REGISTRY_DB_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    connection = sqlite3.connect(
        REGISTRY_DB_PATH
    )

    cursor = connection.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            file_hash TEXT NOT NULL,
            version INTEGER NOT NULL,
            active INTEGER NOT NULL DEFAULT 1,
            ingested_at TEXT NOT NULL
        )
        """
    )

    connection.commit()
    connection.close()


def calculate_file_hash(file_path: str):
    sha256 = hashlib.sha256()

    with open(file_path, "rb") as file:
        while chunk := file.read(8192):
            sha256.update(chunk)

    return sha256.hexdigest()


def get_active_document(filename: str):
    initialize_registry()

    connection = sqlite3.connect(
        REGISTRY_DB_PATH
    )

    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT id, file_hash, version
        FROM documents
        WHERE filename = ?
        AND active = 1
        """,
        (filename,),
    )

    result = cursor.fetchone()

    connection.close()

    return result


def create_vector_store(file_path: str):
    initialize_registry()

    path = Path(file_path)
    filename = path.name

    current_hash = calculate_file_hash(
        file_path
    )

    active_document = get_active_document(
        filename
    )

    if active_document:
        document_id, stored_hash, stored_version = (
            active_document
        )

        if stored_hash == current_hash:
            print(
                f"{filename} is already ingested "
                f"and has not changed."
            )
            return 0

        new_version = stored_version + 1

    else:
        new_version = 1

    documents = load_text_file(
        file_path
    )

    chunks = split_documents(
        documents
    )

    VECTOR_STORE_PATH.mkdir(
        parents=True,
        exist_ok=True,
    )

    vector_store = Chroma(
        collection_name="pulsefit_handbook",
        embedding_function=embeddings,
        persist_directory=str(
            VECTOR_STORE_PATH
        ),
    )

    if active_document:
        old_chunks = vector_store.get(
            where={
                "source": filename
            }
        )

        if old_chunks["ids"]:
            vector_store.delete(
                ids=old_chunks["ids"]
            )

    for chunk in chunks:
        chunk.metadata["source"] = filename
        chunk.metadata["version"] = new_version
        chunk.metadata["file_hash"] = current_hash

    vector_store.add_documents(
        chunks
    )

    connection = sqlite3.connect(
        REGISTRY_DB_PATH
    )

    cursor = connection.cursor()

    if active_document:
        cursor.execute(
            """
            UPDATE documents
            SET active = 0
            WHERE filename = ?
            AND active = 1
            """,
            (filename,),
        )

    cursor.execute(
        """
        INSERT INTO documents (
            filename,
            file_hash,
            version,
            active,
            ingested_at
        )
        VALUES (?, ?, ?, 1, ?)
        """,
        (
            filename,
            current_hash,
            new_version,
            datetime.now().isoformat(),
        ),
    )

    connection.commit()
    connection.close()

    return len(chunks)


def get_document_registry():
    initialize_registry()

    connection = sqlite3.connect(
        REGISTRY_DB_PATH
    )

    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT
            id,
            filename,
            file_hash,
            version,
            active,
            ingested_at
        FROM documents
        ORDER BY filename, version
        """
    )

    rows = cursor.fetchall()

    connection.close()

    return [
        {
            "id": row[0],
            "filename": row[1],
            "file_hash": row[2],
            "version": row[3],
            "active": bool(row[4]),
            "ingested_at": row[5],
        }
        for row in rows
    ]


if __name__ == "__main__":
    chunk_count = create_vector_store(
        "data/PULSEFIT_HANDBOOK.txt"
    )

    if chunk_count > 0:
        print(
            f"Ingested {chunk_count} chunks."
        )