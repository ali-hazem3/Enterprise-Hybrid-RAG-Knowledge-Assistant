from fastapi import FastAPI, HTTPException

from app.agent import run_agent

from app.memory import (
    get_all_interactions as get_all_messages,
)

from app.ingestion import (
    get_document_registry,
)

from app.models import (
    ChatRequest,
    ChatResponse,
)

from app.logger import logger


app = FastAPI(
    title="Multi-Intent RAG + SQL Agent",
    version="1.0.0",
)


@app.get("/health")
def health():
    return {
        "status": "ok"
    }


@app.post(
    "/chat",
    response_model=ChatResponse,
)
def chat(request: ChatRequest):
    logger.info(
        "API chat request | "
        f"session={request.session_id!r}"
    )

    try:
        answer = run_agent(
            question=request.question,
            session_id=request.session_id,
        )

        return ChatResponse(
            session_id=request.session_id,
            answer=answer,
        )

    except Exception as error:
        logger.exception(
            "API chat request failed | "
            f"session={request.session_id!r} | "
            f"error={str(error)!r}"
        )

        raise HTTPException(
            status_code=500,
            detail="Agent request failed.",
        )


@app.get("/memory/{session_id}")
def get_memory(session_id: str):
    messages = get_all_messages(
        session_id=session_id,
    )

    return {
        "session_id": session_id,
        "message_count": len(messages),
        "messages": messages,
    }


@app.get("/documents")
def get_documents():
    documents = get_document_registry()

    return {
        "document_count": len(documents),
        "documents": documents,
    }