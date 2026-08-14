from pydantic import BaseModel


class ChatRequest(BaseModel):
    question: str
    session_id: str


class ChatResponse(BaseModel):
    session_id: str
    answer: str