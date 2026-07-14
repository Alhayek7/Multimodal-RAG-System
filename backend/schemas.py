from typing import List, Optional

from pydantic import BaseModel


class AskRequest(BaseModel):
    question: str


class ChunkResponse(BaseModel):
    content: str
    source: Optional[str] = None
    page: Optional[int] = None
    score: Optional[float] = None


class AskResponse(BaseModel):
    question: str
    no_rag_answer: str
    rag_answer: str
    retrieved_chunks: List[ChunkResponse]
