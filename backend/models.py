"""
models.py — Pydantic data models for all API requests and responses.
FastAPI uses these for automatic validation and OpenAPI docs.
"""

from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime


# ── Auth ───────────────────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=30)
    password: str = Field(..., min_length=6)
    full_name: Optional[str] = None

class LoginRequest(BaseModel):
    username: str
    password: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    username: str
    full_name: str


# ── Chat ───────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=5000)
    session_id: Optional[str] = None     # None → create new session
    use_rag: bool = True                  # whether to retrieve doc context

class ChatResponse(BaseModel):
    response: str
    session_id: str
    sources: List[str] = []              # document names used in RAG
    model_used: str = ""

class Message(BaseModel):
    role: str                             # "user" | "assistant"
    content: str
    timestamp: datetime


# ── Quiz / Study Tools ─────────────────────────────────────────────────

class QuizRequest(BaseModel):
    topic: str = Field(..., min_length=2)
    num_questions: int = Field(5, ge=1, le=20)
    difficulty: str = Field("medium", pattern="^(easy|medium|hard)$")

class SummaryRequest(BaseModel):
    topic: str = Field(..., min_length=2)
    format: str = Field("bullet_points",
                        pattern="^(bullet_points|flashcards|cornell_notes|mind_map)$")

class ExplainRequest(BaseModel):
    concept: str = Field(..., min_length=2)
    level: str = Field("student",
                       pattern="^(eli5|student|advanced|expert)$")


# ── Documents / RAG ────────────────────────────────────────────────────

class DocumentInfo(BaseModel):
    filename: str
    size_kb: float
    chunks: int
    uploaded_at: datetime

class DocumentListResponse(BaseModel):
    documents: List[DocumentInfo]
    total: int


# ── History ────────────────────────────────────────────────────────────

class SessionSummary(BaseModel):
    session_id: str
    title: str
    message_count: int
    created_at: datetime
    last_active: datetime

class HistoryResponse(BaseModel):
    sessions: List[SessionSummary]

class SessionMessages(BaseModel):
    session_id: str
    title: str
    messages: List[Message]
