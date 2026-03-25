from pydantic import BaseModel, Field
from datetime import datetime


# ============ Documents ============

class DocumentResponse(BaseModel):
    """Response after uploading or listing a document."""
    id: str
    name: str
    pages: int
    chunks: int
    status: str  # "processing" | "ready" | "error"
    created_at: datetime


class DocumentListResponse(BaseModel):
    """Response for GET /documents."""
    documents: list[DocumentResponse]


class UploadResponse(BaseModel):
    """Response immediately after upload (before processing)."""
    id: str
    name: str
    status: str = "processing"
    message: str = "Document uploaded and is being processed"


# ============ Chat ============

class HistoryMessage(BaseModel):
    """A single message in conversation history."""
    role: str  # "user" | "assistant"
    content: str


class ChatRequest(BaseModel):
    """Request body for POST /chat."""
    question: str = Field(..., min_length=1, max_length=5000)
    document_ids: list[str] | None = None  # None = search all
    history: list[HistoryMessage] = []  # previous messages for context


class SourceReference(BaseModel):
    """A source chunk used to generate the answer."""
    document_id: str
    document_name: str
    page: int
    chunk_text: str
    relevance_score: float


# ============ General ============

class ErrorResponse(BaseModel):
    """Standard error response."""
    error: str
    message: str


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    vector_db: str
    version: str
