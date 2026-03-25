"""
Chat API endpoint — SSE streaming responses.
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.models.schemas import ChatRequest
from app.services.chat_service import ChatService
from app.services.retrieval_service import RetrievalService
from app.services.embedding_service import EmbeddingService
from app.services.vector_store import VectorStoreService

router = APIRouter(prefix="/api", tags=["chat"])

# Service wiring
embedding_service = EmbeddingService()
vector_store = VectorStoreService()
retrieval_service = RetrievalService(
    embedding_service=embedding_service,
    vector_store=vector_store,
)
chat_service = ChatService(retrieval_service=retrieval_service)


@router.post("/chat")
async def chat(request: ChatRequest):
    """
    Ask a question about uploaded documents.

    Returns a Server-Sent Events (SSE) stream with:
    - chunk events: individual text tokens
    - sources event: document references used
    - done event: stream completion signal
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    # Convert history models to dicts
    history = [{"role": m.role, "content": m.content} for m in request.history]

    return StreamingResponse(
        chat_service.stream_response(
            question=request.question,
            document_ids=request.document_ids,
            history=history,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
