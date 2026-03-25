"""
API endpoints for document management.

POST /api/documents/upload — Upload a PDF
GET  /api/documents        — List all documents
GET  /api/documents/{id}   — Get single document info
DELETE /api/documents/{id} — Delete a document
"""

from fastapi import APIRouter, UploadFile, File, HTTPException
import logging

from app.models.schemas import (
    DocumentResponse,
    DocumentListResponse,
    ErrorResponse,
)
from app.services.document_service import (
    DocumentService,
    InvalidFileError,
    DocumentNotFoundError,
)
from app.services.embedding_service import EmbeddingService
from app.services.ingestion_service import IngestionService
from app.services.vector_store import VectorStoreService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/documents", tags=["documents"])

# Service instances
doc_service = DocumentService()
embedding_service = EmbeddingService()
vector_store = VectorStoreService()
ingestion_service = IngestionService(
    embedding_service=embedding_service,
    vector_store=vector_store,
)


@router.post(
    "/upload",
    response_model=DocumentResponse,
    status_code=201,
    responses={
        413: {"model": ErrorResponse, "description": "File too large"},
        415: {"model": ErrorResponse, "description": "Invalid file type"},
        422: {"model": ErrorResponse, "description": "Processing error"},
    }
)
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a PDF document for processing.

    Full pipeline:
    1. Validate file (size, format, has text)
    2. Extract text from pages (PyMuPDF)
    3. Create document record (SQLite)
    4. Chunk -> Embed -> Store in vector DB (IngestionService)
    5. Update document status to 'ready'
    """
    file_bytes = await file.read()
    filename = file.filename or "unknown.pdf"

    # Step 1: Validate
    try:
        doc_service.validate_file(file_bytes, filename)
    except InvalidFileError as e:
        error_msg = str(e).lower()
        if "too large" in error_msg:
            raise HTTPException(status_code=413, detail=str(e))
        elif "file type" in error_msg:
            raise HTTPException(status_code=415, detail=str(e))
        else:
            raise HTTPException(status_code=422, detail=str(e))

    # Step 2: Extract text
    pages = doc_service.extract_pages(file_bytes)
    logger.info(f"Extracted {len(pages)} pages from '{filename}'")

    # Step 3: Create DB record
    doc_meta = doc_service.create_document(name=filename, pages=len(pages))

    # Step 4: Ingest (chunk -> embed -> store)
    try:
        chunk_count = ingestion_service.ingest(
            pages=pages,
            document_id=doc_meta.id,
            document_name=filename,
        )

        # Step 5: Mark as ready
        doc_service.update_document_status(
            doc_id=doc_meta.id,
            status="ready",
            chunks=chunk_count,
        )
    except Exception as e:
        logger.error(f"Ingestion failed for '{filename}': {e}", exc_info=True)
        doc_service.update_document_status(
            doc_id=doc_meta.id,
            status="error",
            chunks=0,
        )
        raise HTTPException(
            status_code=500,
            detail=f"Document uploaded but processing failed: {str(e)}"
        )

    # Return final state
    doc_meta = doc_service.get_document(doc_meta.id)
    return DocumentResponse(
        id=doc_meta.id,
        name=doc_meta.name,
        pages=doc_meta.pages,
        chunks=doc_meta.chunks,
        status=doc_meta.status,
        created_at=doc_meta.created_at,
    )


@router.get("", response_model=DocumentListResponse)
async def list_documents():
    """List all uploaded documents."""
    docs = doc_service.list_documents()
    return DocumentListResponse(
        documents=[
            DocumentResponse(
                id=d.id, name=d.name, pages=d.pages,
                chunks=d.chunks, status=d.status, created_at=d.created_at,
            )
            for d in docs
        ]
    )


@router.get(
    "/{document_id}",
    response_model=DocumentResponse,
    responses={404: {"model": ErrorResponse}},
)
async def get_document(document_id: str):
    """Get info about a specific document."""
    try:
        doc = doc_service.get_document(document_id)
    except DocumentNotFoundError:
        raise HTTPException(status_code=404, detail="Document not found")
    return DocumentResponse(
        id=doc.id, name=doc.name, pages=doc.pages,
        chunks=doc.chunks, status=doc.status, created_at=doc.created_at,
    )


@router.delete(
    "/{document_id}",
    status_code=204,
    responses={404: {"model": ErrorResponse}},
)
async def delete_document(document_id: str):
    """Delete a document and all associated vectors."""
    try:
        vector_store.delete_by_document(document_id)
        doc_service.delete_document(document_id)
    except DocumentNotFoundError:
        raise HTTPException(status_code=404, detail="Document not found")
