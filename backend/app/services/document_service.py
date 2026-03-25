"""
DocumentService — отвечает за:
1. Парсинг PDF файлов (текст + номера страниц)
2. Валидация файлов (размер, формат, не пустой)
3. Хранение метаданных документов (SQLite)

НЕ знает про embeddings, vectors, chunks.
"""

import fitz  # PyMuPDF
import sqlite3
import uuid
import re
from datetime import datetime, timezone
from pathlib import Path
from dataclasses import dataclass

from app.core.config import settings


@dataclass
class PageContent:
    """Extracted text from a single PDF page."""
    page_number: int
    text: str


@dataclass
class DocumentMetadata:
    """Metadata stored in SQLite about an uploaded document."""
    id: str
    name: str
    pages: int
    chunks: int
    status: str
    created_at: datetime


class DocumentServiceError(Exception):
    """Base error for document operations."""
    pass


class InvalidFileError(DocumentServiceError):
    """File is not a valid PDF or exceeds limits."""
    pass


class DocumentNotFoundError(DocumentServiceError):
    """Document ID not found in database."""
    pass


class DocumentService:
    """Handles PDF parsing and document metadata storage."""

    def __init__(self, db_path: str = "documents.db"):
        self.db_path = db_path
        self._init_db()

    # ==================== Database ====================

    def _init_db(self):
        """Create documents table if it doesn't exist."""
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    pages INTEGER NOT NULL DEFAULT 0,
                    chunks INTEGER NOT NULL DEFAULT 0,
                    status TEXT NOT NULL DEFAULT 'processing',
                    created_at TEXT NOT NULL
                )
            """)
            conn.commit()
        finally:
            conn.close()

    def _get_conn(self) -> sqlite3.Connection:
        """Get a new database connection."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    # ==================== PDF Parsing ====================

    def validate_file(self, file_bytes: bytes, filename: str) -> None:
        """
        Validate the uploaded file before processing.

        Checks:
        1. File is not empty
        2. File size within limits
        3. Filename ends with .pdf
        4. File bytes are actually a valid PDF
        5. PDF is not password-protected
        6. PDF contains extractable text (not just images)

        Raises InvalidFileError with a descriptive message if any check fails.
        """
        # Check 1: Empty file
        if not file_bytes or len(file_bytes) == 0:
            raise InvalidFileError("File is empty")

        # Check 2: File size
        size_mb = len(file_bytes) / (1024 * 1024)
        if size_mb > settings.max_upload_size_mb:
            raise InvalidFileError(
                f"File too large: {size_mb:.1f}MB "
                f"(max {settings.max_upload_size_mb}MB)"
            )

        # Check 3: Filename extension
        if not filename.lower().endswith(".pdf"):
            raise InvalidFileError(
                f"Invalid file type: '{filename}'. Only PDF files are accepted"
            )

        # Check 4 + 5: Valid PDF that can be opened
        try:
            doc = fitz.open(stream=file_bytes, filetype="pdf")
        except Exception:
            raise InvalidFileError(
                "Could not open file. Make sure it's a valid, "
                "non-corrupted PDF"
            )

        if doc.is_encrypted:
            doc.close()
            raise InvalidFileError(
                "Password-protected PDFs are not supported"
            )

        # Check 6: Has extractable text
        has_text = False
        for page in doc:
            if page.get_text("text").strip():
                has_text = True
                break
        doc.close()

        if not has_text:
            raise InvalidFileError(
                "This PDF contains no extractable text. "
                "Scanned documents and image-only PDFs are not yet supported"
            )

    def extract_pages(self, file_bytes: bytes) -> list[PageContent]:
        """
        Extract text content from each page of a PDF.

        Returns a list of PageContent objects, one per page that has text.
        Pages with no text (e.g., full-page images) are skipped.
        """
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        pages = []

        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text").strip()

            if text:
                # Clean up common PDF extraction artifacts
                text = re.sub(r' {2,}', ' ', text)
                text = re.sub(r'\n{3,}', '\n\n', text)

                pages.append(PageContent(
                    page_number=page_num + 1,  # 1-indexed for humans
                    text=text
                ))

        doc.close()
        return pages

    # ==================== CRUD ====================

    def create_document(self, name: str, pages: int) -> DocumentMetadata:
        """
        Create a new document record in the database.
        Called right after upload, before processing.
        Status starts as 'processing'.
        """
        doc_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)

        conn = self._get_conn()
        try:
            conn.execute(
                """INSERT INTO documents (id, name, pages, chunks, status, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (doc_id, name, pages, 0, "processing", now.isoformat())
            )
            conn.commit()
        finally:
            conn.close()

        return DocumentMetadata(
            id=doc_id,
            name=name,
            pages=pages,
            chunks=0,
            status="processing",
            created_at=now,
        )

    def update_document_status(
        self,
        doc_id: str,
        status: str,
        chunks: int = 0
    ) -> None:
        """Update document status after processing completes or fails."""
        conn = self._get_conn()
        try:
            cursor = conn.execute(
                """UPDATE documents SET status = ?, chunks = ?
                   WHERE id = ?""",
                (status, chunks, doc_id)
            )
            if cursor.rowcount == 0:
                raise DocumentNotFoundError(f"Document '{doc_id}' not found")
            conn.commit()
        finally:
            conn.close()

    def get_document(self, doc_id: str) -> DocumentMetadata:
        """Get a single document by ID."""
        conn = self._get_conn()
        try:
            row = conn.execute(
                "SELECT * FROM documents WHERE id = ?", (doc_id,)
            ).fetchone()
        finally:
            conn.close()

        if not row:
            raise DocumentNotFoundError(f"Document '{doc_id}' not found")

        return DocumentMetadata(
            id=row["id"],
            name=row["name"],
            pages=row["pages"],
            chunks=row["chunks"],
            status=row["status"],
            created_at=datetime.fromisoformat(row["created_at"]),
        )

    def list_documents(self) -> list[DocumentMetadata]:
        """List all documents, newest first."""
        conn = self._get_conn()
        try:
            rows = conn.execute(
                "SELECT * FROM documents ORDER BY created_at DESC"
            ).fetchall()
        finally:
            conn.close()

        return [
            DocumentMetadata(
                id=row["id"],
                name=row["name"],
                pages=row["pages"],
                chunks=row["chunks"],
                status=row["status"],
                created_at=datetime.fromisoformat(row["created_at"]),
            )
            for row in rows
        ]

    def delete_document(self, doc_id: str) -> None:
        """Delete a document record. Caller must also delete vectors."""
        conn = self._get_conn()
        try:
            cursor = conn.execute(
                "DELETE FROM documents WHERE id = ?", (doc_id,)
            )
            if cursor.rowcount == 0:
                raise DocumentNotFoundError(f"Document '{doc_id}' not found")
            conn.commit()
        finally:
            conn.close()
