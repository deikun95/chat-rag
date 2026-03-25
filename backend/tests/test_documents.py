"""
Tests for document upload and management.

Run with: pytest tests/ -v
"""

import pytest
from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


# ==================== Helpers ====================

def create_fake_pdf() -> bytes:
    """Create a minimal valid PDF with text for testing."""
    import fitz
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 50), "This is a test document.\n\nIt has two paragraphs of text for testing the upload and parsing pipeline.")
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def create_empty_pdf() -> bytes:
    """Create a PDF with no text (blank pages)."""
    import fitz
    doc = fitz.open()
    doc.new_page()  # blank page, no text
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


# ==================== Tests ====================

class TestHealthCheck:
    def test_health_returns_ok(self):
        response = client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"


class TestUploadDocument:
    def test_upload_valid_pdf(self):
        """Uploading a valid PDF should return 201 with document info."""
        pdf_bytes = create_fake_pdf()
        response = client.post(
            "/api/documents/upload",
            files={"file": ("test.pdf", pdf_bytes, "application/pdf")},
        )
        assert response.status_code == 201
        data = response.json()
        assert data["name"] == "test.pdf"
        assert data["pages"] >= 1
        assert data["status"] == "ready"
        assert "id" in data

    def test_upload_non_pdf_rejected(self):
        """Uploading a non-PDF file should return 415."""
        response = client.post(
            "/api/documents/upload",
            files={"file": ("test.txt", b"hello world", "text/plain")},
        )
        assert response.status_code == 415

    def test_upload_empty_file_rejected(self):
        """Uploading an empty file should return 422."""
        response = client.post(
            "/api/documents/upload",
            files={"file": ("empty.pdf", b"", "application/pdf")},
        )
        assert response.status_code == 422

    def test_upload_empty_pdf_rejected(self):
        """Uploading a PDF with no text should return 422."""
        pdf_bytes = create_empty_pdf()
        response = client.post(
            "/api/documents/upload",
            files={"file": ("blank.pdf", pdf_bytes, "application/pdf")},
        )
        assert response.status_code == 422


class TestListDocuments:
    def test_list_returns_array(self):
        """GET /documents should return a list."""
        response = client.get("/api/documents")
        assert response.status_code == 200
        data = response.json()
        assert "documents" in data
        assert isinstance(data["documents"], list)


class TestDeleteDocument:
    def test_delete_existing_document(self):
        """Deleting an existing document should return 204."""
        # First upload
        pdf_bytes = create_fake_pdf()
        upload_resp = client.post(
            "/api/documents/upload",
            files={"file": ("todelete.pdf", pdf_bytes, "application/pdf")},
        )
        doc_id = upload_resp.json()["id"]

        # Then delete
        delete_resp = client.delete(f"/api/documents/{doc_id}")
        assert delete_resp.status_code == 204

        # Verify it's gone
        get_resp = client.get(f"/api/documents/{doc_id}")
        assert get_resp.status_code == 404

    def test_delete_nonexistent_returns_404(self):
        """Deleting a non-existent document should return 404."""
        response = client.delete("/api/documents/fake-id-12345")
        assert response.status_code == 404
