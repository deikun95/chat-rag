"""
Chat with Docs — FastAPI application entry point.

Run with: uvicorn main:app --reload
"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
import time

from app.core.config import settings
from app.api.documents import router as documents_router
from app.api.chat import router as chat_router
from app.models.schemas import HealthResponse

# ==================== Logging ====================

logging.basicConfig(
    level=logging.INFO if settings.app_env == "production" else logging.DEBUG,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("chatwithdocs")

# ==================== App ====================

app = FastAPI(
    title="Chat with Docs API",
    description="RAG-powered document Q&A system",
    version="0.1.0",
)

# ==================== Middleware ====================

# CORS — allow frontend to make requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration_ms = (time.time() - start_time) * 1000

    logger.info(
        f"{request.method} {request.url.path} "
        f"-> {response.status_code} "
        f"({duration_ms:.0f}ms)"
    )
    return response


# Global exception handler — catch anything unexpected
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "INTERNAL_ERROR",
            "message": "An unexpected error occurred. Please try again.",
        },
    )


# ==================== Routes ====================

app.include_router(documents_router)
app.include_router(chat_router)


@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint for monitoring."""
    return HealthResponse(
        status="ok",
        vector_db="connected",
        version="0.1.0",
    )
