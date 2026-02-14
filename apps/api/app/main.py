"""
Patent Discovery System API

Main application entry point for the FastAPI backend.
Handles patent prior art search, infringement analysis, and landscape summarization.
"""
from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import Any, Dict

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse

from app.api.v1.routes.query import router as query_router
from app.core.logging import configure_logging, get_logger
from app.core.settings import get_settings

settings = get_settings()
configure_logging(settings)
log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan context manager.
    Handles startup and shutdown events.
    """
    log.info("Starting Patent Discovery System API")
    log.info(f"Environment: {settings.env}")
    log.info(f"CORS origins: {settings.cors_allow_origins}")

    if not settings.pinecone_api_key:
        log.warning("PINECONE_API_KEY not set - vector search will fail")
    if not settings.openai_api_key:
        log.warning("OPENAI_API_KEY not set - embeddings will fail")
    if not settings.gemini_api_key:
        log.warning("GEMINI_API_KEY not set - LLM answer generation will fail")
    
    yield
    
    log.info("Shutting down Patent Discovery System API")


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.
    
    Returns:
        Configured FastAPI application instance
    """
    app = FastAPI(
        title="Patent Discovery System API",
        description="AI-powered patent search and analysis system supporting prior art search, "
                    "infringement analysis, and technology landscape summarization",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )
    
    _configure_cors(app)
    _configure_middleware(app)
    _register_exception_handlers(app)
    _register_routes(app)
    
    return app


def _configure_cors(app: FastAPI) -> None:
    """Configure CORS middleware with settings from environment."""
    origins = settings.cors_allow_origins if settings.cors_allow_origins else ["*"]
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID"],
    )
    log.info(f"CORS configured with origins: {origins}")


def _configure_middleware(app: FastAPI) -> None:
    """Configure additional middleware for the application."""
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    
    @app.middleware("http")
    async def add_process_time_header(request: Request, call_next):
        """Add processing time header to all responses."""
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = f"{process_time:.4f}"
        return response

    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        """Log all incoming requests."""
        log.info(f"Request: {request.method} {request.url.path}")
        response = await call_next(request)
        log.info(f"Response: {request.method} {request.url.path} - Status: {response.status_code}")
        return response


def _register_exception_handlers(app: FastAPI) -> None:
    """Register global exception handlers."""
    
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        """Handle request validation errors with detailed error messages."""
        log.warning(f"Validation error for {request.url.path}: {exc.errors()}")
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "error": "Validation Error",
                "detail": exc.errors(),
                "body": exc.body,
            },
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """Handle unexpected exceptions."""
        log.exception(f"Unhandled exception for {request.url.path}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "Internal Server Error",
                "detail": "An unexpected error occurred. Please try again later.",
            },
        )


def _register_routes(app: FastAPI) -> None:
    """Register all API routes."""
    
    @app.get("/health", tags=["Health"])
    async def health_check() -> Dict[str, str]:
        """
        Basic health check endpoint.
        Returns 200 if the service is running.
        """
        return {"status": "healthy", "service": "patent-discovery-api"}
    
    @app.get("/ready", tags=["Health"])
    async def readiness_check() -> Dict[str, Any]:
        """
        Readiness check endpoint.
        Validates that required dependencies are configured.
        """
        ready = True
        issues = []
        
        if not settings.pinecone_api_key:
            ready = False
            issues.append("Pinecone API key not configured")
        
        if not settings.openai_api_key:
            ready = False
            issues.append("OpenAI API key not configured (needed for embeddings)")
        
        if not settings.gemini_api_key:
            ready = False
            issues.append("Gemini API key not configured (needed for LLM generation)")
        
        status_code = status.HTTP_200_OK if ready else status.HTTP_503_SERVICE_UNAVAILABLE
        
        return JSONResponse(
            status_code=status_code,
            content={
                "ready": ready,
                "issues": issues,
                "service": "patent-discovery-api",
            },
        )
    
    app.include_router(
        query_router,
        prefix="/api/v1",
        tags=["Query"],
    )
    
    log.info("Routes registered successfully")

app = create_app()
