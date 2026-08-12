"""
FastAPI application entry point.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_cors_origins
from app.routes import router

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application startup and shutdown lifecycle."""
    # --- Startup ---
    logger.info("RAG application starting up...")
    yield
    # --- Shutdown ---
    logger.info("RAG application shutting down.")


app = FastAPI(
    title="RAG System",
    description="Retrieval-Augmented Generation system with Groq and HuggingFace",
    version="1.0.0",
    lifespan=lifespan,
)

# --- CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_cors_origins(),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- API Routes ---
app.include_router(router)


@app.get("/")
async def service_info():
    """Minimal endpoint for visitors and platform checks."""
    return {"service": "RAG System API", "health": "/api/health"}
