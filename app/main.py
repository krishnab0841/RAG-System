"""
FastAPI application entry point.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from app.config import STATIC_DIR, UPLOAD_DIR
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
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- API Routes ---
app.include_router(router)


@app.get("/")
async def serve_frontend():
    """Serve the frontend SPA."""
    return FileResponse(str(STATIC_DIR / "index.html"))


# --- Static Files (mounted after root route to avoid conflicts) ---
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
