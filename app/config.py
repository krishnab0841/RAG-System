"""
Configuration management for the RAG application.
"""

import os

# Set thread limits before any numerical library imports
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
# Set USER_AGENT to suppress HuggingFace Hub warning
os.environ["USER_AGENT"] = "RAG_System/1.0"

from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# --- Paths ---
BASE_DIR = Path(__file__).resolve().parent.parent

# Render's filesystem is ephemeral. Use its writable temporary directory so
# uploaded source files never end up in the application checkout.
if os.environ.get("VERCEL") or os.environ.get("RENDER"):
    UPLOAD_DIR = Path("/tmp/uploads")
else:
    UPLOAD_DIR = BASE_DIR / "uploads"

STATIC_DIR = BASE_DIR / "static"

# Ensure directories exist
try:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
except OSError:
    pass  # Ignore read-only errors if they still occur

# --- LLM Settings (Groq) ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = "llama-3.3-70b-versatile"

# --- Embedding Settings (HuggingFace Inference API) ---
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY", "")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# --- Chunking Settings ---
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# --- Retrieval Settings ---
TOP_K = 5

# --- Supported File Types ---
SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".docx", ".csv", ".md"}

# Keep request handling within the resources available to a Render web service.
MAX_UPLOAD_SIZE_BYTES = 20 * 1024 * 1024


def get_cors_origins() -> list[str]:
    """Return explicitly configured browser origins for the API."""
    configured_origins = os.getenv("CORS_ORIGINS", "")
    return [origin.strip().rstrip("/") for origin in configured_origins.split(",") if origin.strip()]
