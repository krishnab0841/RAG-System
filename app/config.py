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

# Vercel's serverless environment has a read-only filesystem except for /tmp
if os.environ.get("VERCEL"):
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
