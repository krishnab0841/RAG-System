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
UPLOAD_DIR = BASE_DIR / "uploads"
FAISS_INDEX_DIR = BASE_DIR / "faiss_index"
STATIC_DIR = BASE_DIR / "static"

# Ensure directories exist
UPLOAD_DIR.mkdir(exist_ok=True)
FAISS_INDEX_DIR.mkdir(exist_ok=True)

# --- LLM Settings (Groq) ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = "llama-3.3-70b-versatile"

# --- Embedding Settings ---
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# --- Chunking Settings ---
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# --- Retrieval Settings ---
TOP_K = 5

# --- Supported File Types ---
SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".docx", ".csv", ".md"}
