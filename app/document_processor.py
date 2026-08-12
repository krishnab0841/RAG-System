"""
Document processing pipeline using LangChain.
Uses LangChain document loaders for multi-format parsing
and RecursiveCharacterTextSplitter for semantic chunking.
"""

import hashlib
from pathlib import Path
from urllib.parse import urlparse
import requests

from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    CSVLoader,
    WebBaseLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from app.config import CHUNK_SIZE, CHUNK_OVERLAP, SUPPORTED_EXTENSIONS, UPLOAD_DIR


def generate_doc_id(filename: str, content: str) -> str:
    """Generate a unique document ID from filename and content hash."""
    hash_input = f"{filename}:{content[:500]}"
    return hashlib.md5(hash_input.encode()).hexdigest()[:12]


def _get_loader(file_path: Path):
    """Return the appropriate LangChain document loader for the file type."""
    ext = file_path.suffix.lower()

    loader_map = {
        ".pdf": lambda: PyPDFLoader(str(file_path)),
        ".docx": lambda: Docx2txtLoader(str(file_path)),
        ".txt": lambda: TextLoader(str(file_path), encoding="utf-8"),
        ".md": lambda: TextLoader(str(file_path), encoding="utf-8"),
        ".csv": lambda: CSVLoader(str(file_path), encoding="utf-8"),
    }

    factory = loader_map.get(ext)
    if factory is None:
        raise ValueError(
            f"Unsupported file type: {ext}. Supported: {SUPPORTED_EXTENSIONS}"
        )
    return factory()


def load_document(file_path: Path) -> list[Document]:
    """
    Load a document using the appropriate LangChain loader.

    Returns:
        List of LangChain Document objects with page_content and metadata.
    """
    loader = _get_loader(file_path)
    documents = loader.load()

    # Filter out empty documents
    documents = [doc for doc in documents if doc.page_content.strip()]

    if not documents:
        raise ValueError(f"No text content found in: {file_path.name}")

    return documents


def chunk_document(
    file_path: Path,
    filename: str,
) -> tuple[str, list[Document]]:
    """
    Load, parse, and chunk a document using LangChain.

    Args:
        file_path: Path to the document file.
        filename: Original filename for metadata.

    Returns:
        - doc_id: Unique document identifier.
        - chunks: List of LangChain Document objects (chunked).
    """
    # Step 1: Load the document using LangChain loader
    raw_documents = load_document(file_path)

    # Step 2: Generate a doc_id from the content
    all_text = "\n".join(doc.page_content for doc in raw_documents)
    doc_id = generate_doc_id(filename, all_text)

    # Step 3: Split into chunks using LangChain text splitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(raw_documents)

    # Step 4: Enrich metadata on each chunk
    for i, chunk in enumerate(chunks):
        chunk.metadata.update({
            "doc_id": doc_id,
            "filename": filename,
            "chunk_index": i,
            "file_type": file_path.suffix.lower(),
        })
        # Normalize page number from different loaders
        if "page" in chunk.metadata:
            chunk.metadata["page_number"] = chunk.metadata.pop("page") + 1  # 0-indexed → 1-indexed
        elif "page_number" not in chunk.metadata:
            chunk.metadata["page_number"] = 0

    return doc_id, chunks


def chunk_url(url: str) -> tuple[str, list[Document]]:
    """
    Fetch, load, parse, and chunk content from a URL.

    Args:
        url: The web link / document URL.

    Returns:
        - doc_id: Unique document identifier.
        - chunks: List of chunked LangChain Document objects.
    """
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("Invalid URL format. Please provide a valid HTTP/HTTPS link.")

    # Derive a display name from the URL
    domain = parsed.netloc.replace("www.", "")
    path_name = Path(parsed.path).name
    display_name = path_name if path_name else domain
    if not any(display_name.endswith(ext) for ext in SUPPORTED_EXTENSIONS):
        display_name = f"{display_name}.url"

    # Check if the URL points directly to a PDF or binary supported document
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0"}
    try:
        head_resp = requests.head(url, headers=headers, timeout=10, allow_redirects=True)
        content_type = head_resp.headers.get("Content-Type", "").lower()
    except Exception:
        content_type = ""

    if "application/pdf" in content_type or parsed.path.lower().endswith(".pdf"):
        # Download PDF to temp file and process via PyPDFLoader
        pdf_path = UPLOAD_DIR / f"temp_{hashlib.md5(url.encode()).hexdigest()[:8]}.pdf"
        try:
            resp = requests.get(url, headers=headers, timeout=30)
            resp.raise_for_status()
            with open(pdf_path, "wb") as f:
                f.write(resp.content)
            doc_id, chunks = chunk_document(pdf_path, display_name if display_name.endswith(".pdf") else f"{display_name}.pdf")
            return doc_id, chunks
        finally:
            pdf_path.unlink(missing_ok=True)
    else:
        # Web Page ingestion via WebBaseLoader
        try:
            loader = WebBaseLoader(
                web_paths=[url],
                requests_kwargs={"headers": headers},
            )
            raw_documents = loader.load()
        except Exception as e:
            raise ValueError(f"Failed to fetch content from URL: {str(e)}")

        raw_documents = [doc for doc in raw_documents if doc.page_content.strip()]
        if not raw_documents:
            raise ValueError("No readable text content found at the provided URL.")

        all_text = "\n".join(doc.page_content for doc in raw_documents)
        doc_id = generate_doc_id(display_name, all_text)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        chunks = splitter.split_documents(raw_documents)

        for i, chunk in enumerate(chunks):
            chunk.metadata.update({
                "doc_id": doc_id,
                "filename": display_name,
                "chunk_index": i,
                "file_type": ".url",
                "page_number": 0,
            })

        return doc_id, chunks
