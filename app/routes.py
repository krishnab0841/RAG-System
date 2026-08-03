"""
API route definitions for the RAG application.
"""

import shutil
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse

from app.config import UPLOAD_DIR, SUPPORTED_EXTENSIONS
from app.models import (
    ChatRequest,
    UrlIngestRequest,
    DocumentInfo,
    DocumentListResponse,
    SettingsRequest,
    SettingsResponse,
    HealthResponse,
    UploadResponse,
)
from app.document_processor import chunk_document, chunk_url
from app.vector_store import vector_store
from app.rag_engine import (
    generate_streaming_response,
    get_settings,
    update_settings,
)

router = APIRouter(prefix="/api")


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    settings = get_settings()
    return HealthResponse(
        status="ok",
        documents_count=vector_store.get_document_count(),
        has_api_key=settings["has_api_key"],
    )


@router.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """Upload and ingest a document via LangChain loaders."""
    # Validate file type
    filename = file.filename or "unknown"
    ext = Path(filename).suffix.lower()

    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Supported: {', '.join(SUPPORTED_EXTENSIONS)}",
        )

    # Save the uploaded file
    file_path = UPLOAD_DIR / filename

    # Handle duplicate filenames
    counter = 1
    original_stem = file_path.stem
    while file_path.exists():
        file_path = UPLOAD_DIR / f"{original_stem}_{counter}{ext}"
        counter += 1

    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")
    finally:
        file.file.close()

    # Process and chunk using LangChain document loaders + splitters
    try:
        doc_id, chunks = chunk_document(file_path, file_path.name)
    except ValueError as e:
        file_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Failed to process document")
        file_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Failed to process document: {str(e)}")

    # Add LangChain Documents to vector store
    try:
        chunk_count = vector_store.add_documents(doc_id, chunks)
    except Exception as e:
        logger.exception("Failed to index document")
        file_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Failed to index document: {str(e)}")

    return UploadResponse(
        doc_id=doc_id,
        filename=file_path.name,
        chunk_count=chunk_count,
        message=f"Successfully uploaded and indexed '{file_path.name}' ({chunk_count} chunks)",
    )


@router.post("/ingest-url", response_model=UploadResponse)
async def ingest_url(request: UrlIngestRequest):
    """Fetch content from a URL, parse, and ingest into vector store."""
    url = request.url.strip()
    if not url:
        raise HTTPException(status_code=400, detail="URL cannot be empty")

    try:
        doc_id, chunks = chunk_url(url)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process URL: {str(e)}")

    try:
        chunk_count = vector_store.add_documents(doc_id, chunks)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to index URL content: {str(e)}")

    filename = chunks[0].metadata.get("filename", "url_doc") if chunks else "url_doc"
    return UploadResponse(
        doc_id=doc_id,
        filename=filename,
        chunk_count=chunk_count,
        message=f"Successfully ingested and indexed URL content '{filename}' ({chunk_count} chunks)",
    )


@router.post("/chat")
async def chat(request: ChatRequest):
    """Send a question and get a streamed answer with sources."""
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    return StreamingResponse(
        generate_streaming_response(
            question=request.question,
            top_k=request.top_k,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/documents", response_model=DocumentListResponse)
async def list_documents():
    """List all ingested documents."""
    docs = vector_store.get_all_documents()
    document_list = [
        DocumentInfo(
            doc_id=d["doc_id"],
            filename=d["filename"],
            file_type=d["file_type"],
            chunk_count=d["chunk_count"],
        )
        for d in docs
    ]
    return DocumentListResponse(
        documents=document_list,
        total_count=len(document_list),
    )


@router.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    """Delete a document and its embeddings from FAISS."""
    deleted_count = vector_store.delete_document(doc_id)

    if deleted_count == 0:
        raise HTTPException(status_code=404, detail="Document not found")

    return {
        "message": f"Deleted document with {deleted_count} chunks",
        "doc_id": doc_id,
        "chunks_deleted": deleted_count,
    }


@router.post("/settings", response_model=SettingsResponse)
async def update_app_settings(request: SettingsRequest):
    """Update application settings."""
    update_settings(
        api_key=request.api_key,
        hf_api_key=request.hf_api_key,
        model=request.model,
        top_k=request.top_k,
    )
    settings = get_settings()
    return SettingsResponse(**settings)


@router.get("/settings", response_model=SettingsResponse)
async def get_app_settings():
    """Get current application settings."""
    settings = get_settings()
    return SettingsResponse(**settings)
