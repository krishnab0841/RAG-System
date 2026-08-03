"""
Pydantic models for API request/response validation.
"""

from pydantic import BaseModel
from typing import Optional


class ChatRequest(BaseModel):
    """Request body for the chat endpoint."""
    question: str
    top_k: Optional[int] = 5


class UrlIngestRequest(BaseModel):
    """Request body for URL document ingestion."""
    url: str


class SourceChunk(BaseModel):
    """A retrieved source chunk used in the answer."""
    document_name: str
    chunk_text: str
    page_number: Optional[int] = None
    similarity_score: Optional[float] = None



class DocumentInfo(BaseModel):
    """Information about an ingested document."""
    doc_id: str
    filename: str
    file_type: str
    chunk_count: int
    file_size: Optional[int] = None


class DocumentListResponse(BaseModel):
    """Response body for listing documents."""
    documents: list[DocumentInfo]
    total_count: int


class SettingsRequest(BaseModel):
    """Request body for updating settings."""
    api_key: Optional[str] = None
    hf_api_key: Optional[str] = None
    model: Optional[str] = None
    top_k: Optional[int] = None


class SettingsResponse(BaseModel):
    """Response body for settings."""
    has_api_key: bool
    has_hf_api_key: bool
    model: str
    top_k: int


class HealthResponse(BaseModel):
    """Response body for health check."""
    status: str
    documents_count: int
    has_api_key: bool


class UploadResponse(BaseModel):
    """Response body for document upload."""
    doc_id: str
    filename: str
    chunk_count: int
    message: str
