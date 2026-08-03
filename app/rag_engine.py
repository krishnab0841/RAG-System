"""
RAG Engine using LangChain orchestration.
Uses LangChain's ChatGroq for LLM, ChatPromptTemplate for prompts,
and the FAISS retriever for context retrieval — stream via ChatGroq.stream().
"""

import json
from typing import Optional

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

from app.config import GROQ_API_KEY, GROQ_MODEL, TOP_K
from app.vector_store import vector_store


# --- Runtime Settings (mutable) ---
_settings = {
    "api_key": GROQ_API_KEY,
    "model": GROQ_MODEL,
    "top_k": TOP_K,
}


def get_settings() -> dict:
    """Get current settings."""
    return {
        "has_api_key": bool(_get_api_key()),
        "model": _settings["model"],
        "top_k": _settings["top_k"],
    }


def update_settings(
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    top_k: Optional[int] = None,
):
    """Update runtime settings."""
    if api_key is not None:
        _settings["api_key"] = api_key
    if model is not None:
        _settings["model"] = model
    if top_k is not None:
        _settings["top_k"] = top_k


# --- LangChain Prompt Template ---
RAG_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are an expert AI assistant that answers questions and synthesizes information based on the provided document context.

INSTRUCTIONS:
- Answer the user's question using ONLY the information from the provided context chunks.
- If the context does not contain enough information to answer, say so clearly.
- Use clean Markdown formatting (headers, bold text, bullet points, structured tables) for maximum clarity.
- When referencing specific information, cite the relevant source document.
- Do NOT fabricate or extrapolate information beyond the provided context.

SUMMARIZATION GUIDELINES:
When asked to summarize a document or topic:
1. Concisely capture the document's Purpose and Core Objectives.
2. Outline Main Topics and Key Concepts.
3. Highlight Important Findings and Critical Statistics.
4. Summarize Methodologies, Key Decisions, and Recommendations.
5. Note any mentioned Limitations, Future Scope, and the Overall Conclusion.
6. Strictly eliminate repetitive, boilerplate, or low-value filler content. Structure the summary cleanly with Markdown sections.""",
    ),
    (
        "human",
        """CONTEXT FROM DOCUMENTS:
{context}

USER QUESTION:
{question}

Please answer the question based on the context above.""",
    ),
])


def _format_docs_for_prompt(docs_with_scores: list) -> str:
    """Format retrieved LangChain Documents into a context string for the prompt."""
    context_parts = []
    for i, (doc, score) in enumerate(docs_with_scores, 1):
        filename = doc.metadata.get("filename", "Unknown")
        page = doc.metadata.get("page_number", 0)
        page_info = f" (Page {page})" if page else ""
        similarity = float(1 / (1 + float(score)))  # FAISS L2 distance → similarity [0,1]

        context_parts.append(
            f"--- Source {i}: {filename}{page_info} [Relevance: {similarity:.1%}] ---\n"
            f"{doc.page_content}\n"
        )
    return "\n".join(context_parts)


def _build_sources_payload(docs_with_scores: list) -> list[dict]:
    """Build source citation data from retrieved documents."""
    sources = []
    for doc, score in docs_with_scores:
        text = doc.page_content
        sim_val = round(float(1 / (1 + float(score))), 3)
        sources.append({
            "document_name": doc.metadata.get("filename", "Unknown"),
            "chunk_text": text[:200] + "..." if len(text) > 200 else text,
            "page_number": doc.metadata.get("page_number", 0) or None,
            "similarity_score": float(sim_val),
        })
    return sources


def _get_api_key() -> str:
    """Resolve API key from runtime settings or re-read environment."""
    if _settings["api_key"]:
        return _settings["api_key"]
    from dotenv import load_dotenv
    import os
    load_dotenv()
    return os.getenv("GROQ_API_KEY", "")


def _get_llm() -> ChatGroq:
    """Create a LangChain ChatGroq LLM instance with current settings."""
    api_key = _get_api_key()
    return ChatGroq(
        groq_api_key=api_key,
        model=_settings["model"],
        temperature=0.3,
        max_tokens=2048,
        streaming=True,
    )


def generate_streaming_response(
    question: str,
    top_k: Optional[int] = None,
):
    """
    Full RAG pipeline using LangChain:
      1. Retrieve context via LangChain FAISS retriever
      2. Send source citations
      3. Stream LLM response via ChatGroq.stream()

    Yields SSE-formatted events.
    """
    try:
        api_key = _get_api_key()
        if not api_key:
            yield f"data: {json.dumps({'type': 'error', 'content': 'No API key configured. Please add your Groq API key in Settings.'})}\n\n"
            return

        # --- Step 1: Retrieve relevant context using LangChain FAISS ---
        k = top_k or _settings["top_k"]
        summary_keywords = {"summarize", "summary", "overview", "abstract", "key points", "recap", "synopsis"}
        if any(kw in question.lower() for kw in summary_keywords):
            k = max(k, 12)

        docs_with_scores = vector_store.similarity_search_with_score(
            query=question, top_k=k
        )

        if not docs_with_scores:
            yield f"data: {json.dumps({'type': 'error', 'content': 'No documents found. Please upload some documents first.'})}\n\n"
            return

        # --- Step 2: Send source citations to frontend ---
        sources = _build_sources_payload(docs_with_scores)
        yield f"data: {json.dumps({'type': 'sources', 'content': sources})}\n\n"

        # --- Step 3: Build messages and stream via ChatGroq ---
        llm = _get_llm()
        context_str = _format_docs_for_prompt(docs_with_scores)
        messages = RAG_PROMPT.format_messages(
            context=context_str,
            question=question,
        )

        for chunk in llm.stream(messages):
            if chunk.content:
                yield f"data: {json.dumps({'type': 'token', 'content': chunk.content})}\n\n"

        yield f"data: {json.dumps({'type': 'done', 'content': ''})}\n\n"

    except Exception as e:
        import traceback
        traceback.print_exc()
        error_msg = str(e)
        if "api_key" in error_msg.lower() or "401" in error_msg or "403" in error_msg:
            error_msg = "Invalid API key. Please check your Groq API key in Settings."
        yield f"data: {json.dumps({'type': 'error', 'content': f'LLM Error: {error_msg}'})}\n\n"
