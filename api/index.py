"""
Vercel serverless function entry point.
Exports the FastAPI app instance for Vercel's Python runtime.
"""

from app.main import app

# Vercel looks for a variable named `app` or `handler`
# The FastAPI `app` instance is ASGI-compatible, which Vercel supports.
