# 🧠 RAG System — AI Document & Web Q&A

A full-stack Retrieval-Augmented Generation (RAG) web application built with **FastAPI**, **LangChain**, **FAISS**, and **Groq**.

Upload your documents or ingest web URLs, and ask questions — the AI retrieves relevant context and generates accurate, synthesized answers with source citations.

## 🏗️ Architecture

```text
Frontend (HTML/CSS/JS) → FastAPI Backend → LangChain Pipeline
                                           ├── Document Loaders (PDF, DOCX, TXT, CSV, MD, Web)
                                           ├── Text Splitter (RecursiveCharacterTextSplitter)
                                           ├── Embeddings (HuggingFace all-MiniLM-L6-v2)
                                           ├── Vector Store (FAISS)
                                           ├── Retriever (FAISS similarity search)
                                           └── LLM (Groq — Llama 3.3 70B)
```

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| **Backend** | FastAPI |
| **RAG Framework** | LangChain (loaders, splitters, embeddings, retriever, LLM orchestration) |
| **LLM** | Groq (`llama-3.3-70b-versatile`) |
| **Embeddings** | HuggingFace `all-MiniLM-L6-v2` |
| **Vector DB** | FAISS (local, persistent in-memory via `vector_store`) |
| **Frontend** | Vanilla HTML/CSS/JS (SPA) |
| **Deployment** | Vercel Serverless Functions (`api/index.py`) |

## ✨ Features

- 📄 **Multi-format Document Upload** — Supports PDF, TXT, DOCX, CSV, and Markdown files.
- 🌐 **Web URL Ingestion** — Directly scrape and ingest content from any web link.
- 💬 **Context-Aware Q&A** — Ask questions and receive accurate answers based exclusively on the provided context.
- ⚡ **Real-time Streaming** — Experience fast, token-by-token streaming responses via Server-Sent Events (SSE).
- 📚 **Source Citations** — Transparency with source document citations and similarity scores for every answer.
- 🗂️ **Document Management** — View and delete indexed documents directly from the UI.
- ⚙️ **Dynamic Settings** — Update API keys (Groq & HuggingFace), models, and retrieval configurations on the fly.
- 🌙 **Premium Dark UI** — Beautiful, responsive Glassmorphism design aesthetics.

## 🚀 Local Setup

### 1. Install Dependencies

Ensure you have Python installed, then set up the environment:

```bash
# Clone the repository
git clone https://github.com/krishnab0841/RAG-System.git
cd RAG-System

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

# Install required packages
pip install -r requirements.txt
```

### 2. Configure API Keys

- Get a Groq API Key at [console.groq.com/keys](https://console.groq.com/keys).
- (Optional) Get a HuggingFace API key for higher rate limits on embeddings.

**Option A: Environment Variables**
Create a `.env` file in the root directory:
```env
GROQ_API_KEY=gsk_your_key_here
HUGGINGFACE_API_KEY=your_hf_key_here
```

**Option B: Web UI**
You can also configure your keys dynamically through the **Settings** modal in the frontend application.

### 3. Run the Server

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 4. Access the App

Open your browser and navigate to [http://localhost:8000](http://localhost:8000).

## ☁️ Deployment (Vercel)

This project is configured for serverless deployment on Vercel.

1. Install the Vercel CLI: `npm i -g vercel`
2. Run `vercel` in the project root to deploy.
3. Configure your Environment Variables (`GROQ_API_KEY`) in the Vercel dashboard.

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/health` | Application health check |
| `POST` | `/api/upload` | Upload and ingest a document via LangChain loaders |
| `POST` | `/api/ingest-url` | Scrape and ingest content from a URL |
| `POST` | `/api/chat` | Submit a question and receive a streaming SSE response |
| `GET` | `/api/documents` | Retrieve a list of all indexed documents |
| `DELETE`| `/api/documents/{id}` | Delete a document and its embeddings from the vector store |
| `GET` | `/api/settings` | Get current application settings |
| `POST` | `/api/settings` | Update runtime application settings |

## 📜 License

MIT License
