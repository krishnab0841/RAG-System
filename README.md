# 🧠 RAG System — AI Document Q&A

A full-stack Retrieval-Augmented Generation (RAG) web application built with **LangChain**, **FAISS**, and **Groq**.

Upload your documents and ask questions — the AI retrieves relevant context and generates accurate answers with source citations.

## Architecture

```
Frontend (HTML/CSS/JS) → FastAPI Backend → LangChain Pipeline
                                           ├── Document Loaders (PDF, DOCX, TXT, CSV, MD)
                                           ├── Text Splitter (RecursiveCharacterTextSplitter)
                                           ├── Embeddings (HuggingFace all-MiniLM-L6-v2)
                                           ├── Vector Store (FAISS)
                                           ├── Retriever (FAISS similarity search)
                                           └── LLM (Groq — Llama 3.3 70B)
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| **Backend** | FastAPI |
| **RAG Framework** | LangChain (loaders, splitters, embeddings, retriever, LLM) |
| **LLM** | Groq (Llama 3.3 70B Versatile) |
| **Embeddings** | HuggingFace `all-MiniLM-L6-v2` (local) |
| **Vector DB** | FAISS (local, persistent) |
| **Frontend** | Vanilla HTML/CSS/JS |

## Setup

### 1. Install Dependencies

```bash
cd d:\RAG
pip install -r requirements.txt
```

### 2. Get a Groq API Key

- Go to [console.groq.com/keys](https://console.groq.com/keys)
- Create a free API key

### 3. Configure (Option A: Environment Variable)

```bash
# Copy the example env file
copy .env.example .env

# Edit .env and add your key
GROQ_API_KEY=gsk_your_key_here
```

### 3. Configure (Option B: UI Settings)

You can also enter your API key directly in the web UI under **Settings**.

### 4. Run the Server

```bash
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 5. Open the App

Navigate to [http://localhost:8000](http://localhost:8000)

## Features

- 📄 **Multi-format upload** — PDF, TXT, DOCX, CSV, Markdown
- 💬 **Chat-based Q&A** — Ask questions about your documents
- ⚡ **Streaming responses** — Real-time token-by-token output
- 📚 **Source citations** — See which chunks were used
- 🗂️ **Document management** — View and delete documents
- 🌙 **Premium dark UI** — Glassmorphism design

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/health` | Health check |
| `POST` | `/api/upload` | Upload a document |
| `POST` | `/api/chat` | Ask a question (SSE stream) |
| `GET` | `/api/documents` | List documents |
| `DELETE` | `/api/documents/{id}` | Delete a document |
| `GET/POST` | `/api/settings` | Get/update settings |

## License

MIT
