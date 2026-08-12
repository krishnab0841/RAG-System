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
| **Vector DB** | LangChain `InMemoryVectorStore` backed by Hugging Face embeddings |
| **Frontend** | Vanilla HTML/CSS/JS (SPA) |
| **Backend Deployment** | Render web service |
| **Frontend Deployment** | Vercel static site |

**Live frontend:** [rag-system-iggl.vercel.app](rag-system-iggl.vercel.app)

## ✨ Features

- 📄 **Multi-format Document Upload** — Supports PDF, TXT, DOCX, CSV, and Markdown files.
- 🌐 **Web URL Ingestion** — Directly scrape and ingest content from any web link.
- 💬 **Context-Aware Q&A** — Ask questions and receive accurate answers based exclusively on the provided context.
- ⚡ **Real-time Streaming** — Experience fast, token-by-token streaming responses via Server-Sent Events (SSE).
- 📚 **Source Citations** — Transparency with source document citations and similarity scores for every answer.
- 🗂️ **Document Management** — View and delete indexed documents directly from the UI.
- ⚙️ **Runtime Preferences** — Update the model and retrieval configuration; API keys stay in the hosting environment.
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

### 4. Access the API

Open [http://localhost:8000/docs](http://localhost:8000/docs) for the API documentation, or call [http://localhost:8000/api/health](http://localhost:8000/api/health).

## ☁️ Deploy the backend to Render

The repository includes `render.yaml`, so it can be deployed as a Render Blueprint. The service is explicitly configured for Render's free plan. It uses Python 3.11, installs `requirements.txt`, starts `uvicorn app.main:app`, and checks `/api/health`.

1. Push this repository to GitHub, GitLab, or Bitbucket. Do not commit `.env`.
2. In Render, select **New → Blueprint** and select the repository. Render reads `render.yaml` and creates a free web service. You can also select **New → Web Service**, choose the repository, select the **Free** instance type, and use the same build/start commands from `render.yaml`.
3. Set these environment variables in the service configuration:

   ```text
   GROQ_API_KEY=your_groq_key
   HUGGINGFACE_API_KEY=your_hugging_face_token
   CORS_ORIGINS=https://rag-system-iggl.vercel.app
   ```

   For a custom frontend domain, add it as a comma-separated value too, for example: `https://app.example.com,https://your-frontend.vercel.app`.
4. Deploy and copy the resulting public URL, such as `https://rag-system-backend.onrender.com`.
5. Verify `https://your-render-url/api/health` responds with `status: "ok"`.

Render's filesystem and this application's vector store are ephemeral. Uploaded documents and their index disappear after a restart, redeploy, or scale-out. Add persistent object storage and a managed vector database before using this for durable or multi-user workloads.

Render free web services can spin down after inactivity, so the first request after an idle period can be slow.

## ▲ Connect and deploy the frontend on Vercel

1. In `static/config.js`, set `window.RAG_API_BASE_URL` to the Render URL from the prior section, without a trailing slash.

   ```js
   window.RAG_API_BASE_URL = "https://rag-system-backend.onrender.com";
   ```

2. Import the same repository in Vercel. The included `vercel.json` deploys only the static frontend; it no longer deploys the FastAPI backend as a serverless function.
3. Deploy. Open the Vercel URL and upload a small text file to confirm browser requests reach Render.
4. If the browser reports a CORS error, make the Vercel URL exactly match one of the `CORS_ORIGINS` values in Render, then redeploy the Render service.

The live production frontend is [rag-system-iggl.vercel.app](https://rag-system-iggl.vercel.app/).

For preview deployments, either add each preview URL to `CORS_ORIGINS` or test only against the production Vercel domain.

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
