# StudyBot 📖 — AI Study Assistant

Full-stack AI-powered study assistant with:
- **FastAPI** backend + **SQLite** chat history
- **JWT Authentication** (register / login)
- **LangChain-style RAG** with **FAISS** vector search
- **PDF / TXT / MD** document ingestion
- **Vanilla JS + HTML/CSS** frontend (no React)
- **Voice input** (STT) and **voice output** (TTS)
- Pluggable AI: **Anthropic Claude** · OpenAI GPT-4o · HuggingFace (local)

---

## Project Structure

```
studybot/
│
├── backend/
│   ├── __init__.py         # Package marker
│   ├── main.py             # FastAPI app, all routes
│   ├── ai_engine.py        # AI provider abstraction (Anthropic/OpenAI/HF)
│   ├── rag.py              # Chunking, embeddings, FAISS index
│   ├── pdf_handler.py      # PDF/TXT text extraction
│   ├── models.py           # Pydantic request/response schemas
│   └── config.py           # All settings & environment variables
│
├── frontend/
│   ├── index.html          # Chat UI (single page)
│   ├── style.css           # All styles
│   ├── script.js           # Frontend logic — connects to FastAPI
│   └── assets/             # Icons, images (optional)
│
├── data/
│   ├── documents/          # Uploaded PDFs stored here
│   └── faiss_index/        # Persisted FAISS index + metadata
│
├── database/
│   └── chat_history.db     # SQLite — users, sessions, messages, documents
│
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Clone & install

```bash
git clone <your-repo>
cd studybot
pip install -r requirements.txt
```

### 2. Set environment variables

```bash
# .env  (or export directly)
export ANTHROPIC_API_KEY="sk-ant-..."   # for Claude (default)
# export OPENAI_API_KEY="sk-..."        # for GPT-4o
# export AI_PROVIDER="openai"           # switch provider
export SECRET_KEY="your-secure-secret"
```

### 3. Run the server

```bash
uvicorn backend.main:app --reload --port 8000
```

### 4. Open the app

```
http://localhost:8000
```

API docs (auto-generated):
```
http://localhost:8000/docs    # Swagger UI
http://localhost:8000/redoc   # ReDoc
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/auth/register` | Create account → returns JWT |
| POST | `/auth/login`    | Login → returns JWT |
| POST | `/chat`          | Chat with AI (supports RAG, history) |
| POST | `/quiz`          | Generate MCQ quiz |
| POST | `/summary`       | Generate study summary |
| POST | `/explain`       | Deep concept explanation |
| POST | `/documents/upload` | Upload & index PDF/TXT |
| GET  | `/documents`     | List indexed documents |
| DELETE | `/documents/{filename}` | Remove document from index |
| GET  | `/history`       | List user's chat sessions |
| GET  | `/history/{id}`  | Get all messages in a session |
| DELETE | `/history/{id}` | Delete a session |
| GET  | `/health`        | Server status |

---

## Switching AI Provider

In `backend/config.py` or via environment variable:

```python
AI_PROVIDER = "anthropic"    # Claude (default)
AI_PROVIDER = "openai"       # GPT-4o
AI_PROVIDER = "huggingface"  # Local model (offline)
```

---

## Switching Vector Database

In `backend/rag.py`:

```python
# Default — FAISS (no server needed, persisted to disk)
store = FAISSStore()

# Chroma (uncomment ChromaStore class)
# store = ChromaStore()

# Pinecone (uncomment PineconeStore class, needs API key)
# store = PineconeStore()
```

---

## How RAG Works

```
User uploads PDF
        ↓
pdf_handler.py  — extract text
        ↓
rag.py chunk_text()  — split into 500-char overlapping chunks
        ↓
sentence-transformers  — embed each chunk to float32 vector
        ↓
FAISS index  — store vectors + metadata on disk
        ↓
User asks question
        ↓
embed(query) → FAISS similarity search → top-4 chunks
        ↓
Inject chunks into LLM system prompt
        ↓
AI answers using document context
```

---

## Voice Features

| Feature | Technology | Notes |
|---------|-----------|-------|
| STT (Speech → Text) | Web Speech API | Chrome/Edge; Whisper can replace via backend endpoint |
| TTS (Text → Speech) | Web Speech Synthesis | Built into browser; swap with Coqui TTS endpoint |

---

## Database Schema

```sql
users       — id, username, password_hash, full_name, created_at
sessions    — id, user_id, title, created_at, last_active
messages    — id, session_id, role, content, timestamp
documents   — id, filename, size_kb, chunks, uploaded_by, uploaded_at
```
