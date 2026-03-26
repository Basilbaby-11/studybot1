"""
main.py — StudyBot FastAPI application entry point.

Run with:
    uvicorn backend.main:app --reload --port 8000

API Docs auto-generated at:
    http://localhost:8000/docs   (Swagger UI)
    http://localhost:8000/redoc  (ReDoc)
"""

import sqlite3
import uuid
import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List

from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

import jwt as pyjwt  # pip install PyJWT

from backend.config import (
    APP_TITLE, APP_VERSION, CORS_ORIGINS,
    SECRET_KEY, ALGORITHM, ACCESS_TOKEN_EXPIRE_MINUTES,
    DATABASE_PATH, MAX_HISTORY_PER_SESSION
)
from backend.models import (
    ChatRequest, ChatResponse,
    LoginRequest, RegisterRequest, TokenResponse,
    QuizRequest, SummaryRequest, ExplainRequest,
    DocumentListResponse, DocumentInfo,
    HistoryResponse, SessionSummary, SessionMessages, Message
)
from backend.ai_engine import (
    ask_ai,
    build_chat_system, build_quiz_prompt,
    build_summary_prompt, build_explain_prompt
)
from backend.rag import (
    index_document, retrieve_context,
    remove_document, list_indexed_documents, get_store
)
from backend.pdf_handler import extract_text, save_upload, get_pdf_metadata


# ══════════════════════════════════════════════════════════════════════
#  APP SETUP
# ══════════════════════════════════════════════════════════════════════

app = FastAPI(
    title=APP_TITLE,
    version=APP_VERSION,
    description="AI Study Assistant — FastAPI + RAG + JWT Auth",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the frontend from /  (index.html + static assets)
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


# ══════════════════════════════════════════════════════════════════════
#  DATABASE  (SQLite via stdlib — swap with SQLAlchemy for PostgreSQL)
# ══════════════════════════════════════════════════════════════════════

def get_db():
    conn = sqlite3.connect(str(DATABASE_PATH))
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    db = get_db()
    db.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            username    TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            full_name   TEXT,
            created_at  TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS sessions (
            id          TEXT PRIMARY KEY,
            user_id     INTEGER REFERENCES users(id),
            title       TEXT DEFAULT 'New Conversation',
            created_at  TEXT DEFAULT (datetime('now')),
            last_active TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS messages (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id  TEXT REFERENCES sessions(id) ON DELETE CASCADE,
            role        TEXT NOT NULL,
            content     TEXT NOT NULL,
            timestamp   TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS documents (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            filename    TEXT NOT NULL,
            size_kb     REAL,
            chunks      INTEGER,
            uploaded_by INTEGER REFERENCES users(id),
            uploaded_at TEXT DEFAULT (datetime('now'))
        );
    """)
    db.commit()
    db.close()

init_db()


# ══════════════════════════════════════════════════════════════════════
#  AUTH  (JWT)
# ══════════════════════════════════════════════════════════════════════

security = HTTPBearer(auto_error=False)

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def create_jwt(username: str) -> str:
    payload = {
        "sub": username,
        "exp": datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
        "iat": datetime.utcnow(),
    }
    return pyjwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

def decode_jwt(token: str) -> Optional[str]:
    try:
        payload = pyjwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload.get("sub")
    except pyjwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except pyjwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> Optional[str]:
    """Returns username if token is valid, else None (anonymous allowed)."""
    if credentials is None:
        return None
    return decode_jwt(credentials.credentials)

def require_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> str:
    """Raises 401 if not authenticated."""
    if credentials is None:
        raise HTTPException(status_code=401, detail="Authentication required")
    return decode_jwt(credentials.credentials)


# ── Auth endpoints ────────────────────────────────────────────────────

@app.post("/auth/register", response_model=TokenResponse, tags=["Auth"])
def register(req: RegisterRequest):
    db = get_db()
    existing = db.execute("SELECT id FROM users WHERE username=?",
                          (req.username,)).fetchone()
    if existing:
        raise HTTPException(status_code=400, detail="Username already taken")

    db.execute(
        "INSERT INTO users (username, password_hash, full_name) VALUES (?,?,?)",
        (req.username, hash_password(req.password), req.full_name or req.username)
    )
    db.commit()
    db.close()
    token = create_jwt(req.username)
    return TokenResponse(
        access_token=token,
        username=req.username,
        full_name=req.full_name or req.username
    )

@app.post("/auth/login", response_model=TokenResponse, tags=["Auth"])
def login(req: LoginRequest):
    db = get_db()
    user = db.execute(
        "SELECT * FROM users WHERE username=? AND password_hash=?",
        (req.username, hash_password(req.password))
    ).fetchone()
    db.close()
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = create_jwt(req.username)
    return TokenResponse(
        access_token=token,
        username=user["username"],
        full_name=user["full_name"] or user["username"]
    )


# ══════════════════════════════════════════════════════════════════════
#  CHAT
# ══════════════════════════════════════════════════════════════════════

@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
def chat(
    req: ChatRequest,
    username: Optional[str] = Depends(get_current_user)
):
    db = get_db()

    # ── Session management ───────────────────────────────────────────
    session_id = req.session_id
    if not session_id:
        session_id = str(uuid.uuid4())
        user_id = None
        if username:
            row = db.execute("SELECT id FROM users WHERE username=?",
                             (username,)).fetchone()
            user_id = row["id"] if row else None
        db.execute(
            "INSERT INTO sessions (id, user_id, title) VALUES (?,?,?)",
            (session_id, user_id, req.message[:50])
        )
        db.commit()

    # ── Load history ─────────────────────────────────────────────────
    rows = db.execute(
        "SELECT role, content FROM messages WHERE session_id=? "
        "ORDER BY id DESC LIMIT ?",
        (session_id, MAX_HISTORY_PER_SESSION * 2)
    ).fetchall()
    history = [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]

    # ── RAG context ──────────────────────────────────────────────────
    rag_context, sources = "", []
    if req.use_rag:
        rag_context, sources = retrieve_context(req.message)

    # ── Build messages & call AI ─────────────────────────────────────
    history.append({"role": "user", "content": req.message})
    system  = build_chat_system(rag_context, username or "")
    answer  = ask_ai(history, system)

    # ── Persist ──────────────────────────────────────────────────────
    db.execute(
        "INSERT INTO messages (session_id, role, content) VALUES (?,?,?)",
        (session_id, "user", req.message)
    )
    db.execute(
        "INSERT INTO messages (session_id, role, content) VALUES (?,?,?)",
        (session_id, "assistant", answer)
    )
    db.execute(
        "UPDATE sessions SET last_active=datetime('now') WHERE id=?",
        (session_id,)
    )
    db.commit()
    db.close()

    from backend.config import AI_PROVIDER
    return ChatResponse(
        response=answer,
        session_id=session_id,
        sources=sources,
        model_used=AI_PROVIDER
    )


# ══════════════════════════════════════════════════════════════════════
#  STUDY TOOLS
# ══════════════════════════════════════════════════════════════════════

@app.post("/quiz", tags=["Study Tools"])
def generate_quiz(req: QuizRequest):
    prompt = build_quiz_prompt(req.topic, req.num_questions, req.difficulty)
    result = ask_ai([{"role": "user", "content": prompt}], "You are a quiz generator.")
    return {"quiz": result, "topic": req.topic, "questions": req.num_questions}

@app.post("/summary", tags=["Study Tools"])
def generate_summary(req: SummaryRequest):
    # Include RAG context if documents are indexed
    rag_context, sources = retrieve_context(req.topic)
    system = build_chat_system(rag_context)
    prompt = build_summary_prompt(req.topic, req.format)
    result = ask_ai([{"role": "user", "content": prompt}], system)
    return {"summary": result, "topic": req.topic, "sources": sources}

@app.post("/explain", tags=["Study Tools"])
def explain_concept(req: ExplainRequest):
    rag_context, sources = retrieve_context(req.concept)
    system = build_chat_system(rag_context)
    prompt = build_explain_prompt(req.concept, req.level)
    result = ask_ai([{"role": "user", "content": prompt}], system)
    return {"explanation": result, "concept": req.concept, "sources": sources}


# ══════════════════════════════════════════════════════════════════════
#  DOCUMENTS / RAG
# ══════════════════════════════════════════════════════════════════════

@app.post("/documents/upload", tags=["Documents"])
async def upload_document(
    file: UploadFile = File(...),
    username: Optional[str] = Depends(get_current_user)
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in (".pdf", ".txt", ".md"):
        raise HTTPException(status_code=400,
                            detail="Only .pdf, .txt, .md files are supported")

    content   = await file.read()
    saved_path = save_upload(file.filename, content)
    text       = extract_text(saved_path)

    if not text.strip():
        raise HTTPException(status_code=422, detail="Could not extract text from file")

    chunk_count = index_document(text, file.filename)
    size_kb     = len(content) / 1024

    # Persist document record
    db = get_db()
    user_id = None
    if username:
        row = db.execute("SELECT id FROM users WHERE username=?",
                         (username,)).fetchone()
        user_id = row["id"] if row else None

    db.execute(
        "INSERT INTO documents (filename, size_kb, chunks, uploaded_by) VALUES (?,?,?,?)",
        (file.filename, round(size_kb, 2), chunk_count, user_id)
    )
    db.commit()
    db.close()

    return {
        "message": f"'{file.filename}' indexed successfully",
        "chunks": chunk_count,
        "size_kb": round(size_kb, 2)
    }

@app.get("/documents", response_model=DocumentListResponse, tags=["Documents"])
def list_documents():
    db = get_db()
    rows = db.execute(
        "SELECT filename, size_kb, chunks, uploaded_at FROM documents ORDER BY uploaded_at DESC"
    ).fetchall()
    db.close()
    docs = [
        DocumentInfo(
            filename=r["filename"],
            size_kb=r["size_kb"],
            chunks=r["chunks"],
            uploaded_at=datetime.fromisoformat(r["uploaded_at"])
        ) for r in rows
    ]
    return DocumentListResponse(documents=docs, total=len(docs))

@app.delete("/documents/{filename}", tags=["Documents"])
def delete_document(filename: str, _: str = Depends(require_user)):
    removed = remove_document(filename)
    db = get_db()
    db.execute("DELETE FROM documents WHERE filename=?", (filename,))
    db.commit()
    db.close()
    return {"message": f"Removed '{filename}' ({removed} chunks deleted from index)"}


# ══════════════════════════════════════════════════════════════════════
#  CHAT HISTORY
# ══════════════════════════════════════════════════════════════════════

@app.get("/history", response_model=HistoryResponse, tags=["History"])
def get_history(username: str = Depends(require_user)):
    db = get_db()
    user = db.execute("SELECT id FROM users WHERE username=?",
                      (username,)).fetchone()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    rows = db.execute(
        """SELECT s.id, s.title, s.created_at, s.last_active,
                  COUNT(m.id) as message_count
           FROM sessions s
           LEFT JOIN messages m ON m.session_id = s.id
           WHERE s.user_id=?
           GROUP BY s.id
           ORDER BY s.last_active DESC""",
        (user["id"],)
    ).fetchall()
    db.close()

    sessions = [
        SessionSummary(
            session_id=r["id"],
            title=r["title"],
            message_count=r["message_count"],
            created_at=datetime.fromisoformat(r["created_at"]),
            last_active=datetime.fromisoformat(r["last_active"])
        ) for r in rows
    ]
    return HistoryResponse(sessions=sessions)

@app.get("/history/{session_id}", response_model=SessionMessages, tags=["History"])
def get_session(session_id: str, username: str = Depends(require_user)):
    db = get_db()
    session = db.execute(
        "SELECT * FROM sessions WHERE id=?", (session_id,)
    ).fetchone()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    msgs = db.execute(
        "SELECT role, content, timestamp FROM messages WHERE session_id=? ORDER BY id",
        (session_id,)
    ).fetchall()
    db.close()

    return SessionMessages(
        session_id=session_id,
        title=session["title"],
        messages=[
            Message(
                role=m["role"],
                content=m["content"],
                timestamp=datetime.fromisoformat(m["timestamp"])
            ) for m in msgs
        ]
    )

@app.delete("/history/{session_id}", tags=["History"])
def delete_session(session_id: str, _: str = Depends(require_user)):
    db = get_db()
    db.execute("DELETE FROM messages WHERE session_id=?", (session_id,))
    db.execute("DELETE FROM sessions WHERE id=?", (session_id,))
    db.commit()
    db.close()
    return {"message": "Session deleted"}


# ══════════════════════════════════════════════════════════════════════
#  FRONTEND SERVE
# ══════════════════════════════════════════════════════════════════════
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
@app.get("/", include_in_schema=False)
def serve_frontend():
    index = FRONTEND_DIR / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return JSONResponse({"message": f"{APP_TITLE} v{APP_VERSION} is running",
                         "docs": "/docs"})

@app.get("/health", tags=["Meta"])
def health():
    return {
        "status": "ok",
        "version": APP_VERSION,
        "indexed_documents": len(list_indexed_documents()),
        "total_chunks": get_store().total_chunks,
    }
