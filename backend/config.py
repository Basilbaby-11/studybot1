
"""
config.py — Central configuration for StudyBot
All environment variables and app settings live here.
"""

import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file
# ── Paths ──────────────────────────────────────────────────────────────
BASE_DIR       = Path(__file__).resolve().parent.parent
DATA_DIR       = BASE_DIR / "data" / "documents"
DATABASE_PATH  = BASE_DIR / "database" / "chat_history.db"
FAISS_INDEX    = BASE_DIR / "data" / "faiss_index"

DATA_DIR.mkdir(parents=True, exist_ok=True)
FAISS_INDEX.mkdir(parents=True, exist_ok=True)

# ── AI Model settings ──────────────────────────────────────────────────
# "huggingface" — FREE, no API key, downloads model locally (DEFAULT)
# "anthropic"   — Claude API  (set ANTHROPIC_API_KEY)
# "openai"      — GPT-4o API  (set OPENAI_API_KEY)
AI_PROVIDER    = os.getenv("AI_PROVIDER", "")

ANTHROPIC_API_KEY  = os.getenv("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY     = os.getenv("OPENAI_API_KEY", "")
HF_API_KEY         = os.getenv("HF_API_KEY", "") 
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL   = os.getenv(
    "OPENROUTER_MODEL",
    "openchat/openchat-7b"
)  # HuggingFace Inference API token

# HuggingFace provider mode:
#   "api"   — uses HF Inference API (cloud, needs HF_API_KEY, no local download)  ← default
#   "local" — uses transformers pipeline (free, no API key, downloads model)
HF_MODE        = os.getenv("HF_MODE", "api")

# HuggingFace Inference API model (used when HF_MODE=api)
HF_API_MODEL   = os.getenv("HF_API_MODEL", "google/flan-t5-base")

# HuggingFace local model (used when HF_MODE=local, all FREE, no API key needed)
# "google/flan-t5-base"   ~300 MB  fast, decent answers
# "google/flan-t5-large"  ~800 MB  noticeably better
# "google/flan-t5-xl"     ~3 GB    great quality, needs 8GB RAM
# "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  ~2.2 GB  good chat model
HF_MODEL       = os.getenv("HF_MODEL", "google/flan-t5-base")

# Anthropic model (only used if AI_PROVIDER = "anthropic")
ANTHROPIC_MODEL = "claude-3-5-sonnet-20241022"

# ── Embedding model ────────────────────────────────────────────────────
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ── RAG settings ───────────────────────────────────────────────────────
CHUNK_SIZE    = 500     # characters per chunk
CHUNK_OVERLAP = 50      # overlap between consecutive chunks
TOP_K_RESULTS = 4       # how many chunks to retrieve per query

# ── Auth / JWT ─────────────────────────────────────────────────────────
SECRET_KEY     = os.getenv("SECRET_KEY", "studybot-secret-change-in-prod")
ALGORITHM      = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

# ── App ────────────────────────────────────────────────────────────────
APP_TITLE      = "StudyBot API"
APP_VERSION    = "2.0.0"
CORS_ORIGINS   = ["http://localhost:3000", "http://127.0.0.1:5500", "*"]
MAX_HISTORY_PER_SESSION = 50   # keep last N message pairs in memory
