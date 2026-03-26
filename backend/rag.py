"""
rag.py — Retrieval-Augmented Generation pipeline.

Architecture:
  Document → chunk → embed (sentence-transformers) → FAISS index
  Query    → embed → similarity search → top-K chunks → LLM context

Swap FAISS for Chroma or Pinecone by replacing the VectorStore class below.
"""

import json
import pickle
import re
from pathlib import Path
from typing import List, Tuple, Optional
from datetime import datetime

from backend.config import (
    CHUNK_SIZE, CHUNK_OVERLAP, TOP_K_RESULTS,
    EMBEDDING_MODEL, FAISS_INDEX
)


# ── Chunker ────────────────────────────────────────────────────────────

def chunk_text(text: str,
               chunk_size: int = CHUNK_SIZE,
               overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Split text into overlapping word-based chunks.
    LangChain RecursiveCharacterTextSplitter equivalent.

    Args:
        text:       Raw document text
        chunk_size: Target chunk size in characters
        overlap:    Overlap between adjacent chunks in characters

    Returns:
        List of text chunk strings
    """
    # Split on natural sentence/paragraph boundaries first
    paragraphs = re.split(r'\n{2,}', text)
    chunks: List[str] = []
    current = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        if len(current) + len(para) + 1 <= chunk_size:
            current = (current + "\n\n" + para).strip()
        else:
            if current:
                chunks.append(current)
                # Keep overlap from end of current chunk
                words = current.split()
                overlap_words = int(overlap / 5)  # rough char→word ratio
                current = " ".join(words[-overlap_words:]) + "\n\n" + para
            else:
                # Para itself is larger than chunk_size — split by sentences
                sentences = re.split(r'(?<=[.!?])\s+', para)
                for sent in sentences:
                    if len(current) + len(sent) <= chunk_size:
                        current = (current + " " + sent).strip()
                    else:
                        if current:
                            chunks.append(current)
                        current = sent

    if current.strip():
        chunks.append(current.strip())

    return [c for c in chunks if len(c) > 30]  # drop very short chunks


# ── Embedding (sentence-transformers) ─────────────────────────────────

_embedder = None

def get_embedder():
    """Lazy-load the sentence-transformer model once."""
    global _embedder
    if _embedder is None:
        try:
            from sentence_transformers import SentenceTransformer
            _embedder = SentenceTransformer(EMBEDDING_MODEL)
        except ImportError:
            raise RuntimeError(
                "sentence-transformers not installed. "
                "Run: pip install sentence-transformers"
            )
    return _embedder

def embed(texts: List[str]) -> List[List[float]]:
    """Return float32 embedding vectors for a list of texts."""
    model = get_embedder()
    return model.encode(texts, show_progress_bar=False).tolist()


# ── FAISS Vector Store ─────────────────────────────────────────────────
# To swap to Chroma:  replace FAISSStore with ChromaStore below.
# To swap to Pinecone: replace with PineconeStore below.

class FAISSStore:
    """
    In-process FAISS index with metadata sidecar.
    Persisted to disk at FAISS_INDEX path.
    """

    INDEX_FILE = FAISS_INDEX / "index.faiss"
    META_FILE  = FAISS_INDEX / "meta.pkl"

    def __init__(self):
        self.index   = None   # faiss.IndexFlatIP
        self.chunks  : List[str]  = []
        self.sources : List[str]  = []   # filename for each chunk
        self._load()

    # ── persistence ─────────────────────────

    def _load(self):
        if self.INDEX_FILE.exists() and self.META_FILE.exists():
            try:
                import faiss
                self.index = faiss.read_index(str(self.INDEX_FILE))
                with open(self.META_FILE, "rb") as f:
                    meta = pickle.load(f)
                self.chunks  = meta["chunks"]
                self.sources = meta["sources"]
            except Exception:
                self._reset()

    def _save(self):
        try:
            import faiss
            faiss.write_index(self.index, str(self.INDEX_FILE))
            with open(self.META_FILE, "wb") as f:
                pickle.dump({"chunks": self.chunks, "sources": self.sources}, f)
        except Exception as e:
            print(f"[RAG] Warning: could not save index: {e}")

    def _reset(self):
        self.index   = None
        self.chunks  = []
        self.sources = []

    # ── CRUD ────────────────────────────────

    def add_document(self, chunks: List[str], source_name: str) -> int:
        """
        Embed chunks and add to the FAISS index.
        Returns number of new chunks added.
        """
        if not chunks:
            return 0

        import faiss
        import numpy as np

        vectors = embed(chunks)
        matrix  = np.array(vectors, dtype="float32")
        faiss.normalize_L2(matrix)   # cosine similarity via inner product

        dim = matrix.shape[1]
        if self.index is None:
            self.index = faiss.IndexFlatIP(dim)

        self.index.add(matrix)
        self.chunks.extend(chunks)
        self.sources.extend([source_name] * len(chunks))
        self._save()
        return len(chunks)

    def search(self, query: str, top_k: int = TOP_K_RESULTS
               ) -> List[Tuple[str, str, float]]:
        """
        Semantic search.
        Returns list of (chunk_text, source_filename, score).
        """
        if self.index is None or self.index.ntotal == 0:
            return []

        import faiss
        import numpy as np

        q_vec  = np.array(embed([query]), dtype="float32")
        faiss.normalize_L2(q_vec)
        scores, indices = self.index.search(q_vec, min(top_k, self.index.ntotal))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and score > 0.15:   # relevance threshold
                results.append((self.chunks[idx], self.sources[idx], float(score)))
        return results

    def remove_document(self, source_name: str) -> int:
        """
        Remove all chunks belonging to source_name.
        FAISS doesn't support deletion, so we rebuild the index.
        """
        keep_idx = [i for i, s in enumerate(self.sources) if s != source_name]
        removed  = len(self.chunks) - len(keep_idx)

        if removed == 0:
            return 0

        kept_chunks  = [self.chunks[i]  for i in keep_idx]
        kept_sources = [self.sources[i] for i in keep_idx]
        self._reset()

        if kept_chunks:
            # Re-index remaining chunks
            import faiss, numpy as np
            vectors = embed(kept_chunks)
            matrix  = np.array(vectors, dtype="float32")
            faiss.normalize_L2(matrix)
            dim = matrix.shape[1]
            self.index = faiss.IndexFlatIP(dim)
            self.index.add(matrix)
            self.chunks  = kept_chunks
            self.sources = kept_sources

        self._save()
        return removed

    @property
    def total_chunks(self) -> int:
        return len(self.chunks)

    def unique_sources(self) -> List[str]:
        return list(dict.fromkeys(self.sources))


# ── Chroma Alternative (swap-in) ─────────────────────────────────────
# Uncomment and use ChromaStore instead of FAISSStore to switch backends.
#
# class ChromaStore:
#     def __init__(self):
#         import chromadb
#         self.client = chromadb.PersistentClient(path=str(FAISS_INDEX))
#         self.col = self.client.get_or_create_collection("studybot")
#
#     def add_document(self, chunks, source_name):
#         ids = [f"{source_name}_{i}" for i in range(len(chunks))]
#         vectors = embed(chunks)
#         self.col.add(embeddings=vectors, documents=chunks, ids=ids,
#                      metadatas=[{"source": source_name}]*len(chunks))
#         return len(chunks)
#
#     def search(self, query, top_k=TOP_K_RESULTS):
#         q_vec = embed([query])
#         res = self.col.query(query_embeddings=q_vec, n_results=top_k)
#         return [(doc, meta["source"], 1.0)
#                 for doc, meta in zip(res["documents"][0], res["metadatas"][0])]


# ── Pinecone Alternative (swap-in) ───────────────────────────────────
# class PineconeStore:
#     def __init__(self):
#         import pinecone
#         pinecone.init(api_key=os.getenv("PINECONE_API_KEY"),
#                       environment=os.getenv("PINECONE_ENV"))
#         self.index = pinecone.Index("studybot")
#
#     def add_document(self, chunks, source_name):
#         vectors = embed(chunks)
#         items = [(f"{source_name}_{i}", v, {"text": c, "source": source_name})
#                  for i, (c, v) in enumerate(zip(chunks, vectors))]
#         self.index.upsert(items)
#         return len(chunks)
#
#     def search(self, query, top_k=TOP_K_RESULTS):
#         q_vec = embed([query])
#         res = self.index.query(vector=q_vec[0], top_k=top_k, include_metadata=True)
#         return [(m.metadata["text"], m.metadata["source"], m.score)
#                 for m in res.matches]


# ── Global store instance ─────────────────────────────────────────────
_store: Optional[FAISSStore] = None

def get_store() -> FAISSStore:
    global _store
    if _store is None:
        _store = FAISSStore()
    return _store


# ── Public RAG API ────────────────────────────────────────────────────

def index_document(text: str, source_name: str) -> int:
    """Chunk text and add to the vector store. Returns chunk count."""
    chunks = chunk_text(text)
    return get_store().add_document(chunks, source_name)


def retrieve_context(query: str, top_k: int = TOP_K_RESULTS
                     ) -> Tuple[str, List[str]]:
    """
    Retrieve relevant chunks for a query.

    Returns:
        context_text: Formatted string ready to inject into LLM prompt
        sources:      Unique source document names used
    """
    results = get_store().search(query, top_k)
    if not results:
        return "", []

    lines     = []
    sources   = []
    for i, (chunk, source, score) in enumerate(results, 1):
        lines.append(f"[Context {i} — from '{source}' (relevance: {score:.2f})]:\n{chunk}")
        if source not in sources:
            sources.append(source)

    context = "\n\n".join(lines)
    return context, sources


def remove_document(source_name: str) -> int:
    """Remove a document from the index."""
    return get_store().remove_document(source_name)


def list_indexed_documents() -> List[str]:
    """Return all indexed source names."""
    return get_store().unique_sources()
