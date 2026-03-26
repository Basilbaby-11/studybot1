"""
ai_engine.py — AI model abstraction layer.

DEFAULT PROVIDER: HuggingFace (completely free, no API key, runs locally)
  Model: google/flan-t5-base  — downloads automatically on first run (~300 MB)

Other providers (require API keys):
  • "anthropic"   — Claude via Anthropic SDK
  • "openai"      — GPT-4o via OpenAI SDK

Switch provider in config.py:  AI_PROVIDER = "huggingface"

Usage:
    from backend.ai_engine import ask_ai, build_study_prompt
"""
import requests
from typing import List, Dict, Optional
from backend.config import (
    AI_PROVIDER, ANTHROPIC_API_KEY, ANTHROPIC_MODEL,
    OPENAI_API_KEY, HF_MODEL, HF_API_KEY, HF_API_MODEL, HF_MODE, OPENROUTER_API_KEY, OPENROUTER_MODEL
)

# ── HuggingFace model options (all free, no API key) ─────────────────
# Swap HF_MODEL in config.py to any of these:
#
#  "google/flan-t5-base"        ~300 MB  fast,  good for Q&A
#  "google/flan-t5-large"       ~800 MB  better answers
#  "google/flan-t5-xl"          ~3 GB    much better, needs more RAM
#  "mistralai/Mistral-7B-v0.1"  ~14 GB   best quality, needs GPU
#  "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  ~2.2 GB  good chat model
# ─────────────────────────────────────────────────────────────────────


# ── Prompt Builders ───────────────────────────────────────────────────

SYSTEM_BASE = """You are StudyBot, an expert academic tutor. Your role:
- Answer student questions clearly and accurately
- Use examples, analogies, and step-by-step explanations
- Format responses with headers/bullets when helpful
- Cite retrieved document context when relevant
- Be encouraging and patient"""

def build_chat_system(rag_context: str = "", username: str = "") -> str:
    system = SYSTEM_BASE
    if username:
        system += f"\n\nStudent name: {username}"
    if rag_context:
        system += f"""

The student has uploaded study documents. Use the retrieved context below to answer accurately.
Always prefer information from these documents over general knowledge when relevant.

RETRIEVED DOCUMENT CONTEXT:
{rag_context}
"""
    return system


def build_quiz_prompt(topic: str, num: int, difficulty: str) -> str:
    return f"""Generate exactly {num} {difficulty} multiple-choice questions (MCQs) about: "{topic}"

Format each question EXACTLY like this:
Q[N]. [Clear, specific question]
A) [Option]
B) [Option]
C) [Option]
D) [Option]
✅ Answer: [Letter]) [Brief explanation why this is correct]

Requirements:
- Questions should be exam-quality and unambiguous
- Cover different aspects of the topic
- Progress from easier to harder
- Wrong options should be plausible (not obviously incorrect)"""


def build_summary_prompt(topic: str, fmt: str) -> str:
    fmt_instructions = {
        "bullet_points": "Use organized bullet points with clear headings",
        "flashcards":    "Format as Q&A flashcard pairs (Q: ... / A: ...)",
        "cornell_notes": "Use Cornell Notes format: Main Notes | Cues | Summary",
        "mind_map":      "Create a text-based mind map with main branches and sub-branches",
    }
    style = fmt_instructions.get(fmt, "Use bullet points")
    return f"""Create a comprehensive study summary for: "{topic}"

Format: {style}

Include:
- Core definition and key concept
- 5-8 essential points students must know
- Important terminology with brief definitions
- Real-world applications or examples
- Common exam pitfalls or misconceptions
- Quick memory tips or mnemonics

Keep it concise but complete — suitable for exam revision."""


def build_explain_prompt(concept: str, level: str) -> str:
    level_desc = {
        "eli5":     "Explain like I'm 5 — use extremely simple language, analogies from daily life",
        "student":  "Explain for an undergraduate student — clear but technically accurate",
        "advanced": "Explain for a graduate student — include technical depth and edge cases",
        "expert":   "Explain for a domain expert — use precise terminology and nuanced detail",
    }
    desc = level_desc.get(level, level_desc["student"])
    return f"""Explain the concept: "{concept}"

Level: {desc}

Structure your explanation as:
1. One-sentence definition
2. Intuition / core idea (why does this exist / what problem does it solve?)
3. Step-by-step breakdown (if procedural) or key components (if conceptual)
4. Concrete worked example
5. Common misconceptions
6. Connections to related concepts

Be thorough but focused."""


# ── Provider: Anthropic ───────────────────────────────────────────────

def _ask_anthropic(messages: List[Dict], system: str) -> str:
    try:
        import anthropic
    except ImportError:
        raise RuntimeError("anthropic not installed. Run: pip install anthropic")

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    response = client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=1500,
        system=system,
        messages=messages,
    )
    return response.content[0].text


# ── Provider: OpenAI ──────────────────────────────────────────────────

def _ask_openai(messages: List[Dict], system: str) -> str:
    try:
        from openai import OpenAI
    except ImportError:
        raise RuntimeError("openai not installed. Run: pip install openai")

    client = OpenAI(api_key=OPENAI_API_KEY)
    full_messages = [{"role": "system", "content": system}] + messages
    response = client.chat.completions.create(
        model="gpt-4o",
        max_tokens=1500,
        messages=full_messages,
    )
    return response.choices[0].message.content


# ── Provider: HuggingFace Inference API (cloud — needs HF_API_KEY) ───

def _ask_huggingface_api(messages: List[Dict], system: str) -> str:
    """
    Use the HuggingFace Inference API with Mistral-7B-Instruct (or any
    chat-capable model set in HF_API_MODEL).  Requires HF_API_KEY.
    """
    try:
        from huggingface_hub import InferenceClient
    except ImportError:
        raise RuntimeError(
            "huggingface_hub not installed.\n"
            "Run:  pip install huggingface_hub\n"
            "Then restart the server."
        )

    if not HF_API_KEY:
        raise RuntimeError(
            "HF_API_KEY is not set.  Add it to your .env file:\n"
            "  HF_API_KEY=hf_your_token_here\n"
            "Get a free token at https://huggingface.co/settings/tokens"
        )

    client = InferenceClient(model=HF_API_MODEL, token=HF_API_KEY)

    # Build message list: system + last 6 turns to stay within token limits
    hf_messages = [{"role": "system", "content": system}]
    hf_messages += messages[-6:]

    try:
        response = client.chat_completion(
            messages=hf_messages,
            max_tokens=500,
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()
    except Exception as exc:
        # Surface a clear error rather than a silent empty string
        raise RuntimeError(f"HuggingFace API error: {exc}") from exc


# ── Provider: HuggingFace (local — FREE, no API key needed) ──────────

_hf_pipeline = None
_hf_model_name = None   # track which model is loaded

def _get_hf_pipeline():
    """
    Lazy-load the HuggingFace pipeline.
    First call downloads the model (~300 MB for flan-t5-base).
    Subsequent calls reuse the cached model from disk.
    """
    global _hf_pipeline, _hf_model_name

    if _hf_pipeline is not None and _hf_model_name == HF_MODEL:
        return _hf_pipeline

    try:
        from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
        import torch
    except ImportError:
        raise RuntimeError(
            "HuggingFace libraries not installed.\n"
            "Run:  pip install transformers torch\n"
            "Then restart the server."
        )

    print(f"\n[StudyBot] Loading HuggingFace model: {HF_MODEL}")
    print("[StudyBot] First run downloads the model — this may take a few minutes...")

    # Use GPU if available, otherwise CPU
    device = 0 if torch.cuda.is_available() else -1
    device_name = "GPU" if device == 0 else "CPU"
    print(f"[StudyBot] Running on: {device_name}")

    _hf_pipeline = pipeline(
        "text2text-generation",
        model=HF_MODEL,
        device=device,
        max_new_tokens=512,
        do_sample=True,          # enables more natural varied responses
        temperature=0.7,
        repetition_penalty=1.3,  # reduce repetitive output
    )
    _hf_model_name = HF_MODEL
    print(f"[StudyBot] ✅ Model ready: {HF_MODEL}\n")
    return _hf_pipeline


def _build_hf_prompt(messages: List[Dict], system: str) -> str:
    """
    Build a well-structured prompt for flan-t5 and similar seq2seq models.
    Keeps the last 3 conversation turns to stay within token limits.
    """
    # Truncate system prompt to avoid overflowing context window
    sys_short = system[:600] if len(system) > 600 else system

    # Last 3 user/assistant turns (6 messages max)
    recent = messages[-6:]
    convo  = "\n".join(
        f"{'Student' if m['role'] == 'user' else 'Tutor'}: {m['content']}"
        for m in recent
    )

    # Final prompt — clearly instructs the model what to do
    return (
        f"You are an expert academic tutor. {sys_short}\n\n"
        f"Conversation:\n{convo}\n\n"
        f"Tutor (give a clear, helpful, detailed answer):"
    )


def _ask_huggingface(messages: List[Dict], system: str) -> str:
    pipe   = _get_hf_pipeline()
    prompt = _build_hf_prompt(messages, system)

    # Truncate prompt if it exceeds model's max input length (512 tokens for flan-t5-base)
    max_input_chars = 1800   # ~450 tokens, safe for flan-t5-base
    if len(prompt) > max_input_chars:
        prompt = prompt[-max_input_chars:]

    result = pipe(prompt)
    answer = result[0]["generated_text"].strip()

    # Clean up any prompt echo that some models include
    if "Tutor:" in answer:
        answer = answer.split("Tutor:")[-1].strip()

    return answer if answer else "I'm not sure about that. Could you rephrase your question?"

def generate_chat_response(message,system):
    url = "https://openrouter.ai/api/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": OPENROUTER_MODEL,
        "messages": [
            {"role": "system", "content": "You are a helpful AI tutor."},
            {"role": "user", "content": message}
        ]
    }

    response = requests.post(url, headers=headers, json=data)

    if response.status_code != 200:
        raise Exception(f"OpenRouter error: {response.text}")

    return response.json()["choices"][0]["message"]["content"]
# ── Public API ────────────────────────────────────────────────────────

def ask_ai(messages: List[Dict],
           system: str,
           provider: Optional[str] = None) -> str:
    """
    Route to the configured AI provider.

    Provider selection:
      "anthropic"   — Claude via Anthropic SDK
      "openai"      — GPT-4o via OpenAI SDK
      "huggingface" — HuggingFace (mode controlled by HF_MODE env var):
                        "api"   → Inference API, Mistral-7B (needs HF_API_KEY)
                        "local" → local transformers pipeline (no API key)

    When HF_MODE="api", a failure automatically falls back to the local
    transformers pipeline so the bot stays online.

    Args:
        messages: List of {"role": "user"|"assistant", "content": str}
        system:   System prompt string
        provider: Override AI_PROVIDER from config (optional)

    Returns:
        AI response string
    """
    p = provider or AI_PROVIDER

    if p == "anthropic":
        return _ask_anthropic(messages, system)

    elif p == "openai":
        return _ask_openai(messages, system)
    
    elif p == "openrouter":
        try:
            user_message = messages[-1]["content"]
            return generate_chat_response(messages,system)
        except Exception as exc:
            print(f"[StudyBot] ⚠️  OpenRouter API failed ({exc}).")
            return (
                "I'm having trouble reaching the AI service right now. "
                "Please try again in a moment."
            )
    elif p == "huggingface":
        if HF_MODE == "api":
            try:
                return _ask_huggingface_api(messages, system)
            except Exception as exc:
                # Graceful fallback: log the error and try local pipeline
                print(f"[StudyBot] ⚠️  HF API failed ({exc}); falling back to local model.")
                try:
                    return _ask_huggingface(messages, system)
                except Exception as local_exc:
                    print(f"[StudyBot] ❌ Local fallback also failed: {local_exc}")
                    return (
                        "I'm having trouble reaching the AI service right now. "
                        "Please try again in a moment."
                    )
        else:
            # HF_MODE == "local"
            return _ask_huggingface(messages, system)

    else:
        raise ValueError(
            f"Unknown AI provider: {p}. "
            "Set AI_PROVIDER to 'anthropic', 'openai', or 'huggingface'"
        )
