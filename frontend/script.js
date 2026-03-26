/**
 * script.js — StudyBot Frontend Logic
 *
 * Connects to the FastAPI backend at /api (or localhost:8000 in dev).
 * Handles: Auth, Chat, RAG uploads, History, Quiz/Summary/Explain,
 *          Voice STT, Voice TTS, and in-browser session management.
 */

/* ══════════════════════════════════════════════════════════
   CONFIG
══════════════════════════════════════════════════════════ */

// In production: same origin ("/").  In dev with uvicorn: "http://localhost:8000"
const API_BASE = window.location.hostname === "localhost"
  ? "http://localhost:8000"
  : "";

/* ══════════════════════════════════════════════════════════
   STATE
══════════════════════════════════════════════════════════ */

const state = {
  token:          null,   // JWT
  user:           null,   // { username, full_name }
  currentSession: null,   // active session ID
  sessions:       {},     // local cache { id: { title, msgs, createdAt } }
  selectedVDB:    "faiss",
  autoTTS:        false,
  isRecording:    false,
  recognition:    null,
  loading:        false,
};

/* ══════════════════════════════════════════════════════════
   API HELPERS
══════════════════════════════════════════════════════════ */

async function api(endpoint, options = {}) {
  const headers = { "Content-Type": "application/json", ...options.headers };
  if (state.token) headers["Authorization"] = `Bearer ${state.token}`;

  const res = await fetch(API_BASE + endpoint, {
    ...options,
    headers,
    body: options.body ? JSON.stringify(options.body) : undefined,
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: "Network error" }));
    throw new Error(err.detail || `HTTP ${res.status}`);
  }
  return res.json();
}

async function apiForm(endpoint, formData) {
  const headers = {};
  if (state.token) headers["Authorization"] = `Bearer ${state.token}`;
  const res = await fetch(API_BASE + endpoint, {
    method: "POST",
    headers,
    body: formData,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: "Upload error" }));
    throw new Error(err.detail || `HTTP ${res.status}`);
  }
  return res.json();
}

/* ══════════════════════════════════════════════════════════
   AUTH
══════════════════════════════════════════════════════════ */

async function login() {
  const username = q("#login-user").value.trim();
  const password = q("#login-pass").value;
  if (!username || !password) return toast("⚠️ Fill in all fields");

  try {
    const data = await api("/auth/login", {
      method: "POST",
      body: { username, password },
    });
    applyAuth(data);
    toast("✅ Logged in");
  } catch (e) {
    toast("❌ " + e.message);
  }
}

async function register() {
  const username  = q("#login-user").value.trim();
  const password  = q("#login-pass").value;
  if (!username || !password) return toast("⚠️ Fill in all fields");

  try {
    const data = await api("/auth/register", {
      method: "POST",
      body: { username, password, full_name: username },
    });
    applyAuth(data);
    toast("✅ Registered & logged in");
  } catch (e) {
    toast("❌ " + e.message);
  }
}

function applyAuth(data) {
  state.token = data.access_token;
  state.user  = { username: data.username, full_name: data.full_name };

  q("#logged-out-view").style.display = "none";
  q("#logged-in-view").style.display  = "block";
  q("#user-display-name").textContent = data.full_name;
  q("#user-avatar").textContent       = data.full_name.charAt(0).toUpperCase();
  q("#jwt-pill").style.display        = "inline-flex";

  // Decode expiry from JWT for display
  try {
    const payload = JSON.parse(atob(data.access_token.split(".")[1]));
    q("#jwt-exp").textContent = new Date(payload.exp * 1000).toLocaleTimeString();
  } catch (_) {}

  loadHistory();
  newChat();
}

function logout() {
  state.token         = null;
  state.user          = null;
  state.currentSession = null;
  state.sessions       = {};

  q("#logged-in-view").style.display  = "none";
  q("#logged-out-view").style.display = "block";
  q("#jwt-pill").style.display        = "none";

  clearMessages();
  q("#history-list").innerHTML = "";
  toast("👋 Logged out");
}

/* ══════════════════════════════════════════════════════════
   CHAT HISTORY  (backend + local cache)
══════════════════════════════════════════════════════════ */

async function loadHistory() {
  if (!state.token) return;
  try {
    const data = await api("/history");
    state.sessions = {};
    data.sessions.forEach(s => {
      state.sessions[s.session_id] = {
        id:        s.session_id,
        title:     s.title,
        msgs:      [],
        createdAt: s.created_at,
        msgCount:  s.message_count,
      };
    });
    renderHistory();
  } catch (_) {}
}

async function loadSession(id) {
  state.currentSession = id;
  clearMessages();
  q("#welcome").style.display = "none";

  try {
    const data = await api(`/history/${id}`);
    data.messages.forEach(m => appendMsg(m.role, m.content, false));
    state.sessions[id].msgs = data.messages;
  } catch (e) {
    toast("⚠️ Could not load session: " + e.message);
  }
  renderHistory();
}

async function deleteSession(id, e) {
  e.stopPropagation();
  try {
    await api(`/history/${id}`, { method: "DELETE" });
    delete state.sessions[id];
    if (state.currentSession === id) newChat();
    renderHistory();
    toast("🗑 Session deleted");
  } catch (e) {
    toast("❌ " + e.message);
  }
}

function newChat() {
  state.currentSession = null;
  clearMessages();
  renderHistory();
}

function renderHistory() {
  const list     = q("#history-list");
  const sessions = Object.values(state.sessions).reverse();
  if (!sessions.length) {
    list.innerHTML = `<div style="padding:8px 14px;font-size:11px;color:var(--ink3);font-family:var(--font-mono)">No history yet</div>`;
    return;
  }
  list.innerHTML = sessions.map(s => `
    <div class="history-item ${s.id === state.currentSession ? "active" : ""}" onclick="loadSession('${s.id}')">
      <div class="h-title">
        <button class="h-del-btn" onclick="deleteSession('${s.id}',event)">✕</button>
        ${esc(s.title)}
      </div>
      <div class="h-meta">${s.msgCount || "?"} msgs · ${ago(s.createdAt)}</div>
    </div>
  `).join("");
}

/* ══════════════════════════════════════════════════════════
   MODES
══════════════════════════════════════════════════════════ */

function setMode(mode) {
  document.querySelectorAll(".mode-tab").forEach(t =>
    t.classList.toggle("active", t.dataset.mode === mode)
  );
  ["quiz", "summary", "explain"].forEach(m => {
    q(`#${m}-panel`).classList.toggle("show", m === mode);
  });
}

/* ══════════════════════════════════════════════════════════
   STUDY TOOLS
══════════════════════════════════════════════════════════ */

async function runQuiz() {
  const topic = q("#quiz-topic").value.trim();
  const num   = parseInt(q("#quiz-num").value);
  const diff  = q("#quiz-diff").value.toLowerCase();
  if (!topic) return toast("⚠️ Enter a topic");
  setMode("chat");
  await callTool("/quiz", { topic, num_questions: num, difficulty: diff }, "quiz", `🧪 Generate ${num} ${diff} MCQs on: ${topic}`);
}

async function runSummary() {
  const topic  = q("#summary-topic").value.trim();
  const format = q("#summary-format").value.toLowerCase().replace(/ /g, "_");
  if (!topic) return toast("⚠️ Enter a topic");
  setMode("chat");
  await callTool("/summary", { topic, format }, "summary", `📋 Summary: ${topic}`);
}

async function runExplain() {
  const concept = q("#explain-topic").value.trim();
  const level   = q("#explain-level").value.toLowerCase().replace(/ .*/,"");
  if (!concept) return toast("⚠️ Enter a concept");
  setMode("chat");
  await callTool("/explain", { concept, level }, "explain", `🔬 Explain: ${concept}`);
}

async function callTool(endpoint, body, respField, label) {
  if (state.loading) return;
  appendMsg("user", label);
  showTyping(true);

  try {
    const data = await api(endpoint, { method: "POST", body });
    showTyping(false);
    appendMsg("assistant", data[respField]);
    if (state.autoTTS) speakText(data[respField]);
  } catch (e) {
    showTyping(false);
    appendMsg("assistant", "❌ Error: " + e.message);
  }
}

/* ══════════════════════════════════════════════════════════
   CHAT
══════════════════════════════════════════════════════════ */

async function sendMessage() {
  const input = q("#msg-input");
  const text  = input.value.trim();
  if (!text || state.loading) return;

  input.value = "";
  input.style.height = "auto";
  await sendToBackend(text);
}

async function sendToBackend(text) {
  if (state.loading) return;
  state.loading = true;
  q("#send-btn").disabled = true;
  q("#welcome").style.display = "none";

  appendMsg("user", text);
  showTyping(true);

  try {
    const data = await api("/chat", {
      method: "POST",
      body: {
        message:    text,
        session_id: state.currentSession || undefined,
        use_rag:    true,
      },
    });

    showTyping(false);

    // Update session tracking
    if (!state.currentSession) {
      state.currentSession = data.session_id;
      state.sessions[data.session_id] = {
        id:        data.session_id,
        title:     text.length > 45 ? text.slice(0, 45) + "…" : text,
        msgs:      [],
        createdAt: new Date().toISOString(),
        msgCount:  1,
      };
    }
    if (state.sessions[data.session_id]) {
      state.sessions[data.session_id].msgCount =
        (state.sessions[data.session_id].msgCount || 0) + 2;
    }

    appendMsg("assistant", data.response, true, data.sources || []);
    if (state.autoTTS) speakText(data.response);
    renderHistory();

  } catch (e) {
    showTyping(false);
    appendMsg("assistant", `❌ ${e.message}\n\nMake sure the backend is running:\n  uvicorn backend.main:app --reload`);
  }

  state.loading = false;
  q("#send-btn").disabled = false;
}

function quickSend(text) {
  q("#msg-input").value = text;
  sendMessage();
}

/* ══════════════════════════════════════════════════════════
   MESSAGE DOM HELPERS
══════════════════════════════════════════════════════════ */

function appendMsg(role, content, animate = true, sources = []) {
  const messages = q("#messages");
  const typing   = q("#typing");
  const time     = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });

  const avatarText = role === "user"
    ? (state.user?.full_name?.charAt(0).toUpperCase() || "U")
    : "🎓";

  const sourcesHtml = sources.length
    ? `<div class="msg-sources">${sources.map(s => `<span class="source-tag">📄 ${esc(s)}</span>`).join("")}</div>`
    : "";

  const div = document.createElement("div");
  div.className = `msg ${role}`;
  div.style.animation = animate ? "fadeUp .3s ease" : "none";
  div.innerHTML = `
    <div class="msg-avatar">${avatarText}</div>
    <div class="msg-body">
      <div class="msg-bubble">${esc(content)}</div>
      ${sourcesHtml}
      <div class="msg-meta">${time}</div>
    </div>
  `;
  messages.insertBefore(div, typing);
  messages.scrollTop = messages.scrollHeight;
}

function clearMessages() {
  document.querySelectorAll(".msg").forEach(m => m.remove());
  q("#welcome").style.display = "block";
  showTyping(false);
}

function showTyping(show) {
  q("#typing").classList.toggle("show", show);
  q("#messages").scrollTop = q("#messages").scrollHeight;
}

/* ══════════════════════════════════════════════════════════
   RAG — DOCUMENT UPLOAD
══════════════════════════════════════════════════════════ */

q("#upload-zone").addEventListener("click", () => q("#file-input").click());
q("#upload-zone").addEventListener("dragover",  e => { e.preventDefault(); q("#upload-zone").classList.add("drag"); });
q("#upload-zone").addEventListener("dragleave", () => q("#upload-zone").classList.remove("drag"));
q("#upload-zone").addEventListener("drop", e => {
  e.preventDefault();
  q("#upload-zone").classList.remove("drag");
  handleFiles(e.dataTransfer.files);
});
q("#file-input").addEventListener("change", e => handleFiles(e.target.files));

async function handleFiles(files) {
  for (const file of files) await uploadFile(file);
}

async function uploadFile(file) {
  const proc = q("#processing");
  q("#proc-text").textContent = `Indexing "${file.name}"…`;
  proc.classList.add("show");

  try {
    const form = new FormData();
    form.append("file", file);
    const data = await apiForm("/documents/upload", form);
    toast(`✅ "${file.name}" — ${data.chunks} chunks indexed`);
    loadDocuments();
  } catch (e) {
    toast(`❌ Upload failed: ${e.message}`);
  }
  proc.classList.remove("show");
}

async function loadDocuments() {
  try {
    const data = await api("/documents");
    const list = q("#docs-list");
    list.innerHTML = data.documents.map(d => `
      <div class="doc-card">
        <div class="doc-hdr">
          <span class="doc-icon">${d.filename.endsWith(".pdf") ? "📄" : "📝"}</span>
          <span class="doc-name" title="${esc(d.filename)}">${esc(d.filename)}</span>
          <button class="doc-del" onclick="removeDoc('${esc(d.filename)}')">✕</button>
        </div>
        <div class="doc-meta">${d.size_kb} KB · ${new Date(d.uploaded_at).toLocaleTimeString()}</div>
        <span class="chunk-badge">✓ ${d.chunks} chunks</span>
      </div>
    `).join("");
  } catch (_) {}
}

async function removeDoc(filename) {
  try {
    await api(`/documents/${encodeURIComponent(filename)}`, { method: "DELETE" });
    toast(`🗑 "${filename}" removed`);
    loadDocuments();
  } catch (e) {
    toast("❌ " + e.message);
  }
}

function selectVDB(el, name) {
  document.querySelectorAll(".vdb-btn").forEach(b => b.classList.remove("on"));
  el.classList.add("on");
  state.selectedVDB = name;
  toast(`Vector store: ${name.toUpperCase()}`);
}

/* ══════════════════════════════════════════════════════════
   VOICE — STT  (Web Speech API / Whisper-compatible interface)
══════════════════════════════════════════════════════════ */

function toggleMic() {
  if (state.isRecording) { cancelVoice(); return; }

  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SR) return toast("⚠️ Speech recognition not available in this browser");

  state.recognition = new SR();
  state.recognition.lang = "en-US";
  state.recognition.interimResults = true;
  state.recognition.continuous     = false;

  state.recognition.onstart  = () => {
    state.isRecording = true;
    q("#mic-btn").classList.add("mic-on");
    q("#voice-overlay").classList.add("show");
  };
  state.recognition.onresult = e => {
    let interim = "", final = "";
    for (const r of e.results) {
      if (r.isFinal) final += r[0].transcript;
      else interim += r[0].transcript;
    }
    q("#vc-sub").textContent = interim || final || "Listening…";
    if (final) {
      q("#msg-input").value = final;
      cancelVoice();
    }
  };
  state.recognition.onerror = () => cancelVoice();
  state.recognition.onend   = () => { if (state.isRecording) cancelVoice(); };
  state.recognition.start();
}

function cancelVoice() {
  state.isRecording = false;
  state.recognition?.stop();
  q("#mic-btn").classList.remove("mic-on");
  q("#voice-overlay").classList.remove("show");
  q("#vc-sub").textContent = "Speak now — I'm ready";
}

/* ══════════════════════════════════════════════════════════
   VOICE — TTS  (Web Speech Synthesis / Coqui-compatible)
══════════════════════════════════════════════════════════ */

function toggleTTS() {
  state.autoTTS = !state.autoTTS;
  q("#tts-btn").classList.toggle("tts-on", state.autoTTS);
  q("#tts-btn").title = state.autoTTS ? "Auto-TTS ON (click to disable)" : "Auto Text-to-Speech";
  toast(state.autoTTS ? "🔊 Auto-read ON" : "🔇 Auto-read OFF");
}

function speakText(text) {
  if (!window.speechSynthesis) return;
  stopTTS();
  const utt = new SpeechSynthesisUtterance(
    text.replace(/[*#_`>]/g, "").slice(0, 2000)
  );
  utt.rate  = 0.95;
  utt.pitch = 1.0;
  utt.onstart = () => q("#tts-bar").classList.add("show");
  utt.onend   = () => q("#tts-bar").classList.remove("show");
  utt.onerror = () => q("#tts-bar").classList.remove("show");
  window.speechSynthesis.speak(utt);
}

function stopTTS() {
  window.speechSynthesis?.cancel();
  q("#tts-bar").classList.remove("show");
}

/* ══════════════════════════════════════════════════════════
   PANEL TOGGLE
══════════════════════════════════════════════════════════ */

function togglePanel(side) {
  const panel = q(`#${side}-panel`);
  const btn   = document.querySelector(`button[onclick="togglePanel('${side}')"]`);
  const hide  = panel.style.display !== "none" && panel.style.display !== "";
  panel.style.display = hide ? "none" : "flex";
  panel.style.flexDirection = "column";
  btn.classList.toggle("on", !hide);
}

/* ══════════════════════════════════════════════════════════
   INPUT EVENTS
══════════════════════════════════════════════════════════ */

q("#msg-input").addEventListener("keydown", e => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendMessage(); }
});
q("#msg-input").addEventListener("input", function () {
  this.style.height = "auto";
  this.style.height = Math.min(this.scrollHeight, 110) + "px";
});

/* ══════════════════════════════════════════════════════════
   UTILS
══════════════════════════════════════════════════════════ */

function q(sel)    { return document.querySelector(sel); }
function esc(s)    { return String(s).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/"/g,"&quot;"); }
function toast(msg, ms = 2800) {
  const t = q("#toast");
  t.textContent = msg;
  t.classList.add("show");
  clearTimeout(t._timer);
  t._timer = setTimeout(() => t.classList.remove("show"), ms);
}
function ago(d) {
  const s = Math.floor((Date.now() - new Date(d)) / 1000);
  if (s < 60)   return "just now";
  if (s < 3600) return `${Math.floor(s/60)}m ago`;
  return `${Math.floor(s/3600)}h ago`;
}

/* ══════════════════════════════════════════════════════════
   INIT
══════════════════════════════════════════════════════════ */
document.addEventListener("DOMContentLoaded", () => {
  loadDocuments();
  // Auto-fill demo credentials
  q("#login-user").value = "demo";
  q("#login-pass").value = "password";
});
