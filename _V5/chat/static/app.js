let currentSessionId = null;
let currentHistory = [];
let pendingAssistantTimer = null;
let currentRetrievalLabel = "coarse+association";

const ACTIVATION_STORAGE_KEY = "agentmemory_v5_latest_activation";

async function createSession() {
  const resp = await fetch("/api/session/new", { method: "POST" });
  const data = await resp.json();
  currentSessionId = data.session_id;
  currentHistory = [];
  updateRetrievalLabel("coarse+association");
  updateSessionLabel();
  renderMessages(currentHistory);
  renderAssociationPanel([], []);
  renderCoarsePanel([]);
  persistActivationTrace({});
}

function updateSessionLabel() {
  document.querySelector("#session-label span").textContent = currentSessionId || "-";
}

function updateRetrievalLabel(label) {
  currentRetrievalLabel = label || "coarse+association";
  document.querySelector("#retrieval-label span").textContent = currentRetrievalLabel;
}

function renderMessages(history) {
  const root = document.getElementById("messages");
  root.innerHTML = "";
  history.forEach((turn) => {
    const card = document.createElement("article");
    card.className = `message ${turn.role}${turn.pending ? " pending" : ""}`;
    const role = document.createElement("div");
    role.className = "message-role";
    role.textContent = turn.role;
    const content = document.createElement("div");
    content.className = "message-content";
    content.textContent = turn.content || (turn.pending && turn.status ? turn.status : "");
    card.appendChild(role);
    card.appendChild(content);
    if (turn.pending && turn.status) {
      const status = document.createElement("div");
      status.className = "message-status";
      status.textContent = turn.status;
      card.appendChild(status);
    }
    root.appendChild(card);
  });
  root.scrollTop = root.scrollHeight;
}

function renderMemoryCards(rootId, memoryRefs, pendingText = "", variant = "") {
  const root = document.getElementById(rootId);
  root.innerHTML = "";
  if (pendingText) {
    const card = document.createElement("article");
    card.className = "memory-card pending";
    card.innerHTML = `<h3>正在处理</h3><div class="memory-meta">${pendingText}</div>`;
    root.appendChild(card);
    return;
  }
  if (!memoryRefs || memoryRefs.length === 0) {
    root.innerHTML = '<div class="memory-empty">本轮没有命中参考记忆。</div>';
    return;
  }
  memoryRefs.forEach((ref, idx) => {
    const card = document.createElement("article");
    card.className = `memory-card ${variant}`.trim();
    const title = document.createElement("h3");
    title.textContent = `M${idx + 1} | ${ref.memory_id}`;
    const meta = document.createElement("div");
    meta.className = "memory-meta";
    meta.textContent = `cluster=${ref.cluster_id || "-"} | source=${ref.source || "-"} | score=${Number(ref.score || 0).toFixed(3)}`;
    const text = document.createElement("p");
    text.textContent = ref.display_text || "";
    card.appendChild(title);
    card.appendChild(meta);
    card.appendChild(text);
    root.appendChild(card);
  });
}

function renderAssociationTags(tags) {
  const root = document.getElementById("association-tags");
  root.innerHTML = "";
  if (!tags || tags.length === 0) {
    root.innerHTML = '<div class="tag-empty">本轮没有明显点亮的概念标签。</div>';
    return;
  }
  tags.forEach((tag) => {
    const pill = document.createElement("div");
    pill.className = `tag-pill${tag.via_bridge ? " bridge" : ""}`;
    pill.innerHTML = `
      <span class="tag-level">${tag.level || "-"}</span>
      <span class="tag-name">${tag.name || ""}</span>
      <span class="tag-score">${Number(tag.score || 0).toFixed(2)}</span>
    `;
    root.appendChild(pill);
  });
}

function renderAssociationPanel(memoryRefs, tags, pendingText = "") {
  renderAssociationTags(pendingText ? [] : tags);
  renderMemoryCards("association-list", memoryRefs, pendingText, "association-card");
}

function renderCoarsePanel(memoryRefs, pendingText = "") {
  renderMemoryCards("coarse-list", memoryRefs, pendingText);
}

function persistActivationTrace(trace) {
  try {
    localStorage.setItem(ACTIVATION_STORAGE_KEY, JSON.stringify(trace || {}));
  } catch (_err) {
    // Ignore storage failures; the main chat page can still work without the trace page.
  }
}

async function sendMessage(text) {
  const retrievalLabel = "coarse+association";
  updateRetrievalLabel(retrievalLabel);
  const pendingAssistant = {
    role: "assistant",
    content: "正在回忆…",
    pending: true,
    status: `正在回忆… (${retrievalLabel})`,
  };
  currentHistory = [...currentHistory, { role: "user", content: text }, pendingAssistant];
  renderMessages(currentHistory);
  renderAssociationPanel([], [], `模式=${retrievalLabel} | 正在点亮联想图并回忆关联记忆…`);
  renderCoarsePanel([], `模式=${retrievalLabel} | 正在准备 coarse 参考记忆…`);
  if (pendingAssistantTimer) {
    clearTimeout(pendingAssistantTimer);
  }
  pendingAssistantTimer = setTimeout(() => {
    const last = currentHistory[currentHistory.length - 1];
    if (last && last.pending) {
      last.status = `正在生成回复… (${retrievalLabel})`;
      last.content = "正在生成回复…";
      renderMessages(currentHistory);
      renderAssociationPanel([], [], `模式=${retrievalLabel} | 联想已点亮，正在生成回复…`);
      renderCoarsePanel([], `模式=${retrievalLabel} | coarse 参考已准备，正在生成回复…`);
    }
  }, 700);
  const resp = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: currentSessionId, text }),
  });
  const data = await resp.json();
  if (pendingAssistantTimer) {
    clearTimeout(pendingAssistantTimer);
    pendingAssistantTimer = null;
  }
  if (!resp.ok) {
    throw new Error(data.detail || "request failed");
  }
  currentSessionId = data.session_id;
  updateSessionLabel();
  updateRetrievalLabel(data.retrieval_label || retrievalLabel);
  currentHistory = data.history || [];
  renderMessages(currentHistory);
  renderAssociationPanel(data.association_memory_refs || [], data.association_tags || []);
  renderCoarsePanel(data.coarse_memory_refs || data.memory_refs || []);
  persistActivationTrace(data.association_trace || {});
}

document.getElementById("new-session-btn").addEventListener("click", async () => {
  await createSession();
});

document.getElementById("chat-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const input = document.getElementById("chat-input");
  const text = input.value.trim();
  if (!text) return;
  input.value = "";
  try {
    await sendMessage(text);
  } catch (err) {
    if (pendingAssistantTimer) {
      clearTimeout(pendingAssistantTimer);
      pendingAssistantTimer = null;
    }
    currentHistory = currentHistory.filter((turn) => !turn.pending);
    renderMessages(currentHistory);
    renderAssociationPanel([], []);
    renderCoarsePanel([]);
    alert(err.message || "发送失败");
  }
});

document.getElementById("chat-input").addEventListener("keydown", async (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    document.getElementById("chat-form").requestSubmit();
  }
});

createSession();
