let currentSessionId = null;
let currentHistory = [];
let currentRetrievalLabel = "coarse+association";
let currentQueryText = "";
let currentFeedbackSelections = {};
let currentCandidateRefs = [];
let currentCoarseRefs = [];
let currentAssociationRefs = [];
let currentAssociationTags = [];

const ACTIVATION_STORAGE_KEY = "agentmemory_v5_latest_activation";

async function createSession() {
  const resp = await fetch("/api/session/new", { method: "POST" });
  const data = await resp.json();
  currentSessionId = data.session_id;
  currentHistory = [];
  currentQueryText = "";
  currentFeedbackSelections = {};
  currentCandidateRefs = [];
  currentCoarseRefs = [];
  currentAssociationRefs = [];
  currentAssociationTags = [];
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

function collectCurrentCandidateRefs(coarseRefs, associationRefs) {
  const rows = [];
  (associationRefs || []).forEach((ref) => {
    rows.push({ ...ref, lane: "association" });
  });
  (coarseRefs || []).forEach((ref) => {
    rows.push({ ...ref, lane: "coarse" });
  });
  currentCandidateRefs = rows;
}

function rerenderPanels() {
  renderAssociationPanel(currentAssociationRefs, currentAssociationTags);
  renderCoarsePanel(currentCoarseRefs);
}

async function submitFeedback(ref, lane, feedbackType) {
  if (!currentSessionId || !currentQueryText || !ref || !ref.memory_id) {
    return;
  }
  const key = `${lane}|${ref.memory_id}`;
  if (currentFeedbackSelections[key]) {
    return;
  }
  const resp = await fetch("/api/feedback", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      session_id: currentSessionId,
      query_text: currentQueryText,
      memory_id: ref.memory_id,
      feedback_type: feedbackType,
      lane,
      candidate_refs: currentCandidateRefs,
    }),
  });
  const data = await resp.json();
  if (!resp.ok) {
    throw new Error(data.detail || "feedback failed");
  }
  const selected = data.selected_feedback || {};
  Object.entries(selected).forEach(([memoryId, selectedType]) => {
    currentFeedbackSelections[`coarse|${memoryId}`] = selectedType;
    currentFeedbackSelections[`association|${memoryId}`] = selectedType;
  });
  rerenderPanels();
}

function renderMemoryCards(rootId, memoryRefs, pendingText = "", variant = "", lane = "") {
  const root = document.getElementById(rootId);
  root.innerHTML = "";
  if (pendingText) {
    const card = document.createElement("article");
    card.className = "memory-card pending";
    card.innerHTML = `<h3>正在处理中</h3><div class="memory-meta">${pendingText}</div>`;
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
    const header = document.createElement("div");
    header.className = "memory-card-head";
    const title = document.createElement("h3");
    title.textContent = `M${idx + 1} | ${ref.memory_id}`;
    const actions = document.createElement("div");
    actions.className = "memory-actions";
    const selectedType = currentFeedbackSelections[`${lane}|${ref.memory_id}`];
    const unrelatedBtn = document.createElement("button");
    unrelatedBtn.type = "button";
    unrelatedBtn.className = `feedback-btn unrelated${selectedType ? " selected" : ""}`;
    unrelatedBtn.textContent = selectedType === "unrelated" ? "已标无关" : "无关";
    unrelatedBtn.disabled = Boolean(selectedType);
    unrelatedBtn.addEventListener("click", async () => {
      try {
        await submitFeedback(ref, lane, "unrelated");
      } catch (err) {
        alert(err.message || "反馈失败");
      }
    });
    const forgetBtn = document.createElement("button");
    forgetBtn.type = "button";
    forgetBtn.className = `feedback-btn forget${selectedType ? " selected" : ""}`;
    forgetBtn.textContent = selectedType === "toforget" ? "已标遗忘" : "遗忘";
    forgetBtn.disabled = Boolean(selectedType);
    forgetBtn.addEventListener("click", async () => {
      try {
        await submitFeedback(ref, lane, "toforget");
      } catch (err) {
        alert(err.message || "反馈失败");
      }
    });
    actions.appendChild(unrelatedBtn);
    actions.appendChild(forgetBtn);
    header.appendChild(title);
    header.appendChild(actions);
    const meta = document.createElement("div");
    meta.className = "memory-meta";
    meta.textContent = `cluster=${ref.cluster_id || "-"} | source=${ref.source || "-"} | score=${Number(ref.score || 0).toFixed(3)}`;
    const text = document.createElement("p");
    text.textContent = ref.display_text || "";
    card.appendChild(header);
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
  renderMemoryCards("association-list", memoryRefs, pendingText, "association-card", "association");
}

function renderCoarsePanel(memoryRefs, pendingText = "") {
  renderMemoryCards("coarse-list", memoryRefs, pendingText, "", "coarse");
}

function persistActivationTrace(trace) {
  try {
    localStorage.setItem(ACTIVATION_STORAGE_KEY, JSON.stringify(trace || {}));
  } catch (_err) {
    // ignore storage failures
  }
}

function applyChatPayload(data, text) {
  currentSessionId = data.session_id;
  updateSessionLabel();
  updateRetrievalLabel(data.retrieval_label || "coarse+association");
  currentHistory = data.history || [];
  currentQueryText = text;
  currentCoarseRefs = data.coarse_memory_refs || data.memory_refs || [];
  currentAssociationRefs = data.association_memory_refs || [];
  currentAssociationTags = data.association_tags || [];
  renderMessages(currentHistory);
  collectCurrentCandidateRefs(currentCoarseRefs, currentAssociationRefs);
  rerenderPanels();
  persistActivationTrace(data.association_trace || {});
}

async function requestRetrieve(text, memoryPreferenceEnabled) {
  const resp = await fetch("/api/chat/retrieve", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      session_id: currentSessionId,
      text,
      memory_preference_enabled: Boolean(memoryPreferenceEnabled),
    }),
  });
  const data = await resp.json();
  if (!resp.ok) {
    throw new Error(data.detail || "retrieve failed");
  }
  return data;
}

async function requestRespond(sessionId, retrievalId) {
  const resp = await fetch("/api/chat/respond", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId, retrieval_id: retrievalId }),
  });
  const data = await resp.json();
  if (!resp.ok) {
    throw new Error(data.detail || "respond failed");
  }
  return data;
}

async function sendMessage(text) {
  const retrievalLabel = "coarse+association";
  const retrievalOnly = Boolean(document.getElementById("retrieval-only-toggle")?.checked);
  const memoryPreferenceEnabled = Boolean(document.getElementById("memory-preference-toggle")?.checked);
  currentQueryText = text;
  currentFeedbackSelections = {};
  currentCandidateRefs = [];
  currentCoarseRefs = [];
  currentAssociationRefs = [];
  currentAssociationTags = [];
  updateRetrievalLabel(retrievalLabel);

  currentHistory = [
    ...currentHistory,
    { role: "user", content: text },
    {
      role: "assistant",
      content: "正在召回记忆…",
      pending: true,
      status: `正在召回记忆…(${retrievalLabel})`,
    },
  ];
  renderMessages(currentHistory);
  renderAssociationPanel([], [], `模式=${retrievalLabel} | 正在点亮联想图并召回关联记忆…`);
  renderCoarsePanel([], `模式=${retrievalLabel} | 正在准备 coarse 参考记忆…`);

  const retrievalData = await requestRetrieve(text, memoryPreferenceEnabled);
  applyChatPayload(retrievalData, text);
  const retrievalId = retrievalData.retrieval_id || "";
  if (retrievalOnly || !retrievalId) {
    return;
  }

  currentHistory = [
    ...currentHistory,
    {
      role: "assistant",
      content: "正在生成回复…",
      pending: true,
      status: `正在生成回复…(${currentRetrievalLabel})`,
    },
  ];
  renderMessages(currentHistory);

  const responseData = await requestRespond(currentSessionId, retrievalId);
  applyChatPayload(responseData, text);
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
    currentHistory = currentHistory.filter((turn) => !turn.pending);
    renderMessages(currentHistory);
    rerenderPanels();
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
