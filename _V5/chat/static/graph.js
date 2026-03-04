const levelSelect = document.getElementById("level-select");
const nameInput = document.getElementById("name-input");
const form = document.getElementById("graph-form");
const statusBar = document.getElementById("status-bar");
const resultRoot = document.getElementById("result-root");

function setStatus(text) {
  statusBar.textContent = text;
}

function escapeHtml(text) {
  return String(text || "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function bubble(node, extraMeta = "", className = "") {
  if (!node) {
    return `<article class="bubble bubble-l1 ${className}"><div class="bubble-title">没有</div></article>`;
  }
  const levelClass = `bubble-${String(node.level || "").toLowerCase()}`;
  return `
    <article class="bubble ${levelClass} ${className}">
      <div class="bubble-id">${escapeHtml(node.id)} | ${escapeHtml(node.level)}</div>
      <h2 class="bubble-title">${escapeHtml(node.name)}</h2>
      <div class="bubble-meta">${escapeHtml(extraMeta)}</div>
    </article>
  `;
}

function renderBridgeLane(bridges = {}) {
  const preview = Array.isArray(bridges.preview) ? bridges.preview : [];
  const left = preview.slice(0, 3);
  const right = preview.slice(3, 6);
  const total = Number(bridges.total || 0);
  const note =
    total > 6
      ? `水平桥接 ${total} 个，仅展示 6 个`
      : `水平桥接 ${total} 个`;
  const renderBridgeTags = (items, side) =>
    items
      .map(
        (item) => `
          <div class="bridge-tag bridge-${side}">
            <span class="bridge-tag-name">${escapeHtml(item.name)}</span>
            <span class="bridge-tag-weight">${Number(item.weight || 0).toFixed(3)}</span>
          </div>
        `,
      )
      .join("");
  return `
    <div class="focus-lane">
      <div class="bridge-rail">
        <div class="bridge-line"></div>
        <div class="bridge-tags bridge-tags-left">
          ${renderBridgeTags(left, "left")}
        </div>
        <div class="bridge-tags bridge-tags-right">
          ${renderBridgeTags(right, "right")}
        </div>
      </div>
      <div class="bridge-summary">${escapeHtml(note)}</div>
    </div>
  `;
}

function focusBubble(node, extraMeta = "", bridges = {}) {
  return `
    <section class="focus-node">
      ${renderBridgeLane(bridges)}
      ${bubble(node, extraMeta, "focus-bubble")}
    </section>
  `;
}

function chips(items, emptyText) {
  if (!items || items.length === 0) {
    return `<div class="chip-note">${escapeHtml(emptyText)}</div>`;
  }
  return `
    <div class="chip-cloud">
      ${items
        .map(
          (item) => `
            <div class="chip">
              <span class="chip-level">${escapeHtml(item.level)}</span>
              <span>${escapeHtml(item.name)}</span>
              <span class="chip-level">${Number(item.conf || 0).toFixed(3)}</span>
            </div>
          `,
        )
        .join("")}
    </div>
  `;
}

function renderL1(result) {
  const parent = result.parent;
  const grandparent = result.grandparent;
  resultRoot.innerHTML = `
    <section class="summary-card">
      <h2>L1 结构</h2>
      <p>展示当前概念、它的爸爸 L2、以及爷爷 L3。</p>
    </section>
    <section class="l1-stage">
      <div class="l1-stack">
        ${bubble(grandparent, grandparent ? `conf=${Number(grandparent.conf || 0).toFixed(3)}` : "没有爷爷", "l1-ancestor")}
        ${bubble(parent, parent ? `conf=${Number(parent.conf || 0).toFixed(3)}` : "没有爸爸", "l1-ancestor")}
        ${focusBubble(result.node, "当前命中概念", result.bridges)}
      </div>
    </section>
  `;
}

function renderL2(result) {
  const childTotal = Number(result.child_total || 0);
  const truncated = Boolean(result.children_truncated);
  resultRoot.innerHTML = `
    <section class="summary-card">
      <h2>L2 结构</h2>
      <p>上面是父亲 L3，中间是当前 L2，下面是最多 50 个 L1 儿子。</p>
    </section>
    <section class="l2-stage">
      <div class="bubble-row center">
        ${bubble(result.parent, result.parent ? `conf=${Number(result.parent.conf || 0).toFixed(3)}` : "没有父亲")}
      </div>
      <div class="bubble-row center">
        ${focusBubble(result.node, `L1 children=${childTotal}`, result.bridges)}
      </div>
      <section class="branch-card">
        <h3>L1 儿子</h3>
        ${chips(result.children, "没有儿子。")}
        <div class="chip-note">${truncated ? `仅展示前 50 个，实际共 ${childTotal} 个。` : `当前共 ${childTotal} 个。`}</div>
      </section>
    </section>
  `;
}

function renderL3(result) {
  const branchTotal = Number(result.branch_total || 0);
  const branchNote = result.branches_truncated
    ? `仅展示前 10 个 L2 分支，实际共 ${branchTotal} 个。`
    : `当前共 ${branchTotal} 个 L2 分支。`;
  resultRoot.innerHTML = `
    <section class="summary-card">
      <h2>L3 结构</h2>
      <p>先展示 L3 本体，再展示最多 10 个 L2 分支；每个分支展示最多 10 个 L1 儿子。</p>
    </section>
    <section class="l3-stage">
      <div class="bubble-row center">
        ${focusBubble(result.node, `L2 branches=${branchTotal}`, result.bridges)}
      </div>
      <div class="chip-note">${branchNote}</div>
      <section class="branch-grid">
        ${
          result.branches && result.branches.length
            ? result.branches
                .map((branch) => {
                  const childTotal = Number(branch.child_total || 0);
                  const childNote = branch.children_truncated
                    ? `仅展示前 10 个 L1，实际共 ${childTotal} 个。`
                    : `当前共 ${childTotal} 个 L1。`;
                  return `
                    <article class="branch-card">
                      ${bubble(
                        branch.node,
                        `conf=${Number(branch.conf || 0).toFixed(3)} | L1=${childTotal}`,
                        "compact",
                      )}
                      ${chips(branch.children, "这个分支下面没有 L1。")}
                      <div class="chip-note">${childNote}</div>
                    </article>
                  `;
                })
                .join("")
            : '<article class="branch-card"><div class="chip-note">没有 L2 分支。</div></article>'
        }
      </section>
    </section>
  `;
}

function renderError(detail, fallbackText = "没有命中结果。") {
  const payload = detail && detail.detail ? detail.detail : detail || {};
  const query = payload.query || {};
  const extra =
    payload.error === "ambiguous_name" && Array.isArray(payload.matches) && payload.matches.length
      ? `<div class="chip-note">同名候选：${payload.matches
          .map((item) => `${escapeHtml(item.level)} ${escapeHtml(item.name)} (${escapeHtml(item.id)})`)
          .join(" | ")}</div>`
      : "";
  resultRoot.innerHTML = `
    <section class="error-card">
      <h2>${escapeHtml(fallbackText)}</h2>
      <p>${escapeHtml(query.level || "")} ${escapeHtml(query.name || "")}</p>
      <p>${escapeHtml(payload.error || "")}</p>
      ${extra}
    </section>
  `;
}

async function fetchGraph(level, name) {
  const params = new URLSearchParams({ level, name });
  const resp = await fetch(`/api/graph-view?${params.toString()}`);
  const payload = await resp.json();
  if (!resp.ok) {
    throw payload;
  }
  return payload;
}

async function runLookup(level, name) {
  setStatus(`正在查找 ${level}, ${name} ...`);
  try {
    const payload = await fetchGraph(level, name);
    const result = payload.result;
    if (result.kind === "L1") {
      renderL1(result);
    } else if (result.kind === "L2") {
      renderL2(result);
    } else {
      renderL3(result);
    }
    setStatus(`已命中 ${payload.query.level}, ${payload.query.name}`);
    const url = new URL(window.location.href);
    url.searchParams.set("level", level);
    url.searchParams.set("name", name);
    window.history.replaceState({}, "", url);
  } catch (err) {
    renderError(err, "没有命中这个精确概念。");
    setStatus("没有命中。");
  }
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const level = levelSelect.value;
  const name = nameInput.value.trim();
  if (!name) {
    setStatus("概念名不能为空。");
    return;
  }
  await runLookup(level, name);
});

function bootFromQueryString() {
  const url = new URL(window.location.href);
  const level = (url.searchParams.get("level") || "L1").toUpperCase();
  const name = url.searchParams.get("name") || "";
  if (["L1", "L2", "L3"].includes(level)) {
    levelSelect.value = level;
  }
  if (name) {
    nameInput.value = name;
    runLookup(levelSelect.value, name);
  }
}

bootFromQueryString();
