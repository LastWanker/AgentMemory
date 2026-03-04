const ACTIVATION_STORAGE_KEY = "agentmemory_v5_latest_activation";
const BASE_STAGE_SIZE = 1320;
const LEVEL_RADIUS = { L3: 60, L2: 46, L1: 34, BRIDGE: 30 };
const LEVEL_GAP = { L3_L2: 62, L2_L1: 72, L1_BRIDGE: 94 };
const LAYER_ORDER = ["L3", "L2", "L1", "BRIDGE"];

let currentScale = 1.1;
let activeNodeId = "";

function loadTrace() {
  try {
    return JSON.parse(localStorage.getItem(ACTIVATION_STORAGE_KEY) || "{}");
  } catch (_err) {
    return {};
  }
}

function setScale(value) {
  currentScale = Math.max(0.8, Math.min(1.9, value));
  renderActivation(loadTrace());
}

function centerOut(items) {
  const sorted = [...items].sort((a, b) => {
    const scoreDelta = Number(b.score || 0) - Number(a.score || 0);
    if (Math.abs(scoreDelta) > 1e-9) return scoreDelta;
    return String(a.name || "").localeCompare(String(b.name || ""), "zh-CN");
  });
  const out = new Array(sorted.length);
  const center = Math.floor(sorted.length / 2);
  const slots = [center];
  for (let offset = 1; offset <= sorted.length; offset += 1) {
    if (center - offset >= 0) slots.push(center - offset);
    if (center + offset < sorted.length) slots.push(center + offset);
  }
  sorted.forEach((item, index) => {
    out[slots[index]] = item;
  });
  return out.filter(Boolean);
}

function splitLabel(text) {
  const raw = String(text || "").trim();
  if (!raw) return ["", ""];
  if (raw.length <= 4) return [raw, ""];
  if (raw.length <= 8) {
    const cut = Math.ceil(raw.length / 2);
    return [raw.slice(0, cut), raw.slice(cut)];
  }
  return [raw.slice(0, 4), `${raw.slice(4, 9)}${raw.length > 9 ? "…" : ""}`];
}

function estimateNodeArc(node) {
  const radius = (LEVEL_RADIUS[node.level] || 34) * currentScale;
  const labelWidth = Math.max(68, String(node.name || "").length * 15) * currentScale;
  return Math.max(radius * 2.15, labelWidth) + 10 * currentScale;
}

function isPureBridgeNode(node) {
  const origins = Array.isArray(node.origins) ? node.origins : [];
  return node.level === "L1" && origins.length > 0 && origins.every((origin) => origin === "from_bridge");
}

function normalizeAngle(angle) {
  const twoPi = Math.PI * 2;
  let out = angle;
  while (out < -Math.PI) out += twoPi;
  while (out > Math.PI) out -= twoPi;
  return out;
}

function circularMean(values) {
  if (!values.length) return null;
  let sumSin = 0;
  let sumCos = 0;
  values.forEach((value) => {
    sumSin += Math.sin(value);
    sumCos += Math.cos(value);
  });
  return Math.atan2(sumSin, sumCos);
}

function angleDistance(a, b) {
  return normalizeAngle(a - b);
}

function euclideanSquared(a, b) {
  const dx = a.x - b.x;
  const dy = a.y - b.y;
  return dx * dx + dy * dy;
}

function buildVisibility(view) {
  const nodes = [...(view.nodes || [])].sort((a, b) => Number(b.score || 0) - Number(a.score || 0));
  const edges = view.edges || [];
  const childCounts = new Map();
  const ensureCount = (id) => {
    if (!childCounts.has(id)) childCounts.set(id, 0);
  };
  edges.forEach((edge) => {
    if (edge.kind !== "descend") return;
    ensureCount(edge.source_id);
    childCounts.set(edge.source_id, (childCounts.get(edge.source_id) || 0) + 1);
  });

  const l1Nodes = nodes.filter((node) => node.level === "L1");
  const keepL1Count = Math.max(1, Math.ceil(l1Nodes.length / 2));
  const keepL1 = new Set(l1Nodes.slice(0, keepL1Count).map((node) => node.node_id));
  l1Nodes.forEach((node) => {
    if (node.direct_seed) keepL1.add(node.node_id);
  });

  const visibleIds = new Set();
  nodes.forEach((node) => {
    if (node.level === "L3" && (childCounts.get(node.node_id) || 0) <= 0) return;
    if (node.level === "L2" && (childCounts.get(node.node_id) || 0) <= 0) return;
    if (node.level === "L1" && !keepL1.has(node.node_id)) return;
    visibleIds.add(node.node_id);
  });

  const visibleEdges = edges.filter((edge) => visibleIds.has(edge.source_id) && visibleIds.has(edge.target_id));
  const linkedIds = new Set();
  visibleEdges.forEach((edge) => {
    linkedIds.add(edge.source_id);
    linkedIds.add(edge.target_id);
  });
  const visibleNodes = nodes.filter((node) => visibleIds.has(node.node_id) && (linkedIds.has(node.node_id) || node.direct_seed));
  return { visibleNodes, visibleEdges };
}

function buildLayoutModel(view) {
  const { visibleNodes, visibleEdges } = buildVisibility(view);
  const byId = new Map(visibleNodes.map((node) => [node.node_id, { ...node }]));
  const displayLevelById = new Map();
  const layers = { L3: [], L2: [], L1: [], BRIDGE: [] };

  visibleNodes.forEach((node) => {
    const displayLevel = isPureBridgeNode(node) ? "BRIDGE" : node.level;
    displayLevelById.set(node.node_id, displayLevel);
    const displayNode = displayLevel === node.level ? { ...node } : { ...node, level: displayLevel };
    layers[displayLevel].push(displayNode);
  });

  const layerNodesById = new Map();
  Object.values(layers).forEach((rows) => {
    rows.forEach((node) => layerNodesById.set(node.node_id, node));
  });

  const incident = new Map();
  const addIncident = (layer, nodeId, edgeId) => {
    const key = `${layer}:${nodeId}`;
    if (!incident.has(key)) incident.set(key, []);
    incident.get(key).push(edgeId);
  };

  const edgeList = [];
  visibleEdges.forEach((edge, index) => {
    const sourceLevel = displayLevelById.get(edge.source_id);
    const targetLevel = displayLevelById.get(edge.target_id);
    if (!sourceLevel || !targetLevel) return;
    const edgeRow = {
      id: `e${index}`,
      sourceId: edge.source_id,
      targetId: edge.target_id,
      sourceLayer: sourceLevel,
      targetLayer: targetLevel,
      kind: edge.kind,
      weight: Math.max(0.05, Number(edge.score || 0.5)),
    };
    edgeList.push(edgeRow);
    addIncident(sourceLevel, edge.source_id, edgeRow.id);
    addIncident(targetLevel, edge.target_id, edgeRow.id);
  });

  const edgeById = new Map(edgeList.map((edge) => [edge.id, edge]));

  return {
    nodesById: layerNodesById,
    layers: {
      L3: centerOut(layers.L3),
      L2: centerOut(layers.L2),
      L1: centerOut(layers.L1),
      BRIDGE: centerOut(layers.BRIDGE),
    },
    edges: edgeList,
    edgeById,
    incident,
  };
}

function computeRingRadius(nodes, baseRadius) {
  if (!nodes.length) return baseRadius;
  const totalArc = nodes.reduce((sum, node) => sum + estimateNodeArc(node), 0);
  return Math.max(baseRadius, totalArc / (2 * Math.PI));
}

function computeRings(layers) {
  const ringL3 = computeRingRadius(layers.L3, 136 * currentScale);
  const ringL2 = computeRingRadius(layers.L2, ringL3 + (LEVEL_RADIUS.L3 + LEVEL_RADIUS.L2 + LEVEL_GAP.L3_L2) * currentScale);
  const ringL1 = computeRingRadius(layers.L1, ringL2 + (LEVEL_RADIUS.L2 + LEVEL_RADIUS.L1 + LEVEL_GAP.L2_L1) * currentScale);
  const ringBridge = computeRingRadius(
    layers.BRIDGE,
    ringL1 + (LEVEL_RADIUS.L1 + LEVEL_RADIUS.BRIDGE + LEVEL_GAP.L1_BRIDGE) * currentScale
  );
  return { L3: ringL3, L2: ringL2, L1: ringL1, BRIDGE: ringBridge };
}

function buildState(model) {
  const layers = {
    L3: [...model.layers.L3],
    L2: [...model.layers.L2],
    L1: [...model.layers.L1],
    BRIDGE: [...model.layers.BRIDGE],
  };
  const rings = computeRings(layers);
  const outerRadius = rings.BRIDGE + (LEVEL_RADIUS.BRIDGE + 54) * currentScale;
  const stageSize = Math.max(BASE_STAGE_SIZE, outerRadius * 2 + 180 * currentScale);
  return {
    layers,
    rings,
    rotations: {
      L3: -Math.PI / 2,
      L2: -Math.PI / 2,
      L1: -Math.PI / 2,
      BRIDGE: -Math.PI / 2,
    },
    stageSize,
  };
}

function placeLayer(order, radius, rotation, centerX, centerY) {
  const positions = new Map();
  if (!order.length) return positions;
  const count = order.length;
  order.forEach((node, index) => {
    const angle = normalizeAngle(rotation + (index * Math.PI * 2) / count);
    positions.set(node.node_id, {
      x: centerX + Math.cos(angle) * radius,
      y: centerY + Math.sin(angle) * radius,
      angle,
      radius,
    });
  });
  return positions;
}

function computeAllPositions(state) {
  const centerX = state.stageSize / 2;
  const centerY = state.stageSize / 2;
  const positions = new Map();
  LAYER_ORDER.forEach((layer) => {
    placeLayer(state.layers[layer], state.rings[layer], state.rotations[layer], centerX, centerY).forEach((value, key) => {
      positions.set(key, value);
    });
  });
  return { positions, centerX, centerY };
}

function nodeTargetAngle(layer, nodeId, state, model) {
  const { positions } = computeAllPositions(state);
  const incidentIds = model.incident.get(`${layer}:${nodeId}`) || [];
  const targetAngles = [];
  incidentIds.forEach((edgeId) => {
    const edge = model.edgeById.get(edgeId);
    if (!edge) return;
    const peerId = edge.sourceId === nodeId ? edge.targetId : edge.sourceId;
    const peerPos = positions.get(peerId);
    if (!peerPos) return;
    const repeat = Math.max(1, Math.round(edge.weight * 4));
    for (let i = 0; i < repeat; i += 1) {
      targetAngles.push(peerPos.angle);
    }
  });
  return circularMean(targetAngles);
}

function sortLayerByBarycenter(layer, state, model) {
  const current = state.layers[layer];
  if (current.length <= 2) return [...current];
  const enriched = current.map((node, index) => ({
    node,
    index,
    target: nodeTargetAngle(layer, node.node_id, state, model),
  }));
  const withTarget = enriched.filter((item) => item.target !== null).sort((a, b) => a.target - b.target);
  const withoutTarget = enriched.filter((item) => item.target === null).sort((a, b) => a.index - b.index);
  const merged = [...withTarget.map((item) => item.node), ...withoutTarget.map((item) => item.node)];
  return centerOut(merged);
}

function layerCost(layer, state, model) {
  const { positions } = computeAllPositions(state);
  let total = 0;
  const seen = new Set();
  (state.layers[layer] || []).forEach((node) => {
    const incidentIds = model.incident.get(`${layer}:${node.node_id}`) || [];
    incidentIds.forEach((edgeId) => {
      if (seen.has(edgeId)) return;
      const edge = model.edgeById.get(edgeId);
      if (!edge) return;
      const sourcePos = positions.get(edge.sourceId);
      const targetPos = positions.get(edge.targetId);
      if (!sourcePos || !targetPos) return;
      total += euclideanSquared(sourcePos, targetPos) * edge.weight;
      seen.add(edgeId);
    });
  });
  return total;
}

function improveLayerBySwaps(layer, state, model) {
  let changed = true;
  while (changed) {
    changed = false;
    const order = state.layers[layer];
    if (order.length <= 2) return;
    for (let i = 0; i < order.length - 1; i += 1) {
      const before = layerCost(layer, state, model);
      const swapped = [...order];
      const temp = swapped[i];
      swapped[i] = swapped[i + 1];
      swapped[i + 1] = temp;
      state.layers[layer] = swapped;
      const after = layerCost(layer, state, model);
      if (after + 1e-6 < before) {
        changed = true;
        break;
      }
      state.layers[layer] = order;
    }
  }
}

function optimizeLayerRotation(layer, state, model) {
  const order = state.layers[layer];
  if (!order.length) return;
  let bestRotation = state.rotations[layer];
  let bestCost = layerCost(layer, state, model);
  const steps = Math.max(24, order.length * 6);
  for (let step = 0; step < steps; step += 1) {
    const candidate = -Math.PI + (step * Math.PI * 2) / steps;
    state.rotations[layer] = candidate;
    const cost = layerCost(layer, state, model);
    if (cost < bestCost) {
      bestCost = cost;
      bestRotation = candidate;
    }
  }
  state.rotations[layer] = bestRotation;
}

function optimizeState(state, model) {
  const sweeps = [
    ["L3", "L2", "L1", "BRIDGE"],
    ["BRIDGE", "L1", "L2", "L3"],
  ];
  sweeps.forEach((sequence) => {
    sequence.forEach((layer) => {
      state.layers[layer] = sortLayerByBarycenter(layer, state, model);
      improveLayerBySwaps(layer, state, model);
      optimizeLayerRotation(layer, state, model);
    });
  });
  return state;
}

function buildLayout(view) {
  const model = buildLayoutModel(view);
  const state = optimizeState(buildState(model), model);
  const geometry = computeAllPositions(state);
  const nodes = [];
  LAYER_ORDER.forEach((layer) => {
    state.layers[layer].forEach((node) => nodes.push(node));
  });
  return {
    nodes,
    edges: model.edges,
    positions: geometry.positions,
    stageSize: state.stageSize,
    centerX: geometry.centerX,
    centerY: geometry.centerY,
  };
}

function treePath(source, target, centerX, centerY) {
  const radialOut = 44 * currentScale;
  const sx = centerX + Math.cos(source.angle) * (source.radius + radialOut);
  const sy = centerY + Math.sin(source.angle) * (source.radius + radialOut);
  const tx = centerX + Math.cos(target.angle) * (target.radius - radialOut);
  const ty = centerY + Math.sin(target.angle) * (target.radius - radialOut);
  return `M ${source.x} ${source.y} C ${sx} ${sy}, ${tx} ${ty}, ${target.x} ${target.y}`;
}

function nodeGlow(score) {
  const alpha = Math.max(0.2, Math.min(0.95, score));
  return `drop-shadow(0 0 ${20 + score * 30 * currentScale}px rgba(255,255,255,${alpha * 0.34})) drop-shadow(0 0 ${30 + score * 46 * currentScale}px rgba(96,146,255,${alpha * 0.25}))`;
}

function renderEmpty(message) {
  const svg = document.getElementById("activation-svg");
  svg.innerHTML = "";
  svg.setAttribute("viewBox", `0 0 ${BASE_STAGE_SIZE} ${BASE_STAGE_SIZE}`);
  svg.style.width = `${BASE_STAGE_SIZE}px`;
  svg.style.height = `${BASE_STAGE_SIZE}px`;
  document.getElementById("stage-scale").style.width = `${BASE_STAGE_SIZE}px`;
  document.getElementById("stage-scale").style.height = `${BASE_STAGE_SIZE}px`;
  document.getElementById("activation-query").textContent = message;
  document.getElementById("activation-summary").textContent = "还没有最近一次联想 trace。先回到聊天页发一条消息。";
}

function connectedEdgeIds(edges, nodeId) {
  if (!nodeId) return new Set();
  const out = new Set();
  edges.forEach((edge) => {
    if (edge.sourceId === nodeId || edge.targetId === nodeId) {
      out.add(edge.id);
    }
  });
  return out;
}

function renderActivation(trace) {
  const view = (trace || {}).activation_view || {};
  if (!(view.nodes || []).length) {
    renderEmpty("当前没有可展示的联想 trace。");
    return;
  }

  const layout = buildLayout(view);
  const highlightedEdges = connectedEdgeIds(layout.edges, activeNodeId);
  const svg = document.getElementById("activation-svg");
  svg.innerHTML = `
    <defs>
      <radialGradient id="node-fill" cx="50%" cy="45%" r="72%">
        <stop offset="0%" stop-color="#223146" stop-opacity="0.99" />
        <stop offset="100%" stop-color="#0b0f18" stop-opacity="0.96" />
      </radialGradient>
    </defs>
  `;
  svg.setAttribute("viewBox", `0 0 ${layout.stageSize} ${layout.stageSize}`);
  svg.style.width = `${layout.stageSize}px`;
  svg.style.height = `${layout.stageSize}px`;
  document.getElementById("stage-scale").style.width = `${layout.stageSize}px`;
  document.getElementById("stage-scale").style.height = `${layout.stageSize}px`;

  layout.edges
    .filter((edge) => edge.kind !== "bridge")
    .forEach((edge) => {
      const source = layout.positions.get(edge.sourceId);
      const target = layout.positions.get(edge.targetId);
      if (!source || !target) return;
      const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
      path.setAttribute("class", `flow-line${highlightedEdges.has(edge.id) ? " highlighted" : ""}`);
      path.setAttribute("d", treePath(source, target, layout.centerX, layout.centerY));
      path.setAttribute("stroke-width", String((highlightedEdges.has(edge.id) ? 4.4 : 2.2) * currentScale));
      path.style.opacity = String(
        highlightedEdges.has(edge.id) ? 1 : Math.max(0.24, Math.min(0.76, Number(edge.weight || 0)))
      );
      svg.appendChild(path);
    });

  layout.edges
    .filter((edge) => edge.kind === "bridge")
    .forEach((edge) => {
      const source = layout.positions.get(edge.sourceId);
      const target = layout.positions.get(edge.targetId);
      if (!source || !target) return;
      const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
      path.setAttribute("class", `flow-line bridge${highlightedEdges.has(edge.id) ? " highlighted" : ""}`);
      path.setAttribute("d", `M ${source.x} ${source.y} L ${target.x} ${target.y}`);
      path.setAttribute("stroke-width", String((highlightedEdges.has(edge.id) ? 3.6 : 1.8) * currentScale));
      path.style.opacity = String(
        highlightedEdges.has(edge.id) ? 1 : Math.max(0.24, Math.min(0.8, Number(edge.weight || 0)))
      );
      svg.appendChild(path);
    });

  layout.nodes.forEach((node) => {
    const pos = layout.positions.get(node.node_id);
    if (!pos) return;
    const score = Number(node.score || 0);
    const radius = (LEVEL_RADIUS[node.level] || 34) * currentScale;
    const [line1, line2] = splitLabel(node.name || "");
    const group = document.createElementNS("http://www.w3.org/2000/svg", "g");
    group.setAttribute("class", `node-group${activeNodeId === node.node_id ? " active" : ""}`);
    group.setAttribute("transform", `translate(${pos.x}, ${pos.y})`);
    group.addEventListener("click", (event) => {
      event.stopPropagation();
      activeNodeId = activeNodeId === node.node_id ? "" : node.node_id;
      renderActivation(trace);
    });

    const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    circle.setAttribute("class", "node-circle");
    circle.setAttribute("r", String(radius));
    circle.setAttribute("fill", "url(#node-fill)");
    circle.style.filter = nodeGlow(score);
    circle.style.opacity = String(Math.max(0.74, Math.min(1, 0.82 + score * 0.14)));
    group.appendChild(circle);

    if (node.direct_seed) {
      const ring = document.createElementNS("http://www.w3.org/2000/svg", "circle");
      ring.setAttribute("class", "node-ring");
      ring.setAttribute("r", String(radius + 7 * currentScale));
      group.appendChild(ring);
    }

    const label = document.createElementNS("http://www.w3.org/2000/svg", "text");
    label.setAttribute("class", "node-label");
    label.style.fontSize = `${15 * currentScale}px`;
    const topOffset = line2 ? -8 * currentScale : -2 * currentScale;
    const lineOne = document.createElementNS("http://www.w3.org/2000/svg", "tspan");
    lineOne.setAttribute("x", "0");
    lineOne.setAttribute("y", String(topOffset));
    lineOne.textContent = line1;
    label.appendChild(lineOne);
    if (line2) {
      const lineTwo = document.createElementNS("http://www.w3.org/2000/svg", "tspan");
      lineTwo.setAttribute("x", "0");
      lineTwo.setAttribute("dy", String(16 * currentScale));
      lineTwo.textContent = line2;
      label.appendChild(lineTwo);
    }
    group.appendChild(label);

    const scoreText = document.createElementNS("http://www.w3.org/2000/svg", "text");
    scoreText.setAttribute("class", "node-score");
    scoreText.setAttribute("y", String(radius * 0.56));
    scoreText.style.fontSize = `${11 * currentScale}px`;
    scoreText.textContent = score.toFixed(2);
    group.appendChild(scoreText);

    svg.appendChild(group);
  });

  svg.onclick = () => {
    if (!activeNodeId) return;
    activeNodeId = "";
    renderActivation(trace);
  };

  const counts = view.activation_counts || {};
  document.getElementById("activation-query").textContent = `query: ${view.query || trace.query || "-"}`;
  document.getElementById("activation-summary").textContent =
    `展示节点 ${layout.nodes.length} 个 | 边 ${layout.edges.length} 条 | ` +
    `L3=${counts.L3 || 0} / L2=${counts.L2 || 0} / L1=${counts.L1 || 0} | zoom=${currentScale.toFixed(2)}`;
}

document.getElementById("zoom-in-btn").addEventListener("click", () => setScale(currentScale + 0.12));
document.getElementById("zoom-out-btn").addEventListener("click", () => setScale(currentScale - 0.12));
document.getElementById("zoom-reset-btn").addEventListener("click", () => setScale(1.1));

renderActivation(loadTrace());
