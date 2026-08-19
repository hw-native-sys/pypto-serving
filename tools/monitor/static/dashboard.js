"use strict";

const ui = {
  model: document.getElementById("model-name"),
  status: document.getElementById("connection-status"),
  updated: document.getElementById("last-updated"),
  error: document.getElementById("error-banner"),
  outputTps: document.getElementById("output-tps"),
  prefillTps: document.getElementById("prefill-tps"),
  tpotP50: document.getElementById("tpot-p50"),
  ttftP50: document.getElementById("ttft-p50"),
  running: document.getElementById("running"),
  waiting: document.getElementById("waiting"),
  kvUsage: document.getElementById("kv-usage"),
  cacheHit: document.getElementById("cache-hit"),
  todayDate: document.getElementById("today-date"),
  todayRequests: document.getElementById("today-requests"),
  todayPrompt: document.getElementById("today-prompt"),
  todayOutput: document.getElementById("today-output"),
  todayErrors: document.getElementById("today-errors"),
  dailyTable: document.getElementById("daily-table"),
};

const state = { range: 3600, history: [], daily: [] };
const colors = { output: "#0f807c", prefill: "#2869a6", running: "#16835b", waiting: "#b06b16", grid: "#e7ebef", text: "#66727e" };

function number(value, digits = 0) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "--";
  return new Intl.NumberFormat("en-US", { maximumFractionDigits: digits }).format(Number(value));
}

function milliseconds(seconds) {
  return seconds === null || seconds === undefined ? "--" : number(seconds * 1000, 1);
}

function percent(value) {
  return value === null || value === undefined ? "--" : `${number(value * 100, 1)}%`;
}

async function getJson(url) {
  const response = await fetch(url, { cache: "no-store" });
  if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
  return response.json();
}

function setConnection(collector) {
  const connected = Boolean(collector && collector.connected);
  ui.status.className = `status ${connected ? "connected" : "disconnected"}`;
  ui.status.lastElementChild.textContent = connected ? "Connected" : "Disconnected";
  if (connected) {
    ui.error.hidden = true;
  } else {
    ui.error.textContent = collector && collector.last_error ? collector.last_error : "Metrics target unavailable";
    ui.error.hidden = false;
  }
}

async function refreshSummary() {
  try {
    const data = await getJson("/api/summary?window=300");
    const current = data.current || {};
    const today = data.today || {};
    ui.model.textContent = data.model_name || "Unknown model";
    ui.outputTps.textContent = number(current.generation_tokens_per_second, 1);
    ui.prefillTps.textContent = number(current.prefill_tokens_per_second, 1);
    ui.tpotP50.textContent = milliseconds(current.tpot_p50_seconds);
    ui.ttftP50.textContent = milliseconds(current.ttft_p50_seconds);
    ui.running.textContent = number(current.running);
    ui.waiting.textContent = number(current.waiting);
    ui.kvUsage.textContent = percent(current.kv_cache_usage);
    ui.cacheHit.textContent = percent(current.cache_hit_rate);
    ui.todayDate.textContent = today.day || "--";
    ui.todayRequests.textContent = number(today.request_count);
    ui.todayPrompt.textContent = number(today.prompt_tokens);
    ui.todayOutput.textContent = number(today.generation_tokens);
    ui.todayErrors.textContent = number((today.errors || 0) + (today.aborted || 0));
    ui.updated.textContent = data.timestamp ? `Updated ${new Date(data.timestamp * 1000).toLocaleTimeString()}` : "No samples";
    setConnection(data.collector);
  } catch (error) {
    setConnection({ connected: false, last_error: error.message });
  }
}

async function refreshHistory() {
  try {
    const data = await getJson(`/api/history?range=${state.range}`);
    state.history = data.points || [];
    renderCharts();
  } catch (error) {
    ui.error.textContent = error.message;
    ui.error.hidden = false;
  }
}

async function refreshDaily() {
  try {
    const data = await getJson("/api/daily?days=30");
    state.daily = data.days || [];
    renderDailyTable();
    renderCharts();
  } catch (error) {
    ui.error.textContent = error.message;
    ui.error.hidden = false;
  }
}

function prepareCanvas(canvas) {
  const rect = canvas.getBoundingClientRect();
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.round(rect.width * ratio));
  canvas.height = Math.max(1, Math.round(rect.height * ratio));
  const context = canvas.getContext("2d");
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  return { context, width: rect.width, height: rect.height };
}

function drawEmpty(context, width, height) {
  context.fillStyle = colors.text;
  context.font = "12px system-ui";
  context.textAlign = "center";
  context.fillText("No samples", width / 2, height / 2);
}

function drawLineChart(canvas, points, definitions) {
  const { context, width, height } = prepareCanvas(canvas);
  context.clearRect(0, 0, width, height);
  if (!points.length) return drawEmpty(context, width, height);
  const margin = { top: 12, right: 12, bottom: 28, left: 48 };
  const plotWidth = Math.max(1, width - margin.left - margin.right);
  const plotHeight = Math.max(1, height - margin.top - margin.bottom);
  const values = definitions.flatMap((definition) => points.map((point) => Number(point[definition.key]) || 0));
  const maximum = Math.max(1, ...values) * 1.1;

  context.strokeStyle = colors.grid;
  context.lineWidth = 1;
  context.fillStyle = colors.text;
  context.font = "11px system-ui";
  context.textAlign = "right";
  for (let index = 0; index <= 4; index += 1) {
    const y = margin.top + (plotHeight * index) / 4;
    context.beginPath();
    context.moveTo(margin.left, y);
    context.lineTo(width - margin.right, y);
    context.stroke();
    context.fillText(number(maximum * (1 - index / 4), 1), margin.left - 7, y + 4);
  }

  definitions.forEach((definition) => {
    context.strokeStyle = definition.color;
    context.lineWidth = 2;
    context.beginPath();
    points.forEach((point, index) => {
      const x = margin.left + (plotWidth * index) / Math.max(1, points.length - 1);
      const y = margin.top + plotHeight - ((Number(point[definition.key]) || 0) / maximum) * plotHeight;
      if (index === 0) context.moveTo(x, y); else context.lineTo(x, y);
    });
    context.stroke();
  });

  context.fillStyle = colors.text;
  context.textAlign = "left";
  context.fillText(new Date(points[0].bucket * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }), margin.left, height - 7);
  context.textAlign = "right";
  context.fillText(new Date(points[points.length - 1].bucket * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }), width - margin.right, height - 7);
}

function drawDailyChart(canvas, rows) {
  const { context, width, height } = prepareCanvas(canvas);
  context.clearRect(0, 0, width, height);
  if (!rows.length) return drawEmpty(context, width, height);
  const visible = rows.slice(-14);
  const margin = { top: 10, right: 10, bottom: 32, left: 46 };
  const plotWidth = width - margin.left - margin.right;
  const plotHeight = height - margin.top - margin.bottom;
  const maximum = Math.max(1, ...visible.flatMap((row) => [row.prompt_tokens, row.generation_tokens]));
  const groupWidth = plotWidth / visible.length;
  const barWidth = Math.max(2, Math.min(14, groupWidth * 0.32));

  context.strokeStyle = colors.grid;
  for (let index = 0; index <= 3; index += 1) {
    const y = margin.top + (plotHeight * index) / 3;
    context.beginPath(); context.moveTo(margin.left, y); context.lineTo(width - margin.right, y); context.stroke();
  }
  visible.forEach((row, index) => {
    const center = margin.left + groupWidth * index + groupWidth / 2;
    const promptHeight = (Number(row.prompt_tokens) / maximum) * plotHeight;
    const outputHeight = (Number(row.generation_tokens) / maximum) * plotHeight;
    context.fillStyle = colors.prefill;
    context.fillRect(center - barWidth - 1, margin.top + plotHeight - promptHeight, barWidth, promptHeight);
    context.fillStyle = colors.output;
    context.fillRect(center + 1, margin.top + plotHeight - outputHeight, barWidth, outputHeight);
    if (index === 0 || index === visible.length - 1 || visible.length <= 7) {
      context.fillStyle = colors.text;
      context.font = "10px system-ui";
      context.textAlign = "center";
      context.fillText(row.day.slice(5), center, height - 8);
    }
  });
}

function renderCharts() {
  drawLineChart(document.getElementById("throughput-chart"), state.history, [
    { key: "generation_tps", color: colors.output },
    { key: "prefill_tps", color: colors.prefill },
  ]);
  drawLineChart(document.getElementById("queue-chart"), state.history, [
    { key: "running", color: colors.running },
    { key: "waiting", color: colors.waiting },
  ]);
  drawDailyChart(document.getElementById("daily-chart"), state.daily);
}

function renderDailyTable() {
  if (!state.daily.length) {
    ui.dailyTable.innerHTML = '<tr><td colspan="6" class="empty-cell">No daily data</td></tr>';
    return;
  }
  ui.dailyTable.innerHTML = [...state.daily].reverse().map((row) => {
    const hitRate = row.cache_queries ? `${number((row.cache_hits / row.cache_queries) * 100, 1)}%` : "--";
    return `<tr><td>${row.day}</td><td>${number(row.request_count)}</td><td>${number(row.prompt_tokens)}</td><td>${number(row.generation_tokens)}</td><td>${hitRate}</td><td>${number((row.errors || 0) + (row.aborted || 0))}</td></tr>`;
  }).join("");
}

document.getElementById("range-control").addEventListener("click", (event) => {
  const button = event.target.closest("button[data-range]");
  if (!button) return;
  state.range = Number(button.dataset.range);
  document.querySelectorAll("#range-control button").forEach((item) => item.classList.toggle("active", item === button));
  refreshHistory();
});

document.getElementById("refresh-button").addEventListener("click", () => Promise.all([refreshSummary(), refreshHistory(), refreshDaily()]));
window.addEventListener("resize", renderCharts);

Promise.all([refreshSummary(), refreshHistory(), refreshDaily()]);
setInterval(refreshSummary, 2000);
setInterval(refreshHistory, 5000);
setInterval(refreshDaily, 60000);
