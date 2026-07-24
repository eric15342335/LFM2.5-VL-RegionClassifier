"use strict";

const ids = [
  "file",
  "fileName",
  "shape",
  "retry",
  "cancel",
  "healthPill",
  "healthText",
  "elapsed",
  "status",
  "progressTrack",
  "pipeline",
  "label",
  "probability",
  "coverage",
  "backendTime",
  "engineTime",
  "queueTime",
  "imageDimensions",
  "canvasShell",
  "canvas",
  "canvasPlaceholder",
  "x1",
  "y1",
  "x2",
  "y2",
  "applyCoordinates",
  "cropDimensions",
  "cropPreview",
  "deviceBadge",
  "hardwareSummary",
  "hardwareFacts",
  "timingBars",
  "engineFacts",
  "eventLog",
  "clearEvents",
  "generation",
  "copyResult",
  "copyDebug",
  "rawResult",
  "rawHealth",
];

const element = Object.fromEntries(ids.map((id) => [id, document.getElementById(id)]));
const context = element.canvas.getContext("2d");
const cropContext = element.cropPreview.getContext("2d");
const stageOrder = ["accepted", "decode", "crop", "encode", "queued", "inference", "complete"];

let bitmap = null;
let start = null;
let end = null;
let drawing = false;
let latest = null;
let healthData = null;
let controller = null;
let uploadController = null;
let imageSessionId = null;
let imageLoadSequence = 0;
let requestSequence = 0;
let requestStarted = 0;
let timer = null;
let eventHistory = [];

function setText(target, value) {
  target.textContent = String(value);
}

function clamp(value, minimum, maximum) {
  return Math.min(maximum, Math.max(minimum, value));
}

function formatMilliseconds(value) {
  if (!Number.isFinite(Number(value))) return "—";
  const milliseconds = Number(value);
  return milliseconds >= 1000
    ? `${(milliseconds / 1000).toFixed(2)} s`
    : `${milliseconds.toFixed(1)} ms`;
}

function formatBytes(value) {
  if (!Number.isFinite(Number(value))) return "—";
  const bytes = Number(value);
  const units = ["B", "KiB", "MiB", "GiB", "TiB"];
  let size = bytes;
  let unit = 0;
  while (size >= 1024 && unit < units.length - 1) {
    size /= 1024;
    unit += 1;
  }
  const digits = unit === 0 || size >= 100 ? 0 : size >= 10 ? 1 : 2;
  return `${size.toFixed(digits)} ${units[unit]}`;
}

function formatValue(value) {
  if (value === null || value === undefined || value === "") return "—";
  if (Array.isArray(value)) return value.join(", ");
  if (typeof value === "object") return JSON.stringify(value);
  return String(value);
}

function point(event) {
  const bounds = element.canvas.getBoundingClientRect();
  return {
    x: clamp(
      ((event.clientX - bounds.left) * element.canvas.width) / bounds.width,
      0,
      element.canvas.width,
    ),
    y: clamp(
      ((event.clientY - bounds.top) * element.canvas.height) / bounds.height,
      0,
      element.canvas.height,
    ),
  };
}

function circleEnd(first, last) {
  const side = Math.max(Math.abs(last.x - first.x), Math.abs(last.y - first.y));
  return {
    x: first.x + Math.sign(last.x - first.x || 1) * side,
    y: first.y + Math.sign(last.y - first.y || 1) * side,
  };
}

function finalPoint() {
  return element.shape.value === "circle" ? circleEnd(start, end) : end;
}

function normalizedSelection() {
  const last = finalPoint();
  return {
    x1: clamp(Math.min(start.x, last.x) / element.canvas.width, 0, 1),
    y1: clamp(Math.min(start.y, last.y) / element.canvas.height, 0, 1),
    x2: clamp(Math.max(start.x, last.x) / element.canvas.width, 0, 1),
    y2: clamp(Math.max(start.y, last.y) / element.canvas.height, 0, 1),
  };
}

function syncCoordinateInputs() {
  if (!start || !end) return;
  const selection = normalizedSelection();
  for (const key of ["x1", "y1", "x2", "y2"]) {
    element[key].value = (selection[key] * 100).toFixed(1);
  }
}

function applyCoordinateInputs() {
  if (!bitmap) return;
  const values = Object.fromEntries(
    ["x1", "y1", "x2", "y2"].map((key) => [key, clamp(Number(element[key].value) / 100, 0, 1)]),
  );
  start = { x: values.x1 * element.canvas.width, y: values.y1 * element.canvas.height };
  end = { x: values.x2 * element.canvas.width, y: values.y2 * element.canvas.height };
  latest = null;
  draw();
  classify();
}

function strokeNormalizedBox(box, color, width, dashed = false) {
  const [x1, y1, x2, y2] = box;
  context.save();
  context.strokeStyle = color;
  context.lineWidth = width;
  context.setLineDash(dashed ? [10, 7] : []);
  context.strokeRect(
    x1 * element.canvas.width,
    y1 * element.canvas.height,
    (x2 - x1) * element.canvas.width,
    (y2 - y1) * element.canvas.height,
  );
  context.restore();
}

function drawSelection() {
  if (!start || !end) return;
  const last = finalPoint();
  const x = Math.min(start.x, last.x);
  const y = Math.min(start.y, last.y);
  const width = Math.abs(last.x - start.x);
  const height = Math.abs(last.y - start.y);

  context.save();
  context.strokeStyle = "#fb7185";
  context.fillStyle = "rgb(251 113 133 / 10%)";
  context.lineWidth = Math.max(3, element.canvas.width / 260);
  if (element.shape.value === "circle") {
    context.beginPath();
    context.ellipse(x + width / 2, y + height / 2, width / 2, height / 2, 0, 0, Math.PI * 2);
    context.fill();
    context.stroke();
  } else {
    context.fillRect(x, y, width, height);
    context.strokeRect(x, y, width, height);
  }
  context.restore();
}

function draw() {
  if (!bitmap) return;
  context.clearRect(0, 0, element.canvas.width, element.canvas.height);
  context.drawImage(bitmap, 0, 0, element.canvas.width, element.canvas.height);
  if (latest?.crop?.crop_box) {
    strokeNormalizedBox(latest.crop.crop_box, "#4ade80", 3, true);
  }
  drawSelection();
}

function drawCropPreview(result) {
  if (!bitmap || !result?.crop?.crop_box) return;
  const [x1, y1, x2, y2] = result.crop.crop_box;
  const sourceX = x1 * bitmap.width;
  const sourceY = y1 * bitmap.height;
  const sourceWidth = Math.max(1, (x2 - x1) * bitmap.width);
  const sourceHeight = Math.max(1, (y2 - y1) * bitmap.height);
  const scale = Math.min(1, 520 / sourceWidth, 300 / sourceHeight);

  element.cropPreview.width = Math.max(1, Math.round(sourceWidth * scale));
  element.cropPreview.height = Math.max(1, Math.round(sourceHeight * scale));
  cropContext.clearRect(0, 0, element.cropPreview.width, element.cropPreview.height);
  cropContext.drawImage(
    bitmap,
    sourceX,
    sourceY,
    sourceWidth,
    sourceHeight,
    0,
    0,
    element.cropPreview.width,
    element.cropPreview.height,
  );
  setText(
    element.cropDimensions,
    `${result.crop.crop_size_before_resize.join(" × ")} → ${result.crop.model_image_size.join(" × ")}`,
  );
}

function setProgress(stage, progress, message) {
  const currentIndex = stageOrder.indexOf(stage);
  const bounded = clamp(Number(progress) || 0, 0, 100);
  element.progressTrack.value = bounded;
  if (message) setText(element.status, message);

  for (const item of element.pipeline.querySelectorAll("li")) {
    const index = stageOrder.indexOf(item.dataset.stage);
    item.classList.toggle("done", index < currentIndex || stage === "complete");
    item.classList.toggle("active", index === currentIndex && stage !== "complete");
  }
}

function resetProgress() {
  setProgress("", 0, "Choose an image, then draw around one object.");
  element.status.classList.remove("error");
  for (const item of element.pipeline.querySelectorAll("li")) {
    item.classList.remove("done", "active");
  }
  setText(element.elapsed, "0 ms");
}

function startTimer() {
  stopTimer();
  requestStarted = performance.now();
  timer = window.setInterval(() => {
    setText(element.elapsed, formatMilliseconds(performance.now() - requestStarted));
  }, 100);
}

function stopTimer() {
  if (timer !== null) {
    window.clearInterval(timer);
    timer = null;
  }
}

function setBusy(busy) {
  element.retry.disabled = busy || !bitmap || !imageSessionId || !start || !end;
  element.cancel.disabled = !busy;
  element.file.disabled = busy;
  element.shape.disabled = busy;
  element.applyCoordinates.disabled = busy || !bitmap;
  element.canvas.classList.toggle("busy", busy);
}

function appendFact(fragment, name, value) {
  if (value === null || value === undefined || value === "") return;
  const wrapper = document.createElement("div");
  const term = document.createElement("dt");
  const description = document.createElement("dd");
  setText(term, name);
  setText(description, formatValue(value));
  wrapper.append(term, description);
  fragment.append(wrapper);
}

function renderFacts(target, entries) {
  const fragment = document.createDocumentFragment();
  for (const [name, value] of entries) appendFact(fragment, name, value);
  target.replaceChildren(fragment);
}

function renderHardware(health) {
  const hardware = health.hardware || {};
  const cpu = hardware.cpu || {};
  const memory = hardware.memory || {};
  const cuda = hardware.cuda;
  const device = health.device?.resolved || "unknown";
  const config = health.configuration || {};

  setText(element.deviceBadge, device.toUpperCase());
  const processor = cuda?.name || cpu.model_name || hardware.machine || "Unknown processor";
  const memorySummary = memory.total_bytes
    ? `${formatBytes(memory.total_bytes)} RAM`
    : "RAM unavailable";
  setText(element.hardwareSummary, `${processor} · ${memorySummary}`);

  renderFacts(element.hardwareFacts, [
    ["Architecture", cpu.architecture || hardware.machine],
    ["Logical CPUs", cpu.logical_cpus || hardware.logical_cpus],
    [
      "Topology",
      cpu.cores_per_socket ? `${cpu.cores_per_socket} cores/socket × ${cpu.sockets || 1}` : null,
    ],
    ["Threads/core", cpu.threads_per_core],
    ["ISA", cpu.noteworthy_flags],
    ["L2 / L3", [cpu.l2_cache, cpu.l3_cache].filter(Boolean).join(" / ")],
    [
      "RAM used",
      memory.used_bytes
        ? `${formatBytes(memory.used_bytes)} / ${formatBytes(memory.total_bytes)}`
        : null,
    ],
    [
      "Swap used",
      memory.swap_total_bytes
        ? `${formatBytes(memory.swap_used_bytes)} / ${formatBytes(memory.swap_total_bytes)}`
        : "None",
    ],
    ["Load average", hardware.load_average_1m_5m_15m],
    ["Inference threads", config.threads],
  ]);
}

function renderEngine(health) {
  const engine = health.engine || {};
  const config = health.configuration || {};
  const model = health.model || {};
  renderFacts(element.engineFacts, [
    ["Engine", engine.type],
    ["API model", model.api_model],
    ["GGUF", model.gguf],
    ["Threads", config.threads],
    ["Parallel slots", config.concurrency],
    ["Preprocess workers", config.preprocess_concurrency],
    ["Image tokens", config.image_tokens],
    ["Image side", config.image_side],
    ["Resampling", config.resampling],
    ["Timeout", `${config.request_timeout_seconds} s`],
    ["Ready in", formatMilliseconds(engine.startup?.engine_ready_ms)],
    ["llama.cpp build", engine.binary_version?.split("\n")[0]],
  ]);
}

function renderTimings(performanceData) {
  const entries = [
    ["Session lookup", performanceData.session_lookup_ms],
    ["Decode", performanceData.decode_ms],
    ["Crop", performanceData.crop_ms],
    ["Encode", performanceData.encode_ms],
    ["Queue", performanceData.queue_wait_ms],
    ["Engine", performanceData.engine_http_ms],
  ].filter(([, value]) => Number.isFinite(Number(value)));

  if (!entries.length) return;
  const total = Math.max(Number(performanceData.request_total_ms) || 0, 1);
  const fragment = document.createDocumentFragment();
  for (const [name, value] of entries) {
    const row = document.createElement("div");
    row.className = "timing-row";
    const header = document.createElement("div");
    header.className = "timing-label";
    const label = document.createElement("span");
    const duration = document.createElement("span");
    setText(label, name);
    setText(duration, formatMilliseconds(value));
    header.append(label, duration);

    const track = document.createElement("div");
    track.className = "timing-track";
    const bar = document.createElement("span");
    bar.style.width = `${clamp((Number(value) / total) * 100, 1.5, 100)}%`;
    track.append(bar);
    row.append(header, track);
    fragment.append(row);
  }
  element.timingBars.replaceChildren(fragment);
}

function renderRows(entries) {
  const fragment = document.createDocumentFragment();
  for (const [name, value, raw = false] of entries) {
    const row = document.createElement("tr");
    const key = document.createElement("th");
    const cell = document.createElement("td");
    key.scope = "row";
    setText(key, name);
    setText(cell, typeof value === "string" ? value : JSON.stringify(value, null, 2));
    if (raw) cell.className = "raw-cell";
    row.append(key, cell);
    fragment.append(row);
  }
  element.generation.replaceChildren(fragment);
}

function renderJson(target, value) {
  const code = document.createElement("code");
  if (value === null || value === undefined) {
    setText(code, "No data.");
    target.replaceChildren(code);
    return;
  }

  const json = JSON.stringify(value, null, 2);
  const tokenPattern = /"(?:\\.|[^"\\])*"|\b(?:true|false|null)\b|-?\d+(?:\.\d+)?(?:e[+-]?\d+)?/gi;
  let cursor = 0;
  for (const match of json.matchAll(tokenPattern)) {
    code.append(document.createTextNode(json.slice(cursor, match.index)));
    const token = match[0];
    const span = document.createElement("span");
    if (token.startsWith('"')) {
      const remainder = json.slice(match.index + token.length);
      span.className = /^\s*:/.test(remainder) ? "json-key" : "json-string";
    } else if (token === "null") {
      span.className = "json-null";
    } else if (token === "true" || token === "false") {
      span.className = "json-boolean";
    } else {
      span.className = "json-number";
    }
    setText(span, token);
    code.append(span);
    cursor = match.index + token.length;
  }
  code.append(document.createTextNode(json.slice(cursor)));
  target.replaceChildren(code);
}

function render(result, roundTrip) {
  latest = { ...result, browser: { round_trip_ms: Number(roundTrip.toFixed(3)) } };
  setText(element.elapsed, formatMilliseconds(roundTrip));
  const probability = result.token_probability;
  setText(element.label, result.label);
  setText(
    element.probability,
    probability === null || probability === undefined
      ? "Not returned"
      : `${(Number(probability) * 100).toFixed(1)}%`,
  );
  setText(element.coverage, `${(Number(result.crop.selected_fraction_of_crop) * 100).toFixed(1)}%`);
  setText(element.backendTime, formatMilliseconds(result.performance.request_total_ms));
  setText(element.engineTime, formatMilliseconds(result.performance.engine_http_ms));
  setText(element.queueTime, formatMilliseconds(result.performance.queue_wait_ms));

  renderTimings(result.performance);
  renderRows([
    ["Cleaned label", result.label],
    ["Raw output", result.generation.raw_output, true],
    ["Finish reason", result.generation.finish_reason],
    ["Mean token probability", result.token_probability],
    ["Usage", result.generation.usage, true],
    ["Token logprobs", result.generation.token_logprobs, true],
    ["Engine timings", result.generation.engine_timings, true],
    ["Selection box", result.selection.box, true],
    ["Crop box", result.crop.crop_box, true],
    ["Image session ID", result.image_session_id, true],
    ["Request ID", result.request_id, true],
  ]);
  renderJson(element.rawResult, latest);
  drawCropPreview(result);
  draw();
  element.copyResult.disabled = false;
  element.copyDebug.disabled = false;
}

function clearEventLog() {
  eventHistory = [];
  const empty = document.createElement("li");
  empty.className = "empty-copy";
  setText(empty, "No request events yet.");
  element.eventLog.replaceChildren(empty);
}

function addEvent(event) {
  if (eventHistory.length === 0) element.eventLog.replaceChildren();
  eventHistory.push(event);

  const item = document.createElement("li");
  const heading = document.createElement("div");
  heading.className = "event-heading";
  const stage = document.createElement("strong");
  const time = document.createElement("time");
  setText(stage, event.stage || event.event);
  setText(time, formatMilliseconds(event.elapsed_ms));
  heading.append(stage, time);

  const message = document.createElement("p");
  setText(message, event.message || "");
  item.append(heading, message);

  if (event.detail && Object.keys(event.detail).length) {
    const details = document.createElement("details");
    const summary = document.createElement("summary");
    const code = document.createElement("code");
    setText(summary, "Details");
    setText(code, JSON.stringify(event.detail, null, 2));
    details.append(summary, code);
    item.append(details);
  }
  element.eventLog.append(item);
}

async function consumeNdjson(response, currentRequest) {
  if (!response.ok) {
    const text = await response.text();
    try {
      const body = JSON.parse(text);
      throw new Error(body.detail || response.statusText);
    } catch (error) {
      if (error instanceof SyntaxError)
        throw new Error(text || response.statusText, { cause: error });
      throw error;
    }
  }
  if (!response.body) throw new Error("Streaming response body is unavailable");

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let result = null;

  async function handleLine(line) {
    if (!line.trim()) return;
    const event = JSON.parse(line);
    if (currentRequest !== requestSequence) return;
    if (event.event === "status") {
      setProgress(event.stage, event.progress, event.message);
      addEvent(event);
    } else if (event.event === "result") {
      result = event.result;
    } else if (event.event === "error") {
      const error = new Error(event.message || `Request failed with status ${event.status}`);
      error.status = event.status;
      throw error;
    }
  }

  while (true) {
    const { value, done } = await reader.read();
    buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
    const lines = buffer.split("\n");
    buffer = lines.pop() || "";
    for (const line of lines) await handleLine(line);
    if (done) break;
  }
  await handleLine(buffer);
  if (!result) throw new Error("The stream ended without a result");
  return result;
}

async function stageImage(file, signal) {
  const data = new FormData();
  data.append("image", file);
  const response = await fetch("/image-sessions", {
    method: "POST",
    body: data,
    signal,
  });
  if (!response.ok) {
    const text = await response.text();
    let message = text || response.statusText;
    try {
      message = JSON.parse(text).detail || message;
    } catch {
      // The plain response body is already the most useful message.
    }
    const error = new Error(message);
    error.status = response.status;
    throw error;
  }
  return response.json();
}

function releaseImageSession(sessionId) {
  if (!sessionId) return;
  fetch(`/image-sessions/${encodeURIComponent(sessionId)}`, {
    method: "DELETE",
    keepalive: true,
  }).catch(() => {
    // TTL cleanup is authoritative; deletion is only an eager optimization.
  });
}

function cancelActiveRequest() {
  controller?.abort();
  uploadController?.abort();
}

async function requestClassification(selection, currentRequest) {
  const data = new FormData();
  data.append("session_id", imageSessionId);
  data.append("shape", element.shape.value);
  for (const [key, value] of Object.entries(selection)) data.append(key, String(value));

  const response = await fetch("/classify/stream", {
    method: "POST",
    body: data,
    signal: controller.signal,
  });
  return consumeNdjson(response, currentRequest);
}

async function classify() {
  const file = element.file.files[0];
  if (!file || !imageSessionId || !start || !end) return;
  const selection = normalizedSelection();
  if (selection.x2 - selection.x1 < 0.01 || selection.y2 - selection.y1 < 0.01) {
    element.status.classList.add("error");
    setText(element.status, "Selection is too small.");
    return;
  }

  requestSequence += 1;
  const currentRequest = requestSequence;
  controller?.abort();
  controller = new AbortController();
  clearEventLog();
  setBusy(true);
  element.status.classList.remove("error");
  setProgress("accepted", 2, "Sending request…");
  startTimer();

  try {
    let result;
    try {
      result = await requestClassification(selection, currentRequest);
    } catch (error) {
      if (error.status !== 404 || currentRequest !== requestSequence) throw error;
      setProgress("accepted", 2, "Image session expired; uploading once more…");
      const session = await stageImage(file, controller.signal);
      if (currentRequest !== requestSequence) {
        releaseImageSession(session.session_id);
        return;
      }
      imageSessionId = session.session_id;
      result = await requestClassification(selection, currentRequest);
    }
    if (currentRequest !== requestSequence) return;
    const roundTrip = performance.now() - requestStarted;
    render(result, roundTrip);
    setProgress("complete", 100, `Completed: ${result.label}`);
  } catch (error) {
    if (error.name === "AbortError") {
      setProgress("", 0, "Request cancelled.");
    } else {
      element.status.classList.add("error");
      setText(element.status, error instanceof Error ? error.message : String(error));
      addEvent({
        event: "error",
        stage: "error",
        message: String(error),
        elapsed_ms: performance.now() - requestStarted,
      });
    }
  } finally {
    stopTimer();
    setBusy(false);
  }
}

async function loadHealth() {
  try {
    const response = await fetch("/health");
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    healthData = await response.json();
    renderHardware(healthData);
    renderEngine(healthData);
    renderJson(element.rawHealth, healthData);
    element.healthPill.className = "health-pill ok";
    setText(element.healthText, `${healthData.engine.type} ready`);
    element.copyDebug.disabled = false;
  } catch (error) {
    element.healthPill.className = "health-pill error";
    setText(element.healthText, "Backend unavailable");
    renderJson(element.rawHealth, { error: String(error) });
  }
}

async function loadImage(file) {
  controller?.abort();
  requestSequence += 1;
  uploadController?.abort();
  uploadController = new AbortController();
  imageLoadSequence += 1;
  const currentLoad = imageLoadSequence;
  const oldSessionId = imageSessionId;
  imageSessionId = null;
  releaseImageSession(oldSessionId);
  if (bitmap) {
    bitmap.close();
    bitmap = null;
  }

  element.canvas.width = 1;
  element.canvas.height = 1;
  element.canvasShell.classList.add("empty");
  element.canvasPlaceholder.hidden = false;
  setText(element.fileName, file.name);
  setText(element.imageDimensions, `${formatBytes(file.size)} · uploading`);
  setBusy(true);
  element.status.classList.remove("error");
  setText(element.status, "Uploading image and preparing the local preview…");

  const [bitmapResult, sessionResult] = await Promise.allSettled([
    createImageBitmap(file, { imageOrientation: "from-image" }),
    stageImage(file, uploadController.signal),
  ]);

  if (currentLoad !== imageLoadSequence) {
    if (bitmapResult.status === "fulfilled") bitmapResult.value.close();
    if (sessionResult.status === "fulfilled") releaseImageSession(sessionResult.value.session_id);
    return;
  }
  uploadController = null;

  if (bitmapResult.status === "rejected" || sessionResult.status === "rejected") {
    if (bitmapResult.status === "fulfilled") bitmapResult.value.close();
    if (sessionResult.status === "fulfilled") releaseImageSession(sessionResult.value.session_id);
    throw (
      (bitmapResult.status === "rejected" && bitmapResult.reason) ||
      (sessionResult.status === "rejected" && sessionResult.reason)
    );
  }

  bitmap = bitmapResult.value;
  const session = sessionResult.value;
  if (bitmap.width !== session.image.width || bitmap.height !== session.image.height) {
    bitmap.close();
    bitmap = null;
    releaseImageSession(session.session_id);
    throw new Error("Browser and server disagree on the image orientation");
  }
  imageSessionId = session.session_id;
  const scale = Math.min(1, 1200 / bitmap.width, 1200 / bitmap.height);
  element.canvas.width = Math.max(1, Math.round(bitmap.width * scale));
  element.canvas.height = Math.max(1, Math.round(bitmap.height * scale));
  element.canvasShell.classList.remove("empty");
  element.canvasPlaceholder.hidden = true;
  setText(element.fileName, file.name);
  setText(
    element.imageDimensions,
    `${bitmap.width} × ${bitmap.height} · ${formatBytes(file.size)} · staged`,
  );

  start = null;
  end = null;
  latest = null;
  resetProgress();
  clearEventLog();
  setText(element.label, "—");
  setText(element.probability, "—");
  setText(element.coverage, "—");
  setText(element.backendTime, "—");
  setText(element.engineTime, "—");
  setText(element.queueTime, "—");
  setText(element.cropDimensions, "—");
  element.cropPreview.width = 1;
  element.cropPreview.height = 1;
  cropContext.clearRect(0, 0, 1, 1);
  renderJson(element.rawResult, null);
  element.copyResult.disabled = true;
  setBusy(false);
  draw();
  setText(element.status, "Draw around one object; inference starts when you release.");
}

async function copyText(value, button) {
  const original = button.textContent;
  try {
    if (navigator.clipboard?.writeText) {
      await navigator.clipboard.writeText(value);
    } else {
      const textarea = document.createElement("textarea");
      textarea.value = value;
      textarea.style.position = "fixed";
      textarea.style.opacity = "0";
      document.body.append(textarea);
      textarea.select();
      const copied = document.execCommand("copy");
      textarea.remove();
      if (!copied) throw new Error("Browser copy command failed");
    }
    setText(button, "Copied");
  } catch {
    setText(button, "Copy failed");
  }
  window.setTimeout(() => setText(button, original), 1200);
}

element.file.addEventListener("change", async () => {
  const file = element.file.files[0];
  if (!file) return;
  try {
    await loadImage(file);
  } catch (error) {
    setBusy(false);
    if (error.name === "AbortError") {
      setText(element.status, "Image upload cancelled.");
      return;
    }
    element.status.classList.add("error");
    setText(element.status, `Could not open image: ${error}`);
  }
});

element.canvas.addEventListener("pointerdown", (event) => {
  if (!bitmap || element.canvas.classList.contains("busy")) return;
  element.canvas.setPointerCapture(event.pointerId);
  start = point(event);
  end = start;
  drawing = true;
  latest = null;
  draw();
});

element.canvas.addEventListener("pointermove", (event) => {
  if (!drawing) return;
  end = point(event);
  draw();
});

element.canvas.addEventListener("pointerup", (event) => {
  if (!drawing) return;
  drawing = false;
  end = point(event);
  syncCoordinateInputs();
  draw();
  classify();
});

element.canvas.addEventListener("pointercancel", () => {
  drawing = false;
  draw();
});

element.shape.addEventListener("change", () => {
  draw();
  if (start && end) classify();
});

element.retry.addEventListener("click", classify);
element.cancel.addEventListener("click", cancelActiveRequest);
element.applyCoordinates.addEventListener("click", applyCoordinateInputs);
element.clearEvents.addEventListener("click", clearEventLog);
element.copyResult.addEventListener("click", () =>
  copyText(JSON.stringify(latest, null, 2), element.copyResult),
);
element.copyDebug.addEventListener("click", () => {
  const debug = { health: healthData, latest, events: eventHistory };
  copyText(JSON.stringify(debug, null, 2), element.copyDebug);
});

document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && !element.cancel.disabled) cancelActiveRequest();
});

window.addEventListener("pagehide", () => releaseImageSession(imageSessionId));

(async () => {
  resetProgress();
  setBusy(false);
  await loadHealth();
})();
