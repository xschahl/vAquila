const toast = document.getElementById("toast");
const notificationCenter = document.getElementById("notification-center");
const statusHealth = document.getElementById("status-health");
const runForm = document.getElementById("run-form");
const runSubmitButton = runForm?.querySelector('button[type="submit"]');
const runEstimateCard = document.getElementById("run-estimate-card");
const runEstimateStatus = document.getElementById("run-estimate-status");
const runEstimateMetrics = document.getElementById("run-estimate-metrics");
const inferForm = document.getElementById("infer-form");
const inferOutput = document.getElementById("infer-output");
const inferMetrics = document.getElementById("infer-metrics");
const inferTarget = document.getElementById("infer-target");
const inferEndpointHint = document.getElementById("infer-endpoint-hint");
const inferImagesInput = document.getElementById("infer-images");
const inferImagesPreview = document.getElementById("infer-images-preview");
const inferImagesList = document.getElementById("infer-images-list");
const inferSubmitButton = document.getElementById("infer-submit-button");
const inferStopButton = document.getElementById("infer-stop-button");

const tasksBody = document.getElementById("tasks-body");
const containersBody = document.getElementById("containers-body");
const cacheBody = document.getElementById("cache-body");
const gpuGrid = document.getElementById("gpu-grid");
const gpuSummary = document.getElementById("gpu-summary");
const systemSummary = document.getElementById("system-summary");
const cpuUsageValue = document.getElementById("cpu-usage-value");
const ramUsageValue = document.getElementById("ram-usage-value");
const cpuUsageBar = document.getElementById("cpu-usage-bar");
const ramUsageBar = document.getElementById("ram-usage-bar");
const cpuModelCpuStack = document.getElementById("cpu-model-cpu-stack");
const cpuModelRamStack = document.getElementById("cpu-model-ram-stack");
const cpuModelList = document.getElementById("cpu-model-list");

const runDevice = document.getElementById("run-device");
const runGpuField = document.getElementById("run-gpu-field");
const runBufferField = document.getElementById("run-buffer-field");
const runGpuUtilField = document.getElementById("run-gpu-utilization-field");
const runCpuUtilField = document.getElementById("run-cpu-utilization-field");
const runCpuKvCacheField = document.getElementById("run-cpu-kv-cache-field");
const runGpuInput = document.getElementById("run-gpu-index");
const runBufferInput = document.getElementById("run-buffer-gb");
const runGpuUtilInput = document.getElementById("run-gpu-utilization");
const runCpuUtilInput = document.getElementById("run-cpu-utilization");
const runCpuKvCacheInput = document.getElementById("run-cpu-kv-cache");

const statContainers = document.getElementById("stat-containers");
const statRunning = document.getElementById("stat-running");
const statCache = document.getElementById("stat-cache");
const statTasks = document.getElementById("stat-tasks");

const logsModal = document.getElementById("logs-modal");
const logsTarget = document.getElementById("logs-target");
const logsOutput = document.getElementById("logs-output");
const logsCopy = document.getElementById("logs-copy");
const logsClose = document.getElementById("logs-close");
const logsRefresh = document.getElementById("logs-refresh");
const themeToggle = document.getElementById("theme-toggle");

const MAX_NOTIFICATIONS = 5;
const THEME_STORAGE_KEY = "vaquila.webui.theme";

let selectedLogSource = null;
let logsInterval = null;
let refreshErrorNotified = false;
let lastLogsErrorKey = null;
let tasksSnapshotReady = false;
let themeTransitionTimer = null;
let runEstimateTimer = null;
let runLaunchBlockedReason = "";
let inferAbortController = null;

const taskStates = new Map();

function resolveInitialTheme() {
  let persisted = null;
  try {
    persisted = window.localStorage.getItem(THEME_STORAGE_KEY);
  } catch {
    persisted = null;
  }
  if (persisted === "light" || persisted === "dark") {
    return persisted;
  }

  if (window.matchMedia?.("(prefers-color-scheme: light)")?.matches) {
    return "light";
  }
  return "dark";
}

function applyTheme(theme) {
  const normalized = theme === "light" ? "light" : "dark";
  document.body.classList.add("is-theme-animating");
  document.body.dataset.theme = normalized;
  document.documentElement.style.colorScheme = normalized;

  if (themeTransitionTimer !== null) {
    window.clearTimeout(themeTransitionTimer);
  }
  themeTransitionTimer = window.setTimeout(() => {
    document.body.classList.remove("is-theme-animating");
    themeTransitionTimer = null;
  }, 380);

  if (themeToggle) {
    themeToggle.textContent =
      normalized === "dark" ? "Light mode" : "Dark mode";
    themeToggle.setAttribute("aria-pressed", String(normalized === "light"));
  }
}

function initTheme() {
  const initial = resolveInitialTheme();
  applyTheme(initial);

  themeToggle?.addEventListener("click", () => {
    const current = document.body.dataset.theme === "light" ? "light" : "dark";
    const next = current === "dark" ? "light" : "dark";
    applyTheme(next);
    try {
      window.localStorage.setItem(THEME_STORAGE_KEY, next);
    } catch {
      // Ignore storage errors and keep the in-memory theme state.
    }
  });
}

function setStatus(message, type = "info") {
  toast.textContent = message;
  toast.classList.remove("is-ok", "is-error");
  if (type === "ok") toast.classList.add("is-ok");
  if (type === "error") toast.classList.add("is-error");
}

function notify(
  message,
  type = "info",
  title = "Information",
  duration = 4200,
  options = {},
) {
  if (!notificationCenter) return;

  const { groupKey } = options;

  const toastKey = `${type}::${title}::${message}`;
  const selector = groupKey
    ? `[data-toast-group="${CSS.escape(groupKey)}"]`
    : `[data-toast-key="${CSS.escape(toastKey)}"]`;
  const existingToast = notificationCenter.querySelector(selector);

  if (existingToast) {
    existingToast.remove();
  }

  while (notificationCenter.childElementCount >= MAX_NOTIFICATIONS) {
    notificationCenter.lastElementChild?.remove();
  }

  const toastItem = document.createElement("article");
  toastItem.className = `toast-item toast-${type}`;
  toastItem.dataset.toastKey = toastKey;
  if (groupKey) {
    toastItem.dataset.toastGroup = groupKey;
  }

  const head = document.createElement("div");
  head.className = "toast-head";

  const titleEl = document.createElement("div");
  titleEl.className = "toast-title";
  titleEl.textContent = title;

  const closeButton = document.createElement("button");
  closeButton.type = "button";
  closeButton.className = "toast-close";
  closeButton.textContent = "×";

  const messageEl = document.createElement("div");
  messageEl.className = "toast-message";
  messageEl.textContent = message;

  const progress = document.createElement("div");
  progress.className = "toast-progress";

  const progressBar = document.createElement("span");
  progressBar.className = "toast-progress-bar";
  progressBar.style.setProperty(
    "--toast-duration",
    `${Math.max(duration, 1)}ms`,
  );
  progress.appendChild(progressBar);

  head.appendChild(titleEl);
  head.appendChild(closeButton);
  toastItem.appendChild(head);
  toastItem.appendChild(messageEl);
  toastItem.appendChild(progress);

  let removed = false;
  let timeoutId = null;
  const removeToast = () => {
    if (removed) return;
    removed = true;
    if (timeoutId) {
      window.clearTimeout(timeoutId);
    }
    toastItem.classList.add("is-leaving");
    window.setTimeout(() => toastItem.remove(), 180);
  };

  closeButton.addEventListener("click", removeToast);
  timeoutId = window.setTimeout(removeToast, duration);
  notificationCenter.prepend(toastItem);
}

function rememberTaskState(task) {
  if (!task?.id) return;
  taskStates.set(task.id, {
    status: task.status || "unknown",
    message: task.message || "",
    modelId: task.model_id || "model",
  });
}

function getTaskNotificationConfig(task) {
  const modelId = task.model_id || "model";
  const message = task.message || "Task updated.";
  const groupKey = task.id ? `task:${task.id}` : `task:${modelId}`;

  switch (String(task.status || "").toLowerCase()) {
    case "queued":
      return {
        title: "Task queued",
        type: "info",
        message: `${modelId} has been queued.`,
        duration: 3200,
        groupKey,
      };
    case "running":
      return {
        title: "Task running",
        type: "info",
        message: `${modelId}: ${message}`,
        duration: 3600,
        groupKey,
      };
    case "succeeded":
      return {
        title: "Task completed",
        type: "success",
        message: `${modelId}: ${message}`,
        duration: 5200,
        groupKey,
      };
    case "failed":
      return {
        title: "Task failed",
        type: "error",
        message: `${modelId}: ${message}`,
        duration: 7000,
        groupKey,
      };
    default:
      return {
        title: "Task updated",
        type: "info",
        message: `${modelId}: ${message}`,
        duration: 3600,
        groupKey,
      };
  }
}

function syncTaskNotifications(items) {
  const nextTaskStates = new Map();

  if (!tasksSnapshotReady) {
    (Array.isArray(items) ? items : []).forEach((task) => {
      nextTaskStates.set(task.id, {
        status: task.status || "unknown",
        message: task.message || "",
        modelId: task.model_id || "model",
      });
    });
    taskStates.clear();
    nextTaskStates.forEach((value, key) => taskStates.set(key, value));
    tasksSnapshotReady = true;
    return;
  }

  (Array.isArray(items) ? items : []).forEach((task) => {
    const current = {
      status: task.status || "unknown",
      message: task.message || "",
      modelId: task.model_id || "model",
    };
    nextTaskStates.set(task.id, current);

    const previous = taskStates.get(task.id);
    const statusChanged = !previous || previous.status !== current.status;
    const messageChanged = !previous || previous.message !== current.message;

    if (
      !previous ||
      statusChanged ||
      (current.status === "running" && messageChanged)
    ) {
      const config = getTaskNotificationConfig(task);
      notify(config.message, config.type, config.title, config.duration, {
        groupKey: config.groupKey,
      });
      if (current.status === "failed") {
        setStatus(`${task.model_id}: ${current.message}`, "error");
      }
      if (current.status === "succeeded") {
        setStatus(`${task.model_id}: ${current.message}`, "ok");
      }
    }
  });

  taskStates.clear();
  nextTaskStates.forEach((value, key) => taskStates.set(key, value));
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });

  let payload = {};
  try {
    payload = await response.json();
  } catch {
    payload = {};
  }

  if (!response.ok) {
    throw new Error(payload?.detail ?? `HTTP ${response.status}`);
  }

  return payload;
}

function formatDate(isoString) {
  if (!isoString) return "-";
  const date = new Date(isoString);
  if (Number.isNaN(date.getTime())) return isoString;
  return date.toLocaleString();
}

function formatGiB(bytes) {
  return (Number(bytes || 0) / 1024 ** 3).toFixed(2);
}

function getInstanceId(item) {
  if (item?.instance_id) return String(item.instance_id);
  if (item?.name) {
    const normalized = String(item.name);
    if (normalized.startsWith("vaq-")) {
      return normalized.slice(4);
    }
    return normalized;
  }
  return "unknown";
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function parseFormValue(value) {
  if (value === "") return undefined;
  if (value === "true") return true;
  if (value === "false") return false;
  if (!Number.isNaN(Number(value)) && value.trim() !== "") return Number(value);
  return value;
}

function getFormPayload(form) {
  const payload = {};
  const formData = new FormData(form);
  for (const [key, rawValue] of formData.entries()) {
    payload[key] = parseFormValue(String(rawValue));
  }
  return payload;
}

function formatEstimateRatio(value) {
  if (typeof value !== "number" || Number.isNaN(value)) return "-";
  return value.toFixed(3);
}

function setRunLaunchBlocked(blocked, reason = "") {
  runLaunchBlockedReason = blocked
    ? String(reason || "Launch is blocked.")
    : "";

  if (runSubmitButton) {
    runSubmitButton.disabled = blocked;
    runSubmitButton.title = blocked ? runLaunchBlockedReason : "";
    runSubmitButton.setAttribute("aria-disabled", blocked ? "true" : "false");
  }
}

function setRunEstimateState(state, statusText, options = {}) {
  if (!runEstimateCard) return;

  const icon = document.getElementById("run-estimate-icon");
  const status = document.getElementById("run-estimate-status");
  const barWrap = document.getElementById("estimate-bar-wrap");
  const metricsGrid = document.getElementById("estimate-metrics-grid");
  const footer = document.getElementById("estimate-footer");

  runEstimateCard.classList.remove("state-ok", "state-warn", "state-error");
  if (state === "ok") runEstimateCard.classList.add("state-ok");
  if (state === "warn") runEstimateCard.classList.add("state-warn");
  if (state === "error") runEstimateCard.classList.add("state-error");

  const icons = {
    ok: "✅",
    warn: "⚠️",
    error: "❌",
    loading: "⏳",
    idle: "⚡",
  };
  if (icon) icon.textContent = icons[state] || icons.idle;
  if (status) status.textContent = statusText;

  const showDetails = !!options.breakdown || state === "loading";
  const isLoading = state === "loading";

  if (barWrap) {
    barWrap.style.display = showDetails ? "" : "none";
    barWrap.style.opacity = isLoading ? "0.4" : "1";
    barWrap.style.transition = "opacity 0.2s ease";
  }
  if (metricsGrid) {
    metricsGrid.style.display = showDetails ? "" : "none";
    metricsGrid.style.opacity = isLoading ? "0.4" : "1";
    metricsGrid.style.transition = "opacity 0.2s ease";
  }
  if (footer) {
    footer.style.display = showDetails || options.footerText ? "" : "none";
  }

  // Segmented bar
  const segWeights = document.getElementById("estimate-seg-weights");
  const segKv = document.getElementById("estimate-seg-kv");
  const segOverhead = document.getElementById("estimate-seg-overhead");
  const segFree = document.getElementById("estimate-seg-free");

  if (showDetails && options.segments) {
    const s = options.segments;
    if (segWeights) {
      segWeights.style.width = `${s.weightsPct}%`;
      segWeights.dataset.tooltip = `Weights: ${s.weightsGb} GiB (${s.weightsPct.toFixed(1)}%)`;
    }
    if (segKv) {
      segKv.style.width = `${s.kvPct}%`;
      segKv.dataset.tooltip = `KV cache: ${s.kvGb} GiB (${s.kvPct.toFixed(1)}%)`;
    }
    if (segOverhead) {
      segOverhead.style.width = `${s.overheadPct}%`;
      segOverhead.dataset.tooltip = `Overhead: ${s.overheadGb} GiB (${s.overheadPct.toFixed(1)}%)`;
    }
    if (segFree) {
      segFree.style.width = `${s.freePct}%`;
      segFree.dataset.tooltip = `Free: ${s.freeGb} GiB (${s.freePct.toFixed(1)}%)`;
    }
  } else {
    [segWeights, segKv, segOverhead, segFree].forEach((seg) => {
      if (seg) seg.style.width = "0";
    });
  }

  // Metrics
  const setMetric = (id, value) => {
    const el = document.getElementById(id);
    if (el) el.textContent = value;
  };
  if (options.metrics) {
    const m = options.metrics;
    setMetric("em-weights", m.weightsGb ?? "—");
    setMetric("em-kv", m.kvGb ?? "—");
    setMetric("em-overhead", m.overheadGb ?? "—");
    setMetric("em-total", m.totalGb ?? "—");
    setMetric("em-available", m.availableGb ?? "—");
    setMetric("em-maxseqs", m.maxSeqs ?? "—");
  } else {
    [
      "em-weights",
      "em-kv",
      "em-overhead",
      "em-total",
      "em-available",
      "em-maxseqs",
    ].forEach((id) => setMetric(id, "—"));
  }

  // Confidence
  const confDot = document.getElementById("estimate-confidence-dot");
  const confLabel = document.getElementById("estimate-confidence-label");
  const footerMeta = document.getElementById("estimate-footer-meta");

  if (confDot) {
    confDot.classList.remove(
      "confidence-high",
      "confidence-medium",
      "confidence-low",
    );
    if (options.confidence) {
      confDot.classList.add(`confidence-${options.confidence}`);
    }
  }
  if (confLabel) confLabel.textContent = options.confidenceLabel || "";
  if (footerMeta) footerMeta.textContent = options.footerText || "";
}

function renderRunEstimate(result) {
  setRunLaunchBlocked(result?.port_available === false, result?.message || "");

  if (!result || result.ok !== true) {
    setRunEstimateState(
      "error",
      result?.message || "Unable to compute estimate.",
    );
    return;
  }

  if (result.manual_mode === true) {
    setRunEstimateState(
      "warn",
      result.message || "Manual utilization mode enabled.",
      {
        footerText: `device=${result.device || "gpu"} | gpu_util=${result.gpu_utilization ?? "auto"} | estimation=disabled`,
      },
    );
    return;
  }

  if (String(result.device || "gpu").toLowerCase() === "cpu") {
    setRunEstimateState("ok", result.message || "CPU mode selected.", {
      footerText: `max-num-seqs=${result.requested_max_num_seqs ?? "n/a"} | quantization=${result.quantization ?? "n/a"}`,
    });
    return;
  }

  const breakdown = result.breakdown || {};
  const totalVramGb = Number(result.total_vram_gb || 0);
  const availableGb = Number(result.available_vram_gb || 0);
  const weightsGb = Number(breakdown.weights_gb || 0);
  const kvGb = Number(breakdown.kv_cache_gb || 0);
  const overheadGb = Number(breakdown.runtime_overhead_gb || 0);
  const totalEstGb = Number(breakdown.total_gb || 0);
  const estimatedMaxSeqs = result.estimated_max_num_seqs;
  const requestedMaxSeqs = Number(result.requested_max_num_seqs || 0);
  const fitsRequested =
    typeof estimatedMaxSeqs !== "number" ||
    requestedMaxSeqs <= estimatedMaxSeqs;

  const fits = result.fits_current_settings && fitsRequested;
  const state = fits ? "ok" : "warn";
  const statusText = fits
    ? "Current settings fit available VRAM"
    : "Current settings are likely above available VRAM";

  // Segment percentages (relative to total VRAM)
  const denom = totalVramGb > 0 ? totalVramGb : totalEstGb || 1;
  const weightsPct = Math.min(100, (weightsGb / denom) * 100);
  const kvPct = Math.min(100 - weightsPct, (kvGb / denom) * 100);
  const overheadPct = Math.min(
    100 - weightsPct - kvPct,
    (overheadGb / denom) * 100,
  );
  const usedPct = weightsPct + kvPct + overheadPct;
  const freePct = Math.max(0, 100 - usedPct);
  const freeGb = Math.max(0, denom - totalEstGb).toFixed(2);

  // Confidence mapping
  const confidence = breakdown.estimation_confidence || "medium";
  const sourceLabels = {
    config_explicit: "Config metadata (params count)",
    model_name: "Extracted from model name",
    config_intermediate: "Config analytical (intermediate_size)",
    config_architecture: "Architecture heuristic (12×L×H²)",
    disk_size_fallback: "Disk size fallback estimate",
  };
  const sourceLabel =
    sourceLabels[breakdown.estimation_source] ||
    breakdown.estimation_source ||
    "—";

  setRunEstimateState(state, statusText, {
    breakdown: true,
    segments: {
      weightsPct,
      kvPct,
      overheadPct,
      freePct,
      weightsGb: weightsGb.toFixed(2),
      kvGb: kvGb.toFixed(2),
      overheadGb: overheadGb.toFixed(2),
      freeGb,
    },
    metrics: {
      weightsGb: weightsGb.toFixed(2),
      kvGb: kvGb.toFixed(2),
      overheadGb: overheadGb.toFixed(2),
      totalGb: totalEstGb.toFixed(2),
      availableGb: availableGb.toFixed(2),
      maxSeqs:
        typeof estimatedMaxSeqs === "number" ? String(estimatedMaxSeqs) : "n/a",
    },
    confidence,
    confidenceLabel: `${confidence} confidence — ${sourceLabel}`,
    footerText: `buffer=${result.buffer_gb} GiB | ratio=${formatEstimateRatio(result.required_ratio)}/${formatEstimateRatio(result.max_available_ratio)} | quantization=${result.quantization ?? "auto"}`,
  });
}

async function refreshRunEstimate() {
  if (!runForm) return;

  const payload = getFormPayload(runForm);
  if (!payload.model_id || String(payload.model_id).trim() === "") {
    setRunLaunchBlocked(true, "Model ID is required.");
    setRunEstimateState("warn", "Model ID required", {
      footerText:
        "Enter a model id to compute analytical capacity and VRAM estimate.",
    });
    return;
  }

  const loadingMsg =
    String(payload.device || "gpu").toLowerCase() === "cpu"
      ? "CPU mode selected: checking runtime settings without VRAM constraints."
      : "Evaluating weights, KV cache, and runtime overhead on selected GPU.";
  setRunEstimateState("loading", "Computing estimate…", {
    footerText: loadingMsg,
  });

  try {
    const result = await api("/api/run/estimate", {
      method: "POST",
      body: JSON.stringify(payload),
    });
    renderRunEstimate(result);
  } catch (error) {
    setRunLaunchBlocked(false);
    setRunEstimateState("error", "VRAM estimate failed", {
      footerText: error.message || "Unable to compute run estimate.",
    });
  }
}

function setUsageBar(valueElement, barElement, percent, suffixText = "") {
  if (!valueElement || !barElement) return;

  if (typeof percent !== "number" || Number.isNaN(percent)) {
    valueElement.textContent = "-";
    barElement.style.width = "0%";
    return;
  }

  const clamped = Math.max(0, Math.min(100, percent));
  valueElement.textContent = `${clamped.toFixed(1)}%${suffixText}`;
  barElement.style.width = `${clamped.toFixed(1)}%`;
}

function renderSystemUsage(system) {
  if (!systemSummary) return;

  if (!system || system.available !== true) {
    systemSummary.textContent = "CPU/RAM metrics unavailable.";
    setUsageBar(cpuUsageValue, cpuUsageBar, Number.NaN);
    setUsageBar(ramUsageValue, ramUsageBar, Number.NaN);
    renderCpuModelBreakdown([], Number.NaN, Number.NaN, Number.NaN);
    return;
  }

  const cpuPercent =
    typeof system.cpu_percent === "number" ? system.cpu_percent : Number.NaN;
  const ramPercent =
    typeof system.ram_percent === "number" ? system.ram_percent : Number.NaN;

  const cpuCount = Number(system.cpu_count || 0);
  const cpuName =
    typeof system.cpu_name === "string" && system.cpu_name.trim() !== ""
      ? system.cpu_name.trim()
      : "CPU model unavailable";
  const cpuLabel =
    Number.isFinite(cpuCount) && cpuCount > 0
      ? `${cpuCount} logical cores`
      : "core count unavailable";

  const ramUsed = Number(system.ram_used_bytes || 0);
  const ramTotal = Number(system.ram_total_bytes || 0);
  const ramLabel =
    ramTotal > 0
      ? `${formatGiB(ramUsed)} / ${formatGiB(ramTotal)} GiB`
      : "RAM size unavailable";

  systemSummary.textContent = `${cpuName} • ${cpuLabel} • ${ramLabel}`;
  setUsageBar(cpuUsageValue, cpuUsageBar, cpuPercent);
  setUsageBar(ramUsageValue, ramUsageBar, ramPercent, ` (${ramLabel})`);

  renderCpuModelBreakdown(
    Array.isArray(system.cpu_models) ? system.cpu_models : [],
    cpuCount,
    ramTotal,
    ramUsed,
  );
}

function renderCpuModelBreakdown(
  models,
  cpuCount,
  hostRamTotalBytes,
  hostRamUsedBytes,
) {
  if (!cpuModelCpuStack || !cpuModelRamStack || !cpuModelList) return;

  cpuModelCpuStack.innerHTML = "";
  cpuModelRamStack.innerHTML = "";
  cpuModelList.innerHTML = "";

  const validCpuCount =
    Number.isFinite(cpuCount) && cpuCount > 0 ? cpuCount : 1;
  const validHostRamTotal =
    Number.isFinite(hostRamTotalBytes) && hostRamTotalBytes > 0
      ? hostRamTotalBytes
      : 0;
  const validHostRamUsed =
    Number.isFinite(hostRamUsedBytes) && hostRamUsedBytes > 0
      ? hostRamUsedBytes
      : 0;

  const rows = (Array.isArray(models) ? models : []).map((item, index) => {
    const cpuPercentRaw =
      typeof item.cpu_percent === "number" && !Number.isNaN(item.cpu_percent)
        ? item.cpu_percent
        : 0;
    const hostCpuShare = Math.max(0, cpuPercentRaw / validCpuCount);

    const ramBytes =
      typeof item.ram_used_bytes === "number" &&
      !Number.isNaN(item.ram_used_bytes)
        ? Math.max(0, item.ram_used_bytes)
        : 0;
    const hostRamShare =
      validHostRamTotal > 0
        ? Math.max(0, (ramBytes / validHostRamTotal) * 100)
        : 0;

    const instanceId = getInstanceId(item);
    const displayName = `${String(item.model_id || "Unknown model")} #${instanceId}`;
    const hue = (196 + index * 37) % 360;

    return {
      displayName,
      cpuPercentRaw,
      hostCpuShare,
      ramBytes,
      hostRamShare,
      color: `hsl(${hue} 70% 56%)`,
    };
  });

  if (!rows.length) {
    cpuModelList.innerHTML =
      '<div class="empty-state">No CPU-backed model running.</div>';
    return;
  }

  rows.forEach((row) => {
    const cpuSegment = document.createElement("span");
    cpuSegment.className = "gpu-segment gpu-segment-model";
    cpuSegment.style.width = `${Math.max(0, row.hostCpuShare).toFixed(3)}%`;
    cpuSegment.style.setProperty("--seg-color", row.color);
    cpuSegment.dataset.tooltip = `${row.displayName} - ${row.cpuPercentRaw.toFixed(1)}% container CPU (~${row.hostCpuShare.toFixed(1)}% host)`;
    cpuSegment.setAttribute(
      "aria-label",
      `${row.displayName} CPU usage ${row.cpuPercentRaw.toFixed(1)}% container`,
    );
    cpuModelCpuStack.appendChild(cpuSegment);

    const ramSegment = document.createElement("span");
    ramSegment.className = "gpu-segment gpu-segment-model";
    ramSegment.style.width = `${Math.max(0, row.hostRamShare).toFixed(3)}%`;
    ramSegment.style.setProperty("--seg-color", row.color);
    ramSegment.dataset.tooltip = `${row.displayName} - ${formatGiB(row.ramBytes)} GiB RAM (${row.hostRamShare.toFixed(1)}% host)`;
    ramSegment.setAttribute(
      "aria-label",
      `${row.displayName} RAM usage ${formatGiB(row.ramBytes)} GiB`,
    );
    cpuModelRamStack.appendChild(ramSegment);

    const rowEl = document.createElement("div");
    rowEl.className = "gpu-model-item";
    rowEl.innerHTML = `
      <span class="gpu-model-name">
        <span class="legend-swatch" style="--seg-color:${row.color};"></span>
        ${escapeHtml(row.displayName)}
      </span>
      <span>${row.cpuPercentRaw.toFixed(1)}% CPU • ${formatGiB(row.ramBytes)} GiB RAM</span>
    `;
    cpuModelList.appendChild(rowEl);
  });

  const totalCpuHostShare = rows.reduce(
    (acc, row) => acc + row.hostCpuShare,
    0,
  );
  const totalRamBytes = rows.reduce((acc, row) => acc + row.ramBytes, 0);
  const residualCpuShare = Math.max(0, 100 - totalCpuHostShare);
  const residualRamShare =
    validHostRamTotal > 0
      ? Math.max(
          0,
          (Math.max(0, validHostRamUsed - totalRamBytes) / validHostRamTotal) *
            100,
        )
      : 0;

  if (residualCpuShare > 0.1) {
    const residual = document.createElement("span");
    residual.className = "gpu-segment gpu-segment-system";
    residual.style.width = `${residualCpuShare.toFixed(3)}%`;
    residual.dataset.tooltip = `Other host CPU usage - ${residualCpuShare.toFixed(1)}%`;
    residual.setAttribute(
      "aria-label",
      `Other host CPU usage ${residualCpuShare.toFixed(1)}%`,
    );
    cpuModelCpuStack.appendChild(residual);
  }

  if (residualRamShare > 0.1) {
    const residual = document.createElement("span");
    residual.className = "gpu-segment gpu-segment-system";
    residual.style.width = `${residualRamShare.toFixed(3)}%`;
    residual.dataset.tooltip = `Other host RAM usage - ${residualRamShare.toFixed(1)}%`;
    residual.setAttribute(
      "aria-label",
      `Other host RAM usage ${residualRamShare.toFixed(1)}%`,
    );
    cpuModelRamStack.appendChild(residual);
  }
}

function syncRunDeviceFields() {
  const selectedDevice = String(runDevice?.value || "gpu").toLowerCase();
  const isCpu = selectedDevice === "cpu";

  if (runGpuField) {
    runGpuField.classList.toggle("is-disabled", isCpu);
  }
  if (runBufferField) {
    runBufferField.classList.toggle("is-disabled", isCpu);
  }
  if (runGpuUtilField) {
    runGpuUtilField.classList.toggle("is-disabled", isCpu);
  }
  if (runCpuUtilField) {
    runCpuUtilField.classList.toggle("is-disabled", !isCpu);
  }
  if (runCpuKvCacheField) {
    runCpuKvCacheField.classList.toggle("is-disabled", !isCpu);
  }
  if (runGpuInput) {
    runGpuInput.disabled = isCpu;
  }
  if (runBufferInput) {
    runBufferInput.disabled = isCpu;
  }
  if (runGpuUtilInput) {
    runGpuUtilInput.disabled = isCpu;
    if (isCpu) {
      runGpuUtilInput.value = "";
    }
  }
  if (runCpuUtilInput) {
    runCpuUtilInput.disabled = !isCpu;
    if (!isCpu) {
      runCpuUtilInput.value = "";
    }
  }
  if (runCpuKvCacheInput) {
    runCpuKvCacheInput.disabled = !isCpu;
    if (!isCpu) {
      runCpuKvCacheInput.value = "";
    }
  }
}

function scheduleRunEstimate() {
  if (runEstimateTimer !== null) {
    window.clearTimeout(runEstimateTimer);
  }
  runEstimateTimer = window.setTimeout(() => {
    runEstimateTimer = null;
    refreshRunEstimate();
  }, 360);
}

function makeButton(label, className, onClick) {
  const button = document.createElement("button");
  button.type = "button";
  button.textContent = label;
  button.className = className;
  button.addEventListener("click", onClick);
  return button;
}

function makeCell(label, value) {
  const td = document.createElement("td");
  td.dataset.label = label;
  td.textContent = value ?? "-";
  return td;
}

function makeStatusCell(label, statusValue) {
  const td = document.createElement("td");
  const normalized = String(statusValue || "unknown").toLowerCase();
  td.dataset.label = label;
  const chip = document.createElement("span");
  chip.className = `status-chip status-${normalized}`;
  chip.textContent = String(statusValue || "unknown");
  td.appendChild(chip);
  return td;
}

function createEmptyRow(message, colspan) {
  const tr = document.createElement("tr");
  const td = document.createElement("td");
  td.colSpan = colspan;
  const div = document.createElement("div");
  div.className = "empty-state";
  div.textContent = message;
  td.appendChild(div);
  tr.appendChild(td);
  return tr;
}

function renderInferenceTargets(items) {
  if (!inferTarget || !inferEndpointHint) return;

  const runningContainers = (Array.isArray(items) ? items : []).filter(
    (item) =>
      String(item.status || "").toLowerCase() === "running" && item.host_port,
  );

  inferTarget.innerHTML = "";

  if (!runningContainers.length) {
    const option = document.createElement("option");
    option.value = "";
    option.textContent = "No running model available";
    inferTarget.appendChild(option);
    inferTarget.disabled = true;
    inferEndpointHint.textContent =
      "Start a model first to enable inference from the Web UI.";
    return;
  }

  runningContainers.forEach((container, index) => {
    const instanceId = getInstanceId(container);
    const option = document.createElement("option");
    option.value = container.name;
    option.textContent = `${container.model_id} #${instanceId} • localhost:${container.host_port}`;
    option.dataset.modelId = container.model_id;
    option.dataset.baseUrl = `http://localhost:${container.host_port}`;
    option.dataset.instanceId = instanceId;
    if (index === 0) option.selected = true;
    inferTarget.appendChild(option);
  });

  inferTarget.disabled = false;
  updateInferenceHint();
}

function updateInferenceHint() {
  if (!inferTarget || !inferEndpointHint) return;

  const selectedOption = inferTarget.selectedOptions[0];
  if (!selectedOption || !selectedOption.dataset.baseUrl) {
    inferEndpointHint.textContent =
      "Endpoint will be filled automatically from the selected container.";
    return;
  }

  inferEndpointHint.textContent = `${selectedOption.dataset.modelId} #${selectedOption.dataset.instanceId || "-"} via ${selectedOption.dataset.baseUrl}`;
}

function openLogsModal(source) {
  selectedLogSource = source;
  lastLogsErrorKey = null;
  logsTarget.textContent = source.label;
  logsOutput.textContent = "Loading logs...";
  logsModal.classList.add("open");
  logsModal.setAttribute("aria-hidden", "false");
  notify(`Opened ${source.label}.`, "info", "Logs");
  refreshLogs();

  if (logsInterval) clearInterval(logsInterval);
  logsInterval = setInterval(refreshLogs, 2500);
}

function closeLogsModal() {
  logsModal.classList.remove("open");
  logsModal.setAttribute("aria-hidden", "true");
  selectedLogSource = null;
  if (logsInterval) {
    clearInterval(logsInterval);
    logsInterval = null;
  }
}

async function copyLogsToClipboard() {
  const text = String(logsOutput?.textContent || "").trim();
  if (!text) {
    notify("No logs available to copy.", "warning", "Copy logs");
    setStatus("No logs available to copy.", "error");
    return;
  }

  try {
    if (navigator.clipboard?.writeText) {
      await navigator.clipboard.writeText(text);
    } else {
      const range = document.createRange();
      range.selectNodeContents(logsOutput);
      const selection = window.getSelection();
      selection?.removeAllRanges();
      selection?.addRange(range);
      const copied = document.execCommand("copy");
      selection?.removeAllRanges();
      if (!copied) {
        throw new Error("Clipboard copy command failed.");
      }
    }

    notify("Logs copied to clipboard.", "success", "Copy logs");
    setStatus("Logs copied to clipboard.", "ok");
  } catch (error) {
    notify("Unable to copy logs to clipboard.", "error", "Copy logs", 6500);
    setStatus(
      `Unable to copy logs: ${error.message || "unknown error"}`,
      "error",
    );
  }
}

function compactProgressLogs(rawLogs) {
  const source = String(rawLogs || "");
  if (!source.trim()) return source;

  const lines = source.split(/\r?\n/);
  const compacted = [];
  const progressIndexByKey = new Map();

  let activeDockerImage = "default";
  const dockerPullStartRe =
    /^\[(?:stdout|stderr)\]\s*\[docker\]\s+Pulling image:\s+(.+)$/i;
  const dockerImageReadyRe =
    /^\[(?:stdout|stderr)\]\s*\[docker\]\s+Image ready:\s+(.+)$/i;
  const dockerPullProgressRe =
    /^\[(?:stdout|stderr)\]\s*\[docker\]\s+Pull progress\s+/i;
  const hfProgressRe =
    /^\[(?:stdout|stderr)\]\s*\[startup\]\s+Hugging Face download\s+/i;

  for (const rawLine of lines) {
    const line = String(rawLine || "");
    if (!line.trim()) {
      compacted.push(line);
      continue;
    }

    const pullStartMatch = line.match(dockerPullStartRe);
    if (pullStartMatch) {
      activeDockerImage = pullStartMatch[1].trim() || "default";
      progressIndexByKey.delete(`docker:${activeDockerImage}`);
      compacted.push(line);
      continue;
    }

    const imageReadyMatch = line.match(dockerImageReadyRe);
    if (imageReadyMatch) {
      const imageName = imageReadyMatch[1].trim() || activeDockerImage;
      progressIndexByKey.delete(`docker:${imageName}`);
      compacted.push(line);
      continue;
    }

    if (dockerPullProgressRe.test(line)) {
      const key = `docker:${activeDockerImage}`;
      const previousIndex = progressIndexByKey.get(key);
      if (typeof previousIndex === "number") {
        compacted[previousIndex] = line;
      } else {
        progressIndexByKey.set(key, compacted.length);
        compacted.push(line);
      }
      continue;
    }

    if (hfProgressRe.test(line)) {
      const key = "hf:download";
      const previousIndex = progressIndexByKey.get(key);
      if (typeof previousIndex === "number") {
        compacted[previousIndex] = line;
      } else {
        progressIndexByKey.set(key, compacted.length);
        compacted.push(line);
      }
      continue;
    }

    compacted.push(line);
  }

  return compacted.join("\n");
}

async function refreshLogs() {
  if (!selectedLogSource) return;
  try {
    if (selectedLogSource.type === "task") {
      const data = await api(
        `/api/run/tasks/${encodeURIComponent(selectedLogSource.id)}/logs`,
      );
      logsOutput.textContent = compactProgressLogs(
        data.logs || "No task logs available.",
      );
      lastLogsErrorKey = null;
      setStatus(`Viewing ${selectedLogSource.label}.`, "ok");
      return;
    }

    const data = await api(
      `/api/logs/${encodeURIComponent(selectedLogSource.id)}?tail=500`,
    );
    logsOutput.textContent = compactProgressLogs(
      data.logs || "No container logs available.",
    );
    lastLogsErrorKey = null;
    setStatus(`Viewing ${selectedLogSource.label}.`, "ok");
  } catch (error) {
    logsOutput.textContent = `Failed to load logs: ${error.message}`;
    const errorKey = `${selectedLogSource?.type || "logs"}:${selectedLogSource?.id || "unknown"}:${error.message}`;
    if (lastLogsErrorKey !== errorKey) {
      notify(
        `Unable to load logs: ${error.message}`,
        "error",
        "Logs failed",
        6500,
      );
      lastLogsErrorKey = errorKey;
    }
    setStatus(`Failed to load logs: ${error.message}`, "error");
  }
}

function renderGpu(items) {
  gpuGrid.innerHTML = "";

  if (!Array.isArray(items) || items.length === 0) {
    gpuSummary.textContent = "No GPU detected by NVML.";
    const card = document.createElement("article");
    card.className = "gpu-card";
    card.innerHTML = '<div class="empty-state">No GPU data available.</div>';
    gpuGrid.appendChild(card);
    return;
  }

  let usedTotal = 0;
  let totalTotal = 0;

  items.forEach((gpu) => {
    const usedBytes = Number(gpu.used_bytes || 0);
    const totalBytes = Number(gpu.total_bytes || 0);
    const freeBytes = Number(gpu.free_bytes || 0);
    const ratio = totalBytes > 0 ? (usedBytes / totalBytes) * 100 : 0;

    usedTotal += usedBytes;
    totalTotal += totalBytes;

    const card = document.createElement("article");
    card.className = "gpu-card";

    const modelEntries = Array.isArray(gpu.models)
      ? gpu.models
          .map((model, index) => {
            const modelBytes = Number(model.used_bytes || 0);
            const modelRatio =
              totalBytes > 0 ? Math.max(0, (modelBytes / totalBytes) * 100) : 0;
            const hue = (206 + index * 47) % 360;
            const instanceId = getInstanceId(model);
            const displayName = `${String(model.model_id || "Unknown model")} #${instanceId}`;
            return {
              modelId: String(model.model_id || "Unknown model"),
              displayName,
              bytes: modelBytes,
              ratio: modelRatio,
              color: `hsl(${hue} 72% 56%)`,
            };
          })
          .filter((entry) => entry.bytes > 0)
      : [];

    const modelUsedBytes = modelEntries.reduce(
      (acc, entry) => acc + entry.bytes,
      0,
    );
    const residualUsedBytes = Math.max(0, usedBytes - modelUsedBytes);
    const residualRatio =
      totalBytes > 0 ? (residualUsedBytes / totalBytes) * 100 : 0;
    const freeRatio = totalBytes > 0 ? Math.max(0, 100 - ratio) : 0;

    const segmentParts = [];
    modelEntries.forEach((entry) => {
      segmentParts.push(`
        <span
          class="gpu-segment gpu-segment-model"
          style="width:${entry.ratio.toFixed(3)}%;--seg-color:${entry.color};"
          data-tooltip="${escapeHtml(`${entry.displayName} - ${formatGiB(entry.bytes)} GiB VRAM`)}"
          aria-label="${escapeHtml(`${entry.displayName} uses ${formatGiB(entry.bytes)} GiB VRAM`)}"
        ></span>
      `);
    });

    if (residualRatio > 0.05) {
      segmentParts.push(`
        <span
          class="gpu-segment gpu-segment-system"
          style="width:${residualRatio.toFixed(3)}%;"
          data-tooltip="System or unmanaged usage - ${formatGiB(residualUsedBytes)} GiB VRAM"
          aria-label="System or unmanaged usage: ${formatGiB(residualUsedBytes)} GiB VRAM"
        ></span>
      `);
    }

    if (freeRatio > 0.05) {
      segmentParts.push(`
        <span
          class="gpu-segment gpu-segment-free"
          style="width:${freeRatio.toFixed(3)}%;"
          data-tooltip="Free VRAM - ${formatGiB(freeBytes)} GiB"
          aria-label="Free VRAM: ${formatGiB(freeBytes)} GiB"
        ></span>
      `);
    }

    const modelsMarkup = modelEntries.length
      ? [
          ...modelEntries.map(
            (entry) => `
            <div class="gpu-model-item">
              <span class="gpu-model-name">
                <span class="legend-swatch" style="--seg-color:${entry.color};"></span>
                ${escapeHtml(entry.displayName)}
              </span>
              <span>${formatGiB(entry.bytes)} GiB</span>
            </div>`,
          ),
          residualRatio > 0.05
            ? `<div class="gpu-model-item">
                <span class="gpu-model-name">
                  <span class="legend-swatch legend-swatch-system"></span>
                  System / unmanaged
                </span>
                <span>${formatGiB(residualUsedBytes)} GiB</span>
              </div>`
            : "",
        ].join("")
      : '<div class="empty-state">No running model currently mapped to this GPU.</div>';

    card.innerHTML = `
      <div class="gpu-title">
        <div class="gpu-name">GPU ${gpu.gpu_index} • ${gpu.gpu_name || "Unknown GPU"}</div>
        <span class="badge">${ratio.toFixed(1)}% used</span>
      </div>
      <div class="gpu-meta">
        <span>${formatGiB(usedBytes)} GiB used</span>
        <span>${formatGiB(freeBytes)} GiB free</span>
        <span>${formatGiB(totalBytes)} GiB total</span>
      </div>
      <div class="progress-track gpu-stack">${segmentParts.join("")}</div>
      <div class="gpu-models">${modelsMarkup}</div>
    `;

    gpuGrid.appendChild(card);
  });

  const globalRatio = totalTotal > 0 ? (usedTotal / totalTotal) * 100 : 0;
  gpuSummary.textContent = `Global VRAM: ${formatGiB(usedTotal)} / ${formatGiB(totalTotal)} GiB (${globalRatio.toFixed(1)}%)`;
}

function renderContainers(items) {
  containersBody.innerHTML = "";

  if (!Array.isArray(items) || items.length === 0) {
    containersBody.appendChild(
      createEmptyRow("No managed containers found.", 8),
    );
    return;
  }

  items.forEach((container) => {
    const instanceId = getInstanceId(container);
    const backendValue = String(
      container.compute_backend || "gpu",
    ).toUpperCase();
    let backendLabel = backendValue;
    if (backendValue === "CPU") {
      const details = [];
      if (typeof container.cpu_utilization === "number") {
        details.push(`cpu=${container.cpu_utilization.toFixed(3)}`);
      }
      const kvSpace = String(container.cpu_kv_cache_space || "").trim();
      if (kvSpace !== "") {
        details.push(`kv=${kvSpace}GiB`);
      }
      if (details.length > 0) {
        backendLabel = `${backendValue} (${details.join(", ")})`;
      }
    }
    const tr = document.createElement("tr");
    tr.appendChild(makeCell("Name", container.name));
    tr.appendChild(makeCell("Model", `${container.model_id} #${instanceId}`));
    tr.appendChild(makeStatusCell("Status", container.status));
    tr.appendChild(makeCell("Port", String(container.host_port ?? "-")));
    tr.appendChild(makeCell("Backend", backendLabel));
    tr.appendChild(makeCell("GPU", String(container.gpu_index ?? "-")));

    const actionsTd = document.createElement("td");
    actionsTd.dataset.label = "Actions";
    const actionRow = document.createElement("div");
    actionRow.className = "action-row";
    actionRow.appendChild(
      makeButton("Stop", "small-warning", async () => {
        try {
          notify(
            `Stopping ${container.model_id} #${instanceId}...`,
            "warning",
            "Stop requested",
          );
          setStatus(`Stopping ${container.model_id} #${instanceId}...`, "ok");
          await api("/api/stop", {
            method: "POST",
            body: JSON.stringify({
              model_id: container.model_id,
              container_name: container.name,
              purge_cache: false,
            }),
          });
          notify(
            `Stopped ${container.model_id} #${instanceId}.`,
            "success",
            "Model stopped",
          );
          setStatus(`Stopped ${container.model_id} #${instanceId}.`, "ok");
          await refreshAll();
        } catch (error) {
          notify(`Stop failed: ${error.message}`, "error", "Stop failed", 6500);
          setStatus(`Stop failed: ${error.message}`, "error");
        }
      }),
    );
    actionRow.appendChild(
      makeButton("Remove cache", "small-danger", async () => {
        try {
          notify(
            `Removing cache for ${container.model_id}...`,
            "warning",
            "Cache removal requested",
          );
          setStatus(`Removing cache for ${container.model_id}...`, "ok");
          await api("/api/rm", {
            method: "POST",
            body: JSON.stringify({ model_id: container.model_id }),
          });
          notify(
            `Removed cache for ${container.model_id}.`,
            "success",
            "Cache removed",
          );
          setStatus(`Removed cache for ${container.model_id}.`, "ok");
          await refreshAll();
        } catch (error) {
          notify(
            `Remove failed: ${error.message}`,
            "error",
            "Cache removal failed",
            6500,
          );
          setStatus(`Remove failed: ${error.message}`, "error");
        }
      }),
    );
    actionsTd.appendChild(actionRow);
    tr.appendChild(actionsTd);

    const logsTd = document.createElement("td");
    logsTd.dataset.label = "Logs";
    const logsRow = document.createElement("div");
    logsRow.className = "action-row";
    logsRow.appendChild(
      makeButton("Container", "small-ghost", () =>
        openLogsModal({
          type: "container",
          id: container.name,
          label: `Container • ${container.name}`,
        }),
      ),
    );
    logsTd.appendChild(logsRow);
    tr.appendChild(logsTd);

    containersBody.appendChild(tr);
  });
}

function renderTasks(items) {
  tasksBody.innerHTML = "";

  if (!Array.isArray(items) || items.length === 0) {
    tasksBody.appendChild(createEmptyRow("No launch task recorded yet.", 6));
    return;
  }

  items.forEach((task) => {
    const tr = document.createElement("tr");
    tr.appendChild(makeCell("Model", task.model_id));
    tr.appendChild(makeStatusCell("Status", task.status));
    tr.appendChild(makeCell("Container", task.container_name || "-"));
    tr.appendChild(makeCell("Message", task.message || "-"));
    tr.appendChild(makeCell("Started", formatDate(task.started_at)));

    const logsTd = document.createElement("td");
    logsTd.dataset.label = "Logs";
    const logsRow = document.createElement("div");
    logsRow.className = "action-row";
    logsRow.appendChild(
      makeButton("Task", "small-ghost", () =>
        openLogsModal({
          type: "task",
          id: task.id,
          label: `Task • ${task.model_id}`,
        }),
      ),
    );
    if (task.container_name) {
      logsRow.appendChild(
        makeButton("Container", "small-ghost", () =>
          openLogsModal({
            type: "container",
            id: task.container_name,
            label: `Container • ${task.container_name}`,
          }),
        ),
      );
    }
    logsTd.appendChild(logsRow);
    tr.appendChild(logsTd);

    tasksBody.appendChild(tr);
  });
}

function renderCache(items) {
  cacheBody.innerHTML = "";

  if (!Array.isArray(items) || items.length === 0) {
    cacheBody.appendChild(createEmptyRow("No cached model found.", 4));
    return;
  }

  items.forEach((item) => {
    const tr = document.createElement("tr");
    tr.appendChild(makeCell("Model", item.model_id));
    tr.appendChild(
      makeCell("Size (GiB)", Number(item.size_gib || 0).toFixed(2)),
    );
    tr.appendChild(makeCell("Path", item.path));

    const actionTd = document.createElement("td");
    actionTd.dataset.label = "Action";
    actionTd.appendChild(
      makeButton("Delete", "small-danger", async () => {
        try {
          notify(
            `Deleting cache for ${item.model_id}...`,
            "warning",
            "Cache deletion requested",
          );
          setStatus(`Deleting cache for ${item.model_id}...`, "ok");
          await api(`/api/cache/${encodeURIComponent(item.model_id)}`, {
            method: "DELETE",
          });
          notify(
            `Deleted cache for ${item.model_id}.`,
            "success",
            "Cache deleted",
          );
          setStatus(`Deleted cache for ${item.model_id}.`, "ok");
          await refreshAll();
        } catch (error) {
          notify(
            `Delete failed: ${error.message}`,
            "error",
            "Delete failed",
            6500,
          );
          setStatus(`Delete failed: ${error.message}`, "error");
        }
      }),
    );
    tr.appendChild(actionTd);

    cacheBody.appendChild(tr);
  });
}

async function refreshAll(options = {}) {
  const { notifyOnFailure = false } = options;
  try {
    const [health, containers, tasks, cache, gpu, system] = await Promise.all([
      api("/api/health"),
      api("/api/containers"),
      api("/api/run/tasks"),
      api("/api/cache"),
      api("/api/gpu"),
      api("/api/system"),
    ]);

    const containerItems = containers.items || [];
    const taskItems = tasks.items || [];
    const cacheItems = cache.items || [];
    const gpuItems = gpu.items || [];
    const systemMetrics = system || {};

    syncTaskNotifications(taskItems);
    renderContainers(containerItems);
    renderTasks(taskItems);
    renderCache(cacheItems);
    renderGpu(gpuItems);
    renderSystemUsage(systemMetrics);
    renderInferenceTargets(containerItems);
    scheduleRunEstimate();

    statContainers.textContent = String(containerItems.length);
    statRunning.textContent = String(
      containerItems.filter(
        (item) => String(item.status || "").toLowerCase() === "running",
      ).length,
    );
    statCache.textContent = String(cacheItems.length);
    statTasks.textContent = String(taskItems.length);

    statusHealth.textContent =
      health.status === "ok" ? "Service online" : "Service degraded";
    setStatus("Workspace refreshed.", "ok");
    refreshErrorNotified = false;
  } catch (error) {
    statusHealth.textContent = "Service degraded";
    setStatus(`Refresh failed: ${error.message}`, "error");
    if (notifyOnFailure || !refreshErrorNotified) {
      notify(
        `Refresh failed: ${error.message}`,
        "error",
        "Workspace refresh failed",
        6500,
      );
      refreshErrorNotified = true;
    }
  }
}

runForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  try {
    syncRunDeviceFields();
    if (runLaunchBlockedReason) {
      throw new Error(runLaunchBlockedReason);
    }
    const payload = getFormPayload(runForm);
    notify(`Launching ${payload.model_id}...`, "info", "Launch started");
    setStatus(`Launching ${payload.model_id}...`, "ok");
    const data = await api("/api/run", {
      method: "POST",
      body: JSON.stringify(payload),
    });
    const modelId = data?.task?.model_id || payload.model_id;
    rememberTaskState(data?.task);
    notify(`Launch task queued for ${modelId}.`, "success", "Task queued");
    setStatus(`Launch task queued for ${modelId}.`, "ok");
    await refreshAll({ notifyOnFailure: true });
    scheduleRunEstimate();
  } catch (error) {
    notify(`Run failed: ${error.message}`, "error", "Launch failed", 6500);
    setStatus(`Run failed: ${error.message}`, "error");
  }
});

runForm.addEventListener("input", () => {
  syncRunDeviceFields();
  scheduleRunEstimate();
});

runForm.addEventListener("change", () => {
  syncRunDeviceFields();
  scheduleRunEstimate();
});

// Handle image file input for inference
let inferSelectedImages = [];

function setInferenceRunning(isRunning) {
  if (inferSubmitButton) {
    inferSubmitButton.disabled = isRunning;
    inferSubmitButton.setAttribute("aria-disabled", String(isRunning));
  }

  if (inferStopButton) {
    inferStopButton.disabled = !isRunning;
    inferStopButton.setAttribute("aria-disabled", String(!isRunning));
  }
}

async function fileToBase64(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      resolve(reader.result);
    };
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

function renderInferImagePreview() {
  if (!inferImagesList || !inferImagesPreview) {
    return;
  }

  inferImagesList.innerHTML = "";
  inferImagesPreview.hidden = inferSelectedImages.length === 0;

  inferSelectedImages.forEach((imageUrl, index) => {
    const imageButton = document.createElement("button");
    imageButton.type = "button";
    imageButton.className = "infer-image-chip";
    imageButton.title = `Remove image ${index + 1}`;
    imageButton.setAttribute("aria-label", `Remove image ${index + 1}`);

    const image = document.createElement("img");
    image.src = imageUrl;
    image.alt = `Selected image ${index + 1}`;

    const badge = document.createElement("span");
    badge.className = "infer-image-chip-badge";
    badge.textContent = "Remove";

    imageButton.appendChild(image);
    imageButton.appendChild(badge);
    imageButton.addEventListener("click", () => {
      inferSelectedImages = inferSelectedImages.filter(
        (_, itemIndex) => itemIndex !== index,
      );
      if (inferImagesInput) {
        inferImagesInput.value = "";
      }
      renderInferImagePreview();
    });

    inferImagesList.appendChild(imageButton);
  });
}

async function handleInferImageSelection(event) {
  const files = Array.from(event.target.files || []);
  inferSelectedImages = [];

  for (const file of files) {
    try {
      const base64 = await fileToBase64(file);
      inferSelectedImages.push(base64);
    } catch (error) {
      notify(
        `Failed to read image ${file.name}: ${error.message}`,
        "error",
        "Image read error",
      );
    }
  }

  renderInferImagePreview();
}

if (inferImagesInput) {
  inferImagesInput.addEventListener("change", handleInferImageSelection);
}

inferStopButton?.addEventListener("click", () => {
  if (!inferAbortController) {
    return;
  }

  inferAbortController.abort();
  setStatus("Inference stopped by user.", "error");
  notify("Inference stream stopped.", "info", "Inference stopped", 3200);
});

inferForm.addEventListener("submit", async (event) => {
  event.preventDefault();

  if (inferAbortController) {
    inferAbortController.abort();
  }

  inferAbortController = new AbortController();
  setInferenceRunning(true);

  try {
    const payload = getFormPayload(inferForm);
    const selectedOption = inferTarget?.selectedOptions?.[0];
    if (
      !selectedOption ||
      !selectedOption.dataset.modelId ||
      !selectedOption.dataset.baseUrl
    ) {
      throw new Error("No running model is available for inference.");
    }

    payload.model_id = selectedOption.dataset.modelId;
    payload.base_url = selectedOption.dataset.baseUrl;
    payload.images = inferSelectedImages;

    inferOutput.textContent = "";
    inferMetrics.textContent = "Streaming response...";
    notify(
      `Running inference on ${payload.model_id}...`,
      "info",
      "Inference started",
    );
    setStatus(`Running inference on ${payload.model_id}...`, "ok");

    const response = await fetch("/api/infer/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      signal: inferAbortController.signal,
      body: JSON.stringify(payload),
    });

    if (!response.ok || !response.body) {
      let details = "Streaming endpoint unavailable.";
      try {
        const payloadError = await response.json();
        details = payloadError?.detail || details;
      } catch {
        // Keep fallback details.
      }
      throw new Error(details);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder("utf-8");
    let buffer = "";
    let usage = null;
    let elapsedSeconds = null;

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const frames = buffer.split("\n\n");
      buffer = frames.pop() || "";

      for (const frame of frames) {
        const lines = frame
          .split("\n")
          .map((line) => line.trim())
          .filter((line) => line.startsWith("data:"));

        for (const line of lines) {
          const raw = line.slice(5).trim();
          if (!raw) continue;

          let eventPayload = null;
          try {
            eventPayload = JSON.parse(raw);
          } catch {
            continue;
          }

          if (eventPayload.type === "token") {
            inferOutput.textContent += String(eventPayload.text || "");
            continue;
          }

          if (eventPayload.type === "usage") {
            usage = eventPayload;
            continue;
          }

          if (eventPayload.type === "done") {
            const elapsed = Number(eventPayload.elapsed_seconds);
            if (!Number.isNaN(elapsed) && elapsed > 0) {
              elapsedSeconds = elapsed;
            }
            continue;
          }

          if (eventPayload.type === "error") {
            throw new Error(
              String(eventPayload.message || "Streaming inference failed."),
            );
          }
        }
      }
    }

    const completionTokens = Number(usage?.completion_tokens);
    const promptTokens = Number(usage?.prompt_tokens);
    const totalTokens = Number(usage?.total_tokens);
    const hasTokenUsage =
      !Number.isNaN(completionTokens) &&
      !Number.isNaN(promptTokens) &&
      !Number.isNaN(totalTokens);
    const hasElapsed = typeof elapsedSeconds === "number" && elapsedSeconds > 0;
    const tokensPerSecond =
      hasTokenUsage && hasElapsed ? completionTokens / elapsedSeconds : null;

    if (hasTokenUsage && hasElapsed && typeof tokensPerSecond === "number") {
      inferMetrics.textContent = `prompt=${promptTokens} • completion=${completionTokens} • total=${totalTokens} • speed=${tokensPerSecond.toFixed(2)} tok/s • elapsed=${elapsedSeconds.toFixed(2)}s`;
    } else if (hasTokenUsage) {
      inferMetrics.textContent = `prompt=${promptTokens} • completion=${completionTokens} • total=${totalTokens} • speed=n/a`;
    } else {
      inferMetrics.textContent =
        "Token metrics unavailable for this runtime/image.";
    }

    if (!inferOutput.textContent.trim()) {
      inferOutput.textContent = "(Empty response)";
    }

    notify(
      `Inference completed for ${payload.model_id}.`,
      "success",
      "Inference completed",
    );
    setStatus(`Inference completed for ${payload.model_id}.`, "ok");
  } catch (error) {
    if (error?.name === "AbortError") {
      inferMetrics.textContent = "Inference stopped by user.";
      if (!inferOutput.textContent.trim()) {
        inferOutput.textContent = "Inference stopped.";
      }
      return;
    }

    inferOutput.textContent = "Inference failed.";
    inferMetrics.textContent = "No metrics available.";
    notify(
      `Inference failed: ${error.message}`,
      "error",
      "Inference failed",
      6500,
    );
    setStatus(`Inference failed: ${error.message}`, "error");
  } finally {
    inferAbortController = null;
    setInferenceRunning(false);
  }
});

logsClose.addEventListener("click", closeLogsModal);
logsRefresh.addEventListener("click", async () => {
  notify("Refreshing logs...", "info", "Logs");
  await refreshLogs();
});
logsCopy?.addEventListener("click", async () => {
  await copyLogsToClipboard();
});
inferTarget?.addEventListener("change", () => {
  updateInferenceHint();
  const selectedOption = inferTarget.selectedOptions[0];
  if (selectedOption?.dataset?.modelId) {
    notify(
      `Selected ${selectedOption.dataset.modelId} for inference.`,
      "info",
      "Inference target updated",
      2600,
    );
  }
});
logsModal.addEventListener("click", (event) => {
  if (event.target === logsModal) closeLogsModal();
});

document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && logsModal.classList.contains("open")) {
    closeLogsModal();
  }
});

initTheme();
syncRunDeviceFields();
refreshAll();
scheduleRunEstimate();
setInterval(() => refreshAll({ notifyOnFailure: false }), 6000);
