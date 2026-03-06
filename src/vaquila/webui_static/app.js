const toast = document.getElementById("toast");
const notificationCenter = document.getElementById("notification-center");
const statusHealth = document.getElementById("status-health");
const runForm = document.getElementById("run-form");
const inferForm = document.getElementById("infer-form");
const inferOutput = document.getElementById("infer-output");
const inferTarget = document.getElementById("infer-target");
const inferEndpointHint = document.getElementById("infer-endpoint-hint");

const tasksBody = document.getElementById("tasks-body");
const containersBody = document.getElementById("containers-body");
const cacheBody = document.getElementById("cache-body");
const gpuGrid = document.getElementById("gpu-grid");
const gpuSummary = document.getElementById("gpu-summary");

const statContainers = document.getElementById("stat-containers");
const statRunning = document.getElementById("stat-running");
const statCache = document.getElementById("stat-cache");
const statTasks = document.getElementById("stat-tasks");

const logsModal = document.getElementById("logs-modal");
const logsTarget = document.getElementById("logs-target");
const logsOutput = document.getElementById("logs-output");
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

async function refreshLogs() {
  if (!selectedLogSource) return;
  try {
    if (selectedLogSource.type === "task") {
      const data = await api(
        `/api/run/tasks/${encodeURIComponent(selectedLogSource.id)}/logs`,
      );
      logsOutput.textContent = data.logs || "No task logs available.";
      lastLogsErrorKey = null;
      setStatus(`Viewing ${selectedLogSource.label}.`, "ok");
      return;
    }

    const data = await api(
      `/api/logs/${encodeURIComponent(selectedLogSource.id)}?tail=500`,
    );
    logsOutput.textContent = data.logs || "No container logs available.";
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
      createEmptyRow("No managed containers found.", 7),
    );
    return;
  }

  items.forEach((container) => {
    const instanceId = getInstanceId(container);
    const tr = document.createElement("tr");
    tr.appendChild(makeCell("Name", container.name));
    tr.appendChild(makeCell("Model", `${container.model_id} #${instanceId}`));
    tr.appendChild(makeStatusCell("Status", container.status));
    tr.appendChild(makeCell("Port", String(container.host_port ?? "-")));
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
    const [health, containers, tasks, cache, gpu] = await Promise.all([
      api("/api/health"),
      api("/api/containers"),
      api("/api/run/tasks"),
      api("/api/cache"),
      api("/api/gpu"),
    ]);

    const containerItems = containers.items || [];
    const taskItems = tasks.items || [];
    const cacheItems = cache.items || [];
    const gpuItems = gpu.items || [];

    syncTaskNotifications(taskItems);
    renderContainers(containerItems);
    renderTasks(taskItems);
    renderCache(cacheItems);
    renderGpu(gpuItems);
    renderInferenceTargets(containerItems);

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
  } catch (error) {
    notify(`Run failed: ${error.message}`, "error", "Launch failed", 6500);
    setStatus(`Run failed: ${error.message}`, "error");
  }
});

inferForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  inferOutput.textContent = "Running inference...";
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
    notify(
      `Running inference on ${payload.model_id}...`,
      "info",
      "Inference started",
    );
    setStatus(`Running inference on ${payload.model_id}...`, "ok");
    const data = await api("/api/infer", {
      method: "POST",
      body: JSON.stringify(payload),
    });
    inferOutput.textContent = data.response || data.answer || "";
    notify(
      `Inference completed for ${payload.model_id}.`,
      "success",
      "Inference completed",
    );
    setStatus(`Inference completed for ${payload.model_id}.`, "ok");
  } catch (error) {
    inferOutput.textContent = "Inference failed.";
    notify(
      `Inference failed: ${error.message}`,
      "error",
      "Inference failed",
      6500,
    );
    setStatus(`Inference failed: ${error.message}`, "error");
  }
});

logsClose.addEventListener("click", closeLogsModal);
logsRefresh.addEventListener("click", async () => {
  notify("Refreshing logs...", "info", "Logs");
  await refreshLogs();
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
refreshAll();
setInterval(() => refreshAll({ notifyOnFailure: false }), 6000);
