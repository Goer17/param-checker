const keyInput = document.getElementById("keyInput");
const keyOptions = document.getElementById("keyOptions");
const displayBtn = document.getElementById("displayBtn");
const metaBar = document.getElementById("metaBar");
const canvas = document.getElementById("heatmapCanvas");
const canvasWrap = document.getElementById("canvasWrap");
const tooltip = document.getElementById("tooltip");

let keys = [];
let currentMeta = null;
let currentTile = null;
let zoom = 1;
let panX = 0;
let panY = 0;
let isPanning = false;
let panStartX = 0;
let panStartY = 0;
let panOriginX = 0;
let panOriginY = 0;
let tileRequestId = 0;
let inFlightTileSignature = "";
let hoverAbort = null;
let lastHover = { key: null, row: null, col: null };

const MAX_PREVIEW = 256;
const MAX_TILE = 512;

const ctx = canvas.getContext("2d");

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function lerp(a, b, t) {
  return a + (b - a) * t;
}

function buildPalette() {
  const stops = [
    { t: 0.0, c: [20, 32, 42] },
    { t: 0.25, c: [47, 93, 255] },
    { t: 0.5, c: [139, 92, 255] },
    { t: 0.75, c: [255, 127, 80] },
    { t: 1.0, c: [255, 213, 111] },
  ];
  const palette = new Array(256);

  for (let i = 0; i < 256; i += 1) {
    const t = i / 255;
    let left = stops[0];
    let right = stops[stops.length - 1];
    for (let j = 0; j < stops.length - 1; j += 1) {
      if (t >= stops[j].t && t <= stops[j + 1].t) {
        left = stops[j];
        right = stops[j + 1];
        break;
      }
    }
    const section = right.t - left.t || 1;
    const localT = (t - left.t) / section;
    palette[i] = [
      Math.round(lerp(left.c[0], right.c[0], localT)),
      Math.round(lerp(left.c[1], right.c[1], localT)),
      Math.round(lerp(left.c[2], right.c[2], localT)),
    ];
  }
  return palette;
}

const PALETTE = buildPalette();

function getViewSize() {
  return {
    width: canvas.clientWidth,
    height: canvas.clientHeight,
  };
}

function resizeCanvas() {
  const { width, height } = getViewSize();
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.floor(width * dpr));
  canvas.height = Math.max(1, Math.floor(height * dpr));
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  render();
}

function getLayout() {
  if (!currentMeta) {
    return null;
  }
  const rows = currentMeta.view_shape[0];
  const cols = currentMeta.view_shape[1];
  const { width, height } = getViewSize();
  const usableW = Math.max(40, width - 40);
  const usableH = Math.max(40, height - 40);
  const baseCell = clamp(Math.min(usableW / cols, usableH / rows), 0.2, 40);
  const cell = baseCell * zoom;
  const gridWidth = cols * cell;
  const gridHeight = rows * cell;
  const originX = (width - gridWidth) / 2 + panX;
  const originY = (height - gridHeight) / 2 + panY;
  return { rows, cols, baseCell, cell, gridWidth, gridHeight, originX, originY };
}

function tileCovers(tile, desired) {
  if (!tile) {
    return false;
  }
  if (tile.key !== desired.key || tile.stride !== desired.stride) {
    return false;
  }
  const tileR1 = tile.r0 + tile.rows;
  const tileC1 = tile.c0 + tile.cols;
  const desiredR1 = desired.r0 + desired.rows;
  const desiredC1 = desired.c0 + desired.cols;
  return (
    desired.r0 >= tile.r0 &&
    desired.c0 >= tile.c0 &&
    desiredR1 <= tileR1 &&
    desiredC1 <= tileC1
  );
}

function computeDesiredTile(layout) {
  const padding = 8;
  const left = (-layout.originX) / layout.cell;
  const top = (-layout.originY) / layout.cell;
  const right = left + getViewSize().width / layout.cell;
  const bottom = top + getViewSize().height / layout.cell;

  let r0 = Math.floor(top) - padding;
  let c0 = Math.floor(left) - padding;
  let r1 = Math.ceil(bottom) + padding;
  let c1 = Math.ceil(right) + padding;

  r0 = clamp(r0, 0, layout.rows - 1);
  c0 = clamp(c0, 0, layout.cols - 1);
  r1 = clamp(r1, 1, layout.rows);
  c1 = clamp(c1, 1, layout.cols);

  const baseStride = Math.max(
    1,
    Math.ceil(Math.max(layout.rows / MAX_PREVIEW, layout.cols / MAX_PREVIEW))
  );

  const viewportRows = r1 - r0;
  const viewportCols = c1 - c0;
  const strideByViewport = Math.max(
    1,
    Math.ceil(Math.max(viewportRows / MAX_TILE, viewportCols / MAX_TILE))
  );
  const strideByZoom = Math.max(1, Math.floor(baseStride / zoom));
  const stride = Math.max(1, strideByViewport, strideByZoom);

  return {
    key: currentMeta.key,
    r0,
    c0,
    rows: viewportRows,
    cols: viewportCols,
    stride,
  };
}

function buildTileCanvas(tile) {
  const canvasEl = document.createElement("canvas");
  canvasEl.width = tile.tile_cols;
  canvasEl.height = tile.tile_rows;
  const tctx = canvasEl.getContext("2d");
  const imageData = tctx.createImageData(tile.tile_cols, tile.tile_rows);
  const total = tile.tile_rows * tile.tile_cols;
  const buffer = imageData.data;

  for (let i = 0; i < total; i += 1) {
    const value = tile.data[i];
    const [r, g, b] = PALETTE[value];
    const idx = i * 4;
    buffer[idx] = r;
    buffer[idx + 1] = g;
    buffer[idx + 2] = b;
    buffer[idx + 3] = 255;
  }

  tctx.putImageData(imageData, 0, 0);
  tile.canvas = canvasEl;
}

function render() {
  const { width, height } = getViewSize();
  ctx.clearRect(0, 0, width, height);

  if (!currentMeta) {
    return;
  }

  const layout = getLayout();
  if (!layout) {
    return;
  }

  const desired = computeDesiredTile(layout);
  if (!tileCovers(currentTile, desired)) {
    requestTile(desired);
  }

  if (!currentTile || !currentTile.canvas) {
    return;
  }

  const cellSize = layout.baseCell * zoom * currentTile.stride;
  const tileX = layout.originX + currentTile.c0 * layout.baseCell * zoom;
  const tileY = layout.originY + currentTile.r0 * layout.baseCell * zoom;
  const drawW = currentTile.tile_cols * cellSize;
  const drawH = currentTile.tile_rows * cellSize;

  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(currentTile.canvas, tileX, tileY, drawW, drawH);

  if (cellSize >= 12 && currentTile.stride === 1 && layout.rows * layout.cols <= 30000) {
    ctx.strokeStyle = "rgba(255, 255, 255, 0.08)";
    ctx.lineWidth = 1;
    ctx.beginPath();

    for (let col = 0; col <= layout.cols; col += 1) {
      const x = layout.originX + col * layout.baseCell * zoom;
      ctx.moveTo(x, layout.originY);
      ctx.lineTo(x, layout.originY + layout.gridHeight);
    }

    for (let row = 0; row <= layout.rows; row += 1) {
      const y = layout.originY + row * layout.baseCell * zoom;
      ctx.moveTo(layout.originX, y);
      ctx.lineTo(layout.originX + layout.gridWidth, y);
    }

    ctx.stroke();
  }
}

function updateMeta() {
  if (!currentMeta) {
    metaBar.textContent = "No tensor selected.";
    return;
  }
  metaBar.textContent = `${currentMeta.key} | shape ${currentMeta.original_shape.join("x")} | view ${currentMeta.view_shape.join("x")} | ${currentMeta.dtype} | ${currentMeta.numel} elements`;
}

function resetView() {
  zoom = 1;
  panX = 0;
  panY = 0;
}

function setTooltip(event, row, col, valueText) {
  const rect = canvasWrap.getBoundingClientRect();
  tooltip.textContent = `[${row}, ${col}]\n${valueText}`;
  tooltip.style.left = `${event.clientX - rect.left + 12}px`;
  tooltip.style.top = `${event.clientY - rect.top + 12}px`;
  tooltip.classList.remove("hidden");
}

function hideTooltip() {
  tooltip.classList.add("hidden");
}

async function fetchValue(key, row, col, event) {
  if (hoverAbort) {
    hoverAbort.abort();
  }
  hoverAbort = new AbortController();
  const url = `/api/value?key=${encodeURIComponent(key)}&row=${row}&col=${col}`;
  try {
    const response = await fetch(url, { signal: hoverAbort.signal });
    if (!response.ok) {
      return;
    }
    const data = await response.json();
    if (data.key !== key || data.row !== row || data.col !== col) {
      return;
    }
    setTooltip(event, row, col, data.value.toFixed(6));
  } catch (err) {
    if (err.name !== "AbortError") {
      hideTooltip();
    }
  }
}

function updateTooltip(event) {
  if (!currentMeta) {
    hideTooltip();
    return;
  }

  const layout = getLayout();
  if (!layout || layout.cell < 12) {
    hideTooltip();
    return;
  }

  const rect = canvasWrap.getBoundingClientRect();
  const x = event.clientX - rect.left;
  const y = event.clientY - rect.top;

  const col = Math.floor((x - layout.originX) / layout.cell);
  const row = Math.floor((y - layout.originY) / layout.cell);

  if (col < 0 || col >= layout.cols || row < 0 || row >= layout.rows) {
    hideTooltip();
    return;
  }

  if (lastHover.key === currentMeta.key && lastHover.row === row && lastHover.col === col) {
    return;
  }

  lastHover = { key: currentMeta.key, row, col };
  setTooltip(event, row, col, "...");
  fetchValue(currentMeta.key, row, col, event);
}

function zoomAt(canvasX, canvasY, factor) {
  const layout = getLayout();
  if (!layout) {
    return;
  }

  const oldCell = layout.cell;
  const oldOriginX = layout.originX;
  const oldOriginY = layout.originY;
  const wx = (canvasX - oldOriginX) / oldCell;
  const wy = (canvasY - oldOriginY) / oldCell;

  const nextZoom = clamp(zoom * factor, 0.3, 40);
  if (nextZoom === zoom) {
    return;
  }

  zoom = nextZoom;
  const next = getLayout();
  if (!next) {
    return;
  }

  panX += canvasX - (next.originX + wx * next.cell);
  panY += canvasY - (next.originY + wy * next.cell);
}

async function fetchKeys() {
  const response = await fetch("/api/keys");
  if (!response.ok) {
    metaBar.textContent = "Failed to load keys.";
    return;
  }
  const data = await response.json();
  keys = data.keys || [];
  keyOptions.innerHTML = "";

  keys.forEach((key) => {
    const option = document.createElement("option");
    option.value = key;
    keyOptions.appendChild(option);
  });

  if (keys.length > 0) {
    keyInput.value = keys[0];
    metaBar.textContent = `Loaded ${keys.length} keys.`;
  } else {
    metaBar.textContent = "No tensor keys found.";
  }
}

async function requestTile(desired) {
  if (!desired) {
    return;
  }
  const signature = `${desired.key}|${desired.r0}|${desired.c0}|${desired.rows}|${desired.cols}|${desired.stride}`;
  if (signature === inFlightTileSignature) {
    return;
  }
  inFlightTileSignature = signature;
  const requestId = (tileRequestId += 1);
  const url = `/api/tile?key=${encodeURIComponent(desired.key)}&r0=${desired.r0}&c0=${desired.c0}&rows=${desired.rows}&cols=${desired.cols}&stride=${desired.stride}`;
  try {
    const response = await fetch(url);
    if (!response.ok) {
      inFlightTileSignature = "";
      return;
    }
    const payload = await response.json();
    if (requestId !== tileRequestId) {
      return;
    }

    let data = [];
    if (payload.encoding === "base64") {
      const binary = atob(payload.data);
      data = new Uint8Array(binary.length);
      for (let i = 0; i < binary.length; i += 1) {
        data[i] = binary.charCodeAt(i);
      }
    } else if (payload.encoding === "list") {
      data = new Uint8Array(payload.data.flat());
    }

    currentTile = {
      key: payload.key,
      r0: payload.r0,
      c0: payload.c0,
      rows: payload.rows,
      cols: payload.cols,
      stride: payload.stride,
      tile_rows: payload.tile_rows,
      tile_cols: payload.tile_cols,
      data,
      canvas: null,
    };

    buildTileCanvas(currentTile);
    inFlightTileSignature = "";
    render();
  } catch (err) {
    // ignore network errors
    inFlightTileSignature = "";
  }
}

async function displayKey() {
  const key = keyInput.value.trim();
  if (!key) {
    metaBar.textContent = "Enter a tensor key.";
    return;
  }

  metaBar.textContent = "Loading tensor metadata...";
  hideTooltip();
  currentTile = null;
  inFlightTileSignature = "";

  const response = await fetch(`/api/meta?key=${encodeURIComponent(key)}`);
  if (!response.ok) {
    metaBar.textContent = `Key not found: ${key}`;
    return;
  }

  const data = await response.json();
  currentMeta = data.meta;
  resetView();
  updateMeta();
  render();
}

function onWheel(event) {
  event.preventDefault();
  if (!currentMeta) {
    return;
  }
  const rect = canvasWrap.getBoundingClientRect();
  const canvasX = event.clientX - rect.left;
  const canvasY = event.clientY - rect.top;
  const factor = event.deltaY < 0 ? 1.15 : 0.87;
  zoomAt(canvasX, canvasY, factor);
  render();
  updateTooltip(event);
}

function onMouseDown(event) {
  if (!currentMeta) {
    return;
  }
  isPanning = true;
  panStartX = event.clientX;
  panStartY = event.clientY;
  panOriginX = panX;
  panOriginY = panY;
}

function onMouseMove(event) {
  if (isPanning) {
    panX = panOriginX + (event.clientX - panStartX);
    panY = panOriginY + (event.clientY - panStartY);
    render();
  }
  updateTooltip(event);
}

function onMouseUp() {
  isPanning = false;
}

displayBtn.addEventListener("click", displayKey);
keyInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter") {
    displayKey();
  }
});

canvasWrap.addEventListener("wheel", onWheel, { passive: false });
canvasWrap.addEventListener("mousedown", onMouseDown);
canvasWrap.addEventListener("mousemove", onMouseMove);
canvasWrap.addEventListener("mouseleave", () => {
  onMouseUp();
  hideTooltip();
});
window.addEventListener("mouseup", onMouseUp);
window.addEventListener("resize", resizeCanvas);

fetchKeys().then(() => {
  if (keyInput.value) {
    displayKey();
  }
});
resizeCanvas();
