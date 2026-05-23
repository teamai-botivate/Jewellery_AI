/* ============================================================
   Jewellery Visual Search — Frontend Logic
   ============================================================ */

const API_BASE = window.location.origin;

// ── DOM refs ────────────────────────────────────────────────
const dropZone       = document.getElementById("dropZone");
const fileInput      = document.getElementById("fileInput");
const previewSection = document.getElementById("previewSection");
const previewImg     = document.getElementById("previewImg");
const searchBtn      = document.getElementById("searchBtn");
const clearBtn       = document.getElementById("clearBtn");
const loadingWrap    = document.getElementById("loadingWrap");
const errorBanner    = document.getElementById("errorBanner");
const errorMsg       = document.getElementById("errorMsg");
const resultsSection = document.getElementById("resultsSection");
const resultsGrid    = document.getElementById("resultsGrid");
const resultsCount   = document.getElementById("resultsCount");
const emptyState     = document.getElementById("emptyState");
const topkRow        = document.getElementById("topkRow");
const topkSlider     = document.getElementById("topkSlider");
const topkVal        = document.getElementById("topkVal");

let selectedFile = null;

// ── Top-K slider ─────────────────────────────────────────────
topkSlider.addEventListener("input", () => {
  topkVal.textContent = topkSlider.value;
});

// ── Drag-and-drop ─────────────────────────────────────────────
dropZone.addEventListener("dragenter", (e) => { e.preventDefault(); dropZone.classList.add("drag-over"); });
dropZone.addEventListener("dragover",  (e) => { e.preventDefault(); });
dropZone.addEventListener("dragleave", () => { dropZone.classList.remove("drag-over"); });
dropZone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropZone.classList.remove("drag-over");
  const file = e.dataTransfer.files[0];
  if (file) handleFile(file);
});

dropZone.addEventListener("click", (e) => {
  // The <label for="fileInput"> already opens the picker natively.
  // Without this guard, clicking the label fires the picker twice.
  if (e.target.closest("label")) return;
  fileInput.click();
});

fileInput.addEventListener("change", () => {
  if (fileInput.files[0]) handleFile(fileInput.files[0]);
});

// ── Paste from clipboard ──────────────────────────────────────
document.addEventListener("paste", (e) => {
  const items = e.clipboardData?.items;
  if (!items) return;
  for (const item of items) {
    if (item.type.startsWith("image/")) {
      const file = item.getAsFile();
      if (file) handleFile(file);
      break;
    }
  }
});

// ── Buttons ───────────────────────────────────────────────────
searchBtn.addEventListener("click", () => {
  if (selectedFile) runSearch(selectedFile);
});

clearBtn.addEventListener("click", clearAll);

// ── File handling ─────────────────────────────────────────────
function handleFile(file) {
  if (!file.type.startsWith("image/")) {
    showError("Please select a valid image file (JPG, PNG, WEBP, etc.)");
    return;
  }

  selectedFile = file;
  hideError();
  hideResults();

  const reader = new FileReader();
  reader.onload = (e) => {
    previewImg.src = e.target.result;
    previewSection.style.display = "block";
    topkRow.style.display = "flex";
  };
  reader.readAsDataURL(file);
}

// ── Search ────────────────────────────────────────────────────
async function runSearch(file) {
  if (!file) return;

  hideError();
  hideResults();
  showLoading();
  searchBtn.disabled = true;

  const formData = new FormData();
  formData.append("file", file);

  const topK = parseInt(topkSlider.value, 10);
  const url  = `${API_BASE}/search?top_k=${topK}`;

  try {
    const response = await fetch(url, { method: "POST", body: formData });

    if (!response.ok) {
      let detail = `Server error ${response.status}`;
      try {
        const data = await response.json();
        detail = data.detail || detail;
      } catch (_) {}
      throw new Error(detail);
    }

    const results = await response.json();
    hideLoading();

    if (results.length === 0) {
      emptyState.style.display = "flex";
    } else {
      renderResults(results);
    }
  } catch (err) {
    hideLoading();
    if (err.message.includes("Failed to fetch")) {
      showError("Cannot reach the backend. Make sure the FastAPI server is running at http://localhost:8000");
    } else {
      showError(err.message);
    }
  } finally {
    searchBtn.disabled = false;
  }
}

// ── Render results ────────────────────────────────────────────
function renderResults(results) {
  resultsGrid.innerHTML = "";

  results.forEach((item, index) => {
    const pct         = Math.round(item.score * 100);
    const scoreClass  = pct >= 75 ? "high" : pct >= 50 ? "medium" : "low";
    const barWidth    = Math.max(4, Math.min(100, pct));
    const rank        = index + 1;

    const card = document.createElement("div");
    card.className = "result-card";
    card.style.animationDelay = `${Math.min(index * 0.04, 0.4)}s`;

    card.innerHTML = `
      <div class="result-img-wrap">
        <img
          class="result-img"
          src="${escapeHtml(item.image_url)}"
          alt="Similar jewellery #${rank}"
          loading="lazy"
          onerror="this.closest('.result-card').style.display='none'"
        />
        <div class="result-overlay">
          <span class="result-overlay-text">View full size</span>
        </div>
      </div>
      <div class="score-bar-wrap">
        <div class="score-bar" style="width:${barWidth}%"></div>
      </div>
      <div class="result-info">
        <span class="result-rank">#${rank}</span>
        <span class="score-pill ${scoreClass}">${pct}%</span>
      </div>
    `;

    // Open full image on click
    card.addEventListener("click", () => openLightbox(item.image_url, rank, pct));

    resultsGrid.appendChild(card);
  });

  resultsCount.textContent = `${results.length} result${results.length !== 1 ? "s" : ""} found`;
  resultsSection.style.display = "flex";
}

// ── Lightbox ──────────────────────────────────────────────────
function openLightbox(url, rank, pct) {
  const overlay = document.createElement("div");
  overlay.style.cssText = `
    position:fixed;inset:0;z-index:1000;
    background:rgba(0,0,0,.88);
    display:flex;flex-direction:column;
    align-items:center;justify-content:center;
    gap:1rem;padding:2rem;cursor:zoom-out;
    animation:fadeIn .2s ease;
  `;

  const style = document.createElement("style");
  style.textContent = "@keyframes fadeIn{from{opacity:0}to{opacity:1}}";
  document.head.appendChild(style);

  const img = document.createElement("img");
  img.src = url;
  img.style.cssText = `
    max-width:90vw;max-height:80vh;
    object-fit:contain;border-radius:12px;
    box-shadow:0 20px 80px rgba(0,0,0,.8);
  `;

  const caption = document.createElement("p");
  caption.style.cssText = "color:#c8a96e;font-size:.9rem;";
  caption.textContent = `Result #${rank}  •  ${pct}% similarity`;

  overlay.appendChild(img);
  overlay.appendChild(caption);
  overlay.addEventListener("click", () => {
    document.body.removeChild(overlay);
    document.head.removeChild(style);
  });

  document.body.appendChild(overlay);
}

// ── Utilities ─────────────────────────────────────────────────
function showLoading()  { loadingWrap.style.display = "flex"; }
function hideLoading()  { loadingWrap.style.display = "none"; }

function showError(msg) {
  errorMsg.textContent = msg;
  errorBanner.style.display = "flex";
}
function hideError() { errorBanner.style.display = "none"; }

function hideResults() {
  resultsSection.style.display = "none";
  emptyState.style.display     = "none";
  resultsGrid.innerHTML        = "";
}

function clearAll() {
  selectedFile = null;
  fileInput.value = "";
  previewSection.style.display = "none";
  topkRow.style.display        = "none";
  hideLoading();
  hideError();
  hideResults();
}

function escapeHtml(str) {
  return str
    .replace(/&/g, "&amp;")
    .replace(/"/g, "&quot;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}
