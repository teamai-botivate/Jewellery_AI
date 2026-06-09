/* Admin Dashboard JavaScript */

const API_BASE = window.location.origin;

let selectedFiles = [];
let currentJobId = null;
let statusInterval = null;

// Tab switching
document.querySelectorAll('.tab-btn').forEach(btn => {
  btn.addEventListener('click', (e) => {
    const tabName = e.target.dataset.tab;
    switchTab(tabName);
  });
});

function switchTab(tabName) {
  // Hide all panels
  document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('tab-panel--active'));
  // Deactivate all buttons
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('tab-btn--active'));
  // Show selected panel
  document.getElementById(`${tabName}-tab`).classList.add('tab-panel--active');
  // Activate button
  event.target.classList.add('tab-btn--active');

  if (tabName === 'gallery') {
    loadGallery();
  } else if (tabName === 'jobs') {
    loadJobHistory();
  }
}

// File Upload
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');

uploadArea.addEventListener('dragover', (e) => {
  e.preventDefault();
  uploadArea.style.borderColor = '#c8a96e';
});

uploadArea.addEventListener('dragleave', () => {
  uploadArea.style.borderColor = '#2a2a3d';
});

uploadArea.addEventListener('drop', (e) => {
  e.preventDefault();
  uploadArea.style.borderColor = '#2a2a3d';
  handleFiles(Array.from(e.dataTransfer.files));
});

fileInput.addEventListener('change', () => {
  handleFiles(Array.from(fileInput.files));
});

function handleFiles(files) {
  selectedFiles = files.filter(f => f.type.startsWith('image/'));
  renderFileList();
}

function renderFileList() {
  const fileList = document.getElementById('fileList');
  const filesGrid = document.getElementById('filesGrid');
  const fileCount = document.getElementById('fileCount');

  if (selectedFiles.length === 0) {
    fileList.style.display = 'none';
    return;
  }

  fileList.style.display = 'block';
  fileCount.textContent = selectedFiles.length;
  filesGrid.innerHTML = selectedFiles.map((file, idx) => `
    <div class="file-item" title="${file.name}">
      📄 ${file.name.substring(0, 20)}...
    </div>
  `).join('');
}

function clearFiles() {
  selectedFiles = [];
  fileInput.value = '';
  renderFileList();
}

async function startUpload() {
  if (selectedFiles.length === 0) {
    alert('Please select files first');
    return;
  }

  const formData = new FormData();
  selectedFiles.forEach(f => formData.append('files', f));

  try {
    const response = await fetch(`${API_BASE}/add-image-batch-async`, {
      method: 'POST',
      body: formData
    });

    if (!response.ok) throw new Error('Upload failed');

    const result = await response.json();
    currentJobId = result.job_id;

    showProgress();
    startStatusPolling();
  } catch (err) {
    alert('Error: ' + err.message);
  }
}

function showProgress() {
  document.getElementById('uploadArea').style.display = 'none';
  document.getElementById('fileList').style.display = 'none';
  document.getElementById('progressSection').style.display = 'block';
  document.getElementById('jobIdDisplay').textContent = currentJobId;
}

function startStatusPolling() {
  statusInterval = setInterval(async () => {
    try {
      const response = await fetch(`${API_BASE}/upload-status/${currentJobId}`);
      if (!response.ok) return;

      const status = response.json();
      updateProgress(status);

      if (status.status === 'completed') {
        clearInterval(statusInterval);
        alert(`Upload Complete!\n${status.processed} success\n${status.failed} failed`);
        resetUploadUI();
      }
    } catch (err) {
      console.error('Status check failed:', err);
    }
  }, 2000);
}

async function updateProgress(status) {
  const { total, processed, failed, remaining } = await (await fetch(`${API_BASE}/upload-status/${currentJobId}`)).json();

  const percent = total > 0 ? Math.round((processed / total) * 100) : 0;

  document.getElementById('progressLabel').textContent = `Processing... ${percent}%`;
  document.getElementById('progressPercent').textContent = `${percent}%`;
  document.getElementById('progressFill').style.width = `${percent}%`;
  document.getElementById('processedCount').textContent = processed;
  document.getElementById('failedCount').textContent = failed;
  document.getElementById('totalCount').textContent = total;
}

function resetUploadUI() {
  selectedFiles = [];
  fileInput.value = '';
  document.getElementById('uploadArea').style.display = 'block';
  document.getElementById('fileList').style.display = 'none';
  document.getElementById('progressSection').style.display = 'none';
}

// Gallery
async function loadGallery() {
  const galleryGrid = document.getElementById('galleryGrid');
  const subtitle = document.getElementById('gallerySubtitle');

  galleryGrid.innerHTML = '<p style="text-align: center; color: #8888a8;">Loading images...</p>';

  try {
    const response = await fetch(`${API_BASE}/health`);
    const health = await response.json();
    const count = health.indexed;

    subtitle.textContent = `${count} images indexed in Qdrant Cloud`;

    // Fetch images from Qdrant (scroll)
    // For now, show placeholder
    // In production, you'd need a /get-images endpoint
    galleryGrid.innerHTML = `
      <div style="grid-column: 1/-1; text-align: center; color: #8888a8; padding: 2rem;">
        <p>Gallery view would show ${count} indexed images</p>
        <p style="font-size: 0.9rem; margin-top: 1rem;">Implement GET /images endpoint in backend to fetch and display images</p>
      </div>
    `;
  } catch (err) {
    galleryGrid.innerHTML = '<p style="color: #ef4444;">Error loading gallery</p>';
  }
}

// Job History
async function loadJobHistory() {
  const jobsTable = document.getElementById('jobsTable');
  jobsTable.innerHTML = '<tr><td colspan="7" class="text-center">Loading jobs...</td></tr>';

  try {
    const response = await fetch(`${API_BASE}/jobs-history`);
    if (!response.ok) {
      jobsTable.innerHTML = '<tr><td colspan="7" class="text-center">No jobs found</td></tr>';
      return;
    }

    const jobs = await response.json();

    if (jobs.length === 0) {
      jobsTable.innerHTML = '<tr><td colspan="7" class="text-center">No upload jobs yet</td></tr>';
      return;
    }

    jobsTable.innerHTML = jobs.map(job => {
      const percent = job.total > 0 ? Math.round(((job.processed + job.failed) / job.total) * 100) : 0;
      const statusClass = `status-badge--${job.status}`;

      return `
        <tr>
          <td><code style="font-size: 0.8rem;">${job.job_id.substring(0, 8)}...</code></td>
          <td><span class="status-badge ${statusClass}">${job.status}</span></td>
          <td>${job.total}</td>
          <td>${job.processed}</td>
          <td>${job.failed}</td>
          <td><div style="width: 60px; height: 20px; background: #16162a; border-radius: 4px; overflow: hidden;">
            <div style="height: 100%; background: linear-gradient(90deg, #c8a96e, #e8d4a0); width: ${percent}%"></div>
          </div></td>
          <td><button class="btn btn--secondary" onclick="checkJobStatus('${job.job_id}')" style="padding: 0.4rem 0.8rem; font-size: 0.8rem;">Check</button></td>
        </tr>
      `;
    }).join('');
  } catch (err) {
    jobsTable.innerHTML = '<tr><td colspan="7" class="text-center" style="color: #ef4444;">Error loading jobs</td></tr>';
  }
}

async function checkJobStatus(jobId) {
  try {
    const response = await fetch(`${API_BASE}/upload-status/${jobId}`);
    const status = await response.json();

    alert(`Job: ${jobId}\nStatus: ${status.status}\nProcessed: ${status.processed}/${status.total}\nFailed: ${status.failed}`);
  } catch (err) {
    alert('Error: ' + err.message);
  }
}

// Load gallery on page load
window.addEventListener('load', () => {
  // loadGallery(); // Optional: auto-load on page load
});
