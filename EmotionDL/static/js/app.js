/* app.js — EmoSense Frontend */

const EMOTION_EMOJIS = {
  neutral:   '😐', calm: '😌', happy: '😄', sad: '😢',
  angry:     '😡', fearful: '😨', disgust: '🤢', surprised: '😲'
};

const EMOTION_COLORS = {
  neutral:   '#64748b', calm:     '#0ea5e9', happy:    '#f59e0b',
  sad:       '#6366f1', angry:    '#ef4444', fearful:  '#8b5cf6',
  disgust:   '#10b981', surprised:'#f97316'
};

// ─── DOM references ────────────────────────────────────────────────
const dropZone      = document.getElementById('dropZone');
const audioInput    = document.getElementById('audioInput');
const fileSelected  = document.getElementById('fileSelected');
const fileName      = document.getElementById('fileName');
const fileSize      = document.getElementById('fileSize');
const clearFileBtn  = document.getElementById('clearFile');
const analyzeBtn    = document.getElementById('analyzeBtn');
const uploadForm    = document.getElementById('uploadForm');
const loadingState  = document.getElementById('loadingState');
const resultsPanel  = document.getElementById('resultsPanel');
const demoBanner    = document.getElementById('demoBanner');
const modelBadge    = document.getElementById('modelBadge');
const modelStatus   = document.getElementById('modelStatus');
const wheelEmoji    = document.getElementById('wheelEmoji');

// ─── Model status ──────────────────────────────────────────────────
async function checkModel() {
  try {
    const r = await fetch('/model-info');
    const d = await r.json();
    if (d.loaded) {
      modelStatus.textContent = `DL Model · ${d.accuracy}% acc`;
      modelBadge.querySelector('.dot').style.background = '#22c55e';
    } else {
      modelStatus.textContent = 'Demo mode';
      modelBadge.querySelector('.dot').style.background = '#f59e0b';
      demoBanner.hidden = false;
    }
  } catch {
    modelStatus.textContent = 'Offline';
  }
}
checkModel();

// ─── Tab navigation ────────────────────────────────────────────────
document.querySelectorAll('.pill').forEach(btn => {
  btn.addEventListener('click', () => {
    const tab = btn.dataset.tab;
    document.querySelectorAll('.pill').forEach(p => p.classList.remove('active'));
    document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));
    btn.classList.add('active');
    document.getElementById('tab-' + tab).classList.add('active');
  });
});

// ─── Emotion Wheel ─────────────────────────────────────────────────
function buildWheel() {
  const emotions = Object.keys(EMOTION_EMOJIS);
  const seg = document.getElementById('wheelSegments');
  seg.innerHTML = '';
  const r = 62;
  emotions.forEach((em, i) => {
    const angle = (i / emotions.length) * Math.PI * 2;
    const x = 70 + r * Math.cos(angle);
    const y = 70 + r * Math.sin(angle);
    const dot = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    dot.setAttribute('viewBox','0 0 140 140');
    dot.style.cssText = `position:absolute;width:140px;height:140px;top:0;left:0;`;
    const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
    circle.setAttribute('cx', x); circle.setAttribute('cy', y); circle.setAttribute('r', 5);
    circle.setAttribute('fill', EMOTION_COLORS[em]);
    circle.setAttribute('opacity', '0.5');
    dot.appendChild(circle);
    seg.appendChild(dot);
  });
}
buildWheel();

// ─── File input ────────────────────────────────────────────────────
function formatBytes(b) {
  if (b < 1024) return b + ' B';
  if (b < 1024*1024) return (b/1024).toFixed(1) + ' KB';
  return (b/1024/1024).toFixed(1) + ' MB';
}

function setFile(file) {
  if (!file) return;
  fileName.textContent = file.name;
  fileSize.textContent = formatBytes(file.size);
  dropZone.querySelector('.drop-inner').hidden = true;
  fileSelected.hidden = false;
  analyzeBtn.disabled = false;
}

function clearFile() {
  audioInput.value = '';
  dropZone.querySelector('.drop-inner').hidden = false;
  fileSelected.hidden = true;
  analyzeBtn.disabled = true;
}

dropZone.addEventListener('click', e => {
  if (e.target === clearFileBtn || clearFileBtn.contains(e.target)) return;
  audioInput.click();
});

audioInput.addEventListener('change', () => setFile(audioInput.files[0]));

clearFileBtn.addEventListener('click', e => {
  e.stopPropagation();
  clearFile();
});

dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('drag-over'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
dropZone.addEventListener('drop', e => {
  e.preventDefault(); dropZone.classList.remove('drag-over');
  const file = e.dataTransfer.files[0];
  if (file) {
    const dt = new DataTransfer();
    dt.items.add(file);
    audioInput.files = dt.files;
    setFile(file);
  }
});

// ─── Loading steps animation ───────────────────────────────────────
let stepTimer = null;
function animateSteps() {
  const steps = document.querySelectorAll('.step');
  let current = 0;
  steps.forEach(s => { s.className = 'step'; });
  steps[0].classList.add('active');

  stepTimer = setInterval(() => {
    if (current < steps.length) {
      if (current > 0) steps[current - 1].classList.replace('active', 'done');
      if (current < steps.length) { steps[current].classList.add('active'); }
      current++;
    } else {
      clearInterval(stepTimer);
    }
  }, 600);
}

// ─── Waveform renderer ─────────────────────────────────────────────
function drawWaveform(data, color) {
  const canvas = document.getElementById('waveformCanvas');
  const ctx = canvas.getContext('2d');
  const W = canvas.offsetWidth;
  const H = 60;
  canvas.width = W;

  ctx.clearRect(0, 0, W, H);
  const barW = W / data.length;
  const mid = H / 2;

  data.forEach((v, i) => {
    const h = Math.abs(v) * mid * 6;
    const x = i * barW;
    const alpha = 0.4 + Math.abs(v) * 3;

    ctx.fillStyle = color + Math.round(Math.min(alpha, 1) * 255).toString(16).padStart(2, '0');
    ctx.beginPath();
    ctx.roundRect(x, mid - h/2, barW - 1, h, 2);
    ctx.fill();
  });
}

// ─── Probability bars ──────────────────────────────────────────────
function renderProbBars(allProbs) {
  const container = document.getElementById('probBars');
  container.innerHTML = '';
  const sorted = Object.entries(allProbs).sort((a, b) => b[1] - a[1]);

  sorted.forEach(([emotion, prob]) => {
    const pct = (prob * 100).toFixed(1);
    const color = EMOTION_COLORS[emotion] || '#888';
    const emoji = EMOTION_EMOJIS[emotion] || '🎭';

    const row = document.createElement('div');
    row.className = 'prob-row';
    row.innerHTML = `
      <div class="prob-label">
        <span class="prob-emoji">${emoji}</span>
        <span>${emotion}</span>
      </div>
      <div class="prob-bar-wrap">
        <div class="prob-bar-fill" style="width:0%;background:${color}" data-target="${pct}"></div>
      </div>
      <div class="prob-pct">${pct}%</div>
    `;
    container.appendChild(row);
  });

  // Animate bars after paint
  requestAnimationFrame(() => requestAnimationFrame(() => {
    container.querySelectorAll('.prob-bar-fill').forEach(el => {
      el.style.width = el.dataset.target + '%';
    });
  }));
}

// ─── Show results ──────────────────────────────────────────────────
function showResults(data) {
  const color = data.color || '#4f8ef7';

  // Primary
  document.getElementById('resultEmoji').textContent = data.emoji;
  document.getElementById('resultLabel').textContent = data.emotion.toUpperCase();
  document.getElementById('resultDesc').textContent = data.description;
  document.getElementById('confidenceFill').style.width = data.confidence + '%';
  document.getElementById('confidenceVal').textContent = data.confidence + '%';

  const card = document.getElementById('primaryResult');
  card.classList.add('colored');
  card.style.setProperty('--emotion-color', color);
  document.getElementById('confidenceFill').style.background =
    `linear-gradient(90deg, ${color}, ${color}99)`;

  // Stats
  const s = data.stats;
  document.getElementById('statDuration').textContent = s.duration + 's';
  document.getElementById('statPitch').textContent = s.pitch ? s.pitch.toFixed(0) + ' Hz' : '—';
  document.getElementById('statEnergy').textContent = s.rms.toFixed(4);
  document.getElementById('statZcr').textContent = s.zcr.toFixed(4);

  // Waveform
  if (s.waveform && s.waveform.length) {
    setTimeout(() => drawWaveform(s.waveform, color), 100);
  }

  // Emotion wheel
  wheelEmoji.textContent = data.emoji;

  // Prob bars
  renderProbBars(data.all_probs);

  loadingState.hidden = true;
  resultsPanel.hidden = false;

  if (data.demo_mode) demoBanner.hidden = false;
}

// ─── Form submit ───────────────────────────────────────────────────
uploadForm.addEventListener('submit', async e => {
  e.preventDefault();

  if (!audioInput.files[0]) return;

  // Hide results, show loading
  resultsPanel.hidden = true;
  loadingState.hidden = false;
  analyzeBtn.disabled = true;
  animateSteps();

  const formData = new FormData();
  formData.append('audio', audioInput.files[0]);

  try {
    const res = await fetch('/predict', { method: 'POST', body: formData });
    const data = await res.json();

    clearInterval(stepTimer);
    document.querySelectorAll('.step').forEach(s => s.classList.replace('active', 'done'));

    await new Promise(r => setTimeout(r, 400));

    if (data.error) {
      loadingState.hidden = true;
      alert('Error: ' + data.error);
    } else {
      showResults(data);
    }
  } catch (err) {
    loadingState.hidden = true;
    alert('Network error: ' + err.message);
  } finally {
    analyzeBtn.disabled = false;
  }
});

// ─── Reset button ──────────────────────────────────────────────────
document.getElementById('resetBtn').addEventListener('click', () => {
  resultsPanel.hidden = true;
  clearFile();
  wheelEmoji.textContent = '🎙';
  document.getElementById('primaryResult').classList.remove('colored');
});
