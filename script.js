// ============================================================
// FRAUD DETECTION PORTFOLIO — GLOBAL SCRIPTS
// ============================================================

// ── NAV SCROLL STATE ──
(function () {
  const nav = document.querySelector('.nav');
  if (!nav) return;
  window.addEventListener('scroll', () => {
    nav.classList.toggle('scrolled', window.scrollY > 30);
  }, { passive: true });
})();

// ── SET ACTIVE NAV LINK ──
(function () {
  const links = document.querySelectorAll('.nav-links a');
  const path = window.location.pathname.split('/').pop() || 'index.html';
  links.forEach(link => {
    const href = link.getAttribute('href');
    if (href === path || (path === '' && href === 'index.html')) {
      link.classList.add('active');
    }
  });
})();

// ── PARTICLE CANVAS ──
function initParticles(canvasId) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  let W, H, particles = [];

  function resize() {
    W = canvas.width = canvas.offsetWidth;
    H = canvas.height = canvas.offsetHeight;
  }

  class Particle {
    constructor() { this.reset(); }
    reset() {
      this.x = Math.random() * W;
      this.y = Math.random() * H;
      this.vx = (Math.random() - 0.5) * 0.4;
      this.vy = (Math.random() - 0.5) * 0.4;
      this.r = Math.random() * 1.5 + 0.5;
      this.alpha = Math.random() * 0.4 + 0.1;
      this.decay = Math.random() * 0.003 + 0.001;
    }
    update() {
      this.x += this.vx; this.y += this.vy;
      this.alpha -= this.decay;
      if (this.alpha <= 0 || this.x < 0 || this.x > W || this.y < 0 || this.y > H) this.reset();
    }
    draw() {
      ctx.beginPath();
      ctx.arc(this.x, this.y, this.r, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(192,57,43,${this.alpha})`;
      ctx.fill();
    }
  }

  function connect() {
    for (let i = 0; i < particles.length; i++) {
      for (let j = i + 1; j < particles.length; j++) {
        const dx = particles[i].x - particles[j].x;
        const dy = particles[i].y - particles[j].y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist < 120) {
          ctx.beginPath();
          ctx.moveTo(particles[i].x, particles[i].y);
          ctx.lineTo(particles[j].x, particles[j].y);
          ctx.strokeStyle = `rgba(192,57,43,${0.15 * (1 - dist / 120)})`;
          ctx.lineWidth = 0.5;
          ctx.stroke();
        }
      }
    }
  }

  resize();
  window.addEventListener('resize', resize);
  for (let i = 0; i < 80; i++) particles.push(new Particle());

  function loop() {
    ctx.clearRect(0, 0, W, H);
    connect();
    particles.forEach(p => { p.update(); p.draw(); });
    requestAnimationFrame(loop);
  }
  loop();
}

// ── SCROLL REVEAL ──
function initReveal() {
  const els = document.querySelectorAll('.reveal, .reveal-left, .reveal-right');
  if (!els.length) return;
  const obs = new IntersectionObserver((entries) => {
    entries.forEach(e => {
      if (e.isIntersecting) {
        e.target.classList.add('in-view');
        obs.unobserve(e.target);
      }
    });
  }, { threshold: 0.12, rootMargin: '0px 0px -40px 0px' });
  els.forEach(el => obs.observe(el));
}

// ── ANIMATED COUNTERS ──
function initCounters() {
  const counters = document.querySelectorAll('[data-count]');
  if (!counters.length) return;
  const obs = new IntersectionObserver((entries) => {
    entries.forEach(e => {
      if (e.isIntersecting) {
        const el = e.target;
        const target = parseFloat(el.dataset.count);
        const decimals = el.dataset.decimals ? parseInt(el.dataset.decimals) : 0;
        const suffix = el.dataset.suffix || '';
        const duration = 1800;
        const start = performance.now();
        const animate = (now) => {
          const elapsed = now - start;
          const progress = Math.min(elapsed / duration, 1);
          const ease = 1 - Math.pow(1 - progress, 4);
          const current = (target * ease).toFixed(decimals);
          el.textContent = current + suffix;
          if (progress < 1) requestAnimationFrame(animate);
        };
        requestAnimationFrame(animate);
        obs.unobserve(el);
      }
    });
  }, { threshold: 0.5 });
  counters.forEach(el => obs.observe(el));
}

// ── METRIC BARS ──
function initMetricBars() {
  const bars = document.querySelectorAll('.metric-fill');
  if (!bars.length) return;
  const obs = new IntersectionObserver((entries) => {
    entries.forEach(e => {
      if (e.isIntersecting) {
        e.target.style.width = e.target.dataset.width || '0%';
        obs.unobserve(e.target);
      }
    });
  }, { threshold: 0.3 });
  bars.forEach(bar => obs.observe(bar));
}

// ── 3D CARD TILT ──
function initCardTilt() {
  const cards = document.querySelectorAll('.card-3d');
  cards.forEach(card => {
    card.addEventListener('mousemove', (e) => {
      const rect = card.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;
      const cx = rect.width / 2;
      const cy = rect.height / 2;
      const rx = ((y - cy) / cy) * -6;
      const ry = ((x - cx) / cx) * 6;
      card.style.transform = `perspective(600px) rotateX(${rx}deg) rotateY(${ry}deg) scale(1.01)`;
    });
    card.addEventListener('mouseleave', () => {
      card.style.transform = '';
    });
  });
}

// ── PAGE TRANSITION ──
function initPageTransitions() {
  const overlay = document.createElement('div');
  overlay.className = 'page-transition';
  document.body.appendChild(overlay);

  document.querySelectorAll('a[href]').forEach(link => {
    const href = link.getAttribute('href');
    if (!href || href.startsWith('#') || href.startsWith('http') || href.startsWith('mailto')) return;
    link.addEventListener('click', (e) => {
      e.preventDefault();
      overlay.style.transition = 'transform 0.5s cubic-bezier(0.4,0,0.2,1)';
      overlay.style.transform = 'scaleX(1)';
      overlay.style.transformOrigin = 'left';
      setTimeout(() => {
        window.location.href = href;
      }, 450);
    });
  });

  // Animate out on page load
  overlay.style.transform = 'scaleX(1)';
  overlay.style.transformOrigin = 'right';
  requestAnimationFrame(() => {
    overlay.style.transition = 'transform 0.6s 0.1s cubic-bezier(0.4,0,0.2,1)';
    overlay.style.transform = 'scaleX(0)';
  });
}

// ── MATRIX BG ──
function initMatrix() {
  const container = document.querySelector('.matrix-bg');
  if (!container) return;
  const chars = '01アイウエオカキクケコ$#@%&';
  const cols = Math.floor(window.innerWidth / 20);
  for (let i = 0; i < Math.min(cols, 30); i++) {
    const col = document.createElement('div');
    col.className = 'matrix-col';
    col.style.left = `${i * (100 / Math.min(cols, 30))}%`;
    col.style.animationDuration = `${Math.random() * 8 + 4}s`;
    col.style.animationDelay = `${Math.random() * 5}s`;
    let content = '';
    for (let j = 0; j < 20; j++) {
      content += chars[Math.floor(Math.random() * chars.length)] + '\n';
    }
    col.textContent = content;
    container.appendChild(col);
  }
}

// ── TICKER DUPLICATE ──
function initTicker() {
  const inner = document.querySelector('.ticker-inner');
  if (!inner) return;
  inner.innerHTML += inner.innerHTML;
}

// ── INIT ALL ──
document.addEventListener('DOMContentLoaded', () => {
  initCounters();
  initMetricBars();
  initDetector();
});

/* ── LIVE INTERACTIVE FRAUD DETECTION ENGINE ── */
const DETECTOR_API = 'http://127.0.0.1:5000/api';
const sampleTransactions = new Map();
let activeSampleId = null;

function initDetector() {
  if (!document.getElementById('live-detector')) return;
  updateInputDisplay();
  connectDetectorApi();
}

function setDetectorMessage(message, isError = false) {
  const output = document.getElementById('detector-message');
  if (!output) return;
  output.textContent = message;
  output.classList.toggle('is-error', isError);
}

async function connectDetectorApi() {
  const statusBadge = document.getElementById('api-status-badge');
  const scanButton = document.getElementById('btn-run-scan');
  const presetButtons = document.querySelectorAll('.preset-btn');

  try {
    const healthResponse = await fetch(`${DETECTOR_API}/health`);
    if (!healthResponse.ok) throw new Error(`Health check failed (${healthResponse.status})`);
    const health = await healthResponse.json();
    if (health.models_ready !== true) {
      throw new Error('The API is running, but trained model files are not loaded.');
    }

    const samplesResponse = await fetch(`${DETECTOR_API}/samples`);
    if (!samplesResponse.ok) throw new Error(`Sample loading failed (${samplesResponse.status})`);
    const samples = await samplesResponse.json();
    if (!Array.isArray(samples)) throw new Error('The API returned an invalid sample list.');
    samples.forEach(sample => sampleTransactions.set(sample.id, sample));

    presetButtons.forEach(button => {
      const sampleId = button.dataset.sampleId;
      const sample = sampleTransactions.get(sampleId);
      button.disabled = !sample;
      if (sample) button.textContent = sample.title;
    });
    if (scanButton) scanButton.disabled = false;
    if (statusBadge) statusBadge.textContent = 'MODEL API: READY';
    setDetectorMessage('Connected to the trained models. Sample and custom scans run through the local API.');
  } catch (error) {
    if (statusBadge) statusBadge.textContent = 'MODEL API: OFFLINE';
    presetButtons.forEach(button => { button.disabled = true; });
    if (scanButton) scanButton.disabled = true;
    setDetectorMessage(
      `Live predictions require the trained Python models. Start the API with "python detect_api.py", then refresh this page. ${error.message}`,
      true
    );
  }
}

function updateInputDisplay() {
  activeSampleId = null;
  document.querySelectorAll('.preset-btn').forEach(button => {
    button.classList.remove('active-legit', 'active-fraud');
  });

  const fields = [
    ['inp-amount', 'val-amount', value => `$${Number(value).toFixed(2)}`],
    ['inp-time', 'val-time', value => `${Number(value).toLocaleString()}s`],
    ['inp-v14', 'val-v14', value => Number(value).toFixed(2)],
    ['inp-v4', 'val-v4', value => Number(value).toFixed(2)],
    ['inp-v12', 'val-v12', value => Number(value).toFixed(2)],
    ['inp-v10', 'val-v10', value => Number(value).toFixed(2)]
  ];

  fields.forEach(([inputId, outputId, format]) => {
    const input = document.getElementById(inputId);
    const output = document.getElementById(outputId);
    if (input && output) output.textContent = format(input.value || 0);
  });
}

function loadSamplePreset(sampleId) {
  const sample = sampleTransactions.get(sampleId);
  if (!sample) {
    setDetectorMessage('That sample is not available from the model API.', true);
    return;
  }

  const features = sample.features;
  document.getElementById('inp-amount').value = features.Amount;
  document.getElementById('inp-time').value = features.Time;
  document.getElementById('inp-v14').value = features.V14;
  document.getElementById('inp-v4').value = features.V4;
  document.getElementById('inp-v12').value = features.V12;
  document.getElementById('inp-v10').value = features.V10;
  updateInputDisplay();
  activeSampleId = sampleId;

  document.querySelectorAll('.preset-btn').forEach(button => {
    if (button.dataset.sampleId === sampleId) {
      button.classList.add(sample.label === 'Fraudulent' ? 'active-fraud' : 'active-legit');
    }
  });
  executeFraudScan();
}

async function executeFraudScan() {
  const btn = document.getElementById('btn-run-scan');
  if (btn) {
    btn.disabled = true;
    btn.textContent = 'Running model prediction…';
  }

  const amount = Number(document.getElementById('inp-amount').value);
  const time = Number(document.getElementById('inp-time').value);
  const featureValues = ['inp-v14', 'inp-v4', 'inp-v12', 'inp-v10']
    .map(id => Number(document.getElementById(id).value));
  if (![amount, time, ...featureValues].every(Number.isFinite) || amount < 0 || time < 0) {
    setDetectorMessage('Enter valid non-negative amount and time values before scanning.', true);
    if (btn) {
      btn.disabled = false;
      btn.textContent = 'Run model prediction';
    }
    return;
  }

  const payload = activeSampleId
    ? { sample_id: activeSampleId }
    : {
        amount,
        time,
        V14: featureValues[0],
        V4: featureValues[1],
        V12: featureValues[2],
        V10: featureValues[3]
      };

  try {
    const response = await fetch(`${DETECTOR_API}/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || `Prediction failed (${response.status})`);
    renderScanResults(result);
    setDetectorMessage('Prediction returned by the trained Python models.');
  } catch (error) {
    setDetectorMessage(`Prediction failed: ${error.message}`, true);
  } finally {
    if (btn) {
      btn.disabled = false;
      btn.textContent = 'Run model prediction';
    }
  }
}

function renderScanResults(res) {
  const badge = document.getElementById('result-badge');
  const meterScore = document.getElementById('meter-score-text');
  const meterFill = document.getElementById('meter-bar-fill');
  const rfProbEl = document.getElementById('res-rf-prob');
  const lrProbEl = document.getElementById('res-lr-prob');
  const anomalyContainer = document.getElementById('anomaly-container');
  const actionRec = document.getElementById('action-recommendation');

  const scorePct = (res.consensus_prob * 100).toFixed(2);

  // Meter and Score
  if (meterScore) meterScore.textContent = `${scorePct}%`;
  if (meterFill) {
    meterFill.style.width = `${Math.max(1, Math.min(100, scorePct))}%`;
    meterFill.style.backgroundColor = '#2563a6';
  }

  // Badge
  if (badge) {
    if (res.is_fraud) {
      badge.className = 'result-status-badge badge-fraud';
      badge.textContent = `Fraud flagged — ${res.risk_level} risk`;
    } else {
      badge.className = 'result-status-badge badge-legit';
      badge.textContent = 'No fraud flagged by the model';
    }
  }

  // Model Probabilities
  if (rfProbEl) {
    rfProbEl.textContent = `${(res.rf_fraud_prob * 100).toFixed(2)}%`;
    rfProbEl.style.color = '#2563a6';
  }
  if (lrProbEl) {
    lrProbEl.textContent = `${(res.lr_fraud_prob * 100).toFixed(2)}%`;
    lrProbEl.style.color = '#2563a6';
  }

  // Anomalies
  if (anomalyContainer && res.anomalies) {
    anomalyContainer.innerHTML = '';
    res.anomalies.forEach(a => {
      const isBad = a.isSevere || (typeof a.signal === 'string' && a.signal.includes('Deflection') && !a.signal.includes('Baseline'));
      const item = document.createElement('div');
      item.className = 'anomaly-item';
      const feature = document.createElement('span');
      feature.textContent = a.feature;
      const signal = document.createElement('span');
      signal.className = isBad ? 'anomaly-flag' : 'anomaly-safe';
      signal.textContent = a.signal;
      item.append(feature, signal);
      anomalyContainer.appendChild(item);
    });
  }

  // Recommendation
  if (actionRec) {
    actionRec.textContent = res.recommendation;
    actionRec.style.color = '#2563a6';
  }
}
