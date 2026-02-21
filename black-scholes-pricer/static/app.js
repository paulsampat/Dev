/**
 * Black-Scholes Options Pricer – Frontend
 *
 * Responsibilities:
 *  - Sync slider ↔ number inputs
 *  - On any parameter change: POST /api/price + POST /api/sensitivity (×3 in parallel)
 *  - Render / update six Chart.js sensitivity charts
 *  - Mark current parameter value with a yellow annotation line on each chart
 */

'use strict';

// ── Application state ─────────────────────────────────────────────────────
const state = {
  S: 100.0,
  K: 100.0,
  T: 1.0,
  r: 0.05,         // internal decimal
  sigma: 0.20,     // internal decimal
  option_type: 'call',
};

// ── Chart.js global defaults ──────────────────────────────────────────────
Chart.defaults.color          = '#5c7898';
Chart.defaults.font.family    = "'Menlo','Monaco','Consolas',monospace";
Chart.defaults.font.size      = 10;

// ── Chart registry ────────────────────────────────────────────────────────
const CHARTS = {};

// ── Utilities ─────────────────────────────────────────────────────────────
function debounce(fn, ms) {
  let t;
  return (...args) => { clearTimeout(t); t = setTimeout(() => fn(...args), ms); };
}

function fmt(v, dp = 4) {
  return (v == null || isNaN(v)) ? '—' : Number(v).toFixed(dp);
}

function el(id) { return document.getElementById(id); }

function setStatus(cls, label) {
  el('statusDot').className  = `status-dot ${cls}`;
  el('statusText').textContent = label;
}

// ── API helpers ───────────────────────────────────────────────────────────
async function apiPost(path, body) {
  const res = await fetch(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail ?? `HTTP ${res.status}`);
  }
  return res.json();
}

function fetchPrice() {
  return apiPost('/api/price', state);
}

function fetchSens(vary, points = 100) {
  return apiPost('/api/sensitivity', { ...state, vary, points });
}

// ── Chart factory ─────────────────────────────────────────────────────────
function buildDatasetDef(label, color, dashed = false) {
  return {
    label,
    data: [],
    borderColor: color,
    backgroundColor: color.replace(')', ', 0.07)').replace('rgb', 'rgba'),
    borderWidth: 1.8,
    borderDash: dashed ? [5, 4] : [],
    pointRadius: 0,
    fill: false,
    tension: 0.35,
  };
}

function makeChart(canvasId, title, xLabel, yLabel, datasets, xPct = false) {
  const ctx = el(canvasId).getContext('2d');
  const xFmt = xPct ? (v) => `${(v * 100).toFixed(0)}%` : (v) => v;

  return new Chart(ctx, {
    type: 'line',
    data: { datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,             // instant re-render on update
      interaction: { mode: 'index', intersect: false },
      plugins: {
        title: {
          display: true,
          text: title,
          color: '#c0cfe4',
          font: { size: 11, weight: '600' },
          padding: { bottom: 6 },
        },
        legend: {
          display: datasets.length > 1,
          labels: { color: '#5c7898', boxWidth: 10, padding: 8, font: { size: 9 } },
        },
        tooltip: {
          backgroundColor: '#0d1424',
          borderColor: '#1a2d4a',
          borderWidth: 1,
          titleColor: '#00c8f0',
          bodyColor: '#d4e0f0',
          padding: 8,
          callbacks: {
            title: (items) => {
              const raw = items[0]?.parsed?.x;
              return xPct ? `${(raw * 100).toFixed(1)}%` : String(raw?.toFixed(3) ?? '');
            },
          },
        },
        annotation: {
          annotations: {
            currentLine: {
              type: 'line',
              xMin: 0,
              xMax: 0,
              borderColor: 'rgba(255, 215, 64, 0.9)',
              borderWidth: 1.5,
              borderDash: [5, 4],
            },
          },
        },
      },
      scales: {
        x: {
          type: 'linear',
          grid: { color: 'rgba(26, 45, 74, 0.7)', drawBorder: false },
          ticks: {
            color: '#384d68',
            font: { size: 9 },
            maxTicksLimit: 6,
            callback: xFmt,
          },
          title: {
            display: true,
            text: xLabel,
            color: '#384d68',
            font: { size: 9 },
          },
        },
        y: {
          grid: { color: 'rgba(26, 45, 74, 0.7)', drawBorder: false },
          ticks: { color: '#384d68', font: { size: 9 }, maxTicksLimit: 5 },
          title: {
            display: true,
            text: yLabel,
            color: '#384d68',
            font: { size: 9 },
          },
        },
      },
    },
  });
}

function initCharts() {
  // Price vs Spot  (+ payoff overlay)
  CHARTS.priceVsSpot = makeChart(
    'chartPriceVsSpot', 'Price vs Spot', 'Spot (S)', 'Price',
    [
      buildDatasetDef('BS Price',        '#00c8f0'),
      buildDatasetDef('Payoff at Expiry','#00e676', true),
    ],
  );

  // Price vs Volatility
  CHARTS.priceVsVol = makeChart(
    'chartPriceVsVol', 'Price vs Volatility', 'Volatility (σ)', 'Price',
    [buildDatasetDef('BS Price', '#00c8f0')],
    true,
  );

  // Price vs Time to Expiry
  CHARTS.priceVsTime = makeChart(
    'chartPriceVsTime', 'Price vs Time to Expiry', 'Time (years)', 'Price',
    [buildDatasetDef('BS Price', '#00c8f0')],
  );

  // Delta vs Spot
  CHARTS.deltaVsSpot = makeChart(
    'chartDeltaVsSpot', 'Delta vs Spot', 'Spot (S)', 'Δ Delta',
    [buildDatasetDef('Delta', '#ff9800')],
  );

  // Gamma vs Spot
  CHARTS.gammaVsSpot = makeChart(
    'chartGammaVsSpot', 'Gamma vs Spot', 'Spot (S)', 'Γ Gamma',
    [buildDatasetDef('Gamma', '#ce93d8')],
  );

  // Vega vs Volatility
  CHARTS.vegaVsVol = makeChart(
    'chartVegaVsVol', 'Vega vs Volatility', 'Volatility (σ)', 'ν Vega',
    [buildDatasetDef('Vega', '#ff6b9d')],
    true,
  );
}

// ── Chart updaters ────────────────────────────────────────────────────────
function refreshChart(chart, dataSets, currentX) {
  dataSets.forEach((pts, i) => {
    if (chart.data.datasets[i]) chart.data.datasets[i].data = pts;
  });
  const ann = chart.options.plugins.annotation.annotations.currentLine;
  ann.xMin = currentX;
  ann.xMax = currentX;
  chart.update('none');
}

function toXY(data, metric) {
  return data.map((d) => ({ x: d.x, y: d[metric] }));
}

// ── Results panel ─────────────────────────────────────────────────────────
function updateResults(d) {
  el('resPrice').textContent    = '$' + fmt(d.price, 4);
  el('resDelta').textContent    = fmt(d.delta, 4);
  el('resGamma').textContent    = fmt(d.gamma, 6);
  el('resTheta').textContent    = fmt(d.theta, 4);
  el('resVega').textContent     = fmt(d.vega, 4);
  el('resRho').textContent      = fmt(d.rho, 4);
  el('resD1').textContent       = fmt(d.d1, 4);
  el('resD2').textContent       = fmt(d.d2, 4);
  el('resIntrinsic').textContent= '$' + fmt(d.intrinsic_value, 4);
  el('resTimeValue').textContent= '$' + fmt(d.time_value, 4);
}

// ── Main update cycle ─────────────────────────────────────────────────────
let busy = false;

async function update() {
  if (busy) return;
  busy = true;
  setStatus('loading', 'Pricing…');

  try {
    // Three sensitivity sweeps + price all in parallel
    const [price, sensS, sensT, sensSigma] = await Promise.all([
      fetchPrice(),
      fetchSens('S'),
      fetchSens('T'),
      fetchSens('sigma'),
    ]);

    updateResults(price);

    // Price vs Spot  (price curve + payoff)
    refreshChart(
      CHARTS.priceVsSpot,
      [
        toXY(sensS.data, 'price'),
        sensS.data.map((d) => ({ x: d.x, y: d.payoff ?? 0 })),
      ],
      state.S,
    );

    // Price vs Volatility
    refreshChart(
      CHARTS.priceVsVol,
      [toXY(sensSigma.data, 'price')],
      state.sigma,
    );

    // Price vs Time
    refreshChart(
      CHARTS.priceVsTime,
      [toXY(sensT.data, 'price')],
      state.T,
    );

    // Delta vs Spot
    refreshChart(
      CHARTS.deltaVsSpot,
      [toXY(sensS.data, 'delta')],
      state.S,
    );

    // Gamma vs Spot
    refreshChart(
      CHARTS.gammaVsSpot,
      [toXY(sensS.data, 'gamma')],
      state.S,
    );

    // Vega vs Volatility
    refreshChart(
      CHARTS.vegaVsVol,
      [toXY(sensSigma.data, 'vega')],
      state.sigma,
    );

    setStatus('ok', 'Live');
  } catch (err) {
    console.error('Pricing error:', err);
    setStatus('error', 'Error');
  } finally {
    busy = false;
  }
}

const debouncedUpdate = debounce(update, 250);

// ── Input wiring ──────────────────────────────────────────────────────────
/**
 * @param {string}        sliderId   - range input id
 * @param {string}        numId      - number input id
 * @param {string}        key        - state key
 * @param {function|null} fromInput  - converts displayed value → internal state value
 */
function wireInput(sliderId, numId, key, fromInput = null) {
  const slider = el(sliderId);
  const num    = el(numId);
  if (!slider || !num) return;

  const toState = (v) => (fromInput ? fromInput(v) : v);

  slider.addEventListener('input', () => {
    const val = parseFloat(slider.value);
    state[key] = toState(val);
    num.value  = val;
    debouncedUpdate();
  });

  num.addEventListener('change', () => {
    let val = parseFloat(num.value);
    if (isNaN(val)) return;
    // clamp to slider range
    val = Math.min(Math.max(val, parseFloat(slider.min)), parseFloat(slider.max));
    num.value    = val;
    slider.value = val;
    state[key]   = toState(val);
    debouncedUpdate();
  });
}

function setupControls() {
  // S, K, T: slider value = internal value
  wireInput('sliderS', 'inputS', 'S');
  wireInput('sliderK', 'inputK', 'K');
  wireInput('sliderT', 'inputT', 'T');

  // r, sigma: slider/input in %, state in decimal
  wireInput('sliderR',     'inputR',     'r',     (v) => v / 100);
  wireInput('sliderSigma', 'inputSigma', 'sigma', (v) => v / 100);

  // Option type toggle
  document.querySelectorAll('[data-option-type]').forEach((btn) => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('[data-option-type]').forEach((b) => b.classList.remove('active'));
      btn.classList.add('active');
      state.option_type = btn.dataset.optionType;
      debouncedUpdate();
    });
  });
}

// ── Boot ──────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  initCharts();
  setupControls();
  update();          // initial price on load
});
