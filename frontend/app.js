/* ─── CarbonPilot AI — Frontend JS ───────────────────────────────────────── */

const API = 'http://localhost:8000/api';

let state = {
  priority: 'balanced',
  analysisData: null,
  selectedModel: null,
  impactChart: null,
};

/* ─── Toast ──────────────────────────────────────────────────────────────── */
function toast(msg, type = 'info', duration = 3000) {
  const container = document.getElementById('toast-container');
  const el = document.createElement('div');
  el.className = `toast ${type}`;
  const icon = type === 'success' ? '✅' : type === 'error' ? '❌' : 'ℹ️';
  el.innerHTML = `<span>${icon}</span><span>${msg}</span>`;
  container.appendChild(el);
  setTimeout(() => {
    el.style.animation = 'toastOut .3s ease both';
    setTimeout(() => el.remove(), 300);
  }, duration);
}

/* ─── Copy ───────────────────────────────────────────────────────────────── */
function copyText(elId) {
  const el = document.getElementById(elId);
  const text = el.textContent || el.innerText;
  navigator.clipboard.writeText(text).then(() => {
    toast('Copied to clipboard!', 'success', 2000);
  }).catch(() => {
    toast('Could not copy.', 'error', 2000);
  });
}

/* ─── Clear prompt ───────────────────────────────────────────────────────── */
function clearPrompt() {
  document.getElementById('prompt-input').value = '';
  updateMeta('');
}

function scrollToTop() {
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

/* ─── Token counter live ─────────────────────────────────────────────────── */
function updateMeta(text) {
  const chars = text.length;
  const tokens = Math.max(0, Math.ceil(chars / 4));
  document.getElementById('char-count').textContent = `${chars} chars`;
  document.getElementById('token-estimate').textContent = `~${tokens} tokens`;
}

document.getElementById('prompt-input').addEventListener('input', (e) => {
  updateMeta(e.target.value);
});

/* ─── Priority buttons ───────────────────────────────────────────────────── */
document.getElementById('priority-btns').addEventListener('click', (e) => {
  const btn = e.target.closest('.priority-btn');
  if (!btn) return;
  document.querySelectorAll('.priority-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  state.priority = btn.dataset.value;
});

/* ─── Helpers ────────────────────────────────────────────────────────────── */
function show(id) { document.getElementById(id).classList.remove('hidden'); }
function hide(id) { document.getElementById(id).classList.add('hidden'); }

function setLoading(btnId, spinnerId, textId, loading, label) {
  const btn = document.getElementById(btnId);
  document.getElementById(spinnerId).classList.toggle('hidden', !loading);
  document.getElementById(textId).textContent = loading ? 'Processing…' : label;
  btn.disabled = loading;
}

function fmt(num, decimals = 6) {
  if (num === 0) return '0';
  if (Math.abs(num) < 0.000001) return num.toExponential(2);
  return parseFloat(num.toFixed(decimals)).toString();
}

/* ─── Animate counter ────────────────────────────────────────────────────── */
function animateNum(el, from, to, duration = 700, suffix = '') {
  const start = performance.now();
  const update = (now) => {
    const t = Math.min((now - start) / duration, 1);
    const eased = 1 - Math.pow(1 - t, 3);
    const val = from + (to - from) * eased;
    el.textContent = (Number.isInteger(to) ? Math.round(val) : parseFloat(val.toFixed(4))) + suffix;
    if (t < 1) requestAnimationFrame(update);
  };
  requestAnimationFrame(update);
}

/* ─── Donut chart ────────────────────────────────────────────────────────── */
function renderChart(originalTokens, optimizedTokens) {
  const ctx = document.getElementById('impact-chart');
  if (!ctx) return;
  if (state.impactChart) {
    state.impactChart.destroy();
    state.impactChart = null;
  }
  const saved = Math.max(0, originalTokens - optimizedTokens);
  state.impactChart = new Chart(ctx, {
    type: 'doughnut',
    data: {
      labels: ['Optimized', 'Saved'],
      datasets: [{
        data: [optimizedTokens, saved],
        backgroundColor: ['rgba(34,197,94,.85)', 'rgba(56,189,248,.5)'],
        borderColor: ['#22c55e', '#38bdf8'],
        borderWidth: 1,
        hoverOffset: 4,
      }]
    },
    options: {
      responsive: false,
      cutout: '68%',
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: ctx => ` ${ctx.label}: ${ctx.parsed} tokens`
          }
        }
      },
      animation: { animateRotate: true, duration: 800 }
    }
  });
}

/* ─── Analyze Prompt ─────────────────────────────────────────────────────── */
async function analyzePrompt() {
  const prompt = document.getElementById('prompt-input').value.trim();
  if (!prompt) { toast('Please enter a prompt first.', 'error'); return; }
  if (prompt.length > 8000) { toast('Prompt too long (max 8000 chars).', 'error'); return; }

  setLoading('analyze-btn', 'analyze-spinner', 'analyze-btn-text', true, '🔍 Analyze & Optimize');
  hide('section-analysis');
  hide('section-recommendation');
  hide('section-response');

  try {
    const resp = await fetch(`${API}/analyze`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ prompt, priority: state.priority }),
    });
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({}));
      throw new Error(err.detail || `Server error ${resp.status}`);
    }
    const data = await resp.json();
    state.analysisData = data;
    state.selectedModel = data.recommended_model;
    renderAnalysis(data);
    renderRecommendation(data);
    show('section-analysis');
    show('section-recommendation');
    setTimeout(() => {
      document.getElementById('section-analysis').scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 100);
    toast('Analysis complete!', 'success');
  } catch (e) {
    toast('Error: ' + e.message, 'error', 5000);
    console.error(e);
  } finally {
    setLoading('analyze-btn', 'analyze-spinner', 'analyze-btn-text', false, '🔍 Analyze & Optimize');
  }
}

/* ─── Render Analysis ────────────────────────────────────────────────────── */
function renderAnalysis(data) {
  // Issues
  const issuesBox = document.getElementById('issues-box');
  const isOk = data.issues.length === 1 && data.issues[0].toLowerCase().includes('no major');
  issuesBox.className = `issues-box${isOk ? ' issues-ok' : ''}`;
  const icon = isOk ? '✅' : '⚠';
  issuesBox.innerHTML = `
    <div class="issues-title">${icon} ${isOk ? 'Prompt Analysis' : 'Detected Issues'}</div>
    <ul>${data.issues.map(i => `<li>${escHtml(i)}</li>`).join('')}</ul>`;

  // Prompts
  document.getElementById('orig-prompt-text').textContent = data.original_prompt;
  document.getElementById('opt-prompt-text').textContent = data.optimized_prompt;
  document.getElementById('orig-token-badge').textContent = `${data.original_tokens} tokens`;
  document.getElementById('opt-token-badge').textContent =
    `${data.optimized_tokens} tokens  (−${data.token_reduction_pct}%)`;
  document.getElementById('compare-reduction').textContent =
    data.token_reduction_pct > 0 ? `−${data.token_reduction_pct}%` : '±0%';

  // Similarity bar
  const sim = data.semantic_similarity;
  const simBar = document.getElementById('sim-bar');
  simBar.style.width = '0%';
  setTimeout(() => { simBar.style.width = `${sim}%`; }, 100);
  document.getElementById('sim-value').textContent = `${sim}%`;

  // Metrics
  const om = data.original_metrics;
  const pm = data.optimized_metrics;
  const metrics = [
    { icon: '🔢', label: 'Tokens', before: om.tokens, after: pm.tokens, unit: '', raw: true },
    { icon: '⚡', label: 'Energy (Wh)', before: fmt(om.energy_wh), after: fmt(pm.energy_wh), unit: ' Wh' },
    { icon: '💨', label: 'CO₂ (g)', before: fmt(om.co2_g, 5), after: fmt(pm.co2_g, 5), unit: ' g' },
    { icon: '💧', label: 'Water (ml)', before: fmt(om.water_ml, 4), after: fmt(pm.water_ml, 4), unit: ' ml' },
    { icon: '₹', label: 'Cost (INR)', before: `₹${fmt(om.cost_inr, 5)}`, after: `₹${fmt(pm.cost_inr, 5)}`, unit: '' },
  ];
  document.getElementById('metrics-grid').innerHTML = metrics.map(m => `
    <div class="metric-card">
      <div class="metric-icon">${m.icon}</div>
      <div class="metric-label">${m.label}</div>
      <div class="metric-before">Before: ${m.before}${m.unit}</div>
      <div class="metric-after">After: ${m.after}${m.unit}</div>
    </div>`).join('');

  // Chart
  renderChart(data.original_tokens, data.optimized_tokens);

  // Savings
  const s = data.savings;
  const savingItems = [
    { label: 'Tokens Saved', value: `−${s.tokens} (${s.tokens_pct}%)` },
    { label: 'CO₂ Saved', value: `−${fmt(s.co2_g, 5)} g` },
    { label: 'Energy Saved', value: `−${fmt(s.energy_wh, 6)} Wh` },
    { label: 'Water Saved', value: `−${fmt(s.water_ml, 4)} ml` },
    { label: 'Task Type', value: cap(data.task_complexity) },
  ];
  document.getElementById('savings-banner').innerHTML = savingItems.map((item, i) => `
    ${i > 0 ? '<div class="saving-divider"></div>' : ''}
    <div class="saving-item">
      <div class="saving-label">${item.label}</div>
      <div class="saving-value">${item.value}</div>
    </div>`).join('');
}

/* ─── Render Recommendation & Model Selector ─────────────────────────────── */
function renderRecommendation(data) {
  // 1. AI Recommendation Box
  const reasonText = data.recommendation_reason || 'Optimized for current priority balance and complexity.';
  document.getElementById('rec-box').innerHTML = `
    <div class="rec-icon">🤖</div>
    <div style="flex:1">
      <div class="rec-model-label">Algorithmically Selected Best Model</div>
      <div class="rec-model-name">${escHtml(data.recommended_model_label)}</div>
      <div class="rec-reason"><strong>Selection Proof:</strong> ${escHtml(reasonText)}</div>
      <div class="rec-tags">
        <span class="rec-tag highlight">Priority: ${cap(state.priority)}</span>
        <span class="rec-tag">Complexity: ${cap(data.task_complexity)}</span>
        <span class="rec-tag">Task: ${cap(data.task_type || 'general')}</span>
        <span class="rec-tag">Free Tier ✓</span>
      </div>
    </div>`;

  // 2. Multi-Model Footprint Comparison Grid (Before vs After)
  const compGrid = document.getElementById('comparison-cards-grid');
  if (compGrid && Array.isArray(data.model_comparisons) && data.model_comparisons.length > 0) {
    compGrid.innerHTML = data.model_comparisons.map(item => {
      const isWinner = item.is_recommended;
      const isSelected = item.model_key === state.selectedModel;
      const b = item.before;
      const a = item.after;
      const s = item.savings;

      return `
        <div class="comparison-card ${isWinner ? 'winner' : ''} ${isSelected ? 'selected' : ''}"
             id="comp-card-${safeId(item.model_key)}"
             onclick="selectModel('${escAttr(item.model_key)}')">
          ${isWinner ? '<span class="comp-card-badge">🏆 Best Selection</span>' : ''}
          <div class="comp-card-top">
            <div class="comp-name">${escHtml(item.label)}</div>
            <div class="comp-score">Score: ${item.composite_score}</div>
          </div>
          <div class="comp-arch">${escHtml(item.architecture)} • ${escHtml(item.active_params)} active</div>
          <div class="comp-citation">${escHtml(item.benchmark_citation)}</div>
          
          <div class="comp-table">
            <div class="comp-row">
              <span class="comp-label">Tokens</span>
              <div class="comp-values">
                <span class="comp-before">${b.tokens}</span>
                <span class="comp-arrow">→</span>
                <span class="comp-after">${a.tokens}</span>
                <span class="comp-savings-tag">−${s.pct_reduction}%</span>
              </div>
            </div>
            <div class="comp-row">
              <span class="comp-label">Energy</span>
              <div class="comp-values">
                <span class="comp-before">${fmt(b.energy_wh, 4)} Wh</span>
                <span class="comp-arrow">→</span>
                <span class="comp-after">${fmt(a.energy_wh, 4)} Wh</span>
              </div>
            </div>
            <div class="comp-row">
              <span class="comp-label">Carbon (CO₂)</span>
              <div class="comp-values">
                <span class="comp-before">${fmt(b.co2_g, 4)} g</span>
                <span class="comp-arrow">→</span>
                <span class="comp-after">${fmt(a.co2_g, 4)} g</span>
              </div>
            </div>
            <div class="comp-row">
              <span class="comp-label">Water Cooling</span>
              <div class="comp-values">
                <span class="comp-before">${fmt(b.water_ml, 3)} ml</span>
                <span class="comp-arrow">→</span>
                <span class="comp-after">${fmt(a.water_ml, 3)} ml</span>
              </div>
            </div>
            <div class="comp-row">
              <span class="comp-label">Quality Score</span>
              <div class="comp-values">
                <span class="comp-after">${item.quality_score}/100</span>
              </div>
            </div>
          </div>
        </div>`;
    }).join('');
  }

  // 3. Model Selector Cards
  const registry = data.model_registry;
  document.getElementById('model-selector').innerHTML = Object.entries(registry).map(([key, info]) => {
    const isRec = key === data.recommended_model;
    const isSel = key === state.selectedModel;
    return `
      <div class="model-card ${isSel ? 'selected' : ''} ${isRec ? 'recommended' : ''}"
           id="mcard-${safeId(key)}"
           onclick="selectModel('${escAttr(key)}')">
        <div class="model-card-name">${escHtml(info.label)}</div>
        <div class="model-card-quality">Benchmark Quality: ${info.quality_score}/100</div>
        <div class="quality-bar">
          <div class="quality-fill" style="width:${info.quality_score}%"></div>
        </div>
      </div>`;
  }).join('');
}

function selectModel(key) {
  state.selectedModel = key;
  document.querySelectorAll('.model-card').forEach(c => c.classList.remove('selected'));
  document.querySelectorAll('.comparison-card').forEach(c => c.classList.remove('selected'));
  const mcard = document.getElementById(`mcard-${safeId(key)}`);
  if (mcard) mcard.classList.add('selected');
  const compcard = document.getElementById(`comp-card-${safeId(key)}`);
  if (compcard) compcard.classList.add('selected');
}

/* ─── Run Prompt ─────────────────────────────────────────────────────────── */
async function runPrompt() {
  if (!state.analysisData) return;
  if (!state.selectedModel) { toast('Please select a model.', 'error'); return; }

  setLoading('run-btn', 'run-spinner', 'run-btn-text', true, '🚀 Run Optimized Prompt');
  hide('section-response');

  const promptToRun = state.analysisData.optimized_prompt;

  try {
    const resp = await fetch(`${API}/run`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ prompt: promptToRun, model_key: state.selectedModel }),
    });
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({}));
      throw new Error(err.detail || `Server error ${resp.status}`);
    }
    const data = await resp.json();
    renderResponse(data);
    show('section-response');
    setTimeout(() => {
      document.getElementById('section-response').scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 100);
    toast('Response received!', 'success');
  } catch (e) {
    toast('Error: ' + e.message, 'error', 5000);
    console.error(e);
  } finally {
    setLoading('run-btn', 'run-spinner', 'run-btn-text', false, '🚀 Run Optimized Prompt');
  }
}

/* ─── Render Response ────────────────────────────────────────────────────── */
function renderResponse(data) {
  document.getElementById('response-model-label').textContent = `Response from ${data.model_label}`;

  document.getElementById('response-meta').innerHTML = `
    <span>Model: <strong>${escHtml(data.model_label)}</strong></span>
    <span>Prompt tokens: <strong>${data.prompt_tokens}</strong></span>
    <span>Response tokens: <strong>${data.completion_tokens}</strong></span>
    <span>Total: <strong>${data.total_tokens}</strong></span>`;

  document.getElementById('response-box').textContent = data.response;

  const m = data.metrics;
  document.getElementById('response-metrics').innerHTML = `
    <div class="resp-metric">⚡ Energy: <strong>${fmt(m.energy_wh, 6)} Wh</strong></div>
    <div class="resp-metric">💨 CO₂: <strong>${fmt(m.co2_g, 5)} g</strong></div>
    <div class="resp-metric">💧 Water: <strong>${fmt(m.water_ml, 4)} ml</strong></div>
    <div class="resp-metric">₹ Cost: <strong>₹${fmt(m.cost_inr, 5)}</strong></div>`;
}

/* ─── Reset ──────────────────────────────────────────────────────────────── */
function resetAll() {
  document.getElementById('prompt-input').value = '';
  updateMeta('');
  state.analysisData = null;
  state.selectedModel = null;
  if (state.impactChart) { state.impactChart.destroy(); state.impactChart = null; }
  hide('section-analysis');
  hide('section-recommendation');
  hide('section-response');
  window.scrollTo({ top: 0, behavior: 'smooth' });
  toast('Reset! Enter a new prompt.', 'info', 2000);
}

/* ─── Utils ──────────────────────────────────────────────────────────────── */
function escHtml(str) {
  return String(str).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}
function escAttr(str) { return escHtml(str); }
function safeId(str) { return btoa(str).replace(/[^a-zA-Z0-9]/g, ''); }
function cap(str) { return str ? str.charAt(0).toUpperCase() + str.slice(1) : str; }
