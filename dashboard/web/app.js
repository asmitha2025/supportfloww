// SupportMind Dashboard — app.js
// Interactive demo with real API calls (falls back to simulation if API unavailable)

let API_BASE = window.location.origin;
let apiOnline = false;

function apiCandidates() {
  const candidates = [];
  const url = new URL(window.location.href);
  if (url.hostname === '127.0.0.1' && url.port === '7860') {
    candidates.push('http://127.0.0.1:7862');
    candidates.push('http://127.0.0.1:7861');
  }
  candidates.push(window.location.origin);
  return [...new Set(candidates)];
}

// Category colors
const CAT_COLORS = {
  billing: '#fb923c', technical_support: '#8083ff', account_management: '#89ceff',
  feature_request: '#c0c1ff', compliance_legal: '#f87171', onboarding: '#4ade80',
  general_inquiry: '#94a3b8', churn_risk: '#facc15',
};

// ── Init ──────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  animateCounters();
  initPresets();
  initDropoutViz();
  initScrollAnimations();
  initSmoothScroll();
  checkAPI();
  updateMetrics();
  setInterval(updateMetrics, 5000); // Update every 5 seconds
});

// ── Counter Animation ─────────────────────────────────
function animateCounters() {
  document.querySelectorAll('.stat-card').forEach(card => {
    const counter = card.querySelector('.counter');
    const target = parseFloat(card.dataset.value);
    const duration = 1500;
    const start = performance.now();
    function update(now) {
      const elapsed = now - start;
      const progress = Math.min(elapsed / duration, 1);
      const eased = 1 - Math.pow(1 - progress, 3);
      counter.textContent = Math.round(target * eased * 10) / 10;
      if (progress < 1) requestAnimationFrame(update);
      else counter.textContent = target;
    }
    requestAnimationFrame(update);
  });
}

// ── Presets ────────────────────────────────────────────
// ── Live Telemetry Engine ───────────────────────────
async function updateMetrics() {
  try {
    const res = await fetch(`${API_BASE}/metrics`);
    if (!res.ok) return;
    const data = await res.json();
    
    // Update Counter
    document.getElementById('live-total').textContent = data.total_requests.toLocaleString();
    
    // Update Model Name
    document.getElementById('live-model').textContent = data.model;
    
    // Update Distribution Bar
    const dist = data.routing_distribution;
    document.getElementById('dist-route').style.width = `${dist.route_pct}%`;
    document.getElementById('dist-clarify').style.width = `${dist.clarify_pct}%`;
    document.getElementById('dist-escalate').style.width = `${dist.escalate_pct}%`;
    
    // Update Status Pulse
    const indicator = document.getElementById('live-indicator');
    indicator.style.opacity = '1';
    setTimeout(() => { indicator.style.opacity = '0.8'; }, 500);
    
  } catch (err) {
    console.warn("Metrics sync failed:", err);
  }
}

// ── Presets ────────────────────────────────────────────
function initPresets() {
  document.querySelectorAll('.preset-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.getElementById('ticket-input').value = btn.dataset.text;
    });
  });
}

function initSmoothScroll() {
  document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
      e.preventDefault();
      document.querySelector(this.getAttribute('href')).scrollIntoView({
        behavior: 'smooth'
      });
    });
  });
}

// ── MC Dropout Visualization ──────────────────────────
function initDropoutViz() {
  const grid = document.getElementById('dropout-grid');
  if (!grid) return;
  for (let pass = 0; pass < 20; pass++) {
    const col = document.createElement('div');
    col.className = 'dropout-col';
    for (let neuron = 0; neuron < 12; neuron++) {
      const cell = document.createElement('div');
      cell.className = 'dropout-cell';
      const active = Math.random() > 0.15;
      cell.style.background = active ? 'var(--primary)' : 'rgba(192, 193, 255, 0.05)';
      cell.style.border = active ? 'none' : '1px solid rgba(192, 193, 255, 0.1)';
      col.appendChild(cell);
    }
    grid.appendChild(col);
  }
  // Animate dropout
  setInterval(() => {
    grid.querySelectorAll('.dropout-cell').forEach(cell => {
      const active = Math.random() > 0.15;
      cell.style.background = active ? 'var(--primary)' : 'rgba(192, 193, 255, 0.05)';
      cell.style.border = active ? 'none' : '1px solid rgba(192, 193, 255, 0.1)';
    });
  }, 2000);
}

// ── Scroll Animations ─────────────────────────────────
function initScrollAnimations() {
  const observer = new IntersectionObserver((entries) => {
    entries.forEach(e => { if (e.isIntersecting) e.target.classList.add('visible'); });
  }, { threshold: 0.1 });
  document.querySelectorAll('.section-header, .stat-card, .arch-stage, .bench-card, .ops-card').forEach(el => {
    el.classList.add('fade-in');
    observer.observe(el);
  });
}

// ── API Check ─────────────────────────────────────────
async function checkAPI() {
  for (const candidate of apiCandidates()) {
    try {
      const res = await fetch(`${candidate}/health`, { signal: AbortSignal.timeout(2000) });
      if (!res.ok) continue;
      API_BASE = candidate;
      apiOnline = true;
      const statusEl = document.querySelector('.status-text');
      if (statusEl) statusEl.textContent = API_BASE === window.location.origin ? 'API Connected' : 'Fixed API Connected';
      return;
    } catch {
      // Try the next candidate before falling back to demo mode.
    }
  }
  apiOnline = false;
  const statusEl = document.querySelector('.status-text');
  if (statusEl) statusEl.textContent = 'Demo Mode';
}

// ── Live Metrics ──────────────────────────────────────
async function updateLiveMetrics() {
  if (!apiOnline) return;
  try {
    const res = await fetch(`${API_BASE}/metrics`);
    const data = await res.json();
    
    document.getElementById('live-model').textContent = data.model;
    document.getElementById('live-total').textContent = data.total_requests;
    
    const dist = data.routing_distribution;
    document.getElementById('dist-route').style.width = dist.route_pct + '%';
    document.getElementById('dist-clarify').style.width = dist.clarify_pct + '%';
    document.getElementById('dist-escalate').style.width = dist.escalate_pct + '%';
  } catch (err) {
    console.warn('Metrics update failed:', err);
  }
}

// ── Route Ticket ──────────────────────────────────────
async function routeTicket(extraPayload = {}) {
  const text = document.getElementById('ticket-input').value.trim();
  if (!text) return;

  const btn = document.getElementById('route-btn');
  btn.innerHTML = '<span class="spinner"></span> Routing...';
  btn.disabled = true;

  let result;
  try {
    if (apiOnline) {
      const res = await fetch(`${API_BASE}/route`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text, ...extraPayload }),
      });
      result = await res.json();
    } else {
      result = simulateRouting(text, extraPayload);
    }
    displayResult(result, text);
  } catch (err) {
    result = simulateRouting(text, extraPayload);
    displayResult(result, text);
  }

  btn.innerHTML = '<span class="btn-icon">⚡</span> Route Ticket';
  btn.disabled = false;
}

// ── Display Result ────────────────────────────────────
function displayResult(r, routedText) {
  // Handle edge cases
  if (r.action === 'invalid_input') {
    document.getElementById('result-placeholder').style.display = 'none';
    const content = document.getElementById('result-content');
    content.style.display = 'block';

    const badge = document.getElementById('action-badge');
    badge.textContent = r.error_type.toUpperCase().replace('_', ' ');
    badge.className = 'action-badge clarify'; // yellow

    document.getElementById('action-queue').textContent = r.response;
    document.getElementById('result-reason').textContent = r.response;

    // Hide gauges for invalid input
    document.querySelector('.gauge-row').style.display = 'none';
    document.querySelector('.signals-grid').style.display = 'none';
    document.getElementById('prob-chart').innerHTML = '';
    document.getElementById('clarification-box').style.display = 'none';
    const evidenceGrid = document.getElementById('signal-evidence-grid');
    if (evidenceGrid) evidenceGrid.style.display = 'none';
    const explainBtn = document.getElementById('explain-btn');
    if (explainBtn) explainBtn.style.display = 'none';
    document.getElementById('explanation-box').style.display = 'none';
    return;
  }


  // Show gauges for valid input
  document.querySelector('.gauge-row').style.display = 'grid';
  document.querySelector('.signals-grid').style.display = 'grid';

  document.getElementById('result-placeholder').style.display = 'none';
  const content = document.getElementById('result-content');
  content.style.display = 'block';
  const evidenceGrid = document.getElementById('signal-evidence-grid');
  if (evidenceGrid) evidenceGrid.style.display = 'grid';

  // Action Badge Logic
  const badge = document.getElementById('action-badge');
  const queue = document.getElementById('action-queue');
  
  if (r.action === 'multi_route') {
    badge.textContent = 'MULTI-ROUTE';
    badge.className = 'action-badge';
    badge.style.background = 'linear-gradient(90deg, var(--primary), var(--accent))';
    queue.innerHTML = `
      <div style="display: flex; gap: 8px; margin-top: 4px;">
        <span class="tech-tag" style="background: rgba(192, 193, 255, 0.2)">Primary: ${r.primary_queue}</span>
        <span class="tech-tag" style="background: rgba(255, 255, 255, 0.1)">Secondary: ${r.secondary_queue}</span>
      </div>
    `;
  } else {
    badge.textContent = r.action.toUpperCase();
    badge.className = `action-badge ${r.action}`;
    queue.textContent = r.action === 'route' ? `→ ${r.queue || r.top_category} queue` : 
                         r.action === 'clarify' ? 'Needs 1 clarification question' : 'Immediate human triage';
  }

  // Gauges
  const confPct = Math.min(r.confidence * 100, 100);
  document.getElementById('conf-fill').style.width = confPct + '%';
  document.getElementById('conf-value').textContent = r.confidence.toFixed(4);
  const maxEnt = Math.log(8);
  const entPct = Math.min((r.entropy / maxEnt) * 100, 100);
  document.getElementById('ent-fill').style.width = entPct + '%';
  document.getElementById('ent-value').textContent = r.entropy.toFixed(4);
  if (r.margin !== undefined && document.getElementById('margin-value')) {
    document.getElementById('margin-value').textContent = r.margin.toFixed(4);
  }


  // Prob chart
  const chart = document.getElementById('prob-chart');
  chart.innerHTML = '';
  const probs = r.all_probs || {};
  const sorted = Object.entries(probs).sort((a, b) => b[1] - a[1]);
  const maxProb = sorted.length ? sorted[0][1] : 1;
  sorted.forEach(([cat, prob]) => {
    const row = document.createElement('div');
    row.className = 'prob-row';
    const pct = (prob / Math.max(maxProb, 0.01)) * 100;
    row.innerHTML = `
      <span class="prob-label">${cat.replace(/_/g, ' ')}</span>
      <div class="prob-bar-track"><div class="prob-bar-fill" style="width:${pct}%;background:${CAT_COLORS[cat] || '#6366f1'}"></div></div>
      <span class="prob-val">${(prob * 100).toFixed(1)}%</span>`;
    chart.appendChild(row);
  });

  // Clarification
  const clarBox = document.getElementById('clarification-box');
  if (r.action === 'clarify' && r.clarification) {
    clarBox.style.display = 'block';
    document.getElementById('clarify-question').textContent = r.clarification.question_text;
    const optEl = document.getElementById('clarify-options');
    optEl.innerHTML = '';
    const optionTargets = r.clarification.option_targets || [];
    (r.clarification.options || []).forEach((o, index) => {
      const btn = document.createElement('button');
      btn.className = 'option-btn';
      btn.textContent = o;
      btn.onclick = () => {
        // Provide visual feedback
        document.querySelectorAll('#clarify-options .option-btn').forEach(b => b.disabled = true);
        btn.style.background = 'var(--primary)';
        btn.style.color = '#fff';

        const target = optionTargets[index]
          || inferClarificationTarget(o, r.clarification.relevant_classes || r.top_two_classes || [], index);
        
        // Keep the selection visible and machine-readable for repeat manual runs.
        const input = document.getElementById('ticket-input');
        input.value = input.value.trim() + '\n\n[Clarification: ' + target + ' - ' + o + ']';
        
        // Re-route with new context after a short delay
        setTimeout(() => {
          routeTicket({
            clarification_choice: o,
            clarification_target: target,
            clarification_question_id: r.clarification.question_id,
          });
        }, 800);
      };
      optEl.appendChild(btn);
    });

    // Remove existing badge if any
    const existingBadge = document.getElementById('source-badge');
    if (existingBadge) existingBadge.remove();

    // After displaying the question, add source badge
    const sourceBadge = document.createElement('div');
    sourceBadge.id = 'source-badge';
    sourceBadge.style.cssText = 'font-size:11px;margin-top:8px;opacity:0.6;';
    sourceBadge.textContent = r.clarification.source === 'llm_groq' 
        ? '⚡ Generated by LLaMA3 via Groq' 
        : '📋 Selected from template bank';
    document.getElementById('clarification-box').appendChild(sourceBadge);

    document.getElementById('clarify-gain').textContent =
      `Expected information gain: ${r.clarification.expected_gain?.toFixed(4) || 'N/A'}`;
  } else {
    clarBox.style.display = 'none';
  }

  // Signals
  const slaRiskVal = r.sla_risk || r.sla_breach_probability || 0;
  const slaPct = slaRiskVal * 100;
  document.getElementById('sla-value').textContent = slaPct.toFixed(1) + '%';
  document.getElementById('sla-fill').style.width = slaPct + '%';
  document.getElementById('sla-fill').style.background =
    slaPct > 65 ? 'var(--red)' : slaPct > 35 ? 'var(--yellow)' : 'var(--green)';

  const feat = r.features || {};
  const sent = feat.sentiment_score;
  const sentLabel = feat.sentiment_label || sentimentLabelFromScore(sent);
  const sentimentValue = document.getElementById('sentiment-value');
  sentimentValue.textContent = sentLabel ? sentLabel.toUpperCase() : '—';
  sentimentValue.style.color = sentimentColor(sentLabel, sent);
  const sentimentScore = document.getElementById('sentiment-score');
  if (sentimentScore) {
    const raw = typeof feat.sentiment_raw_score === 'number'
      ? ` raw ${feat.sentiment_raw_score.toFixed(2)}`
      : '';
    sentimentScore.textContent = sent !== undefined ? `score ${sent.toFixed(2)}${raw}` : '—';
  }
  
  const urgScore = numericValue(r.urgency_score, feat.urgency_score, 0);
  const urgLevel = feat.urgency_level || urgencyLevelFromScore(urgScore);
  const urgencyCard = document.getElementById('urgency-value').parentElement;
  const urgencyValue = document.getElementById('urgency-value');
  urgencyValue.textContent = urgLevel.toUpperCase();
  urgencyValue.style.color = urgencyColor(urgLevel);
  const urgencyScore = document.getElementById('urgency-score');
  if (urgencyScore) urgencyScore.textContent = `score ${urgScore.toFixed(2)}`;

  if (urgLevel === 'critical') {
    urgencyCard.style.border = '1px solid var(--red)';
    urgencyCard.style.boxShadow = '0 0 15px rgba(248, 113, 113, 0.2)';
  } else if (urgLevel === 'high') {
    urgencyCard.style.border = '1px solid var(--yellow)';
    urgencyCard.style.boxShadow = '';
  } else {
    urgencyCard.style.border = '';
    urgencyCard.style.boxShadow = '';
  }

  renderEvidenceList('urgency-evidence-list', feat.urgency_evidence || []);
  renderEvidenceList('sentiment-evidence-list', feat.sentiment_evidence || []);

  document.getElementById('latency-value').textContent =
    r.latency_ms ? r.latency_ms + 'ms' : '—';

  // Reason
  let decisionReason = '';
  if (r.clarification_applied) {
    decisionReason = `Clarification answer applied: <strong>${escapeHtml(r.clarification_choice || r.top_category)}</strong>. Routing to <strong>${r.top_category}</strong> without asking another question.`;
  } else if (r.action === 'multi_route') {
    decisionReason = `Multiple distinct intents detected in the request. Primary intent is <strong>${r.primary_queue}</strong>, secondary is <strong>${r.secondary_queue}</strong>.`;
  } else if (r.action === 'clarify') {
    decisionReason = `Model uncertainty is high (entropy: ${r.entropy.toFixed(3)}) or the top two classes are too close (margin: ${r.margin?.toFixed(3)}). A clarification question was generated to refine the intent.`;
  } else if (r.action === 'escalate') {
    decisionReason = `Low model confidence detected (${(r.confidence * 100).toFixed(1)}%). Routing directly to human experts to ensure accuracy.`;
  } else {
    decisionReason = `High-confidence intent detected: <strong>${r.top_category}</strong>. Automatically routing to specialized queue.`;
  }

  document.getElementById('result-reason').innerHTML = `
    <div style="padding: 12px; background: rgba(192, 193, 255, 0.05); border: 1px solid rgba(192, 193, 255, 0.1); border-radius: 8px; margin-top: 16px;">
      <div style="font-size: 11px; text-transform: uppercase; color: var(--primary); margin-bottom: 8px; font-weight: 600;">Decision Reason</div>
      <div style="font-size: 13px; color: var(--on-surface-variant); line-height: 1.5;">${decisionReason}</div>
    </div>
  `;

  // Show explain button for valid input
  const explainBtn = document.getElementById('explain-btn');
  if (explainBtn) {
    explainBtn.style.display = 'flex';
    explainBtn.dataset.text = routedText || document.getElementById('ticket-input').value;
    explainBtn.dataset.category = r.top_category;
  }
  document.getElementById('explanation-box').style.display = 'none';
}

// ── Explain Decision (SHAP) ───────────────────────────
async function explainDecision() {
  const btn = document.getElementById('explain-btn');
  const text = btn.dataset.text;
  const targetClass = btn.dataset.category;
  
  btn.innerHTML = '<span class="spinner"></span> Analyzing tokens...';
  btn.disabled = true;

  try {
    let result;
    if (apiOnline) {
      const res = await fetch(`${API_BASE}/explain`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text, target_class: targetClass }),
      });
      if (!res.ok) throw new Error(`Explain API returned ${res.status}`);
      result = await res.json();
    } else {
      // Simulate SHAP for demo mode
      result = simulateSHAP(text);
    }

    renderSHAP(result);
  } catch (err) {
    console.error('SHAP failed:', err);
    renderSHAP(simulateSHAP(text));
  }

  btn.innerHTML = '<span class="material-symbols-outlined btn-icon">query_stats</span> Analyze Decision';
  btn.disabled = false;
}

function renderSHAP(data) {
  const box = document.getElementById('explanation-box');
  const textEl = document.getElementById('explain-text');
  box.style.display = 'block';
  textEl.innerHTML = '';

  if (data.error) {
    textEl.textContent = 'Error generating explanation: ' + data.error;
    return;
  }

  const source = document.createElement('div');
  source.className = 'explain-source';
  source.textContent = data.source === 'shap_transformer'
    ? 'Transformer SHAP explanation'
    : 'Keyword evidence fallback';
  if (data.note) source.title = data.note;
  textEl.appendChild(source);

  const tokens = data.tokens || [];
  const values = data.values || [];

  tokens.forEach((token, i) => {
    const val = values[i];
    const span = document.createElement('span');
    span.className = 'shap-token';
    span.textContent = token.replace('##', ''); // Simple handling for subwords
    
    // Normalize opacity based on value
    const absVal = Math.abs(val);
    const opacity = Math.min(absVal * 5, 0.8); // Scale for visibility
    
    if (val > 0) {
      span.style.background = `rgba(74, 222, 128, ${opacity})`;
      span.style.borderBottom = `2px solid rgba(74, 222, 128, ${opacity + 0.2})`;
    } else if (val < 0) {
      span.style.background = `rgba(248, 113, 113, ${opacity})`;
      span.style.borderBottom = `2px solid rgba(248, 113, 113, ${opacity + 0.2})`;
    }

    textEl.appendChild(span);
    textEl.appendChild(document.createTextNode(' '));
  });
  
  box.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function simulateSHAP(text) {
  const tokens = text.split(/\s+/);
  const values = tokens.map(() => (Math.random() - 0.4) * 0.2);
  return { tokens, values, source: 'demo_simulated' };
}

function numericValue(...values) {
  for (const value of values) {
    if (typeof value === 'number' && Number.isFinite(value)) return value;
  }
  return 0;
}

function sentimentLabelFromScore(score) {
  if (typeof score !== 'number') return null;
  if (score <= -0.55) return 'frustrated';
  if (score <= -0.2) return 'concerned';
  if (score >= 0.3) return 'positive';
  return 'neutral';
}

function sentimentColor(label, score) {
  const normalized = (label || sentimentLabelFromScore(score) || '').toLowerCase();
  if (normalized === 'frustrated') return 'var(--red)';
  if (normalized === 'concerned') return 'var(--yellow)';
  if (normalized === 'positive') return 'var(--green)';
  return 'var(--text)';
}

function urgencyLevelFromScore(score) {
  if (score >= 0.75) return 'critical';
  if (score >= 0.5) return 'high';
  if (score >= 0.25) return 'medium';
  return 'low';
}

function urgencyColor(level) {
  const normalized = (level || '').toLowerCase();
  if (normalized === 'critical') return 'var(--red)';
  if (normalized === 'high' || normalized === 'medium') return 'var(--yellow)';
  return 'var(--green)';
}

function escapeHtml(value) {
  return String(value)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

function renderEvidenceList(elementId, evidence) {
  const list = document.getElementById(elementId);
  if (!list) return;

  const items = Array.isArray(evidence) ? evidence.filter(Boolean) : [];
  if (!items.length) {
    list.innerHTML = '<div class="evidence-empty">No contextual evidence triggered.</div>';
    return;
  }

  list.innerHTML = items.slice(0, 5).map(item => {
    const [rawType, ...phraseParts] = String(item).split(':');
    const type = rawType ? rawType.replace(/_/g, ' ') : 'signal';
    const phrase = phraseParts.join(':').trim() || item;
    return `
      <div class="evidence-item">
        <span class="evidence-type">${escapeHtml(type)}</span>
        <span class="evidence-phrase">${escapeHtml(phrase)}</span>
      </div>
    `;
  }).join('');
}

function inferClarificationTarget(option, relevantClasses, index) {
  const optionLow = String(option || '').toLowerCase();
  const keywordTargets = [
    ['billing', ['billing', 'invoice', 'payment', 'charge', 'refund', 'credit', 'pricing', 'cost', 'bill']],
    ['technical_support', ['software', 'error', 'technical', 'broken', 'malfunction', 'functionality', 'api', 'integration', 'performance', 'specific issue', 'data movement']],
    ['account_management', ['account', 'plan', 'subscription', 'administrator', 'admin', 'user management', 'regular user', 'settings']],
    ['feature_request', ['new capability', 'feature', 'request', 'enhancement']],
    ['compliance_legal', ['compliance', 'regulatory', 'audit', 'gdpr', 'security', 'data affected']],
    ['onboarding', ['new user', 'onboarding', 'guidance', 'training', 'walkthrough', 'setting up']],
    ['churn_risk', ['continuing', 'switching', 'evaluating options', 'mostly negative']],
    ['general_inquiry', ['general', 'guidance', 'not urgent', 'no specific deadline', 'positive']],
  ];

  for (const [category, keywords] of keywordTargets) {
    if (keywords.some(keyword => optionLow.includes(keyword))) return category;
  }
  return relevantClasses[index] || relevantClasses[0] || 'general_inquiry';
}

function firstPatternHit(text, patterns) {
  for (const pattern of patterns) {
    const match = text.match(pattern);
    if (match) return match[0];
  }
  return null;
}

function inferDemoSignals(t) {
  const urgencySpecs = [
    ['business_impact', 0.30, [
      /\b(?:affecting|impacting|blocking)\s+(?:our\s+)?(?:customers|users|team|business|operations|sales|revenue|payroll|launch|production)\b/,
      /\b(?:customers?|clients?)\s+(?:(?:are|is)\s+)?(?:waiting|blocked|affected|unable)\b/,
      /\b(?:cannot|can't|unable to)\s+(?:process|ship|launch|serve|sell|invoice|onboard|work|access)\b/,
    ]],
    ['deadline_pressure', 0.25, [
      /\b(?:in|within)\s+\d+\s*(?:min|mins|minutes|hour|hours|hrs|days?)\b/,
      /\b(?:by|before)\s+(?:today|tomorrow|eod|end of day|tonight|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b/,
      /\b(?:launch|demo|go-live|renewal|payroll|board meeting|presentation)\b/,
    ]],
    ['production_outage', 0.40, [
      /\bproduction\s+(?:is\s+)?(?:down|blocked|broken|failing|impacted)\b/,
      /\b(?:all|multiple|many)\s+(?:users|customers|accounts|teams)\s+(?:are\s+)?(?:affected|blocked|down|unable)\b/,
      /\b(?:system|service|platform|dashboard|api)\s+(?:is\s+)?(?:down|unavailable|not responding)\b/,
    ]],
    ['access_loss', 0.25, [
      /\b(?:locked out|cannot access|can't access|unable to access|access is blocked)\b/,
      /\b(?:login|sso|authentication)\s+(?:is\s+)?(?:broken|failing|down|not working)\b/,
    ]],
    ['repeat_issue', 0.20, [
      /\b(?:again|still|keeps?|repeated|recurring)\b/,
      /\b(?:second|third|fourth)\s+time\b/,
      /\b(?:raised|reported|opened)\s+(?:this\s+)?(?:before|multiple times|again)\b/,
    ]],
  ];

  const explicitCritical = ['crash', 'blocked', 'down', 'failing', 'cannot access', 'production issue', 'outage', 'emergency', 'critical', 'urgent', 'immediately', 'blocking', 'locked out'];
  const explicitGeneral = ['asap', 'deadline', 'sla', 'escalate', 'priority', 'time-sensitive', 'showstopper', 'presentation'];

  const urgencyEvidence = [];
  const urgencyFlags = [];
  explicitCritical.forEach(word => {
    if (t.includes(word)) {
      urgencyEvidence.push(`explicit_critical: ${word}`);
      urgencyFlags.push(word);
    }
  });
  explicitGeneral.forEach(word => {
    if (t.includes(word)) {
      urgencyEvidence.push(`explicit_general: ${word}`);
      urgencyFlags.push(word);
    }
  });

  let urgencyScore = (explicitCritical.length ? 0 : 0);
  urgencyScore += explicitCritical.filter(word => t.includes(word)).length * 0.25;
  urgencyScore += explicitGeneral.filter(word => t.includes(word)).length * 0.12;
  urgencySpecs.forEach(([label, weight, patterns]) => {
    const phrase = firstPatternHit(t, patterns);
    if (phrase) {
      urgencyScore += weight;
      urgencyEvidence.push(`${label}: ${phrase}`);
      urgencyFlags.push(label);
    }
  });

  if (/\b(?:not urgent|no rush|whenever you can|when you have time)\b/.test(t)) {
    urgencyScore = Math.min(urgencyScore, 0.35);
    urgencyEvidence.push('deescalation: no immediate pressure');
  }

  urgencyScore = Math.max(0, Math.min(1, urgencyScore));

  const sentimentSpecs = [
    ['frustration', -0.30, [
      /\bfrustrat(?:ed|ing|ion)\b/,
      /\bnot happy\b/,
      /\bdisappoint(?:ed|ing|ment)\b/,
      /\bthis is becoming difficult\b/,
      /\bnot ideal\b/,
      /\bunacceptable\b/,
      /\bterrible\b/,
      /\bawful\b/,
    ]],
    ['trust_risk', -0.25, [
      /\b(?:losing|lost)\s+(?:trust|confidence)\b/,
      /\b(?:considering|thinking about)\s+(?:switching|leaving|cancelling|canceling)\b/,
    ]],
    ['polite_negative', -0.22, [
      /\b(?:this|it)\s+is\s+(?:affecting|impacting|blocking)\b/,
      /\b(?:could you please|please)\b.*\b(?:fix|resolve|help)\b.*\b(?:blocking|affecting|stuck|broken|failing)\b/,
      /\b(?:becoming|getting)\s+(?:difficult|hard|painful)\b/,
    ]],
  ];

  const negWords = ['frustrated','broken','terrible','angry','worst','cancel','bad','issue','error', 'invalid', 'locked out'];
  const posWords = ['great','thanks','love','good','happy','please'];
  let rawSentiment = 0;
  negWords.forEach(w => { if (t.includes(w)) rawSentiment -= 0.18; });
  posWords.forEach(w => { if (t.includes(w)) rawSentiment += 0.12; });
  rawSentiment = Math.max(-1, Math.min(1, rawSentiment));

  const sentimentEvidence = [];
  let sentimentScore = rawSentiment;
  sentimentSpecs.forEach(([label, weight, patterns]) => {
    const phrase = firstPatternHit(t, patterns);
    if (phrase) {
      sentimentScore += weight;
      sentimentEvidence.push(`${label}: ${phrase}`);
    }
  });
  sentimentScore = Math.max(-1, Math.min(1, sentimentScore));

  return {
    urgency_score: Math.round(urgencyScore * 10000) / 10000,
    urgency_level: urgencyLevelFromScore(urgencyScore),
    urgency_flags: Array.from(new Set(urgencyFlags)),
    urgency_evidence: urgencyEvidence,
    sentiment_score: Math.round(sentimentScore * 10000) / 10000,
    sentiment_raw_score: Math.round(rawSentiment * 10000) / 10000,
    sentiment_label: sentimentLabelFromScore(sentimentScore),
    sentiment_evidence: sentimentEvidence,
  };
}


// ── Seeded PRNG (deterministic per text) ──────────────
function hashText(str) {
  let h = 0;
  for (let i = 0; i < str.length; i++) {
    h = ((h << 5) - h + str.charCodeAt(i)) | 0;
  }
  return Math.abs(h);
}

function seededRandom(seed) {
  let s = seed;
  return function() {
    s = (s * 1664525 + 1013904223) & 0xffffffff;
    return (s >>> 0) / 0xffffffff;
  };
}

// ── Simulation (when API is offline) ──────────────────
function simulateRouting(text, extraPayload = {}) {
  const t = text.toLowerCase().trim();
  const marker = t.match(/\[clarification:\s*([a-z_]+)\s*-\s*([^\]]+)\]/);
  const clarificationTarget = extraPayload.clarification_target || (marker && marker[1]);
  const clarificationChoice = extraPayload.clarification_choice || (marker && marker[2]);
  const validTargets = Object.keys(CAT_COLORS);

  if (clarificationTarget && validTargets.includes(clarificationTarget)) {
    const allProbs = {};
    validTargets.forEach(cat => { allProbs[cat] = cat === clarificationTarget ? 0.9 : 0.0143; });
    const demoSignals = inferDemoSignals(t);
    return {
      action: 'route',
      confidence: 0.9,
      entropy: 0.35,
      margin: 0.75,
      top_category: clarificationTarget,
      all_probs: allProbs,
      top_two_classes: [clarificationTarget, validTargets.find(cat => cat !== clarificationTarget)],
      queue: clarificationTarget,
      reason: `Clarification answer resolved the ambiguity toward ${clarificationTarget}.`,
      clarification_applied: true,
      clarification_choice: clarificationChoice,
      sla_breach_probability: Math.min(0.95, 0.15 + (demoSignals.urgency_score * 0.45)),
      urgency_score: demoSignals.urgency_score,
      features: {
        ...demoSignals,
        text_complexity_score: Math.round(text.split(' ').length / 5 * 100) / 100,
      },
      latency_ms: 28 + (hashText(t) % 20),
    };
  }
  
  // Basic validation in simulation to match real API behavior
  if (t.length < 10) {
    const greetings = ['hi', 'hello', 'hey', 'test'];
    if (greetings.some(g => t.startsWith(g))) {
        return {
            action: 'invalid_input',
            error_type: 'greeting',
            response: "Hi there! 👋 Could you describe the issue you're experiencing? We're here to help."
        };
    }
    return {
        action: 'invalid_input',
        error_type: 'too_short',
        response: "Could you share a bit more detail about your issue? We're here to help."
    };
  }

  const rng = seededRandom(hashText(t));  // deterministic per text

  const scores = {
    billing: 0.02, technical_support: 0.02, account_management: 0.02,
    feature_request: 0.02, compliance_legal: 0.02, onboarding: 0.02,
    general_inquiry: 0.02, churn_risk: 0.02,
  };

  // Simple keyword scoring
  const kw = {
    billing: ['invoice','billing','payment','charge','refund','price','cost','subscription','plan','pricing','credit'],
    technical_support: ['error','bug','broken','crash','fix','api','endpoint','500','timeout','issue','not working','failed'],
    account_management: ['account','user','access','permission','settings','profile','password','role'],
    feature_request: ['feature','add','implement','suggest','request','capability','enhancement','wish','could you'],
    compliance_legal: ['gdpr','compliance','audit','regulation','privacy','security','data protection','legal'],
    onboarding: ['new user','setup','getting started','onboarding','first time','just signed up','configure','install'],
    general_inquiry: ['how do','what is','question','information','help','guide','documentation'],
    churn_risk: ['cancel','switch','competitor','alternative','frustrated','unacceptable','leaving','terminate','fed up','last straw'],
  };

  Object.entries(kw).forEach(([cat, words]) => {
    words.forEach(w => { if (t.includes(w)) scores[cat] += 0.15 + rng() * 0.05; });
  });

  // Normalize
  const total = Object.values(scores).reduce((a, b) => a + b, 0);
  Object.keys(scores).forEach(k => scores[k] /= total);

  // Add small deterministic noise (simulate MC Dropout variance)
  Object.keys(scores).forEach(k => {
    scores[k] += (rng() - 0.5) * 0.03;
    scores[k] = Math.max(0.001, scores[k]);
  });
  const total2 = Object.values(scores).reduce((a, b) => a + b, 0);
  Object.keys(scores).forEach(k => scores[k] /= total2);

  const sorted = Object.entries(scores).sort((a, b) => b[1] - a[1]);
  const confidence = sorted[0][1];
  const entropy = -Object.values(scores).reduce((s, p) => s + p * Math.log(p + 1e-9), 0);
  const topCat = sorted[0][0];
  const topTwo = [sorted[0][0], sorted[1][0]];
  const margin = sorted[0][1] - sorted[1][1];

  let action, reason;
  const critical_labels = ['compliance_legal', 'account_management'];

  if (critical_labels.includes(topCat)) {
    if (confidence >= 0.90 && margin >= 0.35 && entropy < 0.60) {
      action = 'route';
      reason = `• Safe to auto-route sensitive intent<br>• Confidence: ${(confidence*100).toFixed(1)}%<br>• Margin: ${margin.toFixed(2)}`;
    } else {
      action = 'escalate';
      reason = `• Escalated sensitive intent (${topCat.replace(/_/g,' ')})<br>• Strict confidence/margin threshold not met`;
    }
  } else {
    if (confidence >= 0.85 && margin >= 0.25 && entropy < 0.70) {
      action = 'route';
      reason = `• Strong dominant intent<br>• Confidence: ${(confidence*100).toFixed(1)}%<br>• Margin: ${margin.toFixed(2)}<br>• Safe to auto-route`;
    } else if (confidence >= 0.60 && entropy < 1.05) {
      action = 'clarify';
      reason = `• Medium ambiguity detected<br>• Clarification needed between ${topTwo[0].replace(/_/g,' ')} and ${topTwo[1].replace(/_/g,' ')}<br>• Margin: ${margin.toFixed(2)}`;
    } else {
      action = 'escalate';
      reason = `• High ambiguity / Low confidence (${(confidence*100).toFixed(1)}%)<br>• Multiple overlapping intents detected<br>• Human triage needed`;
    }
  }

  // Clarification question
  let clarification = null;
  if (action === 'clarify') {
    const questions = {
      'billing+technical_support': { question_text: 'Is the main issue related to (A) a software error, or (B) your billing or invoice?', options: ['Software error','Billing/invoice'], expected_gain: 0.71 },
      'technical_support+billing': { question_text: 'Is the main issue related to (A) a software error, or (B) your billing or invoice?', options: ['Software error','Billing/invoice'], expected_gain: 0.71 },
      'feature_request+technical_support': { question_text: 'Are you reporting something broken, or requesting a new capability?', options: ['Something broken','New feature'], expected_gain: 0.68 },
      'technical_support+feature_request': { question_text: 'Are you reporting something broken, or requesting a new capability?', options: ['Something broken','New feature'], expected_gain: 0.68 },
      'churn_risk+account_management': { question_text: 'Are you looking to change your plan, or do you have concerns about continuing?', options: ['Change plan','Concerns about continuing'], expected_gain: 0.74 },
      'account_management+churn_risk': { question_text: 'Are you looking to change your plan, or do you have concerns about continuing?', options: ['Change plan','Concerns about continuing'], expected_gain: 0.74 },
      'onboarding+technical_support': { question_text: 'Is this affecting a new user, or an existing user?', options: ['New user','Existing user'], expected_gain: 0.65 },
      'technical_support+onboarding': { question_text: 'Is this affecting a new user, or an existing user?', options: ['New user','Existing user'], expected_gain: 0.65 },
      'compliance_legal+billing': { question_text: 'Does this relate to a regulatory requirement, or to payment/invoicing?', options: ['Regulatory','Payment'], expected_gain: 0.72 },
      'billing+compliance_legal': { question_text: 'Does this relate to a regulatory requirement, or to payment/invoicing?', options: ['Regulatory','Payment'], expected_gain: 0.72 },
      'technical_support+general_inquiry': { question_text: 'Is this a specific technical problem, or a general question about how something works?', options: ['Specific problem','General question'], expected_gain: 0.66 },
      'general_inquiry+technical_support': { question_text: 'Is this a specific technical problem, or a general question about how something works?', options: ['Specific problem','General question'], expected_gain: 0.66 },
      'billing+general_inquiry': { question_text: 'Is your question about a specific charge on your account, or general pricing information?', options: ['Specific charge','General pricing'], expected_gain: 0.64 },
      'general_inquiry+billing': { question_text: 'Is your question about a specific charge on your account, or general pricing information?', options: ['Specific charge','General pricing'], expected_gain: 0.64 },
      'churn_risk+technical_support': { question_text: 'Is the main concern a technical problem you need fixed, or are you considering leaving the platform?', options: ['Technical problem','Considering leaving'], expected_gain: 0.76 },
      'technical_support+churn_risk': { question_text: 'Is the main concern a technical problem you need fixed, or are you considering leaving the platform?', options: ['Technical problem','Considering leaving'], expected_gain: 0.76 },
    };
    const key = topTwo[0] + '+' + topTwo[1];
    clarification = questions[key] || {
      question_text: 'Could you specify whether this is about a technical issue or an account/billing matter?',
      options: ['Technical issue', 'Account/billing'], expected_gain: 0.62,
    };
    clarification.question_id = 'Q_SIM';
  }

  const demoSignals = inferDemoSignals(t);

  // SLA — deterministic based on text features
  const outageWords = ['down', 'outage', 'crash', 'failing', 'blocked'];
  const outageFlags = outageWords.filter(w => t.includes(w));
  const slaBase = 0.15
    + (demoSignals.sentiment_score < -0.3 ? 0.2 : 0)
    + (demoSignals.urgency_score * 0.45)
    + (outageFlags.length * 0.15);
  const slaBreach = Math.min(Math.round(slaBase * 1000) / 1000, 0.95);

  return {
    action, confidence: Math.round(confidence * 10000) / 10000,
    entropy: Math.round(entropy * 10000) / 10000,
    margin: Math.round(margin * 10000) / 10000,
    top_category: topCat, all_probs: scores,
    top_two_classes: topTwo, queue: topCat,
    reason, clarification,
    sla_breach_probability: slaBreach,
    urgency_score: demoSignals.urgency_score,
    features: {
      ...demoSignals,
      text_complexity_score: Math.round(text.split(' ').length / 5 * 100) / 100,
    },
    latency_ms: 38 + (hashText(t) % 30),
  };
}
