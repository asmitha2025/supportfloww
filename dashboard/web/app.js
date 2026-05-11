// SupportMind Dashboard — app.js
// Interactive demo with real API calls (falls back to simulation if API unavailable)

const API_BASE = window.location.origin;
let apiOnline = false;

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
  checkAPI();
  updateLiveMetrics();
  setInterval(updateLiveMetrics, 5000); // Update every 5 seconds
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

// Initial load and interval
window.addEventListener('DOMContentLoaded', () => {
  checkAPI();
  initPresets();
  updateMetrics();
  setInterval(updateMetrics, 5000);
  
  // Smooth scroll
  document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
      e.preventDefault();
      document.querySelector(this.getAttribute('href')).scrollIntoView({
        behavior: 'smooth'
      });
    });
  });
});

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
  try {
    const res = await fetch(`${API_BASE}/health`, { signal: AbortSignal.timeout(2000) });
    if (res.ok) {
      apiOnline = true;
      const statusEl = document.querySelector('.status-text');
      if (statusEl) statusEl.textContent = 'API Connected';
    }
  } catch {
    apiOnline = false;
    const statusEl = document.querySelector('.status-text');
    if (statusEl) statusEl.textContent = 'Demo Mode';
  }
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
async function routeTicket() {
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
        body: JSON.stringify({ text }),
      });
      result = await res.json();
    } else {
      result = simulateRouting(text);
    }
    displayResult(result, text);
  } catch (err) {
    result = simulateRouting(text);
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
    document.getElementById('prob-chart').innerHTML = '';
    document.getElementById('clarification-box').style.display = 'none';
    const explainBtn = document.getElementById('explain-btn');
    if (explainBtn) explainBtn.style.display = 'none';
    document.getElementById('explanation-box').style.display = 'none';
    return;
  }


  // Show gauges for valid input
  document.querySelector('.gauge-row').style.display = 'grid';

  document.getElementById('result-placeholder').style.display = 'none';
  const content = document.getElementById('result-content');
  content.style.display = 'block';

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
    (r.clarification.options || []).forEach(o => {
      const btn = document.createElement('button');
      btn.className = 'option-btn';
      btn.textContent = o;
      btn.onclick = () => {
        // Provide visual feedback
        document.querySelectorAll('#clarify-options .option-btn').forEach(b => b.disabled = true);
        btn.style.background = 'var(--primary)';
        btn.style.color = '#fff';
        
        // Append clarification to input
        const input = document.getElementById('ticket-input');
        input.value = input.value.trim() + '\n\n[Clarification provided: ' + o + ']';
        
        // Re-route with new context after a short delay
        setTimeout(() => {
          routeTicket();
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
  document.getElementById('sentiment-value').textContent =
    sent !== undefined ? (sent > 0.2 ? '😊 ' : sent < -0.2 ? '😤 ' : '😐 ') + sent.toFixed(2) : '—';
  
  const urgScore = r.urgency_score || feat.urgency_score || 0;
  const urgencyCard = document.getElementById('urgency-value').parentElement;
  if (urgScore > 0.6) {
    document.getElementById('urgency-value').innerHTML = '<span style="color: var(--red); font-weight: bold; animation: pulse 1.5s infinite;">🚨 CRITICAL</span>';
    urgencyCard.style.border = '1px solid var(--red)';
    urgencyCard.style.boxShadow = '0 0 15px rgba(248, 113, 113, 0.2)';
  } else if (urgScore > 0.2) {
    document.getElementById('urgency-value').innerHTML = '<span style="color: var(--yellow); font-weight: bold;">⚡ HIGH</span>';
    urgencyCard.style.border = '1px solid var(--yellow)';
    urgencyCard.style.boxShadow = '';
  } else {
    document.getElementById('urgency-value').textContent = '🟢 Normal';
    urgencyCard.style.border = '';
    urgencyCard.style.boxShadow = '';
  }

  document.getElementById('latency-value').textContent =
    r.latency_ms ? r.latency_ms + 'ms' : '—';

  // Reason
  let decisionReason = '';
  if (r.action === 'multi_route') {
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

  btn.innerHTML = '<span class="material-symbols-outlined btn-icon">query_stats</span> Analyze Decision (SHAP)';
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

  const tokens = data.tokens;
  const values = data.values;

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
  return { tokens, values };
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
function simulateRouting(text) {
  const t = text.toLowerCase().trim();
  
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

  // Sentiment (basic)
  const negWords = ['frustrated','broken','terrible','angry','worst','cancel','bad','issue','error'];
  const posWords = ['great','thanks','love','good','happy','please'];
  let sentScore = 0;
  negWords.forEach(w => { if (t.includes(w)) sentScore -= 0.25; });
  posWords.forEach(w => { if (t.includes(w)) sentScore += 0.2; });
  sentScore = Math.max(-1, Math.min(1, sentScore));

  // Urgency
  const urgencyWords = ['urgent','asap','immediately','critical','blocking','production down'];
  const urgencyFlags = urgencyWords.filter(w => t.includes(w));

  // SLA — deterministic based on text features
  const outageWords = ['down', 'outage', 'crash', 'failing', 'blocked'];
  const outageFlags = outageWords.filter(w => t.includes(w));
  const slaBase = 0.15 + (sentScore < -0.3 ? 0.2 : 0) + (urgencyFlags.length * 0.15) + (outageFlags.length * 0.2);
  const slaBreach = Math.min(Math.round(slaBase * 1000) / 1000, 0.95);

  return {
    action, confidence: Math.round(confidence * 10000) / 10000,
    entropy: Math.round(entropy * 10000) / 10000,
    margin: Math.round(margin * 10000) / 10000,
    top_category: topCat, all_probs: scores,
    top_two_classes: topTwo, queue: topCat,
    reason, clarification,
    sla_breach_probability: slaBreach,
    features: { sentiment_score: sentScore, urgency_flags: urgencyFlags, text_complexity_score: Math.round(text.split(' ').length / 5 * 100) / 100 },
    latency_ms: 38 + (hashText(t) % 30),
  };
}
