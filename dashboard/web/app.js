// SupportMind Dashboard — app.js
// Interactive demo with real API calls (falls back to simulation if API unavailable)

const API_BASE = 'http://localhost:7860';
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
function initPresets() {
  document.querySelectorAll('.preset-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.getElementById('ticket-input').value = btn.dataset.text;
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
    displayResult(result);
  } catch (err) {
    result = simulateRouting(text);
    displayResult(result);
  }

  btn.innerHTML = '<span class="btn-icon">⚡</span> Route Ticket';
  btn.disabled = false;
}

// ── Display Result ────────────────────────────────────
function displayResult(r) {
  document.getElementById('result-placeholder').style.display = 'none';
  const content = document.getElementById('result-content');
  content.style.display = 'block';

  // Action badge
  const badge = document.getElementById('action-badge');
  badge.textContent = r.action.toUpperCase();
  badge.className = 'action-badge ' + r.action;
  const queueEl = document.getElementById('action-queue');
  queueEl.textContent = r.action === 'route' ? `→ ${r.queue || r.top_category} queue` :
                         r.action === 'clarify' ? 'Needs 1 clarification question' : 'Immediate human triage';

  // Gauges
  const confPct = Math.min(r.confidence * 100, 100);
  document.getElementById('conf-fill').style.width = confPct + '%';
  document.getElementById('conf-value').textContent = r.confidence.toFixed(4);
  const maxEnt = Math.log(8);
  const entPct = Math.min((r.entropy / maxEnt) * 100, 100);
  document.getElementById('ent-fill').style.width = entPct + '%';
  document.getElementById('ent-value').textContent = r.entropy.toFixed(4);

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
    document.getElementById('clarify-gain').textContent =
      `Expected information gain: ${r.clarification.expected_gain?.toFixed(4) || 'N/A'}`;
  } else {
    clarBox.style.display = 'none';
  }

  // Signals
  const slaPct = (r.sla_breach_probability || 0) * 100;
  document.getElementById('sla-value').textContent = ((r.sla_breach_probability || 0) * 100).toFixed(1) + '%';
  document.getElementById('sla-fill').style.width = slaPct + '%';
  document.getElementById('sla-fill').style.background =
    slaPct > 60 ? 'var(--red)' : slaPct > 30 ? 'var(--orange)' : 'var(--green)';

  const feat = r.features || {};
  const sent = feat.sentiment_score;
  document.getElementById('sentiment-value').textContent =
    sent !== undefined ? (sent > 0.2 ? '😊 ' : sent < -0.2 ? '😤 ' : '😐 ') + sent.toFixed(2) : '—';
  document.getElementById('urgency-value').textContent =
    feat.urgency_flags ? (feat.urgency_flags.length > 0 ? '🔴 ' + feat.urgency_flags.length + ' flags' : '🟢 Normal') : '—';
  document.getElementById('latency-value').textContent =
    r.latency_ms ? r.latency_ms + 'ms' : '—';

  // Reason
  document.getElementById('result-reason').textContent = r.reason || '';
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
  const t = text.toLowerCase();
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

  let action, reason;
  if (confidence >= 0.80 && entropy <= 0.35) {
    action = 'route';
    reason = `High confidence (${(confidence*100).toFixed(1)}%) with low entropy (${entropy.toFixed(3)})`;
  } else if (confidence >= 0.55) {
    action = 'clarify';
    reason = `Medium confidence (${(confidence*100).toFixed(1)}%) — clarification needed between ${topTwo[0].replace(/_/g,' ')} and ${topTwo[1].replace(/_/g,' ')}`;
  } else {
    action = 'clarify';
    reason = `Borderline confidence (${(confidence*100).toFixed(1)}%) — clarification recommended between ${topTwo[0].replace(/_/g,' ')} and ${topTwo[1].replace(/_/g,' ')}`;
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
  const slaBase = 0.15 + (text.split(' ').length / 200) + (sentScore < -0.3 ? 0.2 : 0) + (urgencyFlags.length * 0.1);
  const slaBreach = Math.min(Math.round(slaBase * 1000) / 1000, 0.95);

  return {
    action, confidence: Math.round(confidence * 10000) / 10000,
    entropy: Math.round(entropy * 10000) / 10000,
    top_category: topCat, all_probs: scores,
    top_two_classes: topTwo, queue: topCat,
    reason, clarification,
    sla_breach_probability: slaBreach,
    features: { sentiment_score: sentScore, urgency_flags: urgencyFlags, text_complexity_score: Math.round(text.split(' ').length / 5 * 100) / 100 },
    latency_ms: 38 + (hashText(t) % 30),
  };
}
