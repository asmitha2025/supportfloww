"""
SupportMind — Full System Validation Suite
Tests every core function for correctness:
  1. Ticket Validator (edge cases)
  2. Feature Extraction (sentiment, urgency, complexity)
  3. Ensemble Router (probability math, entropy, margin, decision logic)
  4. Clarification Engine (information gain, template selection)
  5. SLA Predictor (feature vector, prediction range)
  6. API Endpoints (end-to-end integration)
"""

import os, sys, json, math, time
import requests
import numpy as np

API = "http://localhost:7860"
PASS = 0
FAIL = 0
RESULTS = []

def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        RESULTS.append(f"  ✅ {name}")
    else:
        FAIL += 1
        RESULTS.append(f"  ❌ {name} — {detail}")

def section(title):
    RESULTS.append(f"\n{'='*60}")
    RESULTS.append(f"  {title}")
    RESULTS.append(f"{'='*60}")

def route(text):
    r = requests.post(f"{API}/route", json={"text": text, "customer_id": "test_001"})
    return r.json()

# ══════════════════════════════════════════════════════════
# TEST 1: TICKET VALIDATOR
# ══════════════════════════════════════════════════════════
section("1. TICKET VALIDATOR — Edge Case Handling")

# Empty input
r = route("")
check("Empty input → invalid_input", r["action"] == "invalid_input", f"got {r['action']}")
check("Empty input error_type = 'empty'", r.get("error_type") == "empty", f"got {r.get('error_type')}")

# Greeting
r = route("hi")
check("Greeting 'hi' → invalid_input", r["action"] == "invalid_input", f"got {r['action']}")
check("Greeting error_type = 'greeting' or 'too_short'", r.get("error_type") in ["greeting", "too_short"], f"got {r.get('error_type')}")

# Too short
r = route("help")
check("Short 'help' → invalid_input", r["action"] == "invalid_input", f"got {r['action']}")

# Valid ticket
r = route("My invoice from last month shows $299 but my plan is $199. Please fix this billing error immediately.")
check("Valid billing ticket → NOT invalid_input", r["action"] != "invalid_input", f"got {r['action']}")
check("Valid ticket has confidence > 0", r["confidence"] > 0, f"got {r['confidence']}")
check("Valid ticket has entropy > 0", r["entropy"] > 0, f"got {r['entropy']}")


# ══════════════════════════════════════════════════════════
# TEST 2: PROBABILITY MATH
# ══════════════════════════════════════════════════════════
section("2. PROBABILITY MATHEMATICS — Correctness")

r = route("The API endpoint /v2/export returns a 500 error when batch size exceeds 1000 records. Stack trace attached.")
probs = r.get("all_probs", {})

# Probabilities must sum to ~1.0
prob_sum = sum(probs.values())
check(f"Probabilities sum to 1.0 (got {prob_sum:.6f})", abs(prob_sum - 1.0) < 0.01, f"sum = {prob_sum}")

# All probabilities must be >= 0
all_non_neg = all(v >= 0 for v in probs.values())
check("All probabilities >= 0", all_non_neg, f"probs = {probs}")

# All probabilities must be <= 1
all_leq_one = all(v <= 1.0 for v in probs.values())
check("All probabilities <= 1.0", all_leq_one, f"probs = {probs}")

# Must have exactly 8 categories
check(f"Exactly 8 categories (got {len(probs)})", len(probs) == 8, f"categories = {list(probs.keys())}")

# Verify category names
expected_cats = {'billing', 'technical_support', 'account_management', 'feature_request',
                 'compliance_legal', 'onboarding', 'general_inquiry', 'churn_risk'}
check("Correct category names", set(probs.keys()) == expected_cats, f"got {set(probs.keys())}")


# ══════════════════════════════════════════════════════════
# TEST 3: ENTROPY CALCULATION
# ══════════════════════════════════════════════════════════
section("3. SHANNON ENTROPY — Mathematical Verification")

# Manually compute entropy from the probabilities
manual_entropy = -sum(p * math.log(p + 1e-9) for p in probs.values())
reported_entropy = r["entropy"]
entropy_diff = abs(manual_entropy - reported_entropy)
check(f"Entropy matches manual calculation (diff={entropy_diff:.6f})", entropy_diff < 0.05,
      f"manual={manual_entropy:.4f}, reported={reported_entropy:.4f}")

# Entropy bounds: 0 <= H <= log(8) ≈ 2.079
max_entropy = math.log(8)
check(f"Entropy >= 0 (got {reported_entropy:.4f})", reported_entropy >= 0)
check(f"Entropy <= log(8)={max_entropy:.4f} (got {reported_entropy:.4f})", reported_entropy <= max_entropy + 0.01)

# For a clear technical ticket, entropy should be reasonably low
check(f"Technical ticket entropy < 1.5 (got {reported_entropy:.4f})", reported_entropy < 1.5,
      f"entropy too high for a clear technical ticket")


# ══════════════════════════════════════════════════════════
# TEST 4: CONFIDENCE & MARGIN
# ══════════════════════════════════════════════════════════
section("4. CONFIDENCE & MARGIN — Decision Gate Logic")

confidence = r["confidence"]
margin = r.get("margin", 0)

# Confidence must equal max probability
max_prob = max(probs.values())
check(f"Confidence = max(probs) (conf={confidence:.4f}, max={max_prob:.4f})",
      abs(confidence - max_prob) < 0.01)

# Margin = top1 - top2
sorted_probs = sorted(probs.values(), reverse=True)
expected_margin = sorted_probs[0] - sorted_probs[1]
check(f"Margin = top1 - top2 (margin={margin:.4f}, expected={expected_margin:.4f})",
      abs(margin - expected_margin) < 0.02)

# top_category must match highest probability
top_cat = r["top_category"]
actual_top = max(probs, key=probs.get)
check(f"top_category matches highest prob ('{top_cat}' vs '{actual_top}')", top_cat == actual_top)


# ══════════════════════════════════════════════════════════
# TEST 5: ROUTING DECISION LOGIC
# ══════════════════════════════════════════════════════════
section("5. ROUTING DECISION LOGIC — 3-Tier Gate")

# Test clear billing ticket
r_billing = route("My invoice from last month shows $299 but my plan is $199. Please fix this billing error immediately.")
check(f"Clear billing → action in [route, clarify] (got '{r_billing['action']}')",
      r_billing["action"] in ["route", "clarify"])
check(f"Clear billing → top_category is billing (got '{r_billing['top_category']}')",
      r_billing["top_category"] == "billing")

# Test clear technical ticket
r_tech = route("The API endpoint /v2/export returns a 500 error when batch size exceeds 1000 records. Stack trace attached.")
check(f"Clear technical → top_category is technical_support (got '{r_tech['top_category']}')",
      r_tech["top_category"] == "technical_support")

# Test ambiguous ticket
r_ambig = route("Hey, we have been having issues with the export function since last Tuesday's update. Also our invoice from last month looks incorrect.")
check(f"Ambiguous ticket → action in [clarify, escalate] (got '{r_ambig['action']}')",
      r_ambig["action"] in ["clarify", "escalate"])

# Action must be valid
for test_r in [r_billing, r_tech, r_ambig]:
    check(f"Action '{test_r['action']}' is valid", test_r["action"] in ["route", "clarify", "escalate", "invalid_input"])

# If action=route, queue must not be None
if r_billing["action"] == "route":
    check("Route action has queue set", r_billing.get("queue") is not None)

# If action=clarify, clarification question must exist
if r_ambig["action"] == "clarify":
    check("Clarify action has clarification object", r_ambig.get("clarification") is not None)
    if r_ambig.get("clarification"):
        check("Clarification has question_text", "question_text" in r_ambig["clarification"])
        check("Clarification has options", "options" in r_ambig["clarification"])
        check("Clarification has expected_gain", "expected_gain" in r_ambig["clarification"])
        gain = r_ambig["clarification"]["expected_gain"]
        check(f"Expected gain > 0 (got {gain})", gain > 0)


# ══════════════════════════════════════════════════════════
# TEST 6: FEATURE EXTRACTION
# ══════════════════════════════════════════════════════════
section("6. FEATURE EXTRACTION — NLP Signals")

features = r_billing.get("features", {})
check("Features dict exists", len(features) > 0)

# Sentiment score range [-1, 1]
sent = features.get("sentiment_score", None)
check(f"Sentiment score exists (got {sent})", sent is not None)
if sent is not None:
    check(f"Sentiment in [-1, 1] (got {sent:.4f})", -1.0 <= sent <= 1.0)

# Urgency flags
urgency = features.get("urgency_flags", None)
check("Urgency flags is a list", isinstance(urgency, list))

# Text complexity
complexity = features.get("text_complexity_score", None)
check(f"Text complexity exists (got {complexity})", complexity is not None)
if complexity is not None:
    check(f"Text complexity >= 0 (got {complexity})", complexity >= 0)

# Token count
tokens = features.get("token_count", None)
check(f"Token count exists (got {tokens})", tokens is not None)
if tokens is not None:
    check(f"Token count > 0 (got {tokens})", tokens > 0)

# Test urgency detection
r_urgent = route("URGENT: Production system is completely down. We need help ASAP! This is blocking all our customers.")
urgent_flags = r_urgent.get("features", {}).get("urgency_flags", [])
check(f"Urgency flags detected for urgent ticket (got {urgent_flags})", len(urgent_flags) > 0,
      "Expected urgency flags for URGENT/ASAP keywords")

# Test negative sentiment
r_angry = route("This is absolutely terrible service. I am extremely frustrated and angry. Your product has been broken for weeks and nobody cares.")
angry_sent = r_angry.get("features", {}).get("sentiment_score", 0)
check(f"Negative sentiment for angry ticket (got {angry_sent:.4f})", angry_sent < -0.1,
      "Expected negative sentiment score")


# ══════════════════════════════════════════════════════════
# TEST 7: SLA PREDICTOR
# ══════════════════════════════════════════════════════════
section("7. SLA BREACH PREDICTOR — XGBoost")

sla_prob = r_billing.get("sla_breach_probability", None)
check(f"SLA probability exists (got {sla_prob})", sla_prob is not None)
if sla_prob is not None:
    check(f"SLA probability in [0, 1] (got {sla_prob:.4f})", 0 <= sla_prob <= 1)

# Direct SLA endpoint test
sla_r = requests.post(f"{API}/sla/predict", json={
    "text_complexity_score": 12.5,
    "agent_queue_depth": 25,
    "customer_tier": 1,
    "hour_of_day": 16,
    "day_of_week": 4,
    "similar_ticket_avg_hrs": 8.0,
    "sentiment_score": -0.6,
    "repeat_issue": 1,
    "escalated_before": 1
})
sla_data = sla_r.json()
check(f"SLA endpoint returns 200 (got {sla_r.status_code})", sla_r.status_code == 200)
check("SLA response has breach_probability", "breach_probability" in sla_data)
if "breach_probability" in sla_data:
    bp = sla_data["breach_probability"]
    check(f"SLA breach_probability in [0,1] (got {bp:.4f})", 0 <= bp <= 1)
    # High risk features should yield higher probability
    check(f"High-risk features → elevated breach prob (got {bp:.4f})", bp > 0.01,
          "Expected higher breach probability for risky feature vector")


# ══════════════════════════════════════════════════════════
# TEST 8: CLARIFICATION ENGINE
# ══════════════════════════════════════════════════════════
section("8. CLARIFICATION ENGINE — Information Gain")

clar_r = requests.post(f"{API}/clarify", json={
    "text": "I need help with my account billing and also something is broken"
})
clar_data = clar_r.json()
check(f"Clarify endpoint returns 200 (got {clar_r.status_code})", clar_r.status_code == 200)
check("Clarification has question_text", "question_text" in clar_data)
check("Clarification has options", "options" in clar_data)
if "options" in clar_data:
    check(f"Options count >= 2 (got {len(clar_data['options'])})", len(clar_data["options"]) >= 2)
if "expected_gain" in clar_data:
    eg = clar_data["expected_gain"]
    check(f"Expected gain in [0, 2] (got {eg:.4f})", 0 <= eg <= 2)


# ══════════════════════════════════════════════════════════
# TEST 9: ENSEMBLE DIAGNOSTICS
# ══════════════════════════════════════════════════════════
section("9. ENSEMBLE ROUTER — Model Agreement & Weights")

ensemble = r_tech.get("ensemble", {})
check("Ensemble diagnostics exist", len(ensemble) > 0)
check(f"BERT available = {ensemble.get('bert_available')}", ensemble.get("bert_available") is not None)
check("bert_top category exists", ensemble.get("bert_top") is not None)
check("sklearn_top category exists", ensemble.get("sklearn_top") is not None)
check("agreement field exists", "agreement" in ensemble)

# Weights must sum to 1.0
bw = ensemble.get("bert_weight", 0)
sw = ensemble.get("sklearn_weight", 0)
weight_sum = bw + sw
check(f"Ensemble weights sum to 1.0 (got {weight_sum})", abs(weight_sum - 1.0) < 0.01)

# MC passes must be > 0
mc = r_tech.get("mc_passes", 0)
check(f"MC passes > 0 (got {mc})", mc > 0)


# ══════════════════════════════════════════════════════════
# TEST 10: METRICS & HEALTH ENDPOINTS
# ══════════════════════════════════════════════════════════
section("10. SYSTEM HEALTH & METRICS")

health_r = requests.get(f"{API}/health")
health = health_r.json()
check(f"Health endpoint returns 200 (got {health_r.status_code})", health_r.status_code == 200)
check(f"Health status = 'ok' (got '{health.get('status')}')", health.get("status") == "ok")
check("Health has bert_online field", "bert_online" in health)

metrics_r = requests.get(f"{API}/metrics")
metrics = metrics_r.json()
check(f"Metrics endpoint returns 200 (got {metrics_r.status_code})", metrics_r.status_code == 200)
check(f"Total requests > 0 (got {metrics.get('total_requests')})", metrics.get("total_requests", 0) > 0)
check("Routing distribution exists", "routing_distribution" in metrics)


# ══════════════════════════════════════════════════════════
# TEST 11: LATENCY
# ══════════════════════════════════════════════════════════
section("11. INFERENCE LATENCY")

latency = r_tech.get("latency_ms", 0)
check(f"Latency reported (got {latency}ms)", latency > 0)
check(f"Latency < 10000ms (got {latency}ms)", latency < 10000, "Inference took too long")


# ══════════════════════════════════════════════════════════
# TEST 12: CATEGORY-SPECIFIC ROUTING ACCURACY
# ══════════════════════════════════════════════════════════
section("12. CATEGORY-SPECIFIC ROUTING ACCURACY")

test_cases = [
    ("My monthly invoice has an extra charge of $50 that I did not authorize. Please review and correct.", "billing"),
    ("The export API keeps returning HTTP 500 errors whenever I try to download CSV files larger than 5MB.", "technical_support"),
    ("We need to add three new team members to our enterprise account. How do I manage user roles?", "account_management"),
    ("It would be great if you could add dark mode support to the analytics dashboard.", "feature_request"),
    ("We require GDPR-compliant data processing agreements for our European customers before renewal.", "compliance_legal"),
    ("Just signed up yesterday. How do I import my existing customer data from our old CRM system?", "onboarding"),
    ("I am completely fed up with this product. We are actively evaluating your competitors and will likely cancel.", "churn_risk"),
]

correct = 0
for text, expected_cat in test_cases:
    r = route(text)
    got = r.get("top_category", "unknown")
    is_correct = got == expected_cat
    if is_correct:
        correct += 1
    check(f"'{expected_cat}' ticket → top_category='{got}'", is_correct,
          f"Expected '{expected_cat}', got '{got}'")

accuracy = correct / len(test_cases) * 100
check(f"Overall category accuracy: {accuracy:.1f}% ({correct}/{len(test_cases)})", accuracy >= 50,
      f"Accuracy too low: {accuracy:.1f}%")


# ══════════════════════════════════════════════════════════
# FINAL REPORT
# ══════════════════════════════════════════════════════════
print("\n" + "═"*60)
print("  SUPPORTMIND — FULL SYSTEM VALIDATION REPORT")
print("═"*60)
for line in RESULTS:
    print(line)

print(f"\n{'═'*60}")
print(f"  TOTAL: {PASS + FAIL} tests | ✅ PASSED: {PASS} | ❌ FAILED: {FAIL}")
if FAIL == 0:
    print(f"  🎉 ALL TESTS PASSED — System is production-ready!")
else:
    print(f"  ⚠️  {FAIL} test(s) need attention")
print(f"{'═'*60}\n")
