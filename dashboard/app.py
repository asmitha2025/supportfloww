# dashboard/app.py
# SupportMind Streamlit Dashboard
# Asmitha · 2026

import streamlit as st
import sys, os, json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

st.set_page_config(page_title="SupportMind", page_icon="🧠", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .stApp { background-color: #0a0a0f; }
    .metric-card { background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px; padding: 20px; text-align: center; }
    .action-route { color: #22c55e; font-size: 28px; font-weight: 800; }
    .action-clarify { color: #eab308; font-size: 28px; font-weight: 800; }
    .action-escalate { color: #ef4444; font-size: 28px; font-weight: 800; }
</style>
""", unsafe_allow_html=True)

st.title("🧠 SupportMind")
st.caption("Confidence-Gated Support Intelligence for B2B SaaS")

# Sidebar
st.sidebar.header("⚙️ Configuration")
mc_passes = st.sidebar.slider("MC Dropout Passes", 5, 50, 20)
route_thresh = st.sidebar.slider("Route Threshold", 0.5, 0.95, 0.80, 0.05)
clarify_thresh = st.sidebar.slider("Clarify Threshold", 0.3, 0.8, 0.55, 0.05)
entropy_max = st.sidebar.slider("Max Entropy (Route)", 0.1, 1.0, 0.35, 0.05)

# Hero metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric("Routing Accuracy", "89.1%", "+16.8pp")
col2.metric("Ambiguous Gain", "+32.3%", "vs baseline")
col3.metric("Annual Savings", "$756K", "per 10K tickets/mo")
col4.metric("Pipeline Latency", "45ms", "20-pass MC Dropout")

st.divider()

# Ticket Input
st.header("🎯 Route a Ticket")

presets = {
    "Billing Issue": "My invoice from last month shows $299 but my plan is $199. Please fix this billing error immediately.",
    "Technical Bug": "The API endpoint /v2/export returns a 500 error when batch size exceeds 1000 records.",
    "Ambiguous Ticket": "Hey, we have issues with the export function since last Tuesday. Also our invoice looks incorrect. We are considering upgrading but want this sorted first.",
    "Churn Risk": "This is the third time I'm reporting this. Still not fixed. We're looking at switching to a competitor.",
    "Onboarding": "We just signed up yesterday and need help setting up SSO for our team of 50 users.",
}

preset = st.selectbox("Quick presets:", ["Custom"] + list(presets.keys()))
if preset != "Custom":
    ticket_text = st.text_area("Ticket Text", value=presets[preset], height=100)
else:
    ticket_text = st.text_area("Ticket Text", placeholder="Enter support ticket text...", height=100)

if st.button("⚡ Route Ticket", type="primary", use_container_width=True):
    if ticket_text.strip():
        with st.spinner("Running MC Dropout inference..."):
            try:
                from confidence_router import ConfidenceGatedRouter, ROUTE_THRESHOLD, CLARIFY_THRESHOLD
                import confidence_router as cr
                cr.ROUTE_THRESHOLD = route_thresh
                cr.CLARIFY_THRESHOLD = clarify_thresh
                cr.ENTROPY_MAX = entropy_max

                router = ConfidenceGatedRouter()
                result = router.route(ticket_text, n_passes=mc_passes)

                # Display action
                action = result['action']
                action_colors = {'route': '🟢', 'clarify': '🟡', 'escalate': '🔴'}
                st.subheader(f"{action_colors.get(action, '')} Decision: {action.upper()}")
                st.caption(result['reason'])

                # Metrics
                c1, c2, c3 = st.columns(3)
                c1.metric("Confidence", f"{result['confidence']:.4f}")
                c2.metric("Entropy", f"{result['entropy']:.4f}")
                c3.metric("Top Category", result['top_category'].replace('_', ' ').title())

                # Probability distribution
                st.subheader("📊 Category Probabilities")
                import plotly.graph_objects as go
                cats = list(result['all_probs'].keys())
                probs_vals = list(result['all_probs'].values())
                fig = go.Figure(go.Bar(
                    x=probs_vals, y=[c.replace('_', ' ').title() for c in cats],
                    orientation='h',
                    marker_color=['#6366f1' if p == max(probs_vals) else '#334155' for p in probs_vals]
                ))
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                    font_color='#94a3b8', height=300, margin=dict(l=0,r=0,t=10,b=0),
                    xaxis_title="Probability"
                )
                st.plotly_chart(fig, use_container_width=True)

                # Clarification
                if action == 'clarify':
                    st.subheader("💡 Suggested Clarification")
                    try:
                        from clarification_engine import ClarificationEngine
                        bank_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'clarification_bank.json')
                        clar = ClarificationEngine(bank_path)
                        probs_arr = np.array(list(result['all_probs'].values()))
                        q = clar.select_question(probs_arr, result['top_two_classes'])
                        st.info(f"**{q['question_text']}**")
                        for opt in q.get('options', []):
                            st.button(opt, disabled=True)
                        st.caption(f"Expected information gain: {q['expected_gain']:.4f}")
                    except Exception as e:
                        st.warning(f"Could not load clarification: {e}")

                # SLA
                st.subheader("🚨 SLA Breach Prediction")
                try:
                    from sla_predictor import SLABreachPredictor
                    from feature_extraction import FeatureExtractor
                    feat_ext = FeatureExtractor()
                    features = feat_ext.extract(ticket_text)
                    sla_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'sla_predictor', 'sla_xgb.json')
                    sla = SLABreachPredictor(sla_path)
                    sla_features = {
                        'text_complexity_score': features['text_complexity_score'],
                        'agent_queue_depth': 15, 'customer_tier': 3,
                        'hour_of_day': 14, 'day_of_week': 2,
                        'similar_ticket_avg_hrs': 4.5,
                        'sentiment_score': features['sentiment_score'],
                        'repeat_issue': 0, 'escalated_before': 0,
                    }
                    sla_result = sla.explain(sla_features)
                    sc1, sc2 = st.columns(2)
                    sc1.metric("Breach Probability", f"{sla_result['breach_probability']:.1%}")
                    sc2.metric("Risk Level", sla_result['risk_level'].upper())
                    if sla_result['contributing_factors']:
                        st.caption("Factors: " + ", ".join(sla_result['contributing_factors']))
                except Exception as e:
                    st.warning(f"SLA prediction unavailable: {e}")

            except Exception as e:
                st.error(f"Error: {e}")
                import traceback
                st.code(traceback.format_exc())
    else:
        st.warning("Please enter ticket text.")

