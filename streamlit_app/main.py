# =============================================================================
# STREAMLIT UI - WhatsApp-style chat + Evaluation Dashboard
# -----------------------------------------------------------------------------
# Two tabs:
#   1. Chat                 - WhatsApp-style conversational UI with the
#                             5-agent CrewAI workflow
#   2. Evaluation Dashboard - reads the 4 CSV files produced by
#                             run_evaluation.py and visualises them
# =============================================================================

import os
import requests
import pandas as pd
import streamlit as st

API_URL = "http://localhost:8000/query"

# Paths to the four CSVs produced by run_evaluation.py
EVAL_BASELINE_CSV   = "evaluation_baseline.csv"
EVAL_AGENTIC_CSV    = "evaluation_agentic.csv"
EVAL_COMPARISON_CSV = "evaluation_comparison.csv"
EVAL_SUMMARY_CSV    = "evaluation_summary.csv"


st.set_page_config(
    page_title="Agentic RAG Chat",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================================================================
# CUSTOM CSS - WhatsApp-style bubbles + responsive wrapping
# =============================================================================
st.markdown("""
<style>
    /* User bubble - green like WhatsApp, dark text */
    .user-bubble {
        background-color: #DCF8C6 !important;
        color: #000000 !important;
        padding: 10px 14px;
        border-radius: 12px 12px 2px 12px;
        margin: 8px 0 8px auto;
        max-width: 70%;
        width: fit-content;
        word-wrap: break-word;
        overflow-wrap: break-word;
        white-space: pre-wrap;
        box-shadow: 0 1px 1px rgba(0,0,0,0.1);
        margin-left: auto;
    }

    /* Container for assistant response */
    .agent-response {
        margin: 8px 0;
        max-width: 85%;
    }

    /* Final answer bubble - white background, FORCED dark text */
    .final-bubble {
        background-color: #FFFFFF !important;
        color: #000000 !important;
        padding: 12px 16px;
        border-radius: 12px 12px 12px 2px;
        margin: 4px 0;
        word-wrap: break-word;
        overflow-wrap: break-word;
        white-space: pre-wrap;
        box-shadow: 0 1px 1px rgba(0,0,0,0.1);
        border-left: 3px solid #25D366;
    }

    /* Force dark text on ALL elements INSIDE the final bubble */
    .final-bubble * {
        color: #000000 !important;
    }

    /* Agent card - light grey background, FORCED dark text */
    .agent-card {
        background-color: #F0F2F5 !important;
        color: #000000 !important;
        border-radius: 8px;
        padding: 10px 14px;
        margin: 6px 0;
        word-wrap: break-word;
        overflow-wrap: break-word;
        white-space: pre-wrap;
        font-size: 0.9em;
        border-left: 3px solid #888;
    }

    /* Force dark text on ALL elements INSIDE agent cards */
    .agent-card * {
        color: #000000 !important;
    }

    /* Status pills */
    .pass-pill {
        background-color: #D4EDDA !important;
        color: #155724 !important;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: bold;
        font-size: 0.85em;
    }
    .review-pill {
        background-color: #FFF3CD !important;
        color: #856404 !important;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: bold;
        font-size: 0.85em;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# SESSION STATE
# =============================================================================
if "messages" not in st.session_state:
    st.session_state.messages = []


# =============================================================================
# SIDEBAR - conversation history
# =============================================================================
with st.sidebar:
    st.markdown("### 💬 Conversation History")

    if not st.session_state.messages:
        st.info("No questions yet. Ask one to get started!")
    else:
        user_questions = [
            (i, msg["content"])
            for i, msg in enumerate(st.session_state.messages)
            if msg["role"] == "user"
        ]
        question_labels = [
            f"Q{idx+1}: {q[:50]}{'...' if len(q) > 50 else ''}"
            for idx, (_, q) in enumerate(user_questions)
        ]
        question_labels = ["-- Select a previous question --"] + question_labels

        selected = st.selectbox(
            "Jump to a previous question:",
            options=range(len(question_labels)),
            format_func=lambda x: question_labels[x],
        )

        if selected > 0:
            idx, full_q = user_questions[selected - 1]
            st.markdown(f"**Full question:**")
            st.markdown(f"> {full_q}")

        st.divider()

        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()


# =============================================================================
# MAIN AREA - two tabs
# =============================================================================
tab_chat, tab_eval = st.tabs(["💬 Chat", "📊 Evaluation Dashboard"])


# =============================================================================
# TAB 1 - Chat
# =============================================================================
# =============================================================================
# TAB 1 - Chat
# =============================================================================
with tab_chat:
    st.markdown("## Agentic RAG Chat")
    st.caption("Ask financial questions about SEC 10-K filings. "
               "Five specialist agents will analyze your question.")

    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(
                    f'<div class="user-bubble">{msg["content"]}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown('<div class="agent-response">', unsafe_allow_html=True)
                st.markdown(
                    f'<div class="final-bubble">{msg["content"]}</div>',
                    unsafe_allow_html=True,
                )

                if msg.get("agent_steps"):
                    # Filter out the Retriever -- its raw chunk output is
                    # noisy and duplicates what's already shown in the
                    # "Retrieved context" expander below.
                    visible_steps = [
                        s for s in msg["agent_steps"]
                        if s["agent"] != "Financial Filings Retriever"
                    ]

                    with st.expander(f"🤖 View {len(visible_steps)} agent contributions"):
                        agent_emoji = {
                            "Planning Coordinator": "🎯",
                            "Financial Analyst": "📈",
                            "Portfolio Strategist": "💼",
                            "Risk Assessment Specialist": "⚠️",
                        }
                        for step in visible_steps:
                            emoji = agent_emoji.get(step["agent"], "🤖")
                            st.markdown(f"**{emoji} {step['agent']}**")
                            st.markdown(
                                f'<div class="agent-card">{step["output"]}</div>',
                                unsafe_allow_html=True,
                            )
                            st.markdown("")

                if msg.get("context"):
                    with st.expander("📄 Retrieved context"):
                        st.markdown(
                            f'<div class="agent-card">{msg["context"]}</div>',
                            unsafe_allow_html=True,
                        )

                st.markdown('</div>', unsafe_allow_html=True)

    user_input = st.chat_input("Type your question here...")
    if user_input:
        st.session_state.messages.append({
            "role": "user",
            "content": user_input,
        })
        with st.spinner("🤔 Agents are thinking..."):
            try:
                r = requests.post(API_URL, json={"query": user_input}, timeout=300)
                data = r.json()
            except Exception as e:
                st.error(f"API request failed: {e}")
                st.stop()
        st.session_state.messages.append({
            "role": "assistant",
            "content": data.get("response", ""),
            "agent_steps": data.get("agent_steps", []),
            "context": data.get("retrieved_context", ""),
        })
        st.rerun()


# =============================================================================
# TAB 2 - Evaluation Dashboard
# -----------------------------------------------------------------------------
# Mirrors the structure of run_evaluation.py output:
#   - Summary table (PASS/REVIEW status per metric)
#   - Bar chart comparing Baseline vs Agentic
#   - Side-by-side per-question comparison
#   - Individual baseline and agentic tables
#   - Verdict
# =============================================================================
with tab_eval:
    st.markdown("## Evaluation Dashboard")
    st.markdown(
        "Compare **Baseline RAG** vs **Agentic RAG** on ground-truth questions. "
        "Generate the data by running:"
    )
    st.code("python -m app.evaluation.run_evaluation", language="bash")

    # Check that all four CSVs exist
    required_files = {
        "Baseline scores": EVAL_BASELINE_CSV,
        "Agentic scores": EVAL_AGENTIC_CSV,
        "Comparison": EVAL_COMPARISON_CSV,
        "Summary": EVAL_SUMMARY_CSV,
    }
    missing = [name for name, path in required_files.items() if not os.path.exists(path)]

    if missing:
        st.warning(
            f"Missing CSV files: {', '.join(missing)}. "
            "Run the evaluation script first, then refresh."
        )
    else:
        # Load all four CSVs
        baseline_df   = pd.read_csv(EVAL_BASELINE_CSV)
        agentic_df    = pd.read_csv(EVAL_AGENTIC_CSV)
        comparison_df = pd.read_csv(EVAL_COMPARISON_CSV)
        summary_df    = pd.read_csv(EVAL_SUMMARY_CSV)

        st.success(f"Loaded results for {len(baseline_df)} questions.")

        # -----------------------------------------------------------------
        # SECTION 1: Summary table with PASS/REVIEW status
        # -----------------------------------------------------------------
        st.subheader("📋 Summary - Baseline vs Agentic")
        st.caption("Averaged scores against threshold criteria.")

        # KPI cards for each metric
        cols = st.columns(len(summary_df))
        for i, row in summary_df.iterrows():
            with cols[i]:
                st.metric(
                    label=row["Metric"],
                    value=f"{row['Agentic']:.4f}",
                    delta=f"{row['Improvement']:+.4f} vs Baseline",
                )
                # Status pill
                status_class = "pass-pill" if row["Agentic Status"] == "PASS" else "review-pill"
                st.markdown(
                    f'<span class="{status_class}">Agentic: {row["Agentic Status"]}</span>',
                    unsafe_allow_html=True,
                )
                st.caption(f"Threshold: {row['Threshold']}")

        st.markdown("---")
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

        # -----------------------------------------------------------------
        # SECTION 2: Visual comparison (bar chart)
        # -----------------------------------------------------------------
        st.subheader("📊 Visual Comparison")
        chart_df = summary_df.set_index("Metric")[["Baseline", "Agentic"]]
        st.bar_chart(chart_df)

        # -----------------------------------------------------------------
        # SECTION 3: Verdict
        # -----------------------------------------------------------------
        st.subheader("🏆 Verdict")
        wins = int((summary_df["Improvement"] > 0).sum())
        total = len(summary_df)

        if wins == total:
            st.success(
                f"✅ **Agentic RAG outperformed Baseline on {wins}/{total} metrics.**\n\n"
                "The multi-agent workflow consistently improves the pipeline."
            )
        elif wins > total / 2:
            st.info(
                f"📈 **Agentic RAG outperformed Baseline on {wins}/{total} metrics.**\n\n"
                "The multi-agent workflow improves most metrics."
            )
        else:
            st.warning(
                f"⚠️ **Agentic RAG outperformed Baseline on {wins}/{total} metrics.**\n\n"
                "Mixed results -- consider tuning agent prompts or reviewing tool selection."
            )

        # -----------------------------------------------------------------
        # SECTION 4: Side-by-side comparison per question
        # -----------------------------------------------------------------
        st.subheader("🔍 Side-by-Side Comparison (Per Question)")
        st.caption("Δ columns show how much agentic improved over baseline. "
                   "Positive values (green tint) mean agentic won.")

        # Style the comparison table to highlight positive/negative deltas
        def color_delta(val):
            try:
                v = float(val)
                if v > 0:
                    return "background-color: #D4EDDA; color: #155724;"
                elif v < 0:
                    return "background-color: #F8D7DA; color: #721C24;"
            except (ValueError, TypeError):
                pass
            return ""

        delta_cols = [c for c in comparison_df.columns if c.startswith("Δ")]
        styled_comparison = comparison_df.style.map(color_delta, subset=delta_cols)
        st.dataframe(styled_comparison, use_container_width=True, hide_index=True)

        # -----------------------------------------------------------------
        # SECTION 5: Per-pipeline tables (collapsible)
        # -----------------------------------------------------------------
        with st.expander("📥 Baseline RAG - Per-question scores"):
            st.dataframe(baseline_df, use_container_width=True, hide_index=True)

        with st.expander("🤖 Agentic RAG - Per-question scores"):
            st.dataframe(agentic_df, use_container_width=True, hide_index=True)

        # -----------------------------------------------------------------
        # SECTION 6: Download buttons
        # -----------------------------------------------------------------
        st.markdown("---")
        st.subheader("💾 Download Raw Data")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.download_button(
                label="Baseline CSV",
                data=baseline_df.to_csv(index=False).encode("utf-8"),
                file_name="evaluation_baseline.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with col2:
            st.download_button(
                label="Agentic CSV",
                data=agentic_df.to_csv(index=False).encode("utf-8"),
                file_name="evaluation_agentic.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with col3:
            st.download_button(
                label="Comparison CSV",
                data=comparison_df.to_csv(index=False).encode("utf-8"),
                file_name="evaluation_comparison.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with col4:
            st.download_button(
                label="Summary CSV",
                data=summary_df.to_csv(index=False).encode("utf-8"),
                file_name="evaluation_summary.csv",
                mime="text/csv",
                use_container_width=True,
            )