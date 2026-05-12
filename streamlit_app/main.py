# =============================================================================
# STREAMLIT UI - intentionally minimal
# -----------------------------------------------------------------------------
# Only what the spec requires:
#   - text input
#   - submit button
#   - generated response
#   - retrieved context display
#   - agent workflow display
# =============================================================================

import requests
import streamlit as st

API_URL = "http://localhost:8000/query"

st.set_page_config(page_title="Agentic RAG - 10-K Filings", layout="wide")
st.title("Agentic RAG over SEC 10-K Filings")

user_query = st.text_input("Ask a question about a company's 10-K filing:")

if st.button("Submit") and user_query.strip():
    with st.spinner("Running multi-agent workflow..."):
        try:
            r = requests.post(API_URL, json={"query": user_query}, timeout=300)
            data = r.json()
        except Exception as e:
            st.error(f"API request failed: {e}")
            st.stop()

    st.subheader("Final Response")
    st.write(data.get("response", ""))

    st.subheader("Retrieved Context")
    st.text(data.get("retrieved_context", ""))

    st.subheader("Agent Workflow")
    for step in data.get("agent_steps", []):
        st.markdown(f"**{step['agent']}**")
        st.text(step["output"])
