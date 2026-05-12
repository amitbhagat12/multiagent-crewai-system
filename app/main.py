# =============================================================================
# FASTAPI ENTRYPOINT
# -----------------------------------------------------------------------------
# Single endpoint: POST /query
# Flow: user query -> CrewAI agents -> RAG pipeline -> generated response
# =============================================================================

from fastapi import FastAPI
from pydantic import BaseModel

from app.rag.rag_pipeline import create_rag_pipeline
from app.agents.agents import run_agentic_workflow

app = FastAPI(title="Agentic RAG - Financial 10-K")

# Initialise the RAG pipeline ONCE when the API boots up.
# (Loading dataset + building/loading FAISS is expensive -- don't do it per request.)
create_rag_pipeline()


class Query(BaseModel):
    query: str


@app.get("/")
def home():
    return {"message": "Agentic RAG API running."}


@app.post("/query")
def query(q: Query):
    """
    Run the full multi-agent workflow on a user question.
    Returns the final response, the retrieved context, and the per-agent steps.
    """
    result = run_agentic_workflow(q.query)
    return {
        "response": result["response"],
        "retrieved_context": result["retrieved_context"],
        "agent_steps": result["agent_steps"],
    }
