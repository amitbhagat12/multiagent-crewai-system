# Agentic RAG System — SEC 10-K Filings

A beginner-friendly capstone project that builds an **Agentic Retrieval-Augmented
Generation (RAG)** system over the [`virattt/financial-qa-10K`](https://huggingface.co/datasets/virattt/financial-qa-10K)
dataset using **CrewAI**, **LangChain**, **FAISS**, **FastAPI**, and **Streamlit**.

The system retrieves financial-filing knowledge, analyses it through a chain
of specialised agents (Planner → Retriever → Analyst → Portfolio → Risk),
and returns a grounded response.

---

## Project structure

```
rag-multiagent-system/
├── app/
│   ├── main.py                  # FastAPI app — single /query endpoint
│   ├── rag/
│   │   └── rag_pipeline.py      # Full RAG pipeline (load → chunk → embed → retrieve → generate)
│   ├── agents/
│   │   └── agents.py            # CrewAI agents + RAG-as-tool wiring
│   └── evaluation/
│       ├── bleu_eval.py
│       ├── rouge_eval.py
│       └── ragas_eval.py
├── vectorstores/                # FAISS index is saved here
├── data/                        # (empty — dataset is pulled directly from HuggingFace)
├── streamlit_app/
│   └── main.py                  # Minimal Streamlit UI
├── Dockerfile
├── requirements.txt
├── .env
└── README.md
```

---

## How the pieces fit together

```
User Query
   │
   ▼
FastAPI /query
   │
   ▼
CrewAI Workflow  (Planner → Retriever → Analysis → Portfolio → Risk)
   │
   ▼
RAG Pipeline    (FAISS + BM25 hybrid search → validation → LLM)
   │
   ▼
Final Response  (+ retrieved context + per-agent steps)
```

### Dataset

`virattt/financial-qa-10K` has 5 columns:

| column   | role in the project                            |
|----------|------------------------------------------------|
| context  | text content → chunked + embedded              |
| ticker   | metadata (used for filtered retrieval)         |
| filing   | metadata (source identifier)                   |
| question | used as retrieval reference                    |
| answer   | ground truth → used by evaluation only         |

### RAG pipeline (`app/rag/rag_pipeline.py`)

1. Load the parquet from HuggingFace
2. Convert rows → `Document` with metadata
3. Metadata-aware chunking with `RecursiveCharacterTextSplitter`
4. Embeddings with `sentence-transformers/all-MiniLM-L6-v2`
5. FAISS vectorstore (saved to `vectorstores/`)
6. Hybrid search = semantic (FAISS) + keyword (BM25)
7. Retrieval validation (drop weakly-related chunks)
8. RAG-as-tool functions exposed to agents
9. Response generation with Gemini

### Agents (`app/agents/agents.py`)

| Agent      | Role                                                       |
|------------|------------------------------------------------------------|
| Planner    | Reads the query, decides which agents/tools to run         |
| Retriever  | Calls the RAG tools (the only agent that touches the vectorstore) |
| Analysis   | Interprets retrieved passages                              |
| Portfolio  | Suggests how the company might fit in a portfolio          |
| Risk       | Lists key risks from the filing                            |

Tool chaining is wired through `Task(context=[upstream_task])` — CrewAI feeds
the output of one task into the next, which gives you the "agents as tools"
pattern without extra plumbing.

---

## Setup

```bash
# 1. Clone, then enter the project
cd rag-multiagent-system

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure API keys
# Edit the .env file:
#   GOOGLE_API_KEY=...   (for Gemini, used by the RAG pipeline)
#   OPENAI_API_KEY=...   (for CrewAI agents, OpenAI is its default backend)
```

---

## Run

### 1. Start the FastAPI backend

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

First startup will:
- download the parquet from HuggingFace
- chunk ~7000 contexts
- build a FAISS index (saved to `vectorstores/faiss_store`)

Subsequent startups load the saved index — much faster.

### 2. Start the Streamlit UI (in another terminal)

```bash
streamlit run streamlit_app/main.py
```

### 3. Or call the API directly

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What were Apple key revenue drivers?"}'
```

Response shape:

```json
{
  "response": "...",
  "retrieved_context": "...",
  "agent_steps": [
    {"agent": "Planning Coordinator", "output": "..."},
    {"agent": "Financial Filings Retriever", "output": "..."},
    ...
  ]
}
```

---

## Example prompts

- "What are the main risk factors mentioned by Microsoft?"
- "Summarize Apple's revenue drivers from its 10-K."
- "How does Tesla describe supply chain risk?"
- "Compare how AAPL and MSFT discuss cloud services."

---

## Evaluation

The `app/evaluation/` folder has three small modules:

```python
from app.evaluation.bleu_eval import compare_bleu
from app.evaluation.rouge_eval import compare_rouge
from app.evaluation.ragas_eval import compare_ragas
```

Each `compare_*` function takes a baseline-RAG response and an agentic-RAG
response plus the reference answer (from the dataset's `answer` column) and
returns the scores side-by-side, so you can tell whether the agentic layer
is actually helping over plain RAG.

---

## Docker

```bash
docker build -t agentic-rag .
docker run -p 8000:8000 --env-file .env agentic-rag
```

---

## Design notes (why it's structured this way)

- **`rag_pipeline.py` owns retrieval; `agents.py` owns orchestration.**
  Clean separation makes each file easy to read on its own.
- **Module-level globals** (`_vectorstore`, `_llm`) are populated by
  `create_rag_pipeline()` so the RAG tool functions stay as simple
  top-level functions — easy for CrewAI to wrap as `@tool`s.
- **Conversational memory** is a plain Python list of `(user, ai)` tuples.
  No `ConversationBufferMemory`, no chain wrappers — easy to inspect.
- **Hybrid search + validation** is implemented with deliberate simplicity
  (word-overlap validation, score-merging via a dict) so you can read
  every line and understand what's happening.
