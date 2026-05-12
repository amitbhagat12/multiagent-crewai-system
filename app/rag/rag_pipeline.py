# =============================================================================
# RAG PIPELINE
# -----------------------------------------------------------------------------
# This file handles EVERYTHING related to the RAG (Retrieval-Augmented Generation)
# pipeline. We keep it simple, functional, and beginner-friendly.
#
# What lives here:
#   1. Dataset loading (HuggingFace parquet -> LangChain Documents)
#   2. Metadata-aware chunking (RecursiveCharacterTextSplitter + metadata)
#   3. Embeddings (HuggingFaceEmbeddings)
#   4. Vectorstore creation (FAISS)
#   5. Semantic similarity search (with optional metadata filtering)
#   6. Hybrid search (semantic + keyword/BM25 combined)
#   7. Retrieval validation (drop weak/irrelevant chunks)
#   8. RAG-as-Tool functions (these are called by CrewAI agents)
#   9. Response generation (grounded LLM answer using retrieved context)
# =============================================================================

import os
from langchain_community.chat_models import ChatOpenAI
import pandas as pd

from dotenv import load_dotenv

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_core.prompts import PromptTemplate

load_dotenv()

# -----------------------------------------------------------------------------
# Global config (kept simple - tweak here if you want to experiment)
# -----------------------------------------------------------------------------
DATASET_URL = "hf://datasets/virattt/financial-qa-10K/data/train-00000-of-00001.parquet"
VECTORSTORE_PATH = "vectorstores/faiss_store"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 300
CHUNK_OVERLAP = 60
TOP_K = 4

# These will be populated when create_rag_pipeline() runs.
# Agents (in agents.py) import them as RAG tools.
_vectorstore = None
_bm25_retriever = None
_llm = None
_all_chunks = None


# =============================================================================
# 1. DATASET LOADING
# =============================================================================
def load_financial_dataset():
    """
    Load the virattt/financial-qa-10K parquet file from HuggingFace and
    convert each row into a LangChain Document.

    Schema of the dataset:
        - question : natural-language question about a 10-K filing
        - answer   : ground-truth answer (used for evaluation, not retrieval)
        - context  : the 10-K text snippet (this is what we chunk + embed)
        - ticker   : company stock ticker -> metadata
        - filing   : SEC filing identifier/URL -> metadata
    """
    print("Loading dataset from HuggingFace...")
    df = pd.read_parquet(DATASET_URL)
    print(f"Dataset loaded. Rows: {len(df)}")

    documents = []
    for _, row in df.iterrows():
        # The `context` field is the actual 10-K text -> page_content.
        # The other fields go into metadata so we can filter on them later.
        doc = Document(
            page_content=row["context"],
            metadata={
                "ticker": row["ticker"],
                "filing": row["filing"],
                "question": row["question"],   # useful for retrieval reference
                "answer": row["answer"],       # used by evaluation only
            },
        )
        documents.append(doc)

    print(f"Converted {len(documents)} rows into LangChain Documents.")
    return documents


# =============================================================================
# 2. METADATA-AWARE CHUNKING
# =============================================================================
def chunk_documents(documents):
    """
    Split each Document into smaller chunks while PRESERVING metadata.

    Why metadata-aware chunking matters:
      - lets us filter retrieval by ticker (e.g. only AAPL chunks)
      - improves hybrid search (we can search on ticker-scoped subsets)
      - downstream agents can reason about WHICH company/filing a chunk
        belongs to instead of just seeing raw text
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []
    for doc in documents:
        # split_documents() automatically copies the parent's metadata
        # onto every resulting chunk -> "metadata-aware" out of the box.
        split_chunks = splitter.split_documents([doc])
        chunks.extend(split_chunks)

    print(f"Created {len(chunks)} chunks (chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}).")
    return chunks


# =============================================================================
# 3. EMBEDDINGS + 4. VECTORSTORE
# =============================================================================
def build_or_load_vectorstore(chunks, embeddings):
    """
    If a FAISS index already exists on disk, load it.
    Otherwise, build it from scratch and save it.

    Saving avoids re-embedding 7000 rows every time you restart the API.
    """
    if os.path.exists(VECTORSTORE_PATH):
        print("FAISS store found on disk. Loading existing store...")
        vectorstore = FAISS.load_local(
            VECTORSTORE_PATH,
            embeddings=embeddings,
            allow_dangerous_deserialization=True,
        )
        print("Vectorstore loaded successfully.")
    else:
        print("No FAISS store found. Building a fresh one (this may take a minute)...")
        vectorstore = FAISS.from_documents(documents=chunks, embedding=embeddings)
        vectorstore.save_local(VECTORSTORE_PATH)
        print("Vectorstore built and saved.")
    return vectorstore


# =============================================================================
# 5. SEMANTIC SIMILARITY SEARCH (with optional metadata filtering)
# =============================================================================
def semantic_search(query, k=TOP_K, ticker=None):
    """
    Top-k semantic retrieval over FAISS.

    If `ticker` is given, we filter to chunks of that company only.
    This is the "metadata-aware retrieval" piece -- agents can scope
    a search to a particular company without sifting through everything.
    """
    if _vectorstore is None:
        raise RuntimeError("Vectorstore not initialised. Call create_rag_pipeline() first.")

    if ticker:
        # FAISS supports a `filter` dict for metadata-scoped retrieval.
        results = _vectorstore.similarity_search(query, k=k, filter={"ticker": ticker})
    else:
        results = _vectorstore.similarity_search(query, k=k)
    return results


# =============================================================================
# 6. HYBRID SEARCH (semantic + keyword/BM25)
# =============================================================================
def hybrid_search(query, k=TOP_K):
    """
    Hybrid retrieval combines two retrieval strategies:

      - Semantic (FAISS)      : great for meaning / paraphrases
      - Keyword (BM25)        : great for exact terms, tickers, numbers

    Financial text is full of exact tokens (ticker symbols, dollar
    amounts, segment names) where BM25 wins. Embeddings win on
    paraphrased questions. Combining both gives more robust recall.

    Scoring approach (simple + readable):
      1. Run both retrievers.
      2. Merge results, deduplicating on page_content.
      3. If a chunk appears in BOTH lists, boost its score -- it's
         agreed on by both methods, so it's probably very relevant.
    """
    semantic_results = _vectorstore.similarity_search(query, k=k)
    bm25_results = _bm25_retriever.get_relevant_documents(query)[:k]

    # Score every chunk: 1 point for being in semantic, 1 for being in bm25.
    # A chunk in both ends up with score 2 -> ranked first.
    scored = {}
    for doc in semantic_results:
        scored[doc.page_content] = {"doc": doc, "score": 1}
    for doc in bm25_results:
        if doc.page_content in scored:
            scored[doc.page_content]["score"] += 1   # agreement boost
        else:
            scored[doc.page_content] = {"doc": doc, "score": 1}

    # Sort by score (desc) and return the top-k unique chunks
    ranked = sorted(scored.values(), key=lambda x: x["score"], reverse=True)
    return [item["doc"] for item in ranked[:k]]


# =============================================================================
# 7. RETRIEVAL VALIDATION
# =============================================================================
def validate_retrieval(query, retrieved_docs, min_overlap=1):
    """
    Drop chunks that look weak/irrelevant before sending them to the LLM.

    Beginner-friendly heuristic:
      - Take the meaningful words from the query (length > 3).
      - Keep a chunk only if at least `min_overlap` of those words appear
        in the chunk text (case-insensitive).
      - If validation removes EVERYTHING (overly strict), fall back to
        the original list so the LLM still has something to work with.

    Real-world systems use rerankers (e.g. cross-encoders) for this --
    we use a simple word-overlap check to keep the code readable.
    """
    keywords = [w.lower() for w in query.split() if len(w) > 3]
    if not keywords:
        return retrieved_docs

    validated = []
    for doc in retrieved_docs:
        text = doc.page_content.lower()
        overlap = sum(1 for kw in keywords if kw in text)
        if overlap >= min_overlap:
            validated.append(doc)

    # Safety net: don't return an empty list -- that starves the LLM
    return validated if validated else retrieved_docs


# =============================================================================
# 8. RAG AS TOOL FUNCTIONS
# -----------------------------------------------------------------------------
# These are plain Python functions that CrewAI agents can call as "tools".
# Each one returns a string (easy for LLMs to consume).
# =============================================================================
def rag_tool_retrieve(query: str) -> str:
    """TOOL: simple top-k semantic retrieval. Used by the Retriever Agent."""
    docs = semantic_search(query, k=TOP_K)
    return _format_docs(docs)


def rag_tool_hybrid_search(query: str) -> str:
    """TOOL: hybrid (semantic + keyword) retrieval."""
    docs = hybrid_search(query, k=TOP_K)
    return _format_docs(docs)


def rag_tool_filtered_search(query: str, ticker: str) -> str:
    """TOOL: retrieval scoped to a specific company's filings."""
    docs = semantic_search(query, k=TOP_K, ticker=ticker)
    return _format_docs(docs)


def rag_tool_validated_retrieve(query: str) -> str:
    """TOOL: hybrid retrieval + validation. The highest-quality option."""
    docs = hybrid_search(query, k=TOP_K)
    docs = validate_retrieval(query, docs)
    return _format_docs(docs)


def _format_docs(docs):
    """Helper: turn a list of Documents into a readable string for the LLM."""
    if not docs:
        return "No relevant context found."
    parts = []
    for i, doc in enumerate(docs, start=1):
        ticker = doc.metadata.get("ticker", "N/A")
        filing = doc.metadata.get("filing", "N/A")
        parts.append(
            f"[Chunk {i} | ticker={ticker} | filing={filing}]\n{doc.page_content}"
        )
    return "\n\n".join(parts)


# =============================================================================
# 9. RESPONSE GENERATION
# =============================================================================
PROMPT_TEMPLATE = """
You are a financial analyst assistant. Use ONLY the information in the
'Context' section (extracted from SEC 10-K filings) to answer the question.
If the answer is not in the context, say so honestly -- do not invent facts.

Be clear, concise, and explain your reasoning briefly.

Context:
{context}

Chat History:
{chat_history}

Question: {question}

Answer:
"""


def generate_response(query, chat_history=None):
    """
    The full end-to-end RAG flow:
        retrieve (hybrid) -> validate -> prompt LLM -> answer

    `chat_history` is an optional list of (user_msg, assistant_msg) tuples
    so the same function can be used in a conversational setting.
    """
    if _llm is None:
        raise RuntimeError("Pipeline not initialised. Call create_rag_pipeline() first.")

    # Step 1+2: hybrid retrieval, then validation
    docs = hybrid_search(query, k=TOP_K)
    docs = validate_retrieval(query, docs)
    context_text = _format_docs(docs)

    # Step 3: format chat history for the prompt
    history_text = ""
    if chat_history:
        for user_msg, ai_msg in chat_history:
            history_text += f"User: {user_msg}\nAssistant: {ai_msg}\n"

    # Step 4: build the prompt and call the LLM
    prompt = PromptTemplate(
        input_variables=["context", "chat_history", "question"],
        template=PROMPT_TEMPLATE,
    )
    final_prompt = prompt.format(
        context=context_text,
        chat_history=history_text or "(no prior conversation)",
        question=query,
    )

    response = _llm.invoke(final_prompt)
    answer = response.content if hasattr(response, "content") else str(response)

    return {
        "answer": answer,
        "retrieved_context": [doc.page_content for doc in docs],
        "sources": [
            {"ticker": d.metadata.get("ticker"), "filing": d.metadata.get("filing")}
            for d in docs
        ],
    }


# =============================================================================
# PIPELINE INITIALISER
# -----------------------------------------------------------------------------
# Call this ONCE at app startup. It populates the module-level globals
# (_vectorstore, _bm25_retriever, _llm, _all_chunks) which the tool
# functions and agents then use.
# =============================================================================
def create_rag_pipeline():
    """
    Initialise the full RAG pipeline:
        load dataset -> chunk -> embed -> vectorstore -> BM25 -> LLM
    """
    global _vectorstore, _bm25_retriever, _llm, _all_chunks

    try:
        # 1. Embeddings (HuggingFace -- free, runs locally)
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

        # 2. Dataset -> Documents -> chunks
        documents = load_financial_dataset()
        chunks = chunk_documents(documents)
        _all_chunks = chunks

        # 3. Vectorstore (load existing or build new)
        _vectorstore = build_or_load_vectorstore(chunks, embeddings)

        # 4. BM25 keyword retriever for hybrid search
        print("Building BM25 keyword retriever...")
        _bm25_retriever = BM25Retriever.from_documents(chunks)
        _bm25_retriever.k = TOP_K

        # 5. LLM
        google_api_key = os.getenv("GOOGLE_API_KEY")

        print("GOOGLE_API_KEY Loaded:", bool(google_api_key))

        from langchain_openai import ChatOpenAI

        _llm = ChatOpenAI(
          model="gpt-4o-mini",
          api_key=os.getenv("OPENAI_API_KEY"),
          temperature=0.2
)

        print("RAG pipeline initialised successfully.")
        return {
            "vectorstore": _vectorstore,
            "bm25_retriever": _bm25_retriever,
            "llm": _llm,
            "chunks": _all_chunks,
        }

    except Exception as e:
        print(f"An error occurred while initialising the RAG pipeline: {str(e)}")
        raise e
