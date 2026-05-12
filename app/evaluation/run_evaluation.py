# =============================================================================
# RAG EVALUATION SCRIPT - Baseline vs Agentic
# -----------------------------------------------------------------------------
# Evaluates TWO pipelines on the same ground-truth questions:
#
#   1. Baseline RAG  -- plain retrieval + LLM (no agents)
#   2. Agentic RAG   -- full CrewAI multi-agent workflow
#
# Both are scored with:
#   - ROUGE-L          : lexical overlap with the reference answer
#   - Faithfulness     : answer grounded in retrieved context (RAGAS + fallback)
#   - Answer Relevance : answer addresses the question (RAGAS + fallback)
#
# Run from project root:
#     python -m app.evaluation.run_evaluation
# =============================================================================

import os
import sys
import contextlib
import numpy as np
import pandas as pd

from rouge_score import rouge_scorer

from app.rag.rag_pipeline import create_rag_pipeline, generate_response
from app.agents.agents import run_agentic_workflow


# =============================================================================
# HELPER: suppress noisy stdout during agentic runs
# -----------------------------------------------------------------------------
# CrewAI's verbose=True prints every agent's reasoning, tool calls, and
# intermediate output. That's helpful for debugging in the API, but spammy
# during evaluation (5 agents × 10 questions = wall of text).
#
# This context manager silences stdout for the block of code inside it.
# stderr is left alone, so real errors still surface.
# =============================================================================
@contextlib.contextmanager
def suppress_output():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


# =============================================================================
# 1. GROUND TRUTH
# -----------------------------------------------------------------------------
# Hand-written Q&A pairs based on the virattt/financial-qa-10K dataset.
# Format: (id, category, question, reference_answer)
# =============================================================================
GROUND_TRUTH = [

    ("Q1", "Revenue", "What are Apple's main revenue segments?",
     "Apple's main revenue segments are iPhone, Mac, iPad, Wearables, and Services."),

    ("Q2", "Products", "What products does Apple sell?",
     "Apple sells iPhone, Mac computers, iPad, Apple Watch, AirPods, and various services."),

    ("Q3", "Risk", "What supply chain risks does Apple face?",
     "Apple faces risks from supply chain disruptions, concentrated manufacturing in Asia, and reliance on key component suppliers."),

    ("Q4", "Cloud", "What is Microsoft's cloud business called?",
     "Microsoft's cloud business is called Azure, part of the Intelligent Cloud segment."),

    ("Q5", "Cloud", "What does Amazon's AWS segment do?",
     "AWS provides cloud computing services including storage, compute, databases, and machine learning to enterprises and developers."),

    ("Q6", "Risk", "What are Tesla's main business risks?",
     "Tesla faces risks related to production scaling, supply chain constraints, battery materials, regulatory changes, and competition in electric vehicles."),

    ("Q7", "Revenue", "How does Meta generate revenue?",
     "Meta generates the majority of its revenue from advertising on Facebook, Instagram, and WhatsApp."),

    ("Q8", "Competition", "What competition does Microsoft face?",
     "Microsoft faces competition in cloud services from Amazon AWS and Google Cloud, in productivity from Google Workspace, and in gaming from Sony and Nintendo."),

    ("Q9", "Products", "What is Alphabet's main product?",
     "Alphabet's main product is Google Search, supported by advertising revenue, along with YouTube, Android, and Google Cloud."),

    ("Q10", "Risk", "What regulatory risks does Meta face?",
     "Meta faces regulatory risks related to data privacy, antitrust scrutiny, content moderation requirements, and changes in advertising policies.")
]


# =============================================================================
# 2. INITIALISE THE RAG PIPELINE
# =============================================================================
print("Initialising RAG pipeline...")
create_rag_pipeline()

from app.rag.rag_pipeline import _vectorstore
hf_embeddings = _vectorstore.embeddings


# =============================================================================
# 3a. GENERATE ANSWERS - BASELINE RAG
# -----------------------------------------------------------------------------
# Plain retrieval + LLM. No agents. Each question goes through the
# RAG pipeline directly and returns a single answer.
# =============================================================================
print("\n" + "=" * 50)
print("BASELINE RAG - GENERATING ANSWERS")
print("=" * 50)

baseline_records = []

for qid, category, question, reference in GROUND_TRUTH:
    try:
        with suppress_output():
            result = generate_response(question)
        generated = result.get("answer", "").strip()
        context = "\n\n".join(result.get("retrieved_context", []))
    except Exception as e:
        generated = f"ERROR: {e}"
        context = ""

    baseline_records.append({
        "id": qid,
        "category": category,
        "question": question,
        "reference": reference,
        "generated": generated,
        "context": context,
    })
    print(f"{qid} done.")


# =============================================================================
# 3b. GENERATE ANSWERS - AGENTIC RAG
# -----------------------------------------------------------------------------
# Full CrewAI multi-agent pipeline: Planner -> Retriever -> Analysis ->
# Portfolio -> Risk. CrewAI's verbose output is suppressed here so the
# terminal stays readable.
# =============================================================================
print("\n" + "=" * 50)
print("AGENTIC RAG - GENERATING ANSWERS")
print("=" * 50)

agentic_records = []

for qid, category, question, reference in GROUND_TRUTH:
    try:
        with suppress_output():
            result = run_agentic_workflow(question)
        generated = result.get("response", "").strip()
        context = result.get("retrieved_context", "")
    except Exception as e:
        generated = f"ERROR: {e}"
        context = ""

    agentic_records.append({
        "id": qid,
        "category": category,
        "question": question,
        "reference": reference,
        "generated": generated,
        "context": context,
    })
    print(f"{qid} done.")


# =============================================================================
# 4. ROUGE-L SCORE (applied to both pipelines)
# =============================================================================
scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

for r in baseline_records + agentic_records:
    r["ROUGE-L"] = round(
        scorer.score(r["reference"], r["generated"])["rougeL"].fmeasure, 4
    )


# =============================================================================
# 5. FAITHFULNESS + ANSWER RELEVANCE via RAGAS (applied to both)
# -----------------------------------------------------------------------------
# RAGAS uses an LLM as a judge. If it fails (missing API key, rate limits),
# the fallback in section 6 fills in the missing scores.
# =============================================================================
try:
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy
    from datasets import Dataset

    for record_set, label in [(baseline_records, "baseline"),
                               (agentic_records, "agentic")]:
        ds = Dataset.from_list([
            {
                "question": r["question"],
                "answer": r["generated"],
                "contexts": [r["context"]],
            }
            for r in record_set
        ])

        with suppress_output():
            ragas_df = evaluate(
                ds,
                metrics=[faithfulness, answer_relevancy],
            ).to_pandas()

        for i, r in enumerate(record_set):
            r["faithfulness"] = round(float(ragas_df.loc[i, "faithfulness"]), 4)
            r["answer_relevancy"] = round(float(ragas_df.loc[i, "answer_relevancy"]), 4)

        print(f"RAGAS scores computed for {label} pipeline.")

except Exception as e:
    print("\nRAGAS failed -- using fallback methods.")
    print(f"Reason: {e}")


# =============================================================================
# 6. FALLBACK: simple faithfulness + cosine relevance (applied to both)
# =============================================================================
def cosine(a, b):
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def simple_faithfulness(answer, context):
    if not answer or not context:
        return 0.0
    answer_words = set(answer.lower().split())
    context_lower = context.lower()
    matched = sum(1 for w in answer_words if w in context_lower)
    return round(matched / (len(answer_words) + 1e-9), 4)


for r in baseline_records + agentic_records:
    try:
        r["Relevance"] = round(
            cosine(
                hf_embeddings.embed_query(r["question"]),
                hf_embeddings.embed_query(r["generated"]),
            ),
            4,
        )
    except Exception:
        r["Relevance"] = 0.0

    r["Faithful"] = simple_faithfulness(r["generated"], r["context"])


# =============================================================================
# 7. PER-QUESTION SCORES - TWO TABLES (Baseline + Agentic)
# =============================================================================
baseline_df = pd.DataFrame([
    {
        "ID": r["id"],
        "Category": r["category"],
        "ROUGE-L": r["ROUGE-L"],
        "Faithful": r["Faithful"],
        "Relevance": r["Relevance"],
    }
    for r in baseline_records
])

agentic_df = pd.DataFrame([
    {
        "ID": r["id"],
        "Category": r["category"],
        "ROUGE-L": r["ROUGE-L"],
        "Faithful": r["Faithful"],
        "Relevance": r["Relevance"],
    }
    for r in agentic_records
])

print("\n" + "=" * 65)
print("BASELINE RAG - PER-QUESTION SCORES")
print("=" * 65)
print(baseline_df.to_string(index=False))

print("\n" + "=" * 65)
print("AGENTIC RAG - PER-QUESTION SCORES")
print("=" * 65)
print(agentic_df.to_string(index=False))


# =============================================================================
# 8. SIDE-BY-SIDE COMPARISON TABLE
# =============================================================================
comparison_df = pd.DataFrame({
    "ID": baseline_df["ID"],
    "Category": baseline_df["Category"],
    "ROUGE-L (Base)": baseline_df["ROUGE-L"],
    "ROUGE-L (Agent)": agentic_df["ROUGE-L"],
    "ΔROUGE": (agentic_df["ROUGE-L"] - baseline_df["ROUGE-L"]).round(4),
    "Faith (Base)": baseline_df["Faithful"],
    "Faith (Agent)": agentic_df["Faithful"],
    "ΔFaith": (agentic_df["Faithful"] - baseline_df["Faithful"]).round(4),
    "Rel (Base)": baseline_df["Relevance"],
    "Rel (Agent)": agentic_df["Relevance"],
    "ΔRel": (agentic_df["Relevance"] - baseline_df["Relevance"]).round(4),
})

print("\n" + "=" * 100)
print("SIDE-BY-SIDE COMPARISON (Per Question)")
print("=" * 100)
print(comparison_df.to_string(index=False))


# =============================================================================
# 9. AGGREGATE SUMMARY - Baseline vs Agentic (averaged)
# =============================================================================
baseline_means = baseline_df[["ROUGE-L", "Faithful", "Relevance"]].mean()
agentic_means = agentic_df[["ROUGE-L", "Faithful", "Relevance"]].mean()

thresholds = {
    "ROUGE-L": 0.40,
    "Faithful": 0.80,
    "Relevance": 0.75,
}

summary_df = pd.DataFrame([
    {
        "Metric": metric,
        "Baseline": round(baseline_means[metric], 4),
        "Agentic": round(agentic_means[metric], 4),
        "Improvement": round(agentic_means[metric] - baseline_means[metric], 4),
        "Threshold": threshold,
        "Baseline Status": "PASS" if baseline_means[metric] >= threshold else "REVIEW",
        "Agentic Status": "PASS" if agentic_means[metric] >= threshold else "REVIEW",
    }
    for metric, threshold in thresholds.items()
])

print("\n" + "=" * 100)
print("AGGREGATE SUMMARY - Baseline vs Agentic")
print("=" * 100)
print(summary_df.to_string(index=False))


# =============================================================================
# 10. VERDICT
# =============================================================================
wins = sum(1 for m in thresholds if agentic_means[m] > baseline_means[m])
total = len(thresholds)

print("\n" + "=" * 65)
print("VERDICT")
print("=" * 65)
print(f"Agentic RAG outperformed Baseline on {wins} out of {total} metrics.")
if wins == total:
    print("The multi-agent workflow consistently improves the pipeline.")
elif wins > total / 2:
    print("The multi-agent workflow improves most metrics.")
else:
    print("The multi-agent workflow shows mixed results -- consider")
    print("tuning agent prompts or reviewing tool selection.")


# =============================================================================
# 11. SAVE EVERYTHING TO CSV
# =============================================================================
baseline_df.to_csv("evaluation_baseline.csv", index=False)
agentic_df.to_csv("evaluation_agentic.csv", index=False)
comparison_df.to_csv("evaluation_comparison.csv", index=False)
summary_df.to_csv("evaluation_summary.csv", index=False)

print("\nSaved:")
print("  - evaluation_baseline.csv   (baseline per-question scores)")
print("  - evaluation_agentic.csv    (agentic per-question scores)")
print("  - evaluation_comparison.csv (side-by-side per question)")
print("  - evaluation_summary.csv    (aggregated comparison)")