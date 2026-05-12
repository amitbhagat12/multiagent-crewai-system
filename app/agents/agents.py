# =============================================================================
# AGENTS - CrewAI multi-agent orchestration
# -----------------------------------------------------------------------------
# This file builds the AGENTIC layer on top of the RAG pipeline.
#
# What you'll find here:
#   - 5 CrewAI agents: Planner, Retriever, Analysis, Portfolio, Risk
#   - The RAG functions (from rag_pipeline.py) wrapped as CrewAI Tools
#   - A Planner -> Executor workflow
#   - "Agents as tools" + tool chaining (output of agent A -> input of agent B)
#   - Simple conversational memory (just a Python list - beginner-friendly)
#
# Flow at a glance:
#
#     User Query
#         |
#     Planner Agent     (decides which agents/tools to use)
#         |
#     Retriever Agent   (calls RAG tools -> fetches context)
#         |
#     Analysis Agent    (interprets the retrieved context)
#         |
#     Portfolio Agent   (gives allocation suggestions)
#         |
#     Risk Agent        (highlights risks)
#         |
#     Final Response
# =============================================================================

import os
from dotenv import load_dotenv

from crewai import Agent, Task, Crew, Process
from crewai.tools import tool

# Import the RAG tool functions we defined in rag_pipeline.py
from app.rag.rag_pipeline import (
    rag_tool_retrieve,
    rag_tool_hybrid_search,
    rag_tool_filtered_search,
    rag_tool_validated_retrieve,
)

load_dotenv()


# =============================================================================
# RAG TOOLS (exposed to CrewAI agents)
# -----------------------------------------------------------------------------
# CrewAI agents can only call functions that are decorated with @tool.
# Each tool below is a thin wrapper around the corresponding RAG function.
# The docstrings matter -- the agent's LLM reads them to decide
# WHEN to call which tool.
# =============================================================================
@tool("Semantic Retrieval Tool")
def semantic_retrieval_tool(query: str) -> str:
    """
    Retrieves the most semantically similar chunks from the 10-K vectorstore.
    Use this for general questions about a company's filings.
    Input: a natural-language query string.
    """
    return rag_tool_retrieve(query)


@tool("Hybrid Search Tool")
def hybrid_search_tool(query: str) -> str:
    """
    Combines semantic search with keyword (BM25) search.
    Use this when the question contains exact terms like tickers,
    dollar amounts, or specific segment names.
    Input: a natural-language query string.
    """
    return rag_tool_hybrid_search(query)


@tool("Filtered Retrieval Tool")
def filtered_retrieval_tool(query: str, ticker: str) -> str:
    """
    Retrieves chunks ONLY from filings of the given ticker symbol.
    Use this when the user asks about a specific company.
    Inputs: query (string), ticker (e.g. 'AAPL', 'MSFT').
    """
    return rag_tool_filtered_search(query, ticker)


@tool("Validated Retrieval Tool")
def validated_retrieval_tool(query: str) -> str:
    """
    Highest-quality retrieval: hybrid search followed by a relevance
    validation step that drops weak chunks. Use this when grounding
    quality matters most.
    Input: a natural-language query string.
    """
    return rag_tool_validated_retrieve(query)


# =============================================================================
# AGENT DEFINITIONS
# -----------------------------------------------------------------------------
# Each agent gets:
#   - a role (one-liner job title)
#   - a goal (what it's trying to achieve)
#   - a backstory (persona/context the LLM uses to stay in character)
#   - tools (functions it's allowed to call)
# =============================================================================
def build_agents():
    """Construct and return all 5 agents."""

    # --- 1. PLANNER --------------------------------------------------------
    # Looks at the user query, decides which downstream agents need to run.
    # Doesn't call RAG tools directly -- it delegates.
    planner = Agent(
        role="Planning Coordinator",
        goal=(
            "Read the user's financial question and decide which specialist "
            "agents should handle it, and in what order."
        ),
        backstory=(
            "You are an experienced research lead who breaks down complex "
            "financial queries into a small set of clear, actionable steps."
        ),
        verbose=True,
        allow_delegation=True,   # this is what makes 'agents-as-tools' work
    )

    # --- 2. RETRIEVER ------------------------------------------------------
    # The ONLY agent that talks to the RAG pipeline.
    # All other agents consume what this agent retrieves.
    retriever = Agent(
        role="Financial Filings Retriever",
        goal=(
            "Fetch the most relevant passages from SEC 10-K filings to answer "
            "the user's question. Use the validated retrieval tool when "
            "possible, and the filtered tool when a specific company is named."
        ),
        backstory=(
            "You are a meticulous research analyst who knows how to query "
            "internal knowledge bases and return only well-grounded evidence."
        ),
        tools=[
            semantic_retrieval_tool,
            hybrid_search_tool,
            filtered_retrieval_tool,
            validated_retrieval_tool,
        ],
        verbose=True,
        allow_delegation=False,
    )

    # --- 3. ANALYSIS -------------------------------------------------------
    # Consumes the Retriever's output, extracts insights.
    analysis = Agent(
        role="Financial Analyst",
        goal=(
            "Read retrieved 10-K passages and produce a clear, grounded "
            "summary of what they say about the user's question."
        ),
        backstory=(
            "You are a senior equity analyst with 10+ years of experience "
            "interpreting 10-K filings, MD&A sections, and risk disclosures."
        ),
        verbose=True,
        allow_delegation=False,
    )

    # --- 4. PORTFOLIO ------------------------------------------------------
    # Turns analysis into allocation/recommendation language.
    portfolio = Agent(
        role="Portfolio Strategist",
        goal=(
            "Based on the financial analysis, suggest how the company might "
            "fit into a diversified portfolio (educational guidance only -- "
            "not personalised financial advice)."
        ),
        backstory=(
            "You are a portfolio strategist who translates fundamental "
            "analysis into clear, structured allocation reasoning."
        ),
        verbose=True,
        allow_delegation=False,
    )

    # --- 5. RISK ASSESSMENT ------------------------------------------------
    # Final pass: identifies risks and uncertainties.
    risk = Agent(
        role="Risk Assessment Specialist",
        goal=(
            "Identify the key risks, uncertainties, and red flags implied "
            "by the retrieved 10-K context and the upstream analysis."
        ),
        backstory=(
            "You are a risk officer who reads 10-K Risk Factor sections "
            "for a living and flags concerns other analysts miss."
        ),
        verbose=True,
        allow_delegation=False,
    )

    return planner, retriever, analysis, portfolio, risk


# =============================================================================
# CONVERSATIONAL MEMORY
# -----------------------------------------------------------------------------
# Kept deliberately simple: a Python list of (user_msg, agent_response) tuples.
# We format it as text when we pass it into agent tasks.
# =============================================================================
chat_history = []


def format_history():
    """Turn the chat history into a string the agents can read."""
    if not chat_history:
        return "(no prior conversation)"
    lines = []
    for user_msg, ai_msg in chat_history[-3:]:   # keep only last 3 turns
        lines.append(f"User: {user_msg}")
        lines.append(f"Assistant: {ai_msg}")
    return "\n".join(lines)


# =============================================================================
# MAIN AGENTIC WORKFLOW
# -----------------------------------------------------------------------------
# This is what FastAPI calls. It runs the full Planner -> Executor pipeline
# and returns the final response + intermediate steps.
# =============================================================================
def run_agentic_workflow(user_query: str):
    """
    Orchestrate the full multi-agent flow for one user query.

    How tool chaining works here:
      - Task #1 (Retriever) produces retrieved context.
      - Task #2 (Analysis) sets `context=[retrieval_task]` -> CrewAI
        feeds task #1's output into task #2 automatically.
      - Same for Portfolio (depends on analysis) and Risk (depends on both).
      - This is the "output of one agent becomes input of the next" pattern.
    """
    planner, retriever, analysis, portfolio, risk = build_agents()
    history = format_history()

    # --- TASK 1: Planning -------------------------------------------------
    planning_task = Task(
        description=(
            f"A user asked the following financial question:\n"
            f"'{user_query}'\n\n"
            f"Prior conversation:\n{history}\n\n"
            "Decide what information needs to be retrieved (what company, "
            "what topic, what kind of detail). Briefly list the plan."
        ),
        expected_output="A short numbered plan of steps to answer the question.",
        agent=planner,
    )

    # --- TASK 2: Retrieval (uses RAG tools) -------------------------------
    retrieval_task = Task(
        description=(
            f"Use your retrieval tools to fetch passages from 10-K filings "
            f"that help answer this question:\n'{user_query}'\n\n"
            "Prefer the Validated Retrieval Tool. If a specific ticker is "
            "mentioned, use the Filtered Retrieval Tool with that ticker."
        ),
        expected_output="The raw retrieved 10-K passages with their metadata.",
        agent=retriever,
        context=[planning_task],
    )

    # --- TASK 3: Analysis -------------------------------------------------
    analysis_task = Task(
        description=(
            f"Read the retrieved passages and write a clear, grounded "
            f"analysis answering the user's question:\n'{user_query}'\n\n"
            "Cite the ticker for any specific claim. If the passages do "
            "not contain enough info, say so explicitly."
        ),
        expected_output="A short, grounded analysis (3-6 sentences).",
        agent=analysis,
        context=[retrieval_task],   # consumes Retriever's output
    )

    # --- TASK 4: Portfolio ------------------------------------------------
    portfolio_task = Task(
        description=(
            "Based on the analysis above, give a short educational view "
            "of how this company/topic might fit into a diversified "
            "portfolio. State that this is NOT personalised advice."
        ),
        expected_output="2-4 sentences of portfolio-fit reasoning.",
        agent=portfolio,
        context=[analysis_task],
    )

    # --- TASK 5: Risk -----------------------------------------------------
    risk_task = Task(
        description=(
            "Based on the retrieved context and the analysis, list the "
            "top 2-3 risks or uncertainties the user should be aware of."
        ),
        expected_output="A bulleted list of 2-3 risks with one-line explanations.",
        agent=risk,
        context=[retrieval_task, analysis_task],   # depends on BOTH upstream
    )

    # --- CREW: orchestrates everything ------------------------------------
    crew = Crew(
        agents=[planner, retriever, analysis, portfolio, risk],
        tasks=[planning_task, retrieval_task, analysis_task, portfolio_task, risk_task],
        process=Process.sequential,   # run tasks in order; outputs flow downstream
        verbose=True,
    )

    result = crew.kickoff()
    final_response = str(result)

    # Save to memory
    chat_history.append((user_query, final_response))

    # Collect intermediate outputs so the UI can show them
    agent_steps = []
    for t in [planning_task, retrieval_task, analysis_task, portfolio_task, risk_task]:
        agent_steps.append({
            "agent": t.agent.role,
            "output": str(t.output) if t.output else "",
        })

    # Pull the raw retrieved context out of the retrieval task's output
    retrieved_context = str(retrieval_task.output) if retrieval_task.output else ""

    return {
        "response": final_response,
        "retrieved_context": retrieved_context,
        "agent_steps": agent_steps,
    }
