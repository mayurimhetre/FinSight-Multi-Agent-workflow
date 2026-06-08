"""
FinSight — LangGraph Workflow

Assembles the 4 agents into a LangGraph StateGraph pipeline:

    news_fetcher → rag_agent → analyst_agent → report_writer → END

Usage:
    from src.workflow import build_pipeline, run_pipeline

    pipeline = build_pipeline(config)
    result   = run_pipeline(pipeline, ticker="AAPL")
    print(result["report"])
"""

from __future__ import annotations

from typing import TypedDict

from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langgraph.graph import StateGraph, END

from src.config import FinSightConfig
from src.embeddings import load_index
from src.agents.news_agent import NewsAgent
from src.agents.rag_agent import RAGAgent
from src.agents.analyst_agent import AnalystAgent
from src.agents.report_agent import ReportAgent


# ── Shared state schema ────────────────────────────────────────────────────────

class AgentState(TypedDict):
    ticker:         str
    news:           str
    rag_context:    str
    financial_data: str
    report:         str
    messages:       list


# ── Pipeline builder ───────────────────────────────────────────────────────────

class FinSightPipeline:
    """
    Holds all agents and the compiled LangGraph app.
    Keeps agent instances accessible so Streamlit can call them directly
    (e.g. RAGAgent for the chat tab).
    """

    def __init__(
        self,
        config:      FinSightConfig,
        vectorstore: FAISS,
        llm:         ChatGroq,
    ):
        self.config      = config
        self.vectorstore = vectorstore
        self.llm         = llm

        # ── Instantiate agents ─────────────────────────────────────────────
        self.news_agent    = NewsAgent(llm=llm, config=config)
        self.rag_agent     = RAGAgent(vectorstore=vectorstore, llm=llm, config=config)
        self.analyst_agent = AnalystAgent(llm=llm, config=config)
        self.report_agent  = ReportAgent(llm=llm, config=config)

        # ── Compile graph ──────────────────────────────────────────────────
        self.app = self._build_graph()

    # ── Public: run full pipeline ──────────────────────────────────────────────

    def run(self, ticker: str) -> AgentState:
        """
        Execute the full 4-agent pipeline for a given ticker.

        Returns:
            Final AgentState dict with keys:
            ticker, news, rag_context, financial_data, report, messages.
        """
        initial_state: AgentState = {
            "ticker":         ticker.upper(),
            "news":           "",
            "rag_context":    "",
            "financial_data": "",
            "report":         "",
            "messages":       [],
        }
        return self.app.invoke(initial_state)

    # ── Private: graph construction ────────────────────────────────────────────

    def _build_graph(self) -> StateGraph:
        # ── Node functions (thin wrappers around agent classes) ────────────
        def news_fetcher_node(state: AgentState) -> dict:
            print("\n📰 News Fetcher running...")
            return {"news": self.news_agent.run(state["ticker"])}

        def rag_agent_node(state: AgentState) -> dict:
            print("\n📄 RAG Agent running...")
            return {"rag_context": self.rag_agent.summarise(state["ticker"])}

        def analyst_agent_node(state: AgentState) -> dict:
            print("\n📊 Analyst Agent running...")
            return {"financial_data": self.analyst_agent.run(state["ticker"])}

        def report_writer_node(state: AgentState) -> dict:
            print("\n✍️  Report Writer running...")
            memo = self.report_agent.run(
                ticker=state["ticker"],
                news=state["news"],
                rag_context=state["rag_context"],
                financial_data=state["financial_data"],
            )
            return {"report": memo}

        # ── Wire the graph ─────────────────────────────────────────────────
        graph = StateGraph(AgentState)
        graph.add_node("news_fetcher",  news_fetcher_node)
        graph.add_node("rag_agent",     rag_agent_node)
        graph.add_node("analyst_agent", analyst_agent_node)
        graph.add_node("report_writer", report_writer_node)

        graph.set_entry_point("news_fetcher")
        graph.add_edge("news_fetcher",  "rag_agent")
        graph.add_edge("rag_agent",     "analyst_agent")
        graph.add_edge("analyst_agent", "report_writer")
        graph.add_edge("report_writer", END)

        return graph.compile()


# ── Convenience factory ────────────────────────────────────────────────────────

def build_pipeline(config: FinSightConfig) -> FinSightPipeline:
    """
    Load the FAISS index and instantiate the full FinSightPipeline.

    Call once at app startup; reuse the returned object for all runs.

    Args:
        config: FinSightConfig with groq_api_key and faiss_index_path set.

    Returns:
        Ready-to-use FinSightPipeline.
    """
    config.set_groq_env()

    llm = ChatGroq(
        model=config.llm_model,
        temperature=config.llm_temperature,
    )
    vectorstore = load_index(config)

    print("✅ Pipeline ready.")
    return FinSightPipeline(config=config, vectorstore=vectorstore, llm=llm)
