"""
FinSight — RAG Agent

Responsibilities:
  1. Accept a ticker + question
  2. Filter the FAISS index to only the relevant company's chunks
  3. Run a LangChain RAG chain and return the answer
"""

from __future__ import annotations

from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq

from src.config import FinSightConfig


_RAG_PROMPT = ChatPromptTemplate.from_template("""
You are a financial analyst. Use the following context from the annual report to answer the question.
If you don't know the answer, say so clearly.

Context:
{context}

Question: {question}

Answer:
""")


def _format_docs(docs) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


class RAGAgent:
    """Retrieves relevant passages from 10-K filings and answers questions."""

    def __init__(self, vectorstore: FAISS, llm: ChatGroq, config: FinSightConfig):
        self.vectorstore = vectorstore
        self.llm         = llm
        self.config      = config

    # ── Public API ────────────────────────────────────────────────────────────

    def run(self, ticker: str, question: str) -> str:
        """
        Answer a question grounded in the 10-K for the given ticker.

        Falls back gracefully if the ticker has no 10-K in the index.
        """
        ticker = ticker.upper()

        if ticker not in self.config.rag_supported_tickers:
            return (
                f"No 10-K document is loaded for {ticker}. "
                f"RAG is currently supported for: "
                f"{', '.join(self.config.rag_supported_tickers)}."
            )

        print(f"  [RAGAgent] Retrieving context for {ticker}...")
        company = self._ticker_to_company_key(ticker)
        chain   = self._build_chain(company)
        return chain.invoke(question)

    def summarise(self, ticker: str) -> str:
        """Convenience wrapper: summarise financial performance and risks."""
        return self.run(
            ticker,
            f"Summarize the financial performance and key risks for {ticker}."
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    def _ticker_to_company_key(self, ticker: str) -> str:
        """Map ticker symbol to the metadata key stored in FAISS."""
        mapping = {"AAPL": "apple", "TSLA": "tesla"}
        return mapping.get(ticker, ticker.lower())

    def _build_chain(self, company_key: str):
        retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k":      self.config.rag_top_k,
                "filter": {"company": company_key},
            },
        )
        return (
            {"context": retriever | _format_docs, "question": RunnablePassthrough()}
            | _RAG_PROMPT
            | self.llm
            | StrOutputParser()
        )
