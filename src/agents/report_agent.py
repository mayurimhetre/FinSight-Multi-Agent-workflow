"""
FinSight — Report Writer Agent

Responsibility:
  Aggregate outputs from all other agents and generate a structured
  investment memo in markdown using Groq LLaMA.
"""

from __future__ import annotations

from langchain_groq import ChatGroq

from src.config import FinSightConfig


_MEMO_TEMPLATE = """
You are a senior financial analyst at a top investment bank.
Using the research below, write a professional investment memo for {ticker}.

## RECENT NEWS & SENTIMENT
{news}

## FROM ANNUAL REPORT (RAG)
{rag_context}

## FINANCIAL DATA & RATIO ANALYSIS
{financial_data}

Write a structured investment memo in markdown with these exact sections:

# Investment Memo: {ticker}

## 1. Executive Summary
## 2. Company Overview
## 3. Financial Health
## 4. News & Sentiment Analysis
## 5. Risks
## 6. Valuation Assessment
## 7. Recommendation (Buy / Hold / Sell)

Be concise, professional, and data-driven.
""".strip()


class ReportAgent:
    """Generates a professional investment memo from aggregated agent outputs."""

    def __init__(self, llm: ChatGroq, config: FinSightConfig):
        self.llm    = llm
        self.config = config

    def run(
        self,
        ticker:         str,
        news:           str,
        rag_context:    str,
        financial_data: str,
    ) -> str:
        """
        Generate and return the investment memo as a markdown string.

        Args:
            ticker:         Stock ticker symbol.
            news:           Output from NewsAgent.run().
            rag_context:    Output from RAGAgent.summarise().
            financial_data: Output from AnalystAgent.run().

        Returns:
            Markdown investment memo string.
        """
        print(f"  [ReportAgent] Writing investment memo for {ticker}...")
        prompt = _MEMO_TEMPLATE.format(
            ticker=ticker,
            news=news,
            rag_context=rag_context,
            financial_data=financial_data,
        )
        return self.llm.invoke(prompt).content
