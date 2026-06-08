"""
FinSight — News Fetcher Agent

Responsibilities:
  1. Search DuckDuckGo for latest news about a ticker
  2. Pass raw headlines to Groq LLaMA for structured sentiment analysis
  3. Return a combined news + sentiment string for the report writer
"""

from __future__ import annotations

from langchain_community.tools import DuckDuckGoSearchRun
from langchain_groq import ChatGroq

from src.config import FinSightConfig


class NewsAgent:
    """Fetches live news and produces an AI sentiment summary for a given ticker."""

    def __init__(self, llm: ChatGroq, config: FinSightConfig):
        self.llm    = llm
        self.config = config
        self.search = DuckDuckGoSearchRun()

    # ── Public API ────────────────────────────────────────────────────────────

    def run(self, ticker: str) -> str:
        """
        Fetch news and return a formatted string:
            RAW NEWS HEADLINES:
            ...

            SENTIMENT ANALYSIS:
            ...
        """
        company_name = self.config.company_map.get(ticker.upper(), ticker)
        print(f"  [NewsAgent] Fetching news for {company_name} ({ticker})...")

        raw_news  = self._fetch_news(ticker, company_name)
        sentiment = self._analyse_sentiment(ticker, raw_news)

        return (
            f"RAW NEWS HEADLINES:\n"
            f"{raw_news[:self.config.news_snippet_chars]}...\n\n"
            f"SENTIMENT ANALYSIS:\n{sentiment}"
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    def _fetch_news(self, ticker: str, company_name: str) -> str:
        query = (
            f"{company_name} {ticker} stock news latest "
            f"{self.config.news_query_year}"
        )
        return self.search.run(query)

    def _analyse_sentiment(self, ticker: str, raw_news: str) -> str:
        prompt = f"""
You are a financial news analyst. Analyse the following recent news for {ticker}
and provide a structured sentiment summary.

NEWS:
{raw_news}

Provide:
1. Overall sentiment: Positive / Negative / Neutral
2. Key themes (2-3 bullet points)
3. Any major events or announcements
4. Potential impact on stock price (short-term)

Keep it concise and factual.
""".strip()
        return self.llm.invoke(prompt).content
