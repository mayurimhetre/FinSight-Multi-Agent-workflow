"""
FinSight — Central configuration
All API keys, model names, paths, and constants live here.
"""

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class FinSightConfig:
    # ── LLM ───────────────────────────────────────────────────────────────────
    groq_api_key: Optional[str]  = None
    llm_model:    str            = "llama-3.3-70b-versatile"
    llm_temperature: float       = 0.0

    # ── Embeddings ────────────────────────────────────────────────────────────
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # ── FAISS ────────────────────────────────────────────────────────────────
    faiss_index_path: str = "/content/drive/MyDrive/FinSight/faiss_index"

    # ── RAG retrieval ────────────────────────────────────────────────────────
    rag_top_k: int = 6

    # ── News search ──────────────────────────────────────────────────────────
    news_query_year: str = "2025"
    news_snippet_chars: int = 500

    # ── Supported companies (for RAG filter + display names) ─────────────────
    company_map: dict = field(default_factory=lambda: {
        "AAPL":  "Apple",
        "TSLA":  "Tesla",
        "MSFT":  "Microsoft",
        "GOOGL": "Google",
        "AMZN":  "Amazon",
        "NVDA":  "NVIDIA",
        "META":  "Meta",
    })

    # ── Tickers with 10-K loaded into FAISS ──────────────────────────────────
    rag_supported_tickers: list = field(default_factory=lambda: ["AAPL", "TSLA"])

    def apply_env(self) -> "FinSightConfig":
        """Pull GROQ_API_KEY from environment if not set explicitly."""
        if not self.groq_api_key:
            self.groq_api_key = os.getenv("GROQ_API_KEY", "")
        return self

    def set_groq_env(self):
        """Write the key into os.environ so LangChain picks it up."""
        if self.groq_api_key:
            os.environ["GROQ_API_KEY"] = self.groq_api_key
