"""
FinSight — Analyst Agent

Responsibilities:
  1. Pull live stock data and financial ratios via yfinance
  2. Format them into a structured string
  3. Ask Groq LLaMA for a plain-language interpretation
  4. Return the combined financial analysis block
"""

from __future__ import annotations

import yfinance as yf
from langchain_groq import ChatGroq

from src.config import FinSightConfig


class AnalystAgent:
    """Fetches live financials, calculates ratios, and generates an LLM interpretation."""

    def __init__(self, llm: ChatGroq, config: FinSightConfig):
        self.llm    = llm
        self.config = config

    # ── Public API ────────────────────────────────────────────────────────────

    def run(self, ticker: str) -> str:
        """
        Returns a formatted string containing:
          - Core financial data (price, market cap, revenue, etc.)
          - Detailed ratio table (valuation, profitability, liquidity, leverage, growth)
          - LLM plain-language interpretation
        """
        print(f"  [AnalystAgent] Pulling live data for {ticker}...")
        stock_data = self._get_stock_data(ticker)
        ratios     = self._calculate_ratios(ticker)

        formatted_data   = self._format_stock_data(stock_data)
        formatted_ratios = self._format_ratios(ticker, ratios)
        llm_summary      = self._get_llm_interpretation(ticker, formatted_ratios)

        return (
            f"{formatted_data}\n\n"
            f"{formatted_ratios}\n\n"
            f"LLM INTERPRETATION:\n{llm_summary}"
        )

    def get_raw_stock_data(self, ticker: str) -> dict:
        """Expose raw stock data dict (used by Streamlit for metric cards)."""
        return self._get_stock_data(ticker)

    def get_raw_ratios(self, ticker: str) -> dict:
        """Expose raw ratios dict (used by Streamlit for the ratios tab)."""
        return self._calculate_ratios(ticker)

    # ── Private: data fetching ────────────────────────────────────────────────

    def _get_stock_data(self, ticker: str) -> dict:
        info = yf.Ticker(ticker).info
        return {
            "ticker":           ticker,
            "company_name":     info.get("longName", "N/A"),
            "current_price":    info.get("currentPrice", "N/A"),
            "market_cap":       info.get("marketCap", "N/A"),
            "pe_ratio":         info.get("trailingPE", "N/A"),
            "forward_pe":       info.get("forwardPE", "N/A"),
            "revenue":          info.get("totalRevenue", "N/A"),
            "net_income":       info.get("netIncomeToCommon", "N/A"),
            "debt_to_equity":   info.get("debtToEquity", "N/A"),
            "return_on_equity": info.get("returnOnEquity", "N/A"),
            "profit_margin":    info.get("profitMargins", "N/A"),
            "52_week_high":     info.get("fiftyTwoWeekHigh", "N/A"),
            "52_week_low":      info.get("fiftyTwoWeekLow", "N/A"),
        }

    def _calculate_ratios(self, ticker: str) -> dict:
        info   = yf.Ticker(ticker).info
        ratios = {}
        try:
            def _r(key, multiplier=1):
                val = info.get(key, 0) or 0
                return round(val * multiplier, 2)

            ratios["pe_ratio"]         = _r("trailingPE")
            ratios["forward_pe"]       = _r("forwardPE")
            ratios["price_to_book"]    = _r("priceToBook")
            ratios["price_to_sales"]   = _r("priceToSalesTrailing12Months")
            ratios["ev_to_ebitda"]     = _r("enterpriseToEbitda")
            ratios["profit_margin"]    = _r("profitMargins",    100)
            ratios["operating_margin"] = _r("operatingMargins", 100)
            ratios["roe"]              = _r("returnOnEquity",   100)
            ratios["roa"]              = _r("returnOnAssets",   100)
            ratios["current_ratio"]    = _r("currentRatio")
            ratios["quick_ratio"]      = _r("quickRatio")
            ratios["debt_to_equity"]   = _r("debtToEquity")
            ratios["revenue_growth"]   = _r("revenueGrowth",   100)
            ratios["earnings_growth"]  = _r("earningsGrowth",  100)
            ratios["dividend_yield"]   = _r("dividendYield",   100)
        except Exception as e:
            print(f"  [AnalystAgent] Ratio calculation warning: {e}")
        return ratios

    # ── Private: formatting ───────────────────────────────────────────────────

    def _format_stock_data(self, data: dict) -> str:
        def _fmt(v):
            if isinstance(v, (int, float)):
                if v >= 1e12: return f"${v / 1e12:.2f}T"
                if v >= 1e9:  return f"${v / 1e9:.2f}B"
                if v >= 1e6:  return f"${v / 1e6:.2f}M"
                return f"${v:,.0f}"
            return str(v)

        return (
            f"Company: {data['company_name']} ({data['ticker']})\n"
            f"Current Price:    ${data['current_price']}\n"
            f"Market Cap:       {_fmt(data['market_cap'])}\n"
            f"P/E (Trailing):   {data['pe_ratio']}\n"
            f"P/E (Forward):    {data['forward_pe']}\n"
            f"Total Revenue:    {_fmt(data['revenue'])}\n"
            f"Net Income:       {_fmt(data['net_income'])}\n"
            f"Debt / Equity:    {data['debt_to_equity']}\n"
            f"Return on Equity: {data['return_on_equity']}\n"
            f"Profit Margin:    {data['profit_margin']}\n"
            f"52-Week High:     ${data['52_week_high']}\n"
            f"52-Week Low:      ${data['52_week_low']}"
        )

    def _format_ratios(self, ticker: str, r: dict) -> str:
        return f"""Financial Ratio Analysis — {ticker}

VALUATION:
  P/E Ratio (Trailing):     {r.get('pe_ratio', 'N/A')}
  P/E Ratio (Forward):      {r.get('forward_pe', 'N/A')}
  Price to Book:            {r.get('price_to_book', 'N/A')}
  Price to Sales:           {r.get('price_to_sales', 'N/A')}
  EV / EBITDA:              {r.get('ev_to_ebitda', 'N/A')}

PROFITABILITY:
  Profit Margin:            {r.get('profit_margin', 'N/A')}%
  Operating Margin:         {r.get('operating_margin', 'N/A')}%
  Return on Equity (ROE):   {r.get('roe', 'N/A')}%
  Return on Assets (ROA):   {r.get('roa', 'N/A')}%

LIQUIDITY:
  Current Ratio:            {r.get('current_ratio', 'N/A')}
  Quick Ratio:              {r.get('quick_ratio', 'N/A')}

LEVERAGE:
  Debt to Equity:           {r.get('debt_to_equity', 'N/A')}

GROWTH:
  Revenue Growth:           {r.get('revenue_growth', 'N/A')}%
  Earnings Growth:          {r.get('earnings_growth', 'N/A')}%

DIVIDEND:
  Dividend Yield:           {r.get('dividend_yield', 'N/A')}%"""

    # ── Private: LLM interpretation ───────────────────────────────────────────

    def _get_llm_interpretation(self, ticker: str, formatted_ratios: str) -> str:
        prompt = f"""
You are a senior financial analyst. Analyse the following financial ratios for {ticker}
and provide a plain-language interpretation.

{formatted_ratios}

Provide:
1. Overall financial health assessment (1-2 sentences)
2. Valuation assessment — is the stock cheap or expensive?
3. Profitability assessment — how efficient is the company?
4. Risk assessment — any red flags in liquidity or leverage?
5. One-line verdict: Strong / Neutral / Weak
""".strip()
        return self.llm.invoke(prompt).content
