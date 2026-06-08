"""
FinSight — Streamlit Frontend
Run:  streamlit run app.py
"""

import streamlit as st
import time
import os

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FinSight – AI Investment Analyst",
    page_icon="📈",
    layout="wide",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  [data-testid="stAppViewContainer"] { background: #0d1117; color: #e6edf3; }
  [data-testid="stSidebar"] { background: #161b22; border-right: 1px solid #30363d; }
  .metric-card {
    background: #161b22; border: 1px solid #30363d;
    border-radius: 10px; padding: 16px 20px; text-align: center;
  }
  .metric-card .label { color: #8b949e; font-size: 12px; text-transform: uppercase; letter-spacing: 1px; }
  .metric-card .value { color: #58a6ff; font-size: 24px; font-weight: 700; margin-top: 4px; }
  .metric-card .delta { font-size: 12px; margin-top: 2px; }
  .positive { color: #3fb950; } .negative { color: #f85149; }
  .memo-box {
    background: #161b22; border: 1px solid #30363d;
    border-radius: 10px; padding: 24px; line-height: 1.7;
  }
  .chat-msg-user {
    background: #1f6feb; color: white; padding: 10px 14px;
    border-radius: 10px 10px 2px 10px; margin: 6px 0;
    max-width: 80%; margin-left: auto;
  }
  .chat-msg-bot {
    background: #21262d; color: #e6edf3; padding: 10px 14px;
    border-radius: 10px 10px 10px 2px; margin: 6px 0;
    max-width: 85%; border: 1px solid #30363d;
  }
  .section-header {
    color: #58a6ff; font-size: 13px; font-weight: 600;
    text-transform: uppercase; letter-spacing: 1px;
    border-bottom: 1px solid #30363d; padding-bottom: 6px;
    margin-bottom: 14px; margin-top: 20px;
  }
</style>
""", unsafe_allow_html=True)


# ── Session state defaults ─────────────────────────────────────────────────────
for key, default in {
    "pipeline":       None,
    "result":         None,
    "stock_data":     None,
    "ratios":         None,
    "chat_history":   [],
    "current_ticker": "",
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ── Cached pipeline loader ─────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading pipeline (first run ~30s)...")
def get_pipeline(groq_key: str, faiss_path: str):
    """Load once, reuse across all reruns."""
    from src.config import FinSightConfig
    from src.workflow import build_pipeline

    config = FinSightConfig(
        groq_api_key=groq_key,
        faiss_index_path=faiss_path,
    )
    return build_pipeline(config)


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 📈 FinSight")
    st.markdown("<span style='color:#8b949e;font-size:13px'>AI-Powered Investment Analyst</span>",
                unsafe_allow_html=True)
    st.markdown("---")

    st.markdown('<div class="section-header">Configuration</div>', unsafe_allow_html=True)
    groq_key   = st.text_input("Groq API Key", type="password", placeholder="gsk_...")
    faiss_path = st.text_input(
        "FAISS Index Path",
        value="/content/drive/MyDrive/FinSight/faiss_index",
    )

    st.markdown('<div class="section-header">Analysis Target</div>', unsafe_allow_html=True)
    ticker_options = ["AAPL", "TSLA", "MSFT", "GOOGL", "AMZN"]
    ticker         = st.selectbox("Select Ticker", ticker_options)
    custom_ticker  = st.text_input("Or enter custom ticker", placeholder="e.g. NVDA")
    final_ticker   = custom_ticker.upper().strip() if custom_ticker.strip() else ticker

    st.markdown("---")
    run_btn = st.button("🚀 Run FinSight Analysis", use_container_width=True, type="primary")

    st.markdown("---")
    st.markdown("""
<div style='color:#8b949e;font-size:12px;line-height:1.6'>
<b style='color:#58a6ff'>Pipeline</b><br>
📰 NewsAgent → DuckDuckGo<br>
📄 RAGAgent → FAISS + 10-K<br>
📊 AnalystAgent → yfinance<br>
✍️ ReportAgent → Groq LLaMA
</div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("# 📊 FinSight — AI Investment Analyst")
st.markdown(
    "<span style='color:#8b949e'>Multi-agent financial research · LangGraph · RAG · Groq LLaMA</span>",
    unsafe_allow_html=True,
)
st.markdown("---")


# ══════════════════════════════════════════════════════════════════════════════
# RUN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
if run_btn:
    if not groq_key:
        st.error("⚠️ Please enter your Groq API key in the sidebar.")
    else:
        # Load pipeline (cached after first call)
        pipeline = get_pipeline(groq_key, faiss_path)
        st.session_state["pipeline"] = pipeline

        progress = st.progress(0, text="Starting agents...")
        status   = st.empty()

        try:
            # ── Step 1: News ──────────────────────────────────────────────
            progress.progress(10, text="📰 Fetching & analysing news...")
            status.info("📰 NewsAgent running...")
            news_result = pipeline.news_agent.run(final_ticker)

            # ── Step 2: RAG ───────────────────────────────────────────────
            progress.progress(35, text="📄 Running RAG over 10-K documents...")
            status.info("📄 RAGAgent running...")
            rag_result = pipeline.rag_agent.summarise(final_ticker)

            # ── Step 3: Financials ────────────────────────────────────────
            progress.progress(60, text="📊 Pulling live financials & ratios...")
            status.info("📊 AnalystAgent running...")
            financial_data = pipeline.analyst_agent.run(final_ticker)
            stock_data     = pipeline.analyst_agent.get_raw_stock_data(final_ticker)
            ratios         = pipeline.analyst_agent.get_raw_ratios(final_ticker)

            # ── Step 4: Memo ──────────────────────────────────────────────
            progress.progress(85, text="✍️ Writing investment memo...")
            status.info("✍️ ReportAgent running...")
            memo = pipeline.report_agent.run(
                ticker=final_ticker,
                news=news_result,
                rag_context=rag_result,
                financial_data=financial_data,
            )

            # ── Store results ─────────────────────────────────────────────
            st.session_state["result"] = {
                "report":         memo,
                "news":           news_result,
                "rag_context":    rag_result,
                "financial_data": financial_data,
            }
            st.session_state["stock_data"]     = stock_data
            st.session_state["ratios"]          = ratios
            st.session_state["chat_history"]    = []
            st.session_state["current_ticker"]  = final_ticker

            progress.progress(100, text="✅ Analysis complete!")
            status.success(f"✅ FinSight analysis for **{final_ticker}** is ready!")
            time.sleep(1)
            status.empty()
            st.rerun()

        except Exception as e:
            st.error(f"Analysis error: {e}")
            st.exception(e)
            st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# RESULTS
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state["result"]:
    result     = st.session_state["result"]
    stock_data = st.session_state["stock_data"]
    ratios     = st.session_state["ratios"]
    cur_ticker = st.session_state["current_ticker"]

    # ── Metric cards ──────────────────────────────────────────────────────────
    def fmt_large(v):
        if isinstance(v, (int, float)):
            if v >= 1e12: return f"${v/1e12:.2f}T"
            if v >= 1e9:  return f"${v/1e9:.2f}B"
            if v >= 1e6:  return f"${v/1e6:.2f}M"
            return f"${v:,.0f}"
        return str(v)

    def pct_color(v):
        if isinstance(v, (int, float)):
            cls  = "positive" if v >= 0 else "negative"
            sign = "▲" if v >= 0 else "▼"
            return f'<span class="{cls}">{sign} {abs(v):.1f}%</span>'
        return str(v)

    cols  = st.columns(5)
    cards = [
        ("Current Price",  f"${stock_data['current_price']}",          pct_color(ratios.get("revenue_growth", 0))),
        ("Market Cap",     fmt_large(stock_data["market_cap"]),         ""),
        ("P/E (Trailing)", str(stock_data["pe_ratio"]),                 ""),
        ("Profit Margin",  f"{ratios.get('profit_margin','N/A')}%",     ""),
        ("Revenue Growth", f"{ratios.get('revenue_growth','N/A')}%",    pct_color(ratios.get("earnings_growth", 0))),
    ]
    for col, (label, value, delta) in zip(cols, cards):
        with col:
            st.markdown(f"""
<div class="metric-card">
  <div class="label">{label}</div>
  <div class="value">{value}</div>
  <div class="delta">{delta}</div>
</div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "📝 Investment Memo", "📊 Financial Ratios",
        "📰 News & Sentiment", "💬 Ask FinSight",
    ])

    # ── Tab 1: Memo ───────────────────────────────────────────────────────────
    with tab1:
        _, col_dl = st.columns([5, 1])
        with col_dl:
            st.download_button(
                "⬇️ Download Memo", data=result["report"],
                file_name=f"{cur_ticker}_investment_memo.md",
                mime="text/markdown", use_container_width=True,
            )
        st.markdown(f'<div class="memo-box">{result["report"]}</div>', unsafe_allow_html=True)

    # ── Tab 2: Ratios ─────────────────────────────────────────────────────────
    with tab2:
        r = ratios
        st.markdown('<div class="section-header">Valuation</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("P/E (Trailing)", r.get("pe_ratio", "N/A"))
        c1.metric("P/E (Forward)",  r.get("forward_pe", "N/A"))
        c2.metric("Price / Book",   r.get("price_to_book", "N/A"))
        c2.metric("Price / Sales",  r.get("price_to_sales", "N/A"))
        c3.metric("EV / EBITDA",    r.get("ev_to_ebitda", "N/A"))

        st.markdown('<div class="section-header">Profitability</div>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Profit Margin",    f"{r.get('profit_margin','N/A')}%")
        c2.metric("Operating Margin", f"{r.get('operating_margin','N/A')}%")
        c3.metric("ROE",              f"{r.get('roe','N/A')}%")
        c4.metric("ROA",              f"{r.get('roa','N/A')}%")

        st.markdown('<div class="section-header">Liquidity & Leverage</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("Current Ratio", r.get("current_ratio", "N/A"))
        c2.metric("Quick Ratio",   r.get("quick_ratio", "N/A"))
        c3.metric("Debt / Equity", r.get("debt_to_equity", "N/A"))

        st.markdown('<div class="section-header">Growth & Dividend</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("Revenue Growth",  f"{r.get('revenue_growth','N/A')}%")
        c2.metric("Earnings Growth", f"{r.get('earnings_growth','N/A')}%")
        c3.metric("Dividend Yield",  f"{r.get('dividend_yield','N/A')}%")

        st.markdown('<div class="section-header">LLM Interpretation</div>', unsafe_allow_html=True)
        fd = result.get("financial_data", "")
        if "LLM INTERPRETATION:" in fd:
            interp = fd.split("LLM INTERPRETATION:")[-1].strip()
            st.markdown(f'<div class="memo-box">{interp}</div>', unsafe_allow_html=True)

    # ── Tab 3: News ───────────────────────────────────────────────────────────
    with tab3:
        news_text = result.get("news", "")
        if "SENTIMENT ANALYSIS:" in news_text:
            headlines, sentiment = news_text.split("SENTIMENT ANALYSIS:", 1)
            st.markdown('<div class="section-header">Raw Headlines</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="memo-box" style="font-size:13px">'
                f'{headlines.replace("RAW NEWS HEADLINES:","").strip()}</div>',
                unsafe_allow_html=True,
            )
            st.markdown('<div class="section-header">Sentiment Analysis</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="memo-box">{sentiment.strip()}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="memo-box">{news_text}</div>', unsafe_allow_html=True)

    # ── Tab 4: Chat ───────────────────────────────────────────────────────────
    with tab4:
        st.markdown(
            f"<span style='color:#8b949e;font-size:13px'>"
            f"Ask follow-up questions about <b>{cur_ticker}</b> — "
            f"RAG-powered over the 10-K annual report</span>",
            unsafe_allow_html=True,
        )

        for msg in st.session_state["chat_history"]:
            if msg["role"] == "user":
                st.markdown(f'<div class="chat-msg-user">{msg["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-msg-bot">🤖 {msg["content"]}</div>', unsafe_allow_html=True)

        if not st.session_state["chat_history"]:
            st.markdown('<div class="section-header">Suggested Questions</div>', unsafe_allow_html=True)
            suggestions = [
                f"What are the main revenue segments for {cur_ticker}?",
                f"What risks does {cur_ticker} highlight in its 10-K?",
                f"How has {cur_ticker}'s cash flow changed year over year?",
                f"What is {cur_ticker}'s strategy for the next 3 years?",
            ]
            sq_cols = st.columns(2)
            for i, q in enumerate(suggestions):
                if sq_cols[i % 2].button(q, key=f"sq_{i}", use_container_width=True):
                    st.session_state["pending_question"] = q
                    st.rerun()

        if "pending_question" in st.session_state:
            user_q = st.session_state.pop("pending_question")
            with st.spinner("Searching annual report..."):
                answer = st.session_state["pipeline"].rag_agent.run(cur_ticker, user_q)
            st.session_state["chat_history"].append({"role": "user",      "content": user_q})
            st.session_state["chat_history"].append({"role": "assistant", "content": answer})
            st.rerun()

        user_input = st.chat_input(f"Ask anything about {cur_ticker}...")
        if user_input:
            with st.spinner("Searching annual report..."):
                answer = st.session_state["pipeline"].rag_agent.run(cur_ticker, user_input)
            st.session_state["chat_history"].append({"role": "user",      "content": user_input})
            st.session_state["chat_history"].append({"role": "assistant", "content": answer})
            st.rerun()

        if st.session_state["chat_history"]:
            if st.button("🗑️ Clear chat", key="clear_chat"):
                st.session_state["chat_history"] = []
                st.rerun()

else:
    st.markdown("""
<div style='text-align:center;padding:60px 20px;color:#8b949e'>
  <div style='font-size:64px'>📈</div>
  <h3 style='color:#58a6ff;margin-top:16px'>Ready to analyse</h3>
  <p>Enter your Groq API key, select a ticker, and click <b>Run FinSight Analysis</b>.</p>
  <br>
  <div style='display:flex;justify-content:center;gap:24px;flex-wrap:wrap;font-size:13px'>
    <span>📰 Live news sentiment</span>
    <span>📄 10-K RAG retrieval</span>
    <span>📊 Financial ratios</span>
    <span>✍️ Investment memo</span>
    <span>💬 Follow-up Q&A</span>
  </div>
</div>
""", unsafe_allow_html=True)
