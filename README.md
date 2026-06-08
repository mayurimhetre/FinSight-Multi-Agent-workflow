# 📈 FinSight — AI-Powered Investment Analyst

> A multi-agent financial research system built with LangGraph, RAG, Groq LLaMA, and Streamlit.  
> Generates professional investment memos in minutes by orchestrating 4 specialised AI agents.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![LangGraph](https://img.shields.io/badge/LangGraph-0.2+-orange?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-red?style=flat-square&logo=streamlit)
![Groq](https://img.shields.io/badge/Groq-LLaMA_3.3_70B-purple?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## 🎯 What it does

FinSight takes a stock ticker (e.g. `AAPL`, `TSLA`) and automatically:

1. **Fetches & analyses** the latest news headlines via DuckDuckGo
2. **Retrieves insights** from SEC 10-K annual reports using RAG (FAISS + HuggingFace embeddings)
3. **Pulls live financials** — price, P/E, margins, growth, ratios via yfinance
4. **Writes a structured investment memo** with executive summary, risks, valuation, and Buy/Hold/Sell recommendation

All via a clean Streamlit UI with metric cards, tabs, and a follow-up Q&A chat interface.

---

## 🏗️ Architecture

```
User Input (Ticker)
       │
       ▼
┌─────────────────────────────────────────────────────┐
│                  LangGraph Pipeline                 │
│                                                     │
│  ┌──────────────┐    ┌──────────────┐               │
│  │ News Fetcher │───▶│  RAG Agent  │               │
│  │ DuckDuckGo   │    │ FAISS + 10-K │               │
│  └──────────────┘    └──────┬───────┘               │
│                             │                       │
│                      ┌──────▼───────┐               │
│                      │   Analyst    │               │
│                      │   yfinance   │               │
│                      └──────┬───────┘               │
│                             │                       │
│                      ┌──────▼───────┐               │
│                      │    Report    │               │
│                      │    Writer    │               │
│                      └──────┬───────┘               │
└─────────────────────────────┼───────────────────────┘
                              │
                              ▼
                  Investment Memo (Markdown)
```

| Agent | Tool | Output |
|---|---|---|
| News Fetcher | DuckDuckGo Search | Raw headlines + sentiment analysis |
| RAG Agent | FAISS + MiniLM-L6-v2 | Key insights from 10-K filing |
| Analyst Agent | yfinance + Groq | Financial ratios + LLM interpretation |
| Report Writer | Groq LLaMA 3.3 70B | Full investment memo in markdown |

---

## 🖥️ Streamlit UI

| Tab | Content |
|---|---|
| 📝 Investment Memo | Full structured memo with download button |
| 📊 Financial Ratios | Valuation, profitability, liquidity, growth metrics |
| 📰 News & Sentiment | Headlines + AI sentiment breakdown |
| 💬 Ask FinSight | RAG-powered follow-up Q&A chat over the 10-K |

---

## 🚀 Quick Start

### Prerequisites
- Google Colab (recommended) or local Python 3.10+
- [Groq API key](https://console.groq.com) (free)
- [ngrok account](https://dashboard.ngrok.com) (free, for Colab only)
- Apple and/or Tesla 10-K PDF files saved to Google Drive

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/finsight.git
cd finsight
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Build the FAISS index (run once)
Open `notebooks/FinSight_Pipeline.ipynb` in Google Colab and run the **Phase 2** cells to:
- Load your 10-K PDFs
- Chunk and embed them with HuggingFace MiniLM
- Save the FAISS index to Google Drive

### 4. Launch the Streamlit app (Colab)
```python
from pyngrok import ngrok
import subprocess, time

ngrok.set_auth_token("YOUR_NGROK_TOKEN")   # from dashboard.ngrok.com

subprocess.Popen([
    "streamlit", "run", "/content/drive/MyDrive/FinSight/app.py",
    "--server.port=8501", "--server.headless=true"
])
time.sleep(4)
tunnel = ngrok.connect(8501)
print("🌐 Live at:", tunnel.public_url)
```

Then open the URL, paste your **Groq API key** in the sidebar, select a ticker, and click **Run FinSight Analysis**.

---

## 📁 Project Structure

```
finsight/
├── app.py                          # Streamlit frontend (Day 9)
├── requirements.txt                # All dependencies
├── README.md                       # This file
├── notebooks/
│   └── FinSight_Pipeline.ipynb     # Full Colab notebook (Days 1–8)
└── data/
    └── .gitkeep                    # Add your 10-K PDFs here (not committed)
```

---

## ⚙️ Tech Stack

| Layer | Technology |
|---|---|
| LLM | Groq — LLaMA 3.3 70B Versatile |
| Orchestration | LangGraph (StateGraph) |
| RAG | FAISS + HuggingFace `all-MiniLM-L6-v2` |
| Document loading | LangChain PyPDFLoader |
| Live data | yfinance |
| News search | DuckDuckGo (LangChain tool) |
| Frontend | Streamlit |
| Deployment | Streamlit Cloud / ngrok (Colab) |

---

## 🔧 Configuration

| Parameter | Where | Description |
|---|---|---|
| `GROQ_API_KEY` | Sidebar / env var | Your Groq API key |
| FAISS index path | Sidebar | Path to saved FAISS index |
| Ticker | Sidebar | Stock ticker symbol |

Supported tickers with RAG (10-K loaded): `AAPL`, `TSLA`  
Supported tickers for live data only: any valid yfinance ticker (`MSFT`, `GOOGL`, `AMZN`, etc.)

---

## 📋 10-Day Build Log

| Day | Phase | What was built |
|---|---|---|
| 1 | Environment | Colab setup, Groq connection, LLaMA test |
| 2 | Skeleton | LangGraph StateGraph, 4 node stubs, AgentState |
| 3 | RAG | PDF ingestion, chunking, FAISS index build |
| 4 | RAG | Retrieval chain, filtered Q&A, node integration |
| 5 | Analyst | Live stock data via yfinance |
| 6 | Analyst | Financial ratio calculation + LLM interpretation |
| 7 | News | DuckDuckGo news fetch + sentiment analysis |
| 8 | Report | Investment memo generation with prompt templates |
| 9 | UI | Streamlit frontend — sidebar, metrics, tabs, chat |
| 10 | Deploy | README, requirements, GitHub, Streamlit Cloud |

---

## 🤝 Contributing

Pull requests welcome! Some ideas for extensions:
- Add more tickers / 10-K documents to the FAISS index
- Add a stock price chart (plotly)
- Add portfolio comparison (multiple tickers side by side)
- Deploy to Streamlit Cloud with secrets management

---

## ⚠️ Disclaimer

FinSight is a **portfolio project for educational purposes only**.  
Nothing here constitutes financial advice. Always do your own research before making investment decisions.

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

<p align="center">Built in 10 days · Powered by LangGraph + Groq + Streamlit</p>
