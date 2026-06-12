---
title: Multi Agent Chatbot
emoji: 🐨
colorFrom: blue
colorTo: purple
sdk: docker
app_file: app.py
pinned: false
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

# 🤖 Multi-Agent RAG Chatbot

A sophisticated AI-powered chatbot designed for deep document reasoning and real-time information retrieval. Built on a modular, multi-agent architecture using **LangGraph**, it coordinates specialized LLM-driven agents to provide accurate, grounded, and up-to-date answers.

## 🛠 Features

- **Multi-Agent Coordination:** Uses **LangGraph** to intelligently route queries to the right specialist agent:
  - 🔍 **RAG Agent** — Retrieves and synthesizes knowledge from uploaded PDF documents
  - 🌍 **Web Search Agent** — Fetches real-time data from the internet (powered by DuckDuckGo)
  - 🔢 **Math Agent** — Handles complex calculations with chain-of-thought reasoning
  - 🧠 **Memory Agent** — Recalls and references previous conversation history
  - 🌐 **General Agent** — Answers general knowledge questions using the LLM
- **Smart Router:** Automatically classifies each query using LLM intent detection + keyword fallback — never fails to route
- **Hybrid RAG Pipeline:** Combines **semantic search** (ChromaDB) + **BM25 keyword search** with Reciprocal Rank Fusion for maximum retrieval accuracy
- **Persistent Memory:** SQLite-backed conversation memory keeps context across sessions
- **Real-Time Streaming UI:** Word-by-word response streaming with a premium dark/light themed interface
- **PDF Upload:** Drag-and-drop PDF indexing directly from the sidebar

## 🏗 Tech Stack

| Layer | Technology |
|---|---|
| **Backend** | Python, Flask 3.x |
| **AI Orchestration** | LangGraph, LangChain |
| **LLM Provider** | Groq API (`llama-3.1-8b-instant`) |
| **Embeddings** | HuggingFace `all-MiniLM-L6-v2` |
| **Vector Database** | ChromaDB |
| **Keyword Search** | BM25 (rank-bm25) |
| **Web Search** | DuckDuckGo (`ddgs`) |
| **Memory** | SQLite |
| **Frontend** | Vanilla HTML5, CSS3, JavaScript |
| **Deployment** | Render.com |

## 📁 Project Structure

```text
chatboart 22/
├── app.py                  # Entry point — runs Flask server
├── config.py               # Centralized config (API keys, paths)
├── render.yaml             # Render.com deployment config
├── requirements.txt        # Python dependencies
│
├── agents/
│   ├── base_agent.py       # Abstract base class for all agents
│   ├── router_agent.py     # Routes queries to the correct agent
│   ├── general_agent.py    # General knowledge responses
│   ├── math_agent.py       # Math & calculation specialist
│   ├── rag_agent.py        # PDF document Q&A
│   ├── memory_agent.py     # Conversation history recall
│   └── search_agent.py     # Real-time web search (DuckDuckGo)
│
├── orchestrator/
│   └── graph.py            # LangGraph state machine
│
├── rag/
│   ├── document_loader.py  # PDF → text chunks
│   ├── vector_store.py     # ChromaDB CRUD operations
│   └── retriever.py        # Hybrid BM25 + semantic search
│
├── memory/
│   └── sqlite_memory.py    # Per-session conversation history
│
├── static/
│   ├── style.css           # Full UI styling (dark/light themes)
│   └── script.js           # Chat logic, streaming, markdown rendering
│
└── templates/
    └── index.html          # Main HTML UI (served by Flask)
```

## 🚀 Installation & Setup

### 1. Prerequisites
- Python 3.9+
- A free [Groq API Key](https://console.groq.com/)

### 2. Clone & Install

```bash
# Clone the repository
git clone <your-repo-url>
cd <repo-folder>

# Create virtual environment
python -m venv .venv

# Activate it
.venv\Scripts\activate      # Windows
source .venv/bin/activate   # Linux / Mac

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure Environment

Create a `.env` file in the root directory:

```env
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL_NAME=llama-3.1-8b-instant
EMBEDDING_MODEL=all-MiniLM-L6-v2
CHROMA_DB_PATH=./data/chroma_db
SQLITE_DB_PATH=./data/chat_memory.db
```

### 4. Run the Application

```bash
python app.py
```

Open your browser at `http://127.0.0.1:5000`

## ☁️ Deploying to Render

1. Push this repo to GitHub
2. Go to [render.com](https://render.com) → **New Web Service** → connect your repo
3. Render will auto-detect `render.yaml` and configure everything
4. In Render dashboard → **Environment** → add your `GROQ_API_KEY` secret
5. Deploy!

> **Note:** Render's free tier uses ephemeral storage — uploaded PDFs and ChromaDB data will reset on redeploy. For persistent storage, connect a cloud database.

## 🧪 Agent Routing Examples

| Query | Agent Triggered |
|---|---|
| `"What is 144 / 12?"` | 🔢 Math Agent |
| `"Summarise the uploaded PDF"` | 🔍 RAG Agent |
| `"What did we discuss earlier?"` | 🧠 Memory Agent |
| `"Who is the current US president?"` | 🌍 Web Search Agent |
| `"Explain quantum computing"` | 🌐 General Agent |

## 📄 License

This project is for educational and portfolio purposes.
