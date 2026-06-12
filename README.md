---
title: Kuldeep AI
emoji: 🚀
colorFrom: red
colorTo: green
sdk: docker
pinned: false
license: mit
---

<h1 align="center">
  Kuldeep AI — Personal AI Assistant
</h1>

<p align="center">
  <strong>A humanized, multi-agent AI chatbot powered by an advanced RAG knowledge base</strong><br/>
  <em>Built by Kuldeep Kumar Mishra · AI Engineer · IIIT Lucknow</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10-blue?style=flat-square&logo=python" />
  <img src="https://img.shields.io/badge/Flask-3.x-black?style=flat-square&logo=flask" />
  <img src="https://img.shields.io/badge/LangGraph-Agentic-purple?style=flat-square" />
  <img src="https://img.shields.io/badge/ChromaDB-Vector%20Store-orange?style=flat-square" />
  <img src="https://img.shields.io/badge/Groq-LLM%20API-red?style=flat-square" />
</p>

---

## 👋 What is Kuldeep AI?

**Kuldeep AI** is my personal AI assistant that anyone can talk to and learn about me — my background, projects, skills, achievements, and more. Instead of a static portfolio, this is a **living, conversational AI** that knows everything about me from a carefully curated knowledge base.

It's not just a chatbot. It's a **multi-agent system** with five specialized agents working under the hood, an **advanced RAG pipeline** that retrieves and reasons over my personal knowledge base, and a **clean, beautiful UI** that feels premium and modern.

> *"The goal isn't to build models. The goal is to build intelligence that matters."*
> — Kuldeep Kumar Mishra

---

## ✨ Key Features

### 🧠 Advanced RAG System
- **Hybrid Retrieval** — BM25 keyword search + semantic vector search combined
- **Reciprocal Rank Fusion (RRF)** — intelligently merges results from both retrieval methods
- **Cross-Encoder Re-ranking** — uses `ms-marco-MiniLM-L-6-v2` for precise relevance scoring
- **Anti-Hallucination** — answers only from the knowledge base; declines gracefully if info is unavailable
- **Persistent ChromaDB** — vector store survives server restarts; auto-indexed on startup

### 🤖 Multi-Agent Architecture (LangGraph)
The system intelligently routes every message to the right specialist:

| Agent | Handles |
|-------|---------|
| 🔀 **Router Agent** | Classifies the query and routes it to the right agent |
| 📚 **RAG Agent** | Answers from Kuldeep's personal knowledge base |
| 🧮 **Math Agent** | Solves calculations and equations |
| 🧠 **Memory Agent** | Recalls earlier parts of the conversation |
| 🌐 **Search Agent** | Fetches real-time information from the web |
| 💬 **General Agent** | Handles greetings and general knowledge |

### 🔐 Admin Panel (Private)
- Upload PDFs and TXT files to expand the knowledge base
- View, edit, and delete documents directly in the browser
- Force re-index all documents with one click
- Protected by a secret key — hidden from public users

### 🎨 Beautiful UI
- **Light mode default** — clean white/blue design inspired by modern AI interfaces
- **Dark mode toggle** — switches to a sleek dark theme
- Word-by-word streaming responses
- Responsive design for mobile and desktop

---

## 🏗️ Project Architecture

```
kuldeep-ai/
│
├── app.py                   # App entry point
├── config.py                # Centralized configuration
├── requirements.txt         # Python dependencies
├── run.bat                  # One-click local startup (Windows)
│
├── agents/                  # All AI agents
│   ├── router_agent.py      # Routes queries to the right agent
│   ├── rag_agent.py         # Knowledge base Q&A (anti-hallucination)
│   ├── math_agent.py        # Math & calculations
│   ├── memory_agent.py      # Conversation memory recall
│   ├── search_agent.py      # Real-time web search
│   └── general_agent.py     # General chat & greetings
│
├── rag/                     # RAG pipeline
│   ├── document_loader.py   # PDF & TXT chunking
│   ├── vector_store.py      # ChromaDB wrapper (CRUD)
│   ├── retriever.py         # Hybrid BM25 + semantic + RRF
│   └── reranker.py          # Cross-encoder re-ranking
│
├── orchestrator/
│   └── graph.py             # LangGraph state machine
│
├── api/
│   └── index.py             # Flask routes (chat, admin, docs)
│
├── memory/
│   └── sqlite_memory.py     # Conversation history (SQLite)
│
├── templates/               # HTML pages
│   ├── index.html           # Public chat UI
│   ├── admin.html           # Admin dashboard
│   └── admin_denied.html    # 403 access denied
│
├── static/                  # Frontend assets
│   ├── style.css            # All styles (light + dark theme)
│   ├── script.js            # Chat logic, streaming, theme
│   └── ai-avatar.png        # Bot avatar image
│
├── data/
│   ├── knowledge/           # 📁 Private: .txt/.pdf knowledge files
│   ├── uploads/             # 📁 Private: admin-uploaded files
│   └── chroma_db/           # 📁 Auto-generated: vector embeddings
│
└── utils/
    └── logger.py            # Structured logging
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- A [Groq API key](https://console.groq.com/) (free tier works great)

### 1. Clone and Setup

```bash
git clone <your-repo-url>
cd kuldeep-ai
```

### 2. Create Virtual Environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Mac/Linux
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment

Copy `.env.example` to `.env` and fill in your values:

```bash
cp .env.example .env
```

Edit `.env`:
```env
GROQ_API_KEY=your_groq_api_key_here
ADMIN_SECRET_KEY=your-secret-admin-key
```

### 5. Add Your Knowledge Base

Place your `.txt` or `.pdf` files in `data/knowledge/`. These are auto-indexed on every server start.

```
data/knowledge/
├── about.txt
├── projects.txt
├── skills.txt
├── work_experience.txt
├── achievements.txt
├── social_links.txt
└── faq.txt
```

### 6. Run the App

```bash
# Windows (double-click or run):
run.bat

# Or manually:
python app.py
```

Open your browser at: **http://127.0.0.1:10000**

---

## 🔐 Admin Panel

Access the admin dashboard at:
```
http://127.0.0.1:10000/admin?key=YOUR_ADMIN_SECRET_KEY
```

From the admin panel you can:
- 📤 Upload new PDF or TXT files
- 👁️ View any document's content
- ✏️ Edit TXT files directly in the browser
- 🗑️ Delete documents (also removes from vector store)
- 🔄 Force re-index all files

---

## ⚙️ Configuration Reference

All settings live in `.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `GROQ_API_KEY` | — | Your Groq API key |
| `GROQ_MODEL_NAME` | `llama-3.1-8b-instant` | LLM model to use |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Local embedding model |
| `ADMIN_SECRET_KEY` | — | Password for admin panel |
| `CHUNK_SIZE` | `600` | Characters per document chunk |
| `CHUNK_OVERLAP` | `80` | Overlap between chunks |
| `TOP_K_RETRIEVAL` | `5` | Final chunks sent to LLM |
| `TOP_K_CANDIDATES` | `8` | Candidates before re-ranking |
| `ENABLE_RERANKING` | `true` | Cross-encoder re-ranking |
| `ENABLE_QUERY_EXPANSION` | `false` | Query expansion (slower) |

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **LLM** | Groq (Llama 3.1 8B Instant) |
| **Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` |
| **Re-ranker** | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| **Vector Store** | ChromaDB (persistent) |
| **Retrieval** | BM25 + Semantic + RRF + Cross-Encoder |
| **Orchestration** | LangGraph (state machine) |
| **Backend** | Flask 3.x |
| **Memory** | SQLite (conversation history) |
| **Frontend** | Vanilla HTML/CSS/JS |
| **Fonts** | Google Fonts (Inter + Outfit) |

---

## 👨‍💻 About the Author

**Kuldeep Kumar Mishra** is an AI Engineer specializing in Generative AI, LLMs, RAG systems, Agentic AI, and Machine Learning. Currently pursuing M.Sc. Data Science at IIIT Lucknow (secured AIR 1180 in IIT JAM Mathematics 2024).

- 🔗 [LinkedIn](https://linkedin.com/in/kuldeep-kumar-mishra)
- 🐙 [GitHub](https://github.com/kuldeepkumar)
- 🤗 [Hugging Face](https://huggingface.co/kuldeep)
- 📺 [YouTube — AI Simplified](https://youtube.com/@aisimplified)

---

## 📄 License

This is a personal project. All rights reserved © 2025 Kuldeep Kumar Mishra.
