# DocuChat - Agentic RAG Document Q&A System

A document Q&A system built on **Agentic RAG** — an AI agent that decides whether to retrieve information, grades the quality of what it retrieves, rewrites failed queries, and verifies its own answers before responding. Powered by LangGraph, LangChain, Groq's LLaMA 3.3 70B, and ChromaDB.

---

## What Makes This "Agentic"?

Classic RAG blindly retrieves the top-K chunks and passes them to the LLM. This system uses a **self-correcting agent** that:

1. **Routes** — decides if the question even needs document retrieval, or can be answered from general knowledge
2. **Retrieves** — fetches top-4 chunks from ChromaDB
3. **Grades** — evaluates each chunk for relevance, discards irrelevant ones
4. **Rewrites** — if no relevant chunks found, rewrites the query and retries (up to 2 times)
5. **Generates** — produces an answer using only the verified relevant chunks
6. **Checks hallucination** — verifies the answer is grounded in the retrieved documents; regenerates if not

Every step is visible in the UI under "Agent reasoning trace".

---

## Features

- Upload PDF and DOCX documents
- Automatically chunks and embeds documents using HuggingFace sentence transformers
- Stores embeddings locally in ChromaDB (persists across sessions)
- Agentic pipeline with routing, grading, query rewriting, and hallucination checking
- Chat history with per-message reasoning traces and source chunks
- Load previously ingested documents without re-uploading

---

## Tech Stack

| Component | Technology |
|---|---|
| LLM | LLaMA 3.3 70B via Groq API |
| Embeddings | `all-MiniLM-L6-v2` (HuggingFace, runs locally) |
| Vector Database | ChromaDB (local, persistent) |
| Agent Framework | LangGraph |
| RAG Orchestration | LangChain |
| UI | Streamlit |
| Document Parsing | PyPDF, python-docx |

---

## Project Structure

```
project1/
├── app.py                  # Streamlit UI and chat interface
├── rag/
│   ├── __init__.py
│   ├── ingestion.py        # Document loading, chunking, embedding, storing
│   ├── retrieval.py        # (Legacy) Original simple RAG chain
│   └── agent.py            # Agentic RAG graph (LangGraph state machine)
├── chroma_db/              # Auto-created: persisted vector store
├── requirements.txt
├── .env                    # Your API keys (not committed to git)
├── .env.example            # Template for environment variables
├── INTERVIEW_PREP.md       # Full concept guide for interviews
└── CHANGELOG.md            # What changed from classic RAG to agentic RAG and why
```

---

## How It Works

### Ingestion (runs once per document)

```
PDF / DOCX
    |
    v
Document Loader (PyPDF / Docx2txt)
    |
    v
RecursiveCharacterTextSplitter  (chunk_size=1000, overlap=200)
    |
    v
HuggingFace Embeddings  (all-MiniLM-L6-v2, runs locally, free)
    |
    v
ChromaDB  (persisted locally in ./chroma_db)
```

### Agentic Query Pipeline (runs on every question)

```
User Question
    |
    v
[Router] ──── "General knowledge?" ────────────────► [Generate directly]
    |
    | "Needs document retrieval"
    v
[Retrieve] — top-4 chunks from ChromaDB
    |
    v
[Grade Documents] — LLM scores each chunk: relevant or not
    |
    ├── Relevant chunks found ──────────────────────► [Generate]
    |                                                      |
    └── No relevant chunks                                 v
            |                                    [Check Hallucination]
            ├── retry_count < 2                       |
            |       |                    Grounded ────► Return to user
            v       v                        |
        [Rewrite Query] → [Retrieve]    Not grounded → [Generate] (retry once)
            |
            └── retry_count >= 2 ─────────────────► [Generate] (with no context)
```

---

## Agent Reasoning Trace (visible in the UI)

Every answer shows what the agent did:

```
- Router: retrieve
- Retrieved 4 chunks for: "what is the conclusion?"
- Grader: 3/4 chunks relevant
- Answer generated
- Hallucination check: passed
```

For a general knowledge question (no retrieval needed):
```
- Router: direct_answer
- Answer generated
- Hallucination check: skipped (no docs used)
```

For a failed retrieval with query rewriting:
```
- Router: retrieve
- Retrieved 4 chunks for: "conclusion"
- Grader: 0/4 chunks relevant
- Query rewritten: "final conclusions and key recommendations of the study"
- Retrieved 4 chunks for: "final conclusions and key recommendations of the study"
- Grader: 2/4 chunks relevant
- Answer generated
- Hallucination check: passed
```

---

## Prerequisites

- Python 3.9 or higher
- Anaconda / Miniconda (recommended)
- A free Groq API key — sign up at https://console.groq.com (no credit card required)

---

## Setup and Installation

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd project1
```

### 2. Install dependencies

```bash
/opt/anaconda3/bin/pip install -r requirements.txt
```

### 3. Set up environment variables

```bash
cp .env.example .env
```

Open `.env` and add your Groq API key:

```
GROQ_API_KEY=gsk_your_actual_key_here
```

To get a free Groq API key:
1. Go to https://console.groq.com
2. Sign up with Google or email
3. Navigate to **API Keys** in the sidebar
4. Click **Create API Key** and copy it

---

## Running the App

```bash
/opt/anaconda3/bin/streamlit run app.py --browser.gatherUsageStats false
```

The app will open in your browser at `http://localhost:8501`

---

## Usage

1. **Upload a document** — Click "Browse files" in the left sidebar and select a PDF or DOCX file
2. **Ingest the document** — Click "Ingest Document" and wait for the green success message
3. **Ask questions** — Type your question in the chat box at the bottom
4. **View agent trace** — Expand "Agent reasoning trace" to see every decision the agent made
5. **View sources** — Expand "Source chunks used" to see the verified relevant chunks
6. **Load existing DB** — If you have ingested a document before, click "Load Existing DB" to skip re-uploading

---

## Example Questions to Try

- "What is this document about?"
- "Summarize the main points"
- "What does it say about [specific topic]?"
- "List all the key findings"
- "What is the conclusion?"
- "What is 2 + 2?" *(tests the router — answered directly without retrieval)*

---

## Environment Variables

| Variable | Description |
|---|---|
| `GROQ_API_KEY` | Your Groq API key (required) |

---

## Dependencies

```
langchain
langgraph
langchain-groq
langchain-chroma
langchain-community
langchain-huggingface
langchain-text-splitters
langchain-core
chromadb
pypdf
python-docx
streamlit
python-dotenv
sentence-transformers
pydantic
```

---

## Limitations

- Very large documents (100+ pages) may take longer to ingest
- Each question goes through multiple LLM calls (router, grader, generator, hallucination checker) — slightly slower than classic RAG but significantly more accurate
- The free Groq API tier has rate limits (sufficient for personal and demo use)
- Currently supports one document at a time


