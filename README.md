# Which Embedding and LLM Pair Works Best? — RAG Benchmark

A research benchmark that tests how different **vector embedding models** and **large language models (LLMs)** perform inside a **Retrieval-Augmented Generation (RAG)** system. Performance is measured using the **RAGAS evaluation framework** across 90 question-answer records.

---

## What This Project Does

Standard LLMs sometimes make up facts or give outdated answers. RAG fixes this by pulling relevant text from a document first, then letting the LLM answer based only on what was retrieved.

This project benchmarks **3 embedding models × 3 LLMs across 2 document types** to find out:
- Which embedding model retrieves the most relevant context?
- Which LLM gives the most accurate answers from that context?
- Does document structure (tables vs. narrative text) affect results?

---

## System Architecture

```
  [ PDF Document ]
        │
        ▼
  [ Ingestion & Cleaning ]
  (strip headers, banners, references)
        │
        ▼
  [ Recursive Text Chunker ]
  (600 chars per chunk, 100 char overlap)
        │
        ├──────────────────┬──────────────────┐
        ▼                  ▼                  ▼
  [ MiniLM Index ]   [ BGE Index ]   [ Gemini-Emb Index ]
  (384-d, local)     (384-d, local)  (3072-d, API)
        │                  │                  │
        └──────────────────┼──────────────────┘
                           ▼
               [ Top-3 Retrieval via FAISS ]
               (cosine similarity search)
                           │
        ┌──────────────────┼──────────────────┐
        ▼                  ▼                  ▼
  [ Qwen 3.8 27B ]  [ GPT-OSS 20B ]  [ Gemini 3.5 Flash ]
   (Groq LPU)        (Groq LPU)       (Google API)
        │                  │                  │
        └──────────────────┼──────────────────┘
                           ▼
               [ 90 Generated Answers ]
                           │
                           ▼
        [ Independent Judge: Qwen 3.8 27B on Groq ]
                           │
             ┌─────────────┼─────────────┐
             ▼             ▼             ▼
       Faithfulness   Context Recall   Answer Correctness
       (0–100)        (0–100)          (0–100)
                           │
                           ▼
          [ results/ — CSV, Markdown, PNG charts ]
```

---

## Models Used

### Embedding Models (for Retrieval)

| Model | Dimensions | Type | Notes |
|---|---|---|---|
| `all-MiniLM-L6-v2` | 384-d | Local CPU | Fast, lightweight baseline |
| `bge-small-en-v1.5` | 384-d | Local CPU | Uses asymmetric query instruction prefix |
| `gemini-embedding-001` | 3072-d | Google API | High-dimensional, best retrieval accuracy |

### Generator LLMs (for Answer Generation)

| Model | Provider | Inference | Avg Latency |
|---|---|---|---|
| `qwen/qwen3.8-27b` | Alibaba Cloud | Groq LPU | ~1.3s |
| `openai/gpt-oss-20b` | OpenAI OSS | Groq LPU | ~1.5s |
| `gemini-3.5-flash-lite` | Google | Cloud API | ~3–22s |

### Evaluation Judge
- **Model:** `qwen/qwen3.8-27b` on Groq (independent — not used as a generator in the same test)
- **Metrics:** Faithfulness, Context Recall, Answer Correctness — all on a continuous 0.00–100.00 scale

---

## How RAGAS Evaluation Works

Each generated answer is scored on 3 metrics:

- **Faithfulness** — Are all the claims in the answer directly supported by the retrieved context? Penalises hallucinations.
- **Context Recall** — Did the retriever fetch all the key facts from the document that the question needed?
- **Answer Correctness** — Does the final answer match the verified reference answer, both factually and semantically?

Scores are computed as continuous percentages (e.g. `87.42`) not coarse ratings (e.g. `0` or `1`).

---

## Key Results (Summary)

| Embedding | Corpus Type | Context Recall | Answer Correctness |
|---|---|---|---|
| Gemini-Emb (3072-d) | Tabular Clinical | 76.29% | 74.60% |
| Gemini-Emb (3072-d) | Narrative Systems | 86.36% | 82.38% |
| BGE-small (384-d) | Tabular Clinical | 62.15% | 54.63% |
| MiniLM (384-d) | Narrative Systems | 67.34% | 61.71% |

**Top findings:**
- 3072-d Gemini embeddings outperform 384-d models by up to **+19%** in context recall
- Narrative prose retrieves **~10% better** than tabular data with standard chunkers
- Qwen 3.8 27B (open-weight, free) matches Gemini Flash (paid API) within **3%** correctness at **17× lower latency**

---

## Project Structure

```
RM/
├── config.py            # All settings: model names, chunk sizes, paths
├── ingestion.py         # PDF → cleaned text → deduplicated chunks
├── retrieval.py         # Chunks → FAISS vector index → top-k retrieval
├── generator.py         # Retrieved context + query → LLM answer
├── evaluator.py         # Generated answers → RAGAS judge scores
├── reporter.py          # Scores → CSV, Markdown table, PNG charts
├── run_benchmark.py     # Main runner — orchestrates the full pipeline
├── paper1.pdf           # Corpus A: Tabular clinical dataset
├── paper2.pdf           # Corpus B: Narrative computational systems dataset
├── results/
│   ├── cross_document_audit_trail.csv     # Full 90-record experiment log
│   ├── cross_document_comparison.md       # Summary comparison table
│   ├── benchmark_audit_trail_paper2.csv   # Corpus B detailed records
│   └── benchmark_comparison_paper2.png    # Corpus B bar chart
├── requirements.txt
└── .env.example         # API key template (copy to .env, never commit .env)
```

---

## Quickstart

### 1. Install dependencies

```bash
python -m venv .venv

# Windows
.venv\Scripts\Activate.ps1

# Mac / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Set up API keys

```bash
cp .env.example .env
```

Edit `.env`:
```env
GROQ_API_KEY=your_groq_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here
```

Get keys for free:
- Groq: [console.groq.com](https://console.groq.com)
- Gemini: [aistudio.google.com](https://aistudio.google.com)

### 3. Run the benchmark

```bash
# Test ingestion and retrieval only (no API calls)
python run_benchmark.py --diagnose

# Run full pipeline: generate → evaluate → report
python run_benchmark.py --step all

# Or run each stage separately
python run_benchmark.py --step generate   # produces answers
python run_benchmark.py --step evaluate   # scores with RAGAS judge
python run_benchmark.py --step report     # saves CSV, MD, PNG
```

---

## Tech Stack

| Layer | Library |
|---|---|
| PDF parsing | `pypdf` |
| Local embeddings | `sentence-transformers` |
| Cloud embeddings | `google-genai` |
| Vector search | `faiss-cpu` |
| LLM inference | `groq`, `google-genai` |
| Evaluation | Custom RAGAS-style judge via LLM-as-a-Judge |
| Data export | `pandas`, `matplotlib` |

---

## Requirements

- Python 3.10+
- A **Groq API key** (free tier works)
- A **Gemini API key** (free tier works for embeddings and generation)
- ~2 GB disk space for model weights (downloaded automatically on first run)
