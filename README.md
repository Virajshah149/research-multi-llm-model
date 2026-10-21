<<<<<<< HEAD
# Parkinson's Disease Paper RAG Research Benchmark

A modular, auditable, research-grade benchmark evaluating Retrieval-Augmented Generation (RAG) across multiple embedding models and LLMs over `paper1.pdf` (*Govindu & Palwe, 2023*).

---

## Architecture Overview

```
                      [ paper1.pdf ]
                            │
               [ Ingestion & Deduplication ]
             (600-char chunks, 100-char overlap)
                            │
      ┌─────────────────────┼─────────────────────┐
      ▼                     ▼                     ▼
[ MiniLM FAISS ]      [ BGE-small FAISS ]   [ Gemini-Emb FAISS ]
(all-MiniLM-L6-v2)    (bge-small-en-v1.5)   (text-embedding-004)
      │                     │                     │
      └─────────────────────┼─────────────────────┘
                            │
                 [ Top-3 Controlled Retrieval ]
                            │
      ┌─────────────────────┼─────────────────────┐
      ▼                     ▼                     ▼
 [ LLM1: Qwen 2.5 ]   [ LLM2: Llama 3.3 ]   [ LLM3: Gemini 1.5 Flash ]
                            │
                 27 Answer Records
                            │
               [ Independent Judge: Llama 3.3 70B ]
          (Unbiased: Distinct from generator LLM3)
                            │
    ┌───────────────────────┼───────────────────────┐
    ▼                       ▼                       ▼
Faithfulness          Context Recall         Answer Correctness
    │                       │                       │
    └───────────────────────┼───────────────────────┘
                            │
      [ CSV Audit Trail, Summary Markdown & Figures ]
```

---

## 3 Core Benchmark Questions (from `paper1.pdf`)

1. **`Q1_contribution` (Main Contribution & Approach):**
   *Question:* What is the main contribution of the paper on early detection of Parkinson's disease using machine learning?
   *Focus:* Comparative analysis of ML models (SVM, RF, KNN, LR), 22-attribute baseline, PCA feature reduction, and dataset balancing.

2. **`Q2_algorithms_pca` (Algorithms & PCA Comparison):**
   *Question:* Which machine learning algorithms are evaluated in the paper, and how do their performances compare under the full attribute set versus PCA feature reduction?
   *Focus:* Random Forest achieved 91.83% accuracy on baseline (Table 2); SVM achieved 91.75% accuracy after PCA reduced features to 5 attributes (Table 3).

3. **`Q3_dataset_balancing` (Dataset Balancing & KNN Performance):**
   *Question:* How does dataset balancing affect classifier performance, and which algorithm achieved the best results on the balanced dataset?
   *Focus:* Upsampling minority healthy class from 40 to 109 samples (218 total); KNN achieved 91.83% accuracy, 0.95 recall, and 0.883 ROC-AUC (Table 4).

---

## Quickstart & Execution Steps

### 1. Set Up Virtual Environment & Dependencies

Open terminal inside the `RM` folder:

```bash
# Create and activate virtual environment
python -m venv .venv

# On Windows (PowerShell):
.venv\Scripts\Activate.ps1
# On Linux / Mac:
source .venv/bin/activate

# Install required packages
pip install -r requirements.txt
```

### 2. Configure API Keys

Copy `.env.example` to `.env` and fill in your keys:

```bash
cp .env.example .env
```

Inside `.env`:
```env
GROQ_API_KEY=your_groq_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here
```

### 3. Run Pipeline Stages

* **Step 0: Test Ingestion & Retrieval (No LLM Tokens Used)**
  ```bash
  python run_benchmark.py --diagnose
  ```

* **Step 1: Run Full End-to-End Benchmark (Generate -> Evaluate -> Report)**
  ```bash
  python run_benchmark.py --step all
  ```

* **Step 2 (Or run individually):**
  ```bash
  python run_benchmark.py --step generate  # Generates 27 records (atomic checkpointing)
  python run_benchmark.py --step evaluate  # Scores Faithfulness, Recall, Correctness
  python run_benchmark.py --step report    # Produces CSV, Markdown table & PNG plot
  ```

---

## Output Artifacts

* `checkpoints/benchmark_checkpoint.json`: Real-time atomic state backup. Safe against crashes or rate limits.
* `results/benchmark_audit_trail.csv`: Complete row-level audit trail with questions, retrieved passages, model answers, reference answers, latency, and all metric scores.
* `results/benchmark_summary.md`: Peer-review style comparison tables by Embedding Model and Generator LLM.
* `results/benchmark_comparison.png`: Publication-grade grouped bar charts comparing model performance across all 3 metrics.
=======
>>>>>>>
