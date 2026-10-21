# Parkinson's Disease Voice Detection (Govindu & Palwe, 2023) — RAG Evaluation Report

**Total Records Evaluated:** 45  
**Metrics Tracked:** Faithfulness, Context Recall, Answer Correctness  

---

## 1. Retrieval Performance by Embedding Model

| Embedding Model | Faithfulness (Mean ± Std) | Context Recall (Mean ± Std) | Answer Correctness (Mean ± Std) | Scored Count |
| :--- | :---: | :---: | :---: | :---: |
| **BGE** | 92.57 ± 14.19 | 62.15 ± 19.11 | 54.63 ± 28.19 | 15 |
| **Gemini-Emb** | 96.73 ± 2.77 | 76.29 ± 19.23 | 74.60 ± 27.07 | 15 |
| **MiniLM** | 95.87 ± 2.71 | 61.80 ± 26.21 | 62.33 ± 29.95 | 15 |

---

## 2. Generation Performance by Language Model

| Generator LLM | Faithfulness (Mean ± Std) | Answer Correctness (Mean ± Std) | Avg Latency (s) | Scored Count |
| :--- | :---: | :---: | :---: | :---: |
| **LLM1_Qwen** | 96.37 ± 2.07 | 65.38 ± 22.01 | 1.31s | 15 |
| **LLM2_GPT_OSS** | 91.87 ± 14.03 | 64.15 ± 38.43 | 1.39s | 15 |
| **LLM3_Gemini** | 96.93 ± 3.05 | 62.03 ± 25.96 | 22.05s | 15 |

---

## 3. Metric Breakdown by Benchmark Question

| Question ID | Faithfulness | Context Recall | Answer Correctness |
| :--- | :---: | :---: | :---: |
| **Q1_contribution** | 95.39 | 59.97 | 67.43 |
| **Q2_telemedicine_problem** | 95.61 | 87.24 | 87.12 |
| **Q3_algorithms_diff** | 95.28 | 59.25 | 60.04 |
| **Q4_algorithms_pca_performance** | 91.22 | 52.25 | 36.13 |
| **Q5_dataset_balancing_metrics** | 97.78 | 75.02 | 68.53 |