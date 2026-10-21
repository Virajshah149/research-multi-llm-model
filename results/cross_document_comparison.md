# Cross-Domain RAG Research Benchmark — Comparative Analysis

**Total Records Across All Domains:** 90  

Comparing Retrieval-Augmented Generation across contrasting domains:

- **Parkinson's Disease Voice Detection (Govindu & Palwe, 2023)** (Biomedical Signal Processing & Clinical Diagnostics): 45 records

- **Attention Is All You Need (Vaswani et al., 2017)** (Deep Learning Architecture & Neural Machine Translation): 45 records


---

## 1. Embedding Model Performance by Domain

| Embedding Model | Document Domain | Faithfulness (Mean) | Context Recall (Mean) | Answer Correctness (Mean) |
| :--- | :--- | :---: | :---: | :---: |
| **BGE** | paper1 | 92.57% | 62.15% | 54.63% |
| **BGE** | paper2 | 94.87% | 80.50% | 72.74% |
| **Gemini-Emb** | paper1 | 96.73% | 76.29% | 74.60% |
| **Gemini-Emb** | paper2 | 96.07% | 86.36% | 82.38% |
| **MiniLM** | paper1 | 95.87% | 61.80% | 62.33% |
| **MiniLM** | paper2 | 96.40% | 67.34% | 61.71% |

---

## 2. Generator LLM Performance by Domain

| Generator LLM | Document Domain | Faithfulness (Mean) | Answer Correctness (Mean) | Avg Latency |
| :--- | :--- | :---: | :---: | :---: |
| **LLM1_Qwen** | paper1 | 96.37% | 65.38% | 1.31s |
| **LLM1_Qwen** | paper2 | 96.33% | 74.14% | 11.52s |
| **LLM2_GPT_OSS** | paper1 | 91.87% | 64.15% | 1.39s |
| **LLM2_GPT_OSS** | paper2 | 94.67% | 65.32% | 1.68s |
| **LLM3_Gemini** | paper1 | 96.93% | 62.03% | 22.05s |
| **LLM3_Gemini** | paper2 | 96.33% | 77.37% | 3.83s |