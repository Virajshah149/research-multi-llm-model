# Attention Is All You Need (Vaswani et al., 2017) — RAG Evaluation Report

**Total Records Evaluated:** 45  
**Metrics Tracked:** Faithfulness, Context Recall, Answer Correctness  

---

## 1. Retrieval Performance by Embedding Model

| Embedding Model | Faithfulness (Mean ± Std) | Context Recall (Mean ± Std) | Answer Correctness (Mean ± Std) | Scored Count |
| :--- | :---: | :---: | :---: | :---: |
| **BGE** | 94.87 ± 4.51 | 80.50 ± 19.89 | 72.74 ± 27.80 | 15 |
| **Gemini-Emb** | 96.07 ± 4.32 | 86.36 ± 12.38 | 82.38 ± 17.16 | 15 |
| **MiniLM** | 96.40 ± 2.79 | 67.34 ± 32.86 | 61.71 ± 35.79 | 15 |

---

## 2. Generation Performance by Language Model

| Generator LLM | Faithfulness (Mean ± Std) | Answer Correctness (Mean ± Std) | Avg Latency (s) | Scored Count |
| :--- | :---: | :---: | :---: | :---: |
| **LLM1_Qwen** | 96.33 ± 2.50 | 74.14 ± 29.92 | 11.52s | 15 |
| **LLM2_GPT_OSS** | 94.67 ± 5.89 | 65.32 ± 35.55 | 1.68s | 15 |
| **LLM3_Gemini** | 96.33 ± 2.27 | 77.37 ± 18.29 | 3.83s | 15 |

---

## 3. Metric Breakdown by Benchmark Question

| Question ID | Faithfulness | Context Recall | Answer Correctness |
| :--- | :---: | :---: | :---: |
| **Q1_transformer_architecture** | 95.56 | 80.27 | 70.99 |
| **Q2_scaled_dot_product_attention** | 95.44 | 49.32 | 40.02 |
| **Q3_positional_encoding** | 96.89 | 80.14 | 77.42 |
| **Q4_training_hardware_optimizer** | 98.50 | 99.58 | 99.11 |
| **Q5_wmt_translation_results** | 92.50 | 81.03 | 73.84 |