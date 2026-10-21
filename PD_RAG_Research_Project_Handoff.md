# Parkinson's Paper RAG Research Project — Detailed Handoff

**Purpose:** preserve the project's research design, implementation status, saved artifacts, evaluation findings, known issues, and next steps so work can continue after a Google Colab runtime restart.

**Last documented status:** generated answers recovered from a Drive backup; Gemini RAGAS judge initialized; one-record evaluation produced numeric scores; retrieval function/indexes still need to be restored in the restarted runtime.

## 1. Project objective

Build and document a one-document Retrieval-Augmented Generation (RAG) experiment over one research PDF, then compare the answers produced by three embedding/retrieval configurations and three language models. Use a fixed set of five paper-specific questions, preserve generated answers and retrieved contexts, and evaluate outputs with three RAGAS metrics. The goal is an auditable experiment, not merely a chatbot: model settings, answer records, reference answers, retrieved passages, scores, failures, and checkpoints should be saved and inspectable.

Planned scale:

- 1 research PDF
- 5 questions with manually written reference/ground-truth answers
- 3 embedding configurations
- 3 answer-generation LLMs
- 5 × 3 × 3 = **45 answer-generation records**
- 3 evaluation metrics per answer = **135 metric scores** when the full run succeeds

Scores should not be invented, replaced with zero on failure, or presented as accuracy percentages. A failed metric must be recorded as failed/missing with its error.

## 2. Source paper

**Aditi Govindu and Sushila Palwe.** “Early detection of Parkinson’s disease using machine learning.” *Procedia Computer Science*, volume 218 (2023), pages 249–261. The supplied PDF is `paper1.pdf`.

The paper studies machine-learning classification of Parkinson’s disease (PD) from voice/audio measurements. It discusses remote and non-invasive screening as a potential telemedicine aid for people who have mobility or speech difficulties. The reported data include MDVP / vowel-phonation features, including measures related to fundamental frequency, jitter, shimmer, noise, RPDE, DFA, PPE, and spread. It compares Logistic Regression, Random Forest, Support Vector Machine (SVM), and K-Nearest Neighbors (KNN).

The paper evaluates three approaches:

1. **Baseline/full attribute set:** training with 22 attributes.
2. **PCA-based feature selection:** reducing the feature set to five selected attributes/features described as MDVP, Shimmer, Jitter, PPE, and RPDE.
3. **Balanced dataset:** upsampling the minority healthy class to create 109 records in each class, starting from an imbalanced set described as 109 people with PD and 40 healthy records.

The paper discusses accuracy, precision, recall/sensitivity, confusion matrices, ROC-AUC, and F1. It recommends Random Forest overall, while reporting strong alternative results for SVM after PCA and KNN on the balanced dataset. There are differences in how the paper describes sensitivity/recall across its abstract, tables, and discussion; the RAG ground truths should be checked against the relevant table and section rather than treating every reported number as interchangeable.

Useful numbers from the paper as read from the supplied PDF:

| Approach | Key reported finding |
|---|---|
| 22 attributes (Table 2) | Random Forest accuracy 91.83%; table shows precision 0.95, recall 0.86, ROC-AUC 0.701 |
| PCA / five attributes (Table 3) | SVM accuracy 91.75%; table shows precision 1.0, recall 0.86, ROC-AUC 0.727 |
| Balanced data (Table 4) | KNN accuracy 91.83%; precision and recall 0.95, ROC-AUC 0.883 |

The abstract/discussion also describes Random Forest as achieving 91.83% accuracy and sensitivity 0.95. Because Table 2 separately shows precision 0.95 and recall 0.86, this is a potential internal inconsistency in the source paper that should be acknowledged if discussing the exact metric value.

## 3. Intended experiment design

### Embeddings and retrieval

| Label | Configured model | Type / notes |
|---|---|---|
| MiniLM | `sentence-transformers/all-MiniLM-L6-v2` | Sentence Transformers, local inference |
| BGE | `BAAI/bge-small-en-v1.5` | Sentence Transformers; query prefix: `Represent this sentence for searching relevant passages: ` |
| Gemini Embeddings | `gemini-embedding-001` | Gemini API, document/query task types configured separately |

Each embedding model should embed the same cleaned chunk list and use its own FAISS index. Vectors are L2-normalized; `faiss.IndexFlatIP` is used so inner product corresponds to cosine similarity for normalized vectors. Retrieval returns the top-k matching text chunks and IDs. Current notebook defaults: `CHUNK_SIZE=600` characters, `CHUNK_OVERLAP=100`, `TOP_K=3`.

### Answer-generation LLMs

| Label | Provider | Model ID / current status |
|---|---|---|
| LLM1 | Groq OpenAI-compatible API | `qwen/qwen3.8-27b` (configured in notebook; smoke test previously succeeded) |
| LLM2 | Groq OpenAI-compatible API | `openai/gpt-oss-20b` (configured in notebook; smoke test previously succeeded) |
| LLM3 | Gemini API | User switched to `gemini-3.5-flash-lite` after the previously configured `gemini-2.5-flash-lite` returned a 404/unavailable error. Treat the current ID as the live configuration to verify against the account, not the stale value in the notebook template. |

All generators receive the same prompt pattern and retrieved text: answer only from provided context, state if context does not contain the answer, and be concise (2–5 sentences). No embedding vectors are sent to the LLM. The code strips `<think>...</think>` blocks if returned. Groq output budgets and pauses differ by model; Gemini has its own pause. Rate limits are provider/account dependent.

**Fairness note:** the RAGAS judge is also currently `gemini-3.5-flash-lite`, the same model family/configuration as LLM3. This is workable for prototyping but may create shared-model evaluation bias for LLM3. Keep one judge consistent across all generators, and document this limitation. A judge from a different model/provider would reduce that specific concern if a valid, sufficiently available model/key is available.

## 4. Questions and references

Five primary paper questions are in the notebook's `ground_truth` dictionary. The first, already used in the diagnostic, is:

**Question:** What is the main contribution of the paper on early detection of Parkinson's disease using machine learning?

**Reference answer:** The main contribution is the use and comparison of machine learning techniques for the early detection of Parkinson’s disease using audio/voice data. The paper investigates SVM, Random Forest, KNN, and Logistic Regression models and explores PCA-based feature selection and dataset balancing to improve classification.

Other primary topics represented in the configured ground truths include: the problem addressed by voice-based early detection; the four algorithms and their differences; comparative best performance under full/PCA/balanced approaches; and the evaluation metrics used in the paper. The notebook template has a longer, 10-question design in the original backup, but the current cleaned research design is **5 questions**. Do not silently expand it to 10 if keeping the 45-record experiment.

## 5. Processing and indexing pipeline

1. Mount Google Drive and create `/content/drive/MyDrive/RAG_Project`.
2. Read API keys from Colab Secrets: `GROQ_API_KEY`, `GEMINI_API_KEY` (the Gemini key has already been stored by the user under `GEMINI_API_KEY`). Do not print or hard-code secrets in notebooks or exported files.
3. Load `paper1.pdf` using `pypdf.PdfReader` and extract text page-by-page. Assert that text extraction yields a reasonable amount of text; OCR is not yet part of this implementation.
4. Clean recurring journal headers/footers/noise and stop before the References section so bibliography text does not dominate retrieval.
5. Split cleaned text with `RecursiveCharacterTextSplitter`, chunk size 600 and overlap 100. Drop very short chunks and near-duplicates using normalized text keys; retain page numbers alongside chunks.
6. Embed all chunks with each functioning embedding model, normalize vectors, and create separate FAISS indexes.
7. For each embedding and question, retrieve top 3 chunks. The chunk texts and IDs should be saved in the record.
8. Feed identical retrieved context to each LLM for the relevant embedding/question combination. The three embeddings can return different contexts; for a given embedding/question pair, all three LLMs should receive the same contexts so generation comparison is controlled.
9. Save generated answer, ground truth, contexts, model IDs, status, and errors in a checkpoint and CSV.
10. Score each saved answer on Faithfulness, Context Recall, and Answer Correctness using RAGAS. Score each metric separately so one metric failure does not discard successes in other metrics.

## 6. RAGAS metrics in this project

- **Faithfulness:** judges whether claims in the generated response are supported by retrieved contexts. A high score does not guarantee the answer is complete or correct relative to the reference.
- **Context Recall:** judges whether the retrieved context contains information needed to support the reference answer. A low score can signal missing evidence or a judge/input issue.
- **Answer Correctness:** compares the generated answer against the reference answer using the metric implementation's factual/semantic comparison. It is a bounded score, not the probability that an answer is true and not a classifier accuracy percentage.

Configured implementation in the cleaned notebook pins `ragas==0.2.15` and uses the 0.2.x API (`EvaluationDataset`, `SingleTurnSample`, and `evaluate`). The judge embeddings are MiniLM (`sentence-transformers/all-MiniLM-L6-v2`) via a LangChain embedding wrapper. The score function evaluates one metric at a time. The current judge in the live Colab is Gemini via `ChatGoogleGenerativeAI`, model `gemini-3.5-flash-lite`.

The Gemini warning `uses fixed sampling defaults; the sampling parameter(s) temperature will be ignored` is a warning, not a failure. Remove `temperature=0` from `ChatGoogleGenerativeAI(...)` to avoid that warning; it does not invalidate an otherwise returned numeric score.

## 7. Confirmed run history and first-record scores

### Generation

- The three generation models were smoke-tested and reported working in the earlier runtime.
- The generation loop produced **45 records** (5 questions × 3 embeddings × 3 generators), stored with keys such as `MiniLM||LLM1||0`.
- The generated answers were saved in Drive; answer generation does not need to be repeated to continue evaluation.

### Initial Groq judge attempt

For `MiniLM||LLM1||0`, the first attempt returned Faithfulness `1.0000` and Context Recall `0.5000`, but Answer Correctness failed with a 429 rate limit because the Groq judge requested/expected too many output tokens for the model's output-token-per-minute quota. These old scores used a different judge and should not be compared as if evaluation settings were identical to the Gemini run.

### Gemini judge attempt

For the same first record, Gemini judge returned:

| Metric | Score |
|---|---:|
| Answer Correctness | 0.3263 |
| Faithfulness | 0.8000 |
| Context Recall | 0.0000 |

These are observed results, not fabricated values. They are preliminary one-record scores only; don't report them as overall results for the experiment.

The generated response focuses on Random Forest's 91.83% accuracy and sensitivity 0.95 and telemedicine. The reference answer emphasizes the overall contribution: comparison of four algorithms, PCA feature selection, and dataset balancing. The retrieved contexts contained duplicated abstract/title material in contexts 1 and 2, and context 3 contained the algorithm comparison plus RF results but not the PCA/balancing details. This is evidence that top-3 retrieval may be redundant/incomplete for this question. The `context_recall=0.0000` result is worth double-checking because the third context does support some reference claims; do not assume the judge is certainly right or wrong without further tests.

## 8. Checkpoint recovery history (critical)

Google Colab runtime restarts clear Python variables and live FAISS indexes. They do **not** normally delete files saved in mounted Google Drive. After restart, `records`, `score_one`, `retrieve`, embedder objects, and FAISS indexes need to be reconstructed/loaded/redefined.

Drive project folder:

`/content/drive/MyDrive/RAG_Project`

Files observed in it:

| File | Observed size | Meaning / handling |
|---|---:|---|
| `one_document_checkpoint_e35a32e5fc.json` | 2 bytes, content `{}` | Empty checkpoint. Do not use it as the source of generated answers. |
| `one_document_checkpoint_fd7f5dde33.json` | 146,879 bytes; 45 top-level records | Good backup; it was read successfully and contains keys including `MiniLM||LLM1||0`, etc. Preserve it. |
| `one_document_rag_results_fd7f5dde33.csv` | 132,829 bytes | Matching results export; preserve it. |

After a restart, the backup was loaded into `backup_data` and then `records = backup_data`; the sample `MiniLM||LLM1||0` exists and has three retrieved contexts. The current saved records are in memory if the current notebook session has not restarted again. The RAGAS setup was restored and the Gemini judge evaluation for this record succeeded. However, `retrieve()` later raised `NameError`, proving the embedding/index/retrieval setup had not been re-run in that runtime.

**Protect the backup:** do not save to `one_document_checkpoint_fd7f5dde33.json`. Do not let the code write `{}` to the backup path. Use a new, explicit evaluation checkpoint, e.g. `/content/drive/MyDrive/RAG_Project/one_document_gemini_judge_checkpoint.json`, and persist after each successfully completed metric.

Note: the notebook's checkpoint filename is based on an experiment signature hash. Changing model IDs/configuration can cause a different filename (e.g. `e35a32e5fc`) from the pre-existing backup (`fd7f5dde33`). An empty new-signature file does not mean the 45-record backup disappeared.

## 9. Current live Colab status

As of the latest recorded interaction:

- Google Drive is already mounted at `/content/drive`.
- The recovered `records` dict had 45 records loaded from the `fd7f5dde33` backup.
- `langchain-google-genai` was installed successfully after an import error, and the Gemini judge initialization succeeded.
- The active live judge uses `gemini-3.5-flash-lite`; `gemini-3.8-flash` was attempted but errored in that account/setup, so it was not used.
- The RAGAS setup objects and `score_one()` were redefined after restart; a single record's three metrics completed.
- `retrieve()` is currently undefined in the live session, which means `embed_query`, `faiss_indexes`, `chunks`, `chunk_pages`, and/or the cell defining retrieval need to be rebuilt.
- The next action is retrieval debugging; do not rerun generation.

Since the currently cleaned notebook configuration still shows the older Gemini generator candidate `gemini-2.5-flash-lite` and Groq judge `qwen/qwen3.8-27b`, update those config cells if you want the notebook to match the currently tested live setup. Keep a note of what was changed and when. The model string seen in code is not proof of current account access; verify through the model-list/preflight call.

## 10. Next steps, in order

### A. Protect and verify saved records

Use the non-empty backup and assert it contains 45 records:

```python
import json, os
PROJECT_DIR = "/content/drive/MyDrive/RAG_Project"
BACKUP_JSON = os.path.join(PROJECT_DIR, "one_document_checkpoint_fd7f5dde33.json")
with open(BACKUP_JSON, "r", encoding="utf-8") as f:
    records = json.load(f)
assert len(records) == 45
print(len(records), list(records)[:5])
```

Do not run the generation loop or the cell that initializes a fresh empty `records` dictionary and saves it over the backup. If saving evaluation results, use a new destination.

### B. Restore retrieval for debugging

Rerun only the setup cells needed to restore: imports and config; PDF loading/text extraction; chunk cleaning and chunking; retry helper; embeddings and FAISS index creation; then the `retrieve()` definition. Because the embedding/index setup also recomputes vectors, start with MiniLM if possible. Avoid executing the generation smoke-test loop inside/after the retrieval cell if trying to minimize API calls. A Colab restart loses indexes stored only in RAM; cached embeddings under `/content/emb_cache` may also be gone, because `/content` is ephemeral unless copied to Drive.

Then rerun the PCA/balancing diagnostic query with `retrieve("MiniLM", debug_query, k=10)` and inspect returned passages. Check whether top chunks cover the PCA and balanced-data sections, whether chunking/cleanup split the relevant details, and whether the index has all chunks.

### C. Debug the zero Context Recall

1. Inspect retrieved context and reference side by side (already done for the record).
2. Confirm the correct `ground_truth` string and the correct contexts are passed to `SingleTurnSample`.
3. Test whether MiniLM retrieval can find passages for “PCA”, “five attributes”, “upsampling”, “balanced dataset”, and “Table 3/Table 4”.
4. Consider reducing redundant contexts through chunk deduplication, overlap tuning, or a larger top-k for a diagnostic. If changing production top-k, document the change and apply it consistently across compared models.
5. Re-run the same one-record RAGAS metric after verifying the inputs. Do not force the score to be nonzero or edit the ground truth merely to improve metrics.

### D. Save evaluation scores independently and resume safely

Use a separate evaluation output file. Persist each metric as soon as it succeeds. Store missing/failed values as null/NaN plus an error/status field rather than `0.0`, because zero is a valid score. Never silently discard the prior backup. Consider a key composed from record key plus metric name so each metric can be resumed individually.

### E. Validate before the full run

- Run all three metrics for one record and review the outputs.
- Check at least one representative record from each embedding and each generator.
- Confirm score count, missing-score count, metric/model configuration, and reproducible output CSV.
- Only then run up to 135 metric evaluations. The actual number of judge API calls may exceed 135 because metrics can make multiple LLM calls. Use rate limits/backoff and checkpoint after each score.

## 11. Methodology and interpretation safeguards

- Keep the same evaluator across all generation-model outputs in a reported comparison, and document the judge model/version. If the judge is the same exact model as one generator, state that as a limitation.
- Keep the same question set, reference answers, retrieval k, chunking, prompts, and scoring method across comparisons unless a planned ablation changes one factor at a time.
- Compare embeddings by holding question and LLM constant; compare LLMs by holding embedding and retrieved contexts constant.
- Use means and coverage counts only on successfully scored entries. Report how many of 45 answers have a valid score per metric.
- Scores of 1, 0.5, or 0 can be legitimate; interpret in light of examples and evaluator reliability. Never characterize Answer Correctness as “probability true” or Context Recall as the generator's factual correctness.
- Include failure counts and API/provider errors transparently. A quota failure is not a metric score.
- The current three-metric experiment is an answer-evaluation study for one document, not a clinical validation study. It does not train or validate a Parkinson’s diagnostic model.

## 12. Key paths and labels

- Uploaded paper in working environment: `/mnt/data/paper1.pdf` (the user's Colab source is expected to be uploaded into `/content` during a fresh run).
- Original notebook copy: `/mnt/data/RAG_Project_Backup.ipynb`.
- Cleaned notebook artifact: `/mnt/data/RAG_Research_Colab_Complete.ipynb`.
- Drive project directory: `/content/drive/MyDrive/RAG_Project`.
- Preserved generated-answer backup: `one_document_checkpoint_fd7f5dde33.json`.
- Matching CSV: `one_document_rag_results_fd7f5dde33.csv`.
- Empty checkpoint to avoid using as source: `one_document_checkpoint_e35a32e5fc.json`.
- Proposed separate evaluation checkpoint: `one_document_gemini_judge_checkpoint.json`.

## 13. Paper citation

Govindu, A., & Palwe, S. (2023). Early detection of Parkinson’s disease using machine learning. *Procedia Computer Science, 218*, 249–261.

---
**Handoff principle:** preserve generated artifacts; reconstruct ephemeral runtime state as needed; validate retrieval/metrics on a small example; record real scores and failures; only then scale up.
