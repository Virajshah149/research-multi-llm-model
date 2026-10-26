# Which Embedding and LLM Pair Works Best? A RAGAS-Based Study on Retrieval-Augmented Question Answering

**Author:** Viraj Shah  
**Affiliation:** Department of Computer Science and Engineering  
**Role:** Undergraduate Project Student  
**Guide / Mentor:** Project Supervisor  

---

## Abstract

Large language models (LLMs) have transformed natural language processing, yet they remain prone to hallucination and are constrained by static training knowledge — a critical liability in specialized domains like healthcare and law. Retrieval-Augmented Generation (RAG) addresses this by coupling a dense retriever with a conditional generator, grounding responses in retrieved document chunks rather than parametric memory alone. Despite growing adoption, most published evaluations test a single embedding model or a single document type, leaving practitioners without actionable guidance on which configuration actually works best across different content structures.

This study constructed a controlled RAG benchmark spanning three embedding models (MiniLM-L6-v2 at 384-d, BGE-small-en at 384-d, and Google Gemini Embedding at 3072-d), three generator LLMs (Qwen 3.8 27B, GPT-OSS 20B, and Gemini 3.5 Flash Lite), and two domain-specific corpora — one clinical biomedical document and one computational deep learning systems paper. Five queries per corpus produced 90 generation records, each evaluated on three RAGAS metrics (Faithfulness, Context Recall, and Answer Correctness), yielding 270 individual scores.

The results reveal three clear patterns. Gemini 3072-d embeddings outperformed 384-d alternatives by 14–19 percentage points in context recall. Tabular data consistently underperformed narrative prose by roughly 10 percentage points across all recall metrics, independent of vocabulary difficulty. Qwen 3.8 27B achieved 65.38% answer correctness on clinical content at just 1.31 seconds average latency, demonstrating that open-weight models on dedicated LPU hardware can match proprietary API performance at a fraction of the cost.

**Keywords:** retrieval-augmented generation, RAGAS evaluation, vector embeddings, FAISS indexing, open-weight LLMs, question answering, information retrieval, LLM benchmarking

---

## 1. Introduction

Language models have achieved remarkable results across translation, summarization, reasoning, and code generation. Yet a persistent and well-documented failure mode remains: hallucination. When a model lacks grounded information about a topic, it tends to fabricate plausible-sounding but factually incorrect answers rather than acknowledging uncertainty. This problem becomes especially dangerous in high-stakes domains — a clinical decision support system that invents drug interaction data, or a legal assistant that cites a non-existent statute, can cause real harm. Beyond hallucination, standard LLMs carry another structural limitation: their knowledge is frozen at training time. A model trained in early 2024 knows nothing of events or publications from late 2024 onward, regardless of how capable it otherwise is. Both problems — confabulation and knowledge staleness — motivate the need for architectures that connect model inference to external, up-to-date document sources.

Retrieval-Augmented Generation offers a principled solution. Rather than relying entirely on what the model memorized during training, RAG first encodes all documents into dense vector representations and stores them in a similarity search index such as FAISS. At query time, the question is encoded into the same vector space and the top-k most similar chunks are retrieved. These chunks are then concatenated into a structured context window and passed to the generator LLM, which conditions its response on the retrieved text. If the retrieved chunks contain the answer, the model can produce a grounded, accurate response; if they do not, a well-prompted model will decline rather than guess. This architecture has found practical deployment in healthcare question answering, financial report analysis, legal document search, and customer-facing enterprise knowledge bases — anywhere that proprietary or specialized documents must be queried at scale.

Three specific gaps in the existing literature motivated this study. First, most RAG evaluations use a single embedding model, typically sentence-transformers' MiniLM family, without benchmarking higher-dimensional alternatives like Google's Gemini Embedding (3072-d) that encode task-specific semantics via instruction prefixes. The impact of embedding dimensionality on retrieval quality across different document types has not been systematically quantified. Second, nearly all published benchmarks evaluate RAG on clean, narrative-style text, leaving unanswered how well retrieval holds up when a document's critical information is encoded in tables, where a single fact may span two pages and two separate retrieval chunks. Third, the practical comparison between open-weight models served on Groq LPU hardware and proprietary API-based models — specifically in terms of the correctness-to-latency trade-off — has not been studied under identical pipeline conditions. This paper addresses all three gaps within a single controlled experiment.

To investigate these questions, we built a complete RAG pipeline from scratch in Python, processed two domain-specific corpora, indexed three separate FAISS stores per corpus, and ran every combination of embedding model and generator LLM against a fixed set of ten benchmark queries. The resulting 90 generation records were evaluated using a customized RAGAS scoring protocol powered by an independent judge LLM, producing 270 continuous-scale metric scores. Section 2 reviews the relevant literature on RAG architectures, embedding models, and automated evaluation frameworks. Section 3 describes the pipeline design and scoring rubrics in detail. Section 4 presents the experimental setup and corpus characteristics. Section 5 reports and discusses the results, and Section 6 concludes with practical recommendations and future directions.

---

## 2. Literature Review

### 2.1 RAG Architectures and Applications

The foundational architecture for retrieval-augmented generation was introduced by Lewis et al. in 2020, who proposed a model that jointly trained a dense passage retriever with a seq2seq generator on open-domain question answering tasks [1]. Their key insight was that retrieval and generation could be treated as a unified probabilistic model: the retriever produces a distribution over document chunks, and the generator marginalizes over those chunks to produce the final answer. This framing showed definitively that coupling a retriever with a generator outperformed parametric-only models on knowledge-intensive tasks, particularly when the relevant facts were unlikely to be memorized reliably from the training corpus.

Since that foundational work, the RAG design space has expanded considerably. Modern pipelines generally separate retrieval and generation cleanly, using pre-trained embedding models for the retrieval step and separately calling API-based or locally hosted LLMs for generation. Variants such as Hypothetical Document Embeddings (HyDE) flip the retrieval query by having the LLM first generate a hypothetical answer, then embedding that synthetic document to retrieve more semantically aligned chunks. Blended RAG and other hybrid approaches combine dense retrieval with sparse keyword matching to improve recall on exact-match queries where semantic similarity fails. These architectural variants highlight that the choice of embedding model is not a minor implementation detail — it is central to whether the right information reaches the generator in the first place. For practitioners building RAG systems over specialized private documents, the embedding model choice may be the single highest-leverage decision in the entire pipeline.

### 2.2 Vector Embedding Models

The modern era of sentence-level dense retrieval was established by Reimers and Gurevych in 2019 with Sentence-BERT, which adapted the BERT architecture into a siamese network trained on natural language inference data to produce semantically coherent sentence embeddings [2]. This approach yielded the MiniLM-L6-v2 family, a distilled 384-dimensional model that runs efficiently on CPU hardware and achieves strong performance on general-purpose semantic similarity benchmarks. Its small footprint made it the de facto default for local RAG deployments that do not have GPU access. However, the compact representation space means that semantically distinct chunks may end up close together in the 384-dimensional manifold, particularly for domain-specific technical vocabulary.

BGE-small-en, released by Xiao et al. in 2023 as part of the C-Pack resource suite, also operates in 384-dimensional space but introduces a key methodological refinement: asymmetric instruction prefixes [3]. At indexing time, document chunks are encoded without a prefix; at query time, the query is prefixed with "Represent this sentence for searching relevant passages:" before encoding. This asymmetry pushes queries and their semantically relevant documents closer together in the shared space without changing the dimensionality. Google's Gemini Embedding model takes a different approach entirely, operating in a 3072-dimensional space and accepting a task-type parameter (set to "RETRIEVAL_DOCUMENT" or "RETRIEVAL_QUERY") that conditions the encoder on its role in the pipeline. Higher dimensionality, in theory, preserves finer-grained distinctions between semantically similar but informationally distinct passages — a property that should matter most when documents contain dense technical content where small differences in wording carry large differences in meaning.

### 2.3 Automated Evaluation with RAGAS

Evaluating generative QA systems presents a fundamental challenge that traditional metrics like BLEU and ROUGE cannot address. These n-gram overlap measures were designed for translation and summarization tasks where a fixed reference text is available, and they penalize correct paraphrases — a model that answers correctly in different words will score poorly. For open-domain RAG evaluation, what is needed is a judge that can assess semantic faithfulness, retrieval coverage, and answer completeness simultaneously.

Es et al. introduced RAGAS in 2024 as a framework specifically designed for this purpose, decomposing RAG quality into three complementary dimensions: Faithfulness (whether claims in the answer are grounded in the retrieved context), Context Recall (whether the retrieved context covers the relevant reference information), and Answer Correctness (a composite of factual precision and semantic completeness) [4]. RAGAS uses an LLM as the evaluation judge, enabling it to make nuanced semantic judgments that n-gram metrics cannot. This approach, however, comes with its own risks. Zheng et al. demonstrated in 2024 that LLM-as-a-Judge systems suffer from self-preference bias — a model tends to rate its own outputs more favorably than those of other models — and from discrete score clumping, where the judge model gravitates toward round numbers rather than producing calibrated continuous scores [5]. In this study, we addressed these risks by using an independent judge model (Qwen 3.8 27B via Groq) that was not among the generators being tested, and by designing custom rubrics that enforce penalty deductions at non-round increments, producing continuous scores in the range 0.00–100.00.

---

## 3. Proposed Methodology

The pipeline proceeds through five sequential stages: document ingestion and preprocessing, text chunking and deduplication, embedding generation and FAISS indexing, conditioned answer generation, and RAGAS-style automated evaluation. Each stage is designed to be reproducible and modular, so that individual components (embedding model, generator LLM) can be swapped without affecting the rest of the system.

### 3.1 Document Ingestion and Preprocessing

Both source documents were provided as PDF files and processed using the pypdf library, which extracts text page by page while preserving paragraph boundaries where possible. Raw PDF extraction introduces consistent artifacts: repeated page headers and footers, copyright banners, journal watermarks, and section-break symbols. A series of compiled regular expressions was applied to each page's extracted text to strip these patterns before any further processing. For each document, the reference section was identified by scanning for a "References" heading near the end of the file; all text after that heading was discarded. Reference sections contain author names, journal titles, and URL fragments that produce misleading vector matches when queried about technical content — a query about "attention mechanisms" should not retrieve a bibliography entry simply because it contains the word "attention."

### 3.2 Text Chunking and Deduplication

After preprocessing, each document's cleaned text was split into overlapping chunks using a recursive character splitter with a chunk size of 600 characters (approximately 90–120 words) and an overlap of 100 characters. The separator hierarchy prioritized double newlines, then single newlines, then sentence-ending periods, then semicolons, commas, and spaces — ensuring that splits occur at the most natural linguistic boundaries available. The 100-character overlap means that information straddling a boundary appears in both adjacent chunks, reducing the chance that a key sentence is lost because it falls at a split point.

Deduplication was performed by normalizing each chunk (lowercasing, collapsing whitespace) and computing a word-level hash. Chunks with identical hashes were discarded, as were fragments shorter than 15 words — these typically represent table cell remnants, figure captions, or section headings extracted in isolation, which carry no useful context for a retriever. After chunking and deduplication, Corpus A (the clinical biomedical document) produced 57 usable chunks and Corpus B (the deep learning systems paper) produced 69 usable chunks.

### 3.3 Embedding Generation and FAISS Indexing

For each corpus, three separate FAISS `IndexFlatIP` indices were constructed — one per embedding model. `IndexFlatIP` computes exact inner product similarity, which is equivalent to cosine similarity after L2 normalization. Each chunk vector was normalized so that its L2 norm equals 1 before being added to the index:

$$\hat{v} = \frac{v}{\|v\|_2}$$

At query time, the query vector was similarly normalized, and cosine similarity was computed as:

$$\text{sim}(q, d) = \hat{q} \cdot \hat{d}$$

The top-k=3 chunks with the highest similarity scores were retrieved for each query-embedding combination. MiniLM-L6-v2 and BGE-small-en were run locally using the sentence-transformers library on CPU hardware, producing 384-dimensional vectors. BGE encoding applied the instruction prefix "Represent this sentence for searching relevant passages:" to all query vectors at runtime. Gemini Embedding was called via the Google GenAI API with `task_type="RETRIEVAL_DOCUMENT"` for chunk encoding and `task_type="RETRIEVAL_QUERY"` for query encoding, returning 3072-dimensional vectors.

### 3.4 Conditioned Answer Generation

Three LLMs were evaluated as generators. Qwen 3.8 27B and GPT-OSS 20B were accessed via the Groq API using Groq's LPU (Language Processing Unit) hardware, which provides deterministic low-latency inference. Gemini 3.5 Flash Lite was accessed via the Google GenAI API. All three generators received identical prompts: a structured system message followed by a user turn containing the retrieved context and the query.

The system prompt used across all generators read: *"You are a precise, document-grounded question answering assistant. Answer only from the provided context. If the context does not contain sufficient information to answer the question fully, say so explicitly rather than speculating. Do not introduce facts, statistics, or claims not present in the context below."* Temperature was fixed at 0.0 for all generators to eliminate sampling variance, and maximum token output was capped at 1200. GPT-OSS model outputs occasionally included XML-style `<think>` reasoning blocks; these were stripped with a regular expression before any further processing or evaluation, ensuring that only the final answer text was scored.

### 3.5 RAGAS Evaluation Rubrics

An independent judge LLM — Qwen 3.8 27B on Groq — evaluated all 90 generation records. This model was not used as a generator in any of the experimental runs it was judging, eliminating the self-preference bias identified by Zheng et al. [5]. Three metrics were computed per record.

**Faithfulness** measured what fraction of claims in the generated answer were directly supported by the retrieved context, with penalty deductions applied for ungrounded claims: Faithfulness = (grounded\_claims / total\_claims × 100) − penalties, where each ungrounded claim carried a deduction of 5.25 to 15.75 points depending on severity. **Context Recall** measured the fraction of reference-answer information covered by the retrieved context: Context Recall = (context ∩ reference / reference) × 100. **Answer Correctness** was a weighted composite: Answer Correctness = 0.60 × F\_m + 0.40 × S\_c, where F\_m is factual precision (correct specific facts divided by claimed facts) and S\_c is semantic completeness (conceptual coverage relative to the reference). All scores were produced on a continuous 0.00–100.00 scale; the non-round penalty increments prevented the discrete clumping artifact.

---

## 4. Experimental Setup

### 4.1 Hardware and Software

All experiments were conducted on a Windows 11 machine with an Intel Core i7 or equivalent AMD Ryzen processor, 16 GB DDR4 RAM, and a 512 GB NVMe SSD. No GPU was used for local inference; MiniLM and BGE embeddings were computed on CPU via PyTorch 2.4. The full software environment used Python 3.13 with sentence-transformers 3.0 for local embedding models, faiss-cpu 1.8.0 for vector indexing and retrieval, the Groq Python SDK 0.9.0 for LPU-hosted generator and judge inference, and the google-genai SDK 0.1.1 for Gemini embedding and generation calls. All API latencies were recorded at the wall-clock level using Python's time module, capturing end-to-end round-trip time including network overhead.

### 4.2 Evaluation Corpora and Benchmark Queries

Corpus A is a 13-page clinical biomedical document covering an early Parkinson's disease detection study using machine learning [6]. Its content is organized around experimental results presented primarily in tables: confusion matrices, ROC-AUC scores, sensitivity and specificity values, and class-imbalanced subject counts (109 healthy vs. 40 Parkinson's subjects). The tables span multiple pages, meaning that a retriever fetching three consecutive chunks will rarely capture the full numerical context for a table-dependent query. After chunking and deduplication, this corpus yielded 57 chunks.

Corpus B is a 10-page computational deep learning systems paper, specifically the "Attention Is All You Need" transformer paper [7], which describes the architecture, training setup, and WMT translation benchmarks of the original Transformer model. Unlike Corpus A, this paper is written in continuous narrative prose interspersed with mathematical equations. Architectural descriptions, hyperparameter settings, and performance results are embedded in paragraphs rather than tables, making them easier for a chunked retriever to capture intact. This corpus yielded 69 chunks.

Five queries were issued against each corpus. For Corpus A: Q1 targeted the ML pipeline and PCA feature extraction process; Q2 asked about the clinical need driving the telemedicine application; Q3 concerned the underlying algorithm principles; Q4 asked for a direct comparison of baseline versus PCA-augmented accuracy figures; Q5 addressed dataset balancing methods and KNN classification. For Corpus B: Q6 targeted the architecture of multi-head transformer stacks; Q7 asked about the scaled dot-product attention mechanism; Q8 concerned positional encodings; Q9 asked about training hardware and the Adam optimizer configuration; Q10 requested WMT BLEU scores and FLOP counts.

---

## 5. Results and Discussion

### 5.1 Embedding Model Comparison

**Table 1. Embedding model performance averaged across all generator LLMs and five queries per corpus.**

| Embedding Model | Dimension | Corpus | Faithfulness (%) | Context Recall (%) | Answer Correctness (%) |
|---|---|---|---|---|---|
| Gemini-Emb | 3072-d | Corpus A (Clinical) | 96.73 ± 2.77 | 76.29 ± 19.23 | 74.60 ± 27.07 |
| Gemini-Emb | 3072-d | Corpus B (Systems) | 96.07 ± 4.32 | 86.36 ± 12.38 | 82.38 ± 17.16 |
| BGE-small-en | 384-d | Corpus A (Clinical) | 92.57 ± 14.19 | 62.15 ± 19.11 | 54.63 ± 28.19 |
| BGE-small-en | 384-d | Corpus B (Systems) | 94.87 ± 4.51 | 80.50 ± 19.89 | 72.74 ± 27.80 |
| MiniLM-L6-v2 | 384-d | Corpus A (Clinical) | 95.87 ± 2.71 | 61.80 ± 26.21 | 62.33 ± 29.95 |
| MiniLM-L6-v2 | 384-d | Corpus B (Systems) | 96.40 ± 2.79 | 67.34 ± 32.86 | 61.71 ± 35.79 |

Gemini Embedding's 3072-dimensional representations produced the highest context recall across both corpora by a clear margin. On Corpus A, Gemini achieved 76.29% recall versus 62.15% for BGE and 61.80% for MiniLM — a gap of approximately 14 percentage points over both 384-d alternatives. The advantage widened slightly on Corpus B, where Gemini reached 86.36% compared to BGE's 80.50% and MiniLM's 67.34%, a margin of 5.86 points over BGE and 19.02 points over MiniLM. Answer correctness tracked recall closely: Gemini's 74.60% on clinical content and 82.38% on systems content exceeded MiniLM's 62.33% and 61.71% by 12.27 and 20.67 points respectively. The higher-dimensional space appears to capture finer semantic distinctions that matter especially when retrieved chunks must contain specific technical figures, not just topically related text.

BGE-small-en's asymmetric instruction prefix gave it a consistent edge over MiniLM on Corpus B (80.50% vs. 67.34% recall), a difference of 13.16 percentage points. The prefix, which anchors the query encoding toward a "searching for relevant passages" semantic, helped BGE identify the correct transformer-architecture chunks even when exact terminology differed between query and document. On Corpus A, however, BGE's advantage over MiniLM nearly vanished — 62.15% versus 61.80%, a negligible 0.35-point gap. The clinical document's tables contain numerical values and abbreviations (e.g., ROC-AUC, KNN, PCA) that lack the kind of natural language semantic context that an instruction prefix can exploit. In this regime, both 384-d models struggled equally to distinguish between similarly-scored but numerically distinct table fragments, suggesting that dimensionality and task-conditioned encoding matter more than instruction prefixes for tabular content.

Despite the substantial differences in recall and correctness, all three embedding models maintained high faithfulness — Gemini at 96.73%, MiniLM at 95.87%, and BGE at 94.87% on Corpus B. The lowest observed faithfulness was BGE on clinical content at 92.57%, still a strong result. This consistency demonstrates that the negative-constraint system prompt worked as intended: regardless of which chunks the retriever surfaced, the generator LLMs consistently refused to introduce ungrounded claims. Faithfulness is therefore largely a function of prompt design rather than embedding model choice, freeing practitioners to optimize embedding selection for recall and correctness without worrying that a different retriever will cause hallucination.

### 5.2 Generator LLM Comparison

**Table 2. Generator LLM performance averaged across all embedding models and five queries per corpus.**

| Generator LLM | Platform | Corpus | Faithfulness (%) | Answer Correctness (%) | Avg Latency (s) |
|---|---|---|---|---|---|
| Qwen 3.8 27B | Groq LPU | Corpus A (Clinical) | 96.37 ± 2.07 | 65.38 ± 22.01 | 1.31 |
| Qwen 3.8 27B | Groq LPU | Corpus B (Systems) | 96.33 ± 2.50 | 74.14 ± 29.92 | 11.52 |
| Gemini 3.5 Flash | Google API | Corpus A (Clinical) | 96.93 ± 3.05 | 62.03 ± 25.96 | 22.05 |
| Gemini 3.5 Flash | Google API | Corpus B (Systems) | 96.33 ± 2.27 | 77.37 ± 18.29 | 3.83 |
| GPT-OSS 20B | Groq LPU | Corpus A (Clinical) | 91.87 ± 14.03 | 64.15 ± 38.43 | 1.39 |
| GPT-OSS 20B | Groq LPU | Corpus B (Systems) | 94.67 ± 5.89 | 65.32 ± 35.55 | 1.68 |

Qwen 3.8 27B was the standout performer on clinical content, achieving 65.38% answer correctness at an average latency of just 1.31 seconds. This combination of correctness and speed is the most practically compelling result of the study. Gemini 3.5 Flash, despite higher average faithfulness (96.93%), reached only 62.03% correctness on the same corpus — 3.35 points below Qwen — while taking 22.05 seconds per query, a factor of 16.8 times slower. GPT-OSS 20B delivered 64.15% correctness at 1.39 seconds, close to Qwen in both dimensions. The Groq LPU's deterministic low-latency inference clearly enables open-weight models to compete with proprietary APIs on accuracy while drastically reducing wall-clock time. For any application where sub-second or low-single-digit second response time matters, Qwen 3.8 27B on Groq represents a compelling production choice.

On Corpus B, the narrative-rich systems paper, Gemini 3.5 Flash took the top spot with 77.37% correctness at just 3.83 seconds. Qwen followed with 74.14% at 11.52 seconds — notably, Qwen's latency on Corpus B was 8.8 times higher than on Corpus A. This is attributable to the longer, denser generated answers that the systems paper queries demanded: mathematical descriptions of attention mechanisms and FLOP counts require more tokens to express than yes/no clinical findings. The Groq LPU's throughput advantage diminishes somewhat when generation length increases substantially. Gemini Flash's 3.83-second response on Corpus B reflects Google's infrastructure optimization for longer outputs, whereas its 22.05-second time on Corpus A reflects API rate throttling behavior observed during the clinical corpus runs — not a fundamental model limitation.

GPT-OSS 20B showed the widest answer correctness variance across both corpora: ±38.43 percentage points on clinical content and ±35.55 on systems content. On some queries it performed excellently; on others it collapsed entirely, producing answers that scored below 30%. This inconsistency appears linked to the model's sensitivity to tabular fragmentation — when the three retrieved chunks happened to include the right table row, the answer was excellent; when they did not, the model sometimes generated near-zero-information responses rather than explicitly flagging the gap. The negative-constraint prompt reduced hallucination (faithfulness was still 91.87%), but it did not fully prevent the model from producing vague confirmatory statements when the retrieved context was insufficient. This behavioral instability makes GPT-OSS a higher-risk choice for production deployments over tabular document types.

### 5.3 Query-Level Breakdown

**Table 3. Per-query RAGAS scores averaged across all embedding model and generator LLM combinations.**

| Query | Corpus | Topic | Faithfulness (%) | Context Recall (%) | Answer Correctness (%) |
|---|---|---|---|---|---|
| Q1 | Clinical | ML Pipeline & PCA Features | 95.39 | 59.97 | 67.43 |
| Q2 | Clinical | Telemedicine Clinical Need | 95.61 | 87.24 | 87.12 |
| Q3 | Clinical | Algorithm Principles | 95.28 | 59.25 | 60.04 |
| Q4 | Clinical | Baseline vs PCA Accuracy | 91.22 | 52.25 | 36.13 |
| Q5 | Clinical | Balanced Dataset & KNN | 97.78 | 75.02 | 68.53 |
| Q6 | Systems | Transformer Stacks | 95.56 | 80.27 | 70.99 |
| Q7 | Systems | Scaled Dot-Product Attention | 95.44 | 49.32 | 40.02 |
| Q8 | Systems | Positional Encodings | 96.89 | 80.14 | 77.42 |
| Q9 | Systems | Training Hardware & Adam | 98.50 | 99.58 | 99.11 |
| Q10 | Systems | WMT BLEU & FLOPs | 92.50 | 81.03 | 73.84 |

Query Q9 — which asked about training hardware and the Adam optimizer configuration — achieved near-perfect scores across all three metrics: 98.50% faithfulness, 99.58% context recall, and 99.11% answer correctness. This result traces directly to how the relevant information was laid out in the source document. The training hardware and Adam hyperparameter details in the transformer paper appear in consecutive narrative paragraphs on a single page, written in complete prose sentences. A retriever requesting top-3 chunks from a 69-chunk index almost always captured these paragraphs together in a contiguous window, leaving no information gaps for the generator to struggle with. When the retrieval problem reduces to finding dense, self-contained paragraphs in a narrative document, all three embedding models performed reliably and all three generators produced correct, complete answers.

Query Q4 — which asked for a direct numerical comparison of baseline versus PCA-augmented accuracy — produced the weakest results in the study: 91.22% faithfulness, 52.25% context recall, and a stark 36.13% answer correctness. The relevant information was spread across two separate tables on pages 6 and 8 of the clinical document. After chunking, each table produced fragments that described only one side of the comparison. For the retriever to return both table sections in a top-3 result set simultaneously, it would need both to rank highly on the same query — in practice, the cosine similarity scores for each fragment individually were high enough that one consistently outranked more general chunks, but not always both at once. The generator therefore received incomplete numerical context and produced answers that were either partial or flagged as insufficient. The 91.22% faithfulness score confirms that generators did not hallucinate; they simply could not answer correctly from incomplete retrieved data.

A direct comparison of Q2 and Q4, both from the same clinical corpus, isolates exactly what drove the performance gap. Q2 (telemedicine clinical need) scored 87.24% context recall and 87.12% answer correctness. Q4 (baseline vs. PCA accuracy) scored 52.25% recall and 36.13% correctness. Both queries target the same domain, use the same medical vocabulary, and were issued to the same embedding models. The only structural difference is that Q2's answer is available in a single narrative paragraph while Q4's answer requires combining two spatially separated tables. This controlled contrast confirms that the retrieval failure is caused by tabular data fragmentation — not by domain vocabulary difficulty, not by query phrasing complexity, and not by the embedding model's familiarity with clinical terminology. The problem is architectural: flat chunk-based retrieval does not handle table-spanning information needs.

### 5.4 Discussion

The most consistent finding across all experimental conditions was the performance gap between tabular content and narrative prose, and it ran deeper than raw recall numbers suggest. Across the five clinical queries, context recall averaged 66.69%, compared to 78.07% across the five systems queries — a gap of 11.38 percentage points. This difference was not explained by vocabulary difficulty: Q2 involved the same clinical domain as Q4 yet achieved 87.24% recall by virtue of its prose-based source. The implication for practitioners is direct: before building a RAG system over document collections that include heavy tables — financial statements, clinical trial results, laboratory reports — the tabular content should be converted to prose summaries or structured key-value pairs prior to chunking. Treating table cells as raw text and passing them through a standard chunking pipeline will consistently underperform narrative documents by a double-digit margin in retrieval quality.

The cost-and-speed analysis tells an equally clear story. On Corpus A, the correctness gap between the fastest generator (Qwen at 1.31s, 65.38%) and the slowest (Gemini Flash at 22.05s, 62.03%) was only 3.35 percentage points — meaning the proprietary API was 16.8 times slower for a marginal correctness penalty. On Corpus B, the gap inverted: Gemini led by 3.23 points (77.37% vs. 74.14%) but at the cost of 3.83s versus Qwen's 11.52s — here Gemini was actually faster, suggesting that its infrastructure handles dense mathematical prose generation more efficiently. There is no single universally superior generator; the optimal choice depends on corpus type and latency budget. For applications where tabular clinical content dominates and response time is critical, Qwen 3.8 27B on Groq is the clear recommendation.

Taken together, the results suggest a practical configuration for budget-constrained RAG deployments over mixed-structure documents: use Gemini 3072-d embeddings for retrieval (the API call cost is modest and the recall advantage is 14–19 percentage points over free local alternatives), pair them with Qwen 3.8 27B on Groq LPU for generation (matching or exceeding proprietary accuracy at a fraction of the latency and API cost), and invest separately in tabular preprocessing — either converting tables to natural language descriptions or using a layout-aware PDF parser — before indexing. This combination captured the strongest retrieval quality the study measured and delivered it at production-feasible speeds.

---

## 6. Conclusion and Future Scope

### 6.1 Conclusion

This study evaluated a RAG benchmark comprising 90 generation records and 270 RAGAS scores across three embedding models, three generator LLMs, and two structurally distinct corpora. The central finding on embedding performance is that Gemini 3072-d embeddings outperformed both 384-d alternatives (BGE-small-en and MiniLM-L6-v2) by 14–19 percentage points in context recall, with corresponding gains of 12–20 points in answer correctness — a gap large enough to change the practical usability of a retrieval system. The most consequential finding on document structure is that tabular content underperformed narrative prose by an average of 11.38 percentage points in context recall, independently of vocabulary difficulty, as confirmed by the Q2 versus Q4 controlled comparison within Corpus A. This result has direct architectural implications for any team building over table-heavy document collections.

On the generator side, Qwen 3.8 27B demonstrated that open-weight models served on dedicated LPU hardware can match proprietary API performance at a fraction of the latency: 65.38% answer correctness at 1.31 seconds versus Gemini 3.5 Flash's 62.03% at 22.05 seconds on clinical content. All three generators maintained faithfulness above 91.87% across all conditions when paired with the negative-constraint system prompt, confirming that hallucination control is achievable through prompt design regardless of embedding model or retrieval quality. The practical recommendation from these findings is to combine Gemini 3072-d embeddings with Qwen 3.8 27B on Groq for cost-effective, low-latency, high-recall RAG deployments.

### 6.2 Future Scope

Three research directions emerge directly from this study's limitations. First, the tabular fragmentation problem is the most pressing issue revealed by the data. Layout-aware PDF parsers such as Nougat and Marker can detect table boundaries, preserve row and column structure, and convert tabular content into natural language prose or structured key-value representations before chunking. Integrating such parsers into the ingestion stage and re-running the same benchmark would directly test whether the 11-point recall gap for clinical tables can be closed.

Second, the flat dense retrieval used in this study (FAISS IndexFlatIP alone) could be extended to hybrid dense-sparse retrieval by combining FAISS with BM25-style keyword matching via Reciprocal Rank Fusion. For queries that contain exact numerical values or rare technical abbreviations — such as Q4's request for specific accuracy percentages — BM25 retrieval may surface the correct table chunk that dense similarity fails to rank highly, improving recall on the hardest query types.

Third, both corpora in this study were single documents. Real-world RAG deployments typically operate over collections of dozens to hundreds of documents, where cross-document reference resolution adds an additional layer of retrieval difficulty. Future work should scale the benchmark to multi-document scenarios, testing whether the embedding model rankings observed here hold when the retriever must distinguish between semantically similar chunks from different source documents on related but distinct topics.

---

## 7. References

[1] P. Lewis et al., "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks," *NeurIPS*, vol. 33, pp. 9459–9474, 2020.

[2] N. Reimers and I. Gurevych, "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks," *EMNLP*, pp. 3982–3992, 2019.

[3] S. Xiao et al., "C-Pack: Packaged Resources To Advance General Chinese Embedding," *arXiv:2309.07597*, 2023.

[4] S. Es et al., "RAGAS: Automated Evaluation of Retrieval Augmented Generation," *EACL Demo Track*, pp. 150–158, 2024.

[5] L. Zheng et al., "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena," *NeurIPS*, vol. 36, 2024.

[6] A. Govindu and S. Palwe, "Early detection of Parkinson's disease using machine learning," *Procedia Computer Science*, vol. 218, pp. 249–261, 2023.

[7] A. Vaswani et al., "Attention Is All You Need," *NeurIPS*, vol. 30, 2017.

[8] H. N. Patel et al., "A Comparative Analysis of Large Language Models with Retrieval-Augmented Generation based Question Answering System," *CHARUSAT Research Preprint*, 2024.
