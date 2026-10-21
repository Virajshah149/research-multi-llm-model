"""
Configuration module for the Parkinson's Disease Paper RAG Research Benchmark.
Centralizes dataset paths, models, hyper-parameters, and standard benchmark questions.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables if .env exists
load_dotenv()

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
PAPER1_PDF = BASE_DIR / "paper1.pdf"
PAPER2_PDF = BASE_DIR / "paper2.pdf"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
RESULTS_DIR = BASE_DIR / "results"

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Default paths for backwards compatibility
PDF_PATH = PAPER1_PDF
CHECKPOINT_FILE = CHECKPOINT_DIR / "benchmark_checkpoint.json"
RESULTS_CSV = RESULTS_DIR / "benchmark_audit_trail.csv"
SUMMARY_MD = RESULTS_DIR / "benchmark_summary.md"
CHART_PNG = RESULTS_DIR / "benchmark_comparison.png"

# API Keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# RAG Hyperparameters
CHUNK_SIZE = 600
CHUNK_OVERLAP = 100
TOP_K = 3
GENERATOR_TEMPERATURE = 0.0
GENERATOR_MAX_TOKENS = 1200

# Embedding Models Configuration
EMBEDDING_CONFIGS = {
    "MiniLM": {
        "model_id": "sentence-transformers/all-MiniLM-L6-v2",
        "type": "sentence_transformers",
        "query_prefix": "",
        "dimension": 384,
    },
    "BGE": {
        "model_id": "BAAI/bge-small-en-v1.5",
        "type": "sentence_transformers",
        "query_prefix": "Represent this sentence for searching relevant passages: ",
        "dimension": 384,
    },
    "Gemini-Emb": {
        "model_id": "gemini-embedding-001",
        "type": "gemini",
        "query_prefix": "",
        "dimension": 3072,
    },
}

# Answer Generation LLMs (The 3 competitors)
GENERATOR_MODELS = {
    "LLM1_Qwen": {
        "provider": "groq",
        "model_id": "qwen/qwen3.8-27b",
        "display_name": "Qwen 3.8 27B (Groq)",
    },
    "LLM2_GPT_OSS": {
        "provider": "groq",
        "model_id": "openai/gpt-oss-20b",
        "display_name": "GPT-OSS 20B (Groq)",
    },
    "LLM3_Gemini": {
        "provider": "gemini",
        "model_id": "gemini-3.5-flash-lite",
        "display_name": "Gemini 3.5 Flash Lite",
    },
}

# High-Throughput Independent Judge for RAGAS Evaluation
# Uses Groq's Qwen 3.8 27B: immune to Gemini's 15 RPM free-tier limit, fast (~0.8s) & unbiased
JUDGE_CONFIG = {
    "provider": "groq",
    "model_id": "qwen/qwen3.8-27b",
    "temperature": 0.0,
    "max_tokens": 2500,
}

# The 5 Curated Benchmark Questions and Reference Ground Truths (from paper1.pdf - Parkinson's Disease)
BENCHMARK_QUESTIONS_PAPER1 = [
    {
        "id": "Q1_contribution",
        "question": "What is the main contribution of the paper on early detection of Parkinson's disease using machine learning?",
        "ground_truth": (
            "The main contribution is the comparative evaluation of machine learning techniques "
            "(Logistic Regression, Random Forest, SVM, and KNN) for early, non-invasive detection "
            "of Parkinson's disease from vocal/speech phonation measurements. The paper investigates "
            "three distinct approaches: full 22-attribute baseline training, PCA-based feature reduction "
            "to five selected attributes (MDVP, Shimmer, Jitter, PPE, and RPDE), and dataset balancing "
            "via minority class upsampling."
        ),
        "target_sections": ["Abstract", "Introduction", "Proposed Methodology"],
    },
    {
        "id": "Q2_telemedicine_problem",
        "question": "What problem in Parkinson's disease detection does this study address using voice data, and why is remote screening necessary?",
        "ground_truth": (
            "The paper addresses the challenge of detecting Parkinson's disease at an early stage "
            "in a non-invasive and remote manner. Because patients with Parkinson's experience mobility "
            "challenges and speech difficulties, frequent physical visits to specialized clinical centers "
            "for disease monitoring can be a major hurdle. The study explores voice and audio measurements "
            "(MDVP vowel phonation) as viable digital biomarkers to enable remote screening and progress "
            "tracking through telemedicine."
        ),
        "target_sections": ["Introduction", "Biomarker Discussion"],
    },
    {
        "id": "Q3_algorithms_diff",
        "question": "Which machine learning algorithms are evaluated in the Parkinson's disease study and how do their fundamental mechanisms differ?",
        "ground_truth": (
            "The paper evaluates four machine learning algorithms: Logistic Regression, Random Forest, "
            "Support Vector Machine (SVM), and K-Nearest Neighbors (KNN). Logistic Regression models categorical "
            "outcomes using a logistic sigmoid probability curve. Random Forest builds an ensemble of multiple "
            "decision trees and determines class assignment via majority voting. SVM constructs optimal hyperplanes "
            "using kernel transformations to separate non-linearly separable classes. KNN is a non-parametric, "
            "instance-based classifier that categorizes samples based on similarity distance."
        ),
        "target_sections": ["Model Training", "Proposed Methodology"],
    },
    {
        "id": "Q4_algorithms_pca_performance",
        "question": "Which machine learning algorithms achieved the highest performance under the full 22-attribute set versus the PCA-reduced feature set, and what were their reported metrics?",
        "ground_truth": (
            "On the complete 22-attribute baseline dataset (Approach 1), Random Forest achieved the highest "
            "accuracy of 91.83% with precision 0.95 and recall 0.86 (Table 2). When PCA reduced the feature set "
            "to five major attributes (MDVP, Shimmer, Jitter, PPE, RPDE) in Approach 2, Support Vector Machine (SVM) "
            "with a linear kernel achieved the best performance with an accuracy of 91.75%, precision of 1.0, and recall of 0.86 (Table 3)."
        ),
        "target_sections": ["Approach 1 (Table 2)", "Approach 2 (Table 3)"],
    },
    {
        "id": "Q5_dataset_balancing_metrics",
        "question": "How does dataset balancing affect classifier performance, which algorithm performed best on the balanced dataset, and what evaluation metrics were reported?",
        "ground_truth": (
            "The original dataset was imbalanced with 109 Parkinson's records and only 40 healthy records. "
            "The authors upsampled the minority healthy class to 109 records, creating a balanced dataset of 218 instances (Approach 3). "
            "On this balanced dataset, K-Nearest Neighbors (KNN) demonstrated the highest performance, achieving 91.83% accuracy, "
            "0.95 precision, 0.95 recall, and an ROC-AUC of 0.883 (Table 4). The paper also utilizes confusion matrices, F1 score, "
            "and ROC curves to evaluate trade-offs between clinical sensitivity and specificity."
        ),
        "target_sections": ["Approach 3 (Table 4)", "Results and Discussion"],
    },
]

# The 5 Curated Benchmark Questions and Reference Ground Truths (from paper2.pdf - Attention Is All You Need)
BENCHMARK_QUESTIONS_PAPER2 = [
    {
        "id": "Q1_transformer_architecture",
        "question": "What is the primary architectural contribution of the Transformer, how does it dispense with recurrence and convolutions, and what are its encoder and decoder layer stack depths?",
        "ground_truth": (
            "The Transformer is the first transduction model relying entirely on self-attention to compute representations "
            "of its input and output without using sequence-aligned RNNs or convolution. It uses an encoder-decoder architecture "
            "where both the encoder and decoder are composed of stacks of N = 6 identical layers. The encoder contains multi-head "
            "self-attention and position-wise feed-forward sub-layers with residual connections and layer normalization. "
            "The decoder inserts a third sub-layer that performs multi-head attention over the encoder output and employs masking "
            "in self-attention to ensure predictions for position i depend only on known outputs at positions less than i."
        ),
        "target_sections": ["Abstract", "Introduction", "Model Architecture (Section 3.1)"],
    },
    {
        "id": "Q2_scaled_dot_product_attention",
        "question": "Explain the mathematical formula for Scaled Dot-Product Attention, why the scaling factor 1/sqrt(d_k) is applied, and what parameter values are used for d_model, number of heads h, and d_k, d_v in the base model?",
        "ground_truth": (
            "Scaled Dot-Product Attention is computed as Attention(Q, K, V) = softmax((Q * K^T) / sqrt(d_k)) * V. "
            "The scaling factor 1/sqrt(d_k) is introduced because for large values of d_k, the dot products grow large in magnitude, "
            "pushing the softmax function into regions with extremely small gradients. In the base model, Multi-Head Attention "
            "employs h = 8 parallel attention heads, with model dimension d_model = 512, and projection dimensions "
            "d_k = d_v = d_model / h = 64."
        ),
        "target_sections": ["Section 3.2.1 (Scaled Dot-Product Attention)", "Section 3.2.2 (Multi-Head Attention)"],
    },
    {
        "id": "Q3_positional_encoding",
        "question": "Why does the Transformer require positional encodings, what mathematical sinusoidal functions are used to compute them, and why were sinusoidal encodings chosen over learned embeddings?",
        "ground_truth": (
            "Since the Transformer contains no recurrence and no convolution, positional encodings must be added to the input embeddings "
            "to inject sequence order information. They use sine and cosine functions of different frequencies: "
            "PE(pos, 2i) = sin(pos / 10000^(2i/d_model)) and PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model)), with wavelengths forming "
            "a geometric progression from 2*pi to 10000*2*pi. Sinusoidal encodings were chosen because they hypothesized it allows "
            "the model to attend by relative positions easily (since for any fixed offset k, PE(pos+k) can be represented as a linear "
            "function of PE(pos)) and may allow the model to extrapolate to sequence lengths longer than those encountered during training."
        ),
        "target_sections": ["Section 3.5 (Positional Encoding)"],
    },
    {
        "id": "Q4_training_hardware_optimizer",
        "question": "What training hardware was used, what were the training step counts and runtimes for the base and big models, and what were the optimizer settings (Adam parameters and learning rate schedule)?",
        "ground_truth": (
            "Models were trained on one machine with 8 NVIDIA P100 GPUs. The base model was trained for 100,000 steps taking 0.4 seconds "
            "per step (12 hours). The big model was trained for 300,000 steps taking 1.0 second per step (3.5 days). "
            "Optimization used Adam with beta1 = 0.9, beta2 = 0.98, and epsilon = 10^-9. The learning rate was varied according to "
            "lrate = d_model^(-0.5) * min(step_num^(-0.5), step_num * warmup_steps^(-1.5)) with warmup_steps = 4000."
        ),
        "target_sections": ["Section 5.1 (Training Data and Batching)", "Section 5.2 (Hardware and Schedule)", "Section 5.3 (Optimizer)"],
    },
    {
        "id": "Q5_wmt_translation_results",
        "question": "What empirical BLEU scores did the Transformer (base and big models) achieve on the WMT 2014 English-to-German and English-to-French translation tasks, and how did its training FLOPs compare to prior models?",
        "ground_truth": (
            "On the WMT 2014 English-to-German task, the big Transformer established a new state-of-the-art BLEU score of 28.4 "
            "(outperforming the best previous models by more than 2.0 BLEU), while the base model achieved 27.3 BLEU. "
            "On the WMT 2014 English-to-French task, the big model achieved a state-of-the-art BLEU score of 41.8 (41.0 for single model), "
            "outperforming all previous single models and ensembles at a fraction of the training cost (2.3 * 10^19 FLOPs, "
            "over an order of magnitude lower training cost than competing models like ConvS2S and MoE)."
        ),
        "target_sections": ["Section 6.1 (Machine Translation)", "Table 2"],
    },
]

# Legacy alias
BENCHMARK_QUESTIONS = BENCHMARK_QUESTIONS_PAPER1

# Document Registry for Multi-Document Benchmarks
DOCUMENTS = {
    "paper1": {
        "key": "paper1",
        "title": "Parkinson's Disease Voice Detection (Govindu & Palwe, 2023)",
        "domain": "Biomedical Signal Processing & Clinical Diagnostics",
        "pdf_path": PAPER1_PDF,
        "checkpoint_file": CHECKPOINT_DIR / "benchmark_checkpoint_paper1.json",
        "results_csv": RESULTS_DIR / "benchmark_audit_trail_paper1.csv",
        "summary_md": RESULTS_DIR / "benchmark_summary_paper1.md",
        "chart_png": RESULTS_DIR / "benchmark_comparison_paper1.png",
        "questions": BENCHMARK_QUESTIONS_PAPER1,
    },
    "paper2": {
        "key": "paper2",
        "title": "Attention Is All You Need (Vaswani et al., 2017)",
        "domain": "Deep Learning Architecture & Neural Machine Translation",
        "pdf_path": PAPER2_PDF,
        "checkpoint_file": CHECKPOINT_DIR / "benchmark_checkpoint_paper2.json",
        "results_csv": RESULTS_DIR / "benchmark_audit_trail_paper2.csv",
        "summary_md": RESULTS_DIR / "benchmark_summary_paper2.md",
        "chart_png": RESULTS_DIR / "benchmark_comparison_paper2.png",
        "questions": BENCHMARK_QUESTIONS_PAPER2,
    },
}
