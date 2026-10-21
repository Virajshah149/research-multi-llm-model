"""
Master Orchestrator for Multi-Document RAG Research Benchmark.
Supports Parkinson's Disease Voice Detection (paper1.pdf) and
Attention Is All You Need / Transformers (paper2.pdf).

Usage:
  python run_benchmark.py --doc paper2 --step all       # Runs full benchmark on Attention Is All You Need
  python run_benchmark.py --doc paper1 --step report     # Re-exports reports for Parkinson's
  python run_benchmark.py --doc all    --step report     # Compiles cross-document comparative analysis
"""

import sys
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
import argparse
from typing import Dict, Any

from config import (
    DOCUMENTS,
    EMBEDDING_CONFIGS,
    GENERATOR_MODELS,
    GROQ_API_KEY,
    GEMINI_API_KEY,
    RESULTS_DIR,
)
from ingestion import build_deduplicated_chunks
from retrieval import RetrievalPipeline, diagnose_retrieval
from generator import LLMClientManager, load_checkpoint, save_checkpoint
from evaluator import evaluate_all_records
from reporter import export_all_reports, compile_dataframe, generate_markdown_report


def check_api_keys():
    """Warns user if required API keys are missing before making network calls."""
    missing = []
    if not GROQ_API_KEY:
        missing.append("GROQ_API_KEY (Needed for LLM1, LLM2, and the Independent Judge)")
    if not GEMINI_API_KEY:
        missing.append("GEMINI_API_KEY (Needed for Gemini Embeddings and LLM3)")

    if missing:
        print("\n" + "!" * 70)
        print("WARNING: Missing environment variables:")
        for m in missing:
            print(f"  - {m}")
        print("Please configure them in your .env file or environment before executing LLM calls.")
        print("!" * 70 + "\n")


def run_diagnose(doc_cfg: Dict[str, Any]):
    """Runs PDF extraction and retrieval diagnostics without using LLM generation tokens."""
    pdf_path = doc_cfg["pdf_path"]
    questions = doc_cfg["questions"]
    print("=" * 70)
    print(f"PHASE 1: INGESTION AND CHUNKING DIAGNOSTIC ({doc_cfg['title']})")
    print("=" * 70)
    chunks = build_deduplicated_chunks(pdf_path)
    print(f"[Ingestion] Extracted {len(chunks)} deduplicated chunks from {pdf_path.name}.")

    pipeline = RetrievalPipeline(chunks)
    for q_item in questions:
        diagnose_retrieval(pipeline, q_item["question"], k=3)


def run_generation(pipeline: RetrievalPipeline, records: Dict[str, Any], doc_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Generates answers across the 5 questions x 3 embeddings x 3 LLMs = 45 records."""
    client_mgr = LLMClientManager()
    questions = doc_cfg["questions"]
    checkpoint_file = doc_cfg["checkpoint_file"]
    total_runs = len(questions) * len(EMBEDDING_CONFIGS) * len(GENERATOR_MODELS)
    current = 0

    print("\n" + "=" * 70)
    print(f"PHASE 2: CONTROLLED ANSWER GENERATION ({total_runs} RECORDS - {doc_cfg['title']})")
    print("=" * 70)

    for q_idx, q_item in enumerate(questions):
        q_id = q_item["id"]
        question = q_item["question"]
        ground_truth = q_item["ground_truth"]

        for emb_name in EMBEDDING_CONFIGS.keys():
            # Retrieve top-k passages once per (question, embedding) pair
            retrieved = pipeline.retrieve(emb_name, question, k=3)
            context_texts = [r["text"] for r in retrieved]

            # Feed identical context to each generator LLM
            for gen_key, gen_cfg in GENERATOR_MODELS.items():
                current += 1
                rec_key = f"{emb_name}||{gen_key}||{q_id}"

                if (
                    rec_key in records
                    and records[rec_key].get("status") == "SUCCESS"
                    and len(records[rec_key].get("generated_answer", "").strip()) >= 15
                ):
                    print(f"[{current}/{total_runs}] Skipping {rec_key} (already generated).")
                    continue

                print(f"[{current}/{total_runs}] Generating: {rec_key}...")
                try:
                    res = client_mgr.generate_answer(gen_key, question, context_texts)
                    records[rec_key] = {
                        "record_key": rec_key,
                        "document": doc_cfg["key"],
                        "document_title": doc_cfg["title"],
                        "document_domain": doc_cfg["domain"],
                        "question_id": q_id,
                        "question": question,
                        "ground_truth": ground_truth,
                        "embedding_model": emb_name,
                        "generator_llm": gen_key,
                        "generator_provider": gen_cfg["provider"],
                        "retrieved_contexts": context_texts,
                        "retrieved_metadata": retrieved,
                        "generated_answer": res["answer"],
                        "latency_seconds": res["latency_seconds"],
                        "status": "SUCCESS",
                        "evaluation": records.get(rec_key, {}).get("evaluation", {}),
                    }
                    save_checkpoint(records, path=checkpoint_file)
                except Exception as e:
                    print(f"   [Error] Failed generating for {rec_key}: {e}")
                    records[rec_key] = {
                        "record_key": rec_key,
                        "document": doc_cfg["key"],
                        "question_id": q_id,
                        "embedding_model": emb_name,
                        "generator_llm": gen_key,
                        "status": f"FAILED: {str(e)}",
                    }
                    save_checkpoint(records, path=checkpoint_file)

    print(f"\n[Generation] {total_runs} records saved to {checkpoint_file}.")
    return records


def run_document_benchmark(doc_key: str, step: str, diagnose: bool = False):
    """Executes the benchmark lifecycle for a given document."""
    doc_cfg = DOCUMENTS[doc_key]
    checkpoint_file = doc_cfg["checkpoint_file"]
    pdf_path = doc_cfg["pdf_path"]

    print(f"\n{'#' * 75}")
    print(f"BENCHMARK TARGET: {doc_cfg['title']}")
    print(f"DOMAIN: {doc_cfg['domain']}")
    print(f"SOURCE FILE: {pdf_path}")
    print(f"{'#' * 75}\n")

    if diagnose:
        run_diagnose(doc_cfg)
        return

    records = load_checkpoint(path=checkpoint_file)

    # Step: Ingest & Generate
    if step in ["all", "generate"]:
        print(f"\n[Pipeline] Ingesting and indexing {pdf_path.name}...")
        chunks = build_deduplicated_chunks(pdf_path)
        pipeline = RetrievalPipeline(chunks)
        records = run_generation(pipeline, records, doc_cfg)

    # Step: Evaluate
    if step in ["all", "evaluate"]:
        if not records:
            print("[Evaluator] Error: No generation records found in checkpoint. Run with --step generate first.")
            sys.exit(1)
        records = evaluate_all_records(records, save_every_step=True, checkpoint_file=checkpoint_file)

    # Step: Report
    if step in ["all", "report"]:
        if not records:
            print("[Reporter] Error: No records found to report.")
            sys.exit(1)
        print("\n" + "=" * 70)
        print(f"PHASE 3: COMPILING AUDIT TRAIL, SUMMARY REPORT & FIGURES ({doc_cfg['title']})")
        print("=" * 70)
        export_all_reports(
            records,
            results_csv=doc_cfg["results_csv"],
            summary_md=doc_cfg["summary_md"],
            chart_png=doc_cfg["chart_png"],
            report_title=f"{doc_cfg['title']} — RAG Evaluation Report",
        )
        print(f"\nBenchmark experiment for {doc_cfg['title']} successfully completed!")


def export_cross_document_comparison():
    """Merges and compares results across both documents for high-level research analysis."""
    import pandas as pd

    all_dfs = []
    for doc_key, doc_cfg in DOCUMENTS.items():
        if doc_cfg["checkpoint_file"].exists():
            recs = load_checkpoint(doc_cfg["checkpoint_file"])
            if recs:
                df = compile_dataframe(recs)
                df["document_key"] = doc_key
                df["document_title"] = doc_cfg["title"]
                df["document_domain"] = doc_cfg["domain"]
                all_dfs.append(df)

    if not all_dfs:
        print("[Cross-Document] No checkpoint data found for comparison.")
        return

    combined_df = pd.concat(all_dfs, ignore_index=True)
    cross_csv = RESULTS_DIR / "cross_document_audit_trail.csv"
    combined_df.to_csv(cross_csv, index=False, encoding="utf-8")
    print(f"[Cross-Document] Combined audit trail saved to {cross_csv}")

    # Comparative markdown report
    cross_md_path = RESULTS_DIR / "cross_document_comparison.md"
    md = []
    md.append("# Cross-Domain RAG Research Benchmark — Comparative Analysis\n")
    md.append(f"**Total Records Across All Domains:** {len(combined_df)}  \n")
    md.append("Comparing Retrieval-Augmented Generation across contrasting domains:\n")
    for doc_key, doc_cfg in DOCUMENTS.items():
        count = len(combined_df[combined_df["document_key"] == doc_key])
        md.append(f"- **{doc_cfg['title']}** ({doc_cfg['domain']}): {count} records\n")
    md.append("\n---\n")

    # Table: Cross-domain embedding performance
    md.append("## 1. Embedding Model Performance by Domain\n")
    emb_cross = combined_df.groupby(["embedding_model", "document_key"]).agg(
        faithfulness_mean=("faithfulness", "mean"),
        context_recall_mean=("context_recall", "mean"),
        answer_correctness_mean=("answer_correctness", "mean"),
    ).reset_index()

    md.append("| Embedding Model | Document Domain | Faithfulness (Mean) | Context Recall (Mean) | Answer Correctness (Mean) |")
    md.append("| :--- | :--- | :---: | :---: | :---: |")
    for _, r in emb_cross.iterrows():
        md.append(
            f"| **{r['embedding_model']}** | {r['document_key']} | "
            f"{r['faithfulness_mean']:.2f}% | {r['context_recall_mean']:.2f}% | {r['answer_correctness_mean']:.2f}% |"
        )
    md.append("\n---\n")

    # Table: Cross-domain generator LLM performance
    md.append("## 2. Generator LLM Performance by Domain\n")
    gen_cross = combined_df.groupby(["generator_llm", "document_key"]).agg(
        faithfulness_mean=("faithfulness", "mean"),
        answer_correctness_mean=("answer_correctness", "mean"),
        avg_latency=("latency_seconds", "mean"),
    ).reset_index()

    md.append("| Generator LLM | Document Domain | Faithfulness (Mean) | Answer Correctness (Mean) | Avg Latency |")
    md.append("| :--- | :--- | :---: | :---: | :---: |")
    for _, r in gen_cross.iterrows():
        md.append(
            f"| **{r['generator_llm']}** | {r['document_key']} | "
            f"{r['faithfulness_mean']:.2f}% | {r['answer_correctness_mean']:.2f}% | {r['avg_latency']:.2f}s |"
        )

    with open(cross_md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print(f"[Cross-Document] Cross-document comparative report saved to {cross_md_path}")


def main():
    parser = argparse.ArgumentParser(description="Multi-Document RAG Research Benchmark")
    parser.add_argument(
        "--doc",
        choices=["paper1", "paper2", "all"],
        default="paper2",
        help="Specify document target: paper1 (Parkinson's), paper2 (Transformers), or all",
    )
    parser.add_argument(
        "--step",
        choices=["all", "generate", "evaluate", "report"],
        default="all",
        help="Specify which pipeline stage to execute",
    )
    parser.add_argument(
        "--diagnose",
        action="store_true",
        help="Run only PDF text extraction and retrieval diagnostic without LLM calls",
    )
    args = parser.parse_args()

    check_api_keys()

    if args.doc in ["paper1", "paper2"]:
        run_document_benchmark(args.doc, args.step, args.diagnose)
    elif args.doc == "all":
        for d in ["paper1", "paper2"]:
            run_document_benchmark(d, args.step, args.diagnose)
        export_cross_document_comparison()


if __name__ == "__main__":
    main()
