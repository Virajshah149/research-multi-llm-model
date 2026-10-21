"""
Reporting, Statistical Analysis, and Visualization Engine.
Exports auditable CSV trails, publication-standard Markdown comparison tables,
and comparative bar charts across embedding models and generator LLMs.
"""

import json
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from config import RESULTS_CSV, SUMMARY_MD, CHART_PNG, BENCHMARK_QUESTIONS, EMBEDDING_CONFIGS, GENERATOR_MODELS


def compile_dataframe(records: Dict[str, Any]) -> pd.DataFrame:
    """Flattens experiment JSON records into a clean tabular DataFrame."""
    rows = []
    for rec_key, data in records.items():
        eval_data = data.get("evaluation", {})
        f_info = eval_data.get("faithfulness", {})
        r_info = eval_data.get("context_recall", {})
        c_info = eval_data.get("answer_correctness", {})

        row = {
            "record_key": rec_key,
            "question_id": data.get("question_id"),
            "embedding_model": data.get("embedding_model"),
            "generator_llm": data.get("generator_llm"),
            "generator_provider": data.get("generator_provider"),
            "latency_seconds": data.get("latency_seconds"),
            "faithfulness": f_info.get("score"),
            "context_recall": r_info.get("score"),
            "answer_correctness": c_info.get("score"),
            "question": data.get("question"),
            "ground_truth": data.get("ground_truth"),
            "generated_answer": data.get("generated_answer"),
            "retrieved_context_count": len(data.get("retrieved_contexts", [])),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def generate_markdown_report(
    df: pd.DataFrame,
    output_path: Path = SUMMARY_MD,
    report_title: str = "RAG Research Benchmark — Final Evaluation Report",
) -> str:
    """Generates an auditable, peer-review style Markdown research summary."""
    md = []
    md.append(f"# {report_title}\n")
    md.append(f"**Total Records Evaluated:** {len(df)}  ")
    md.append(f"**Metrics Tracked:** Faithfulness, Context Recall, Answer Correctness  \n")
    md.append("---\n")

    # Table 1: Embedding Model Comparison (holding generation constant)
    md.append("## 1. Retrieval Performance by Embedding Model\n")
    emb_stats = df.groupby("embedding_model").agg(
        faithfulness_mean=("faithfulness", "mean"),
        faithfulness_std=("faithfulness", "std"),
        context_recall_mean=("context_recall", "mean"),
        context_recall_std=("context_recall", "std"),
        answer_correctness_mean=("answer_correctness", "mean"),
        answer_correctness_std=("answer_correctness", "std"),
        valid_count=("faithfulness", "count"),
    ).reset_index()

    md.append("| Embedding Model | Faithfulness (Mean ± Std) | Context Recall (Mean ± Std) | Answer Correctness (Mean ± Std) | Scored Count |")
    md.append("| :--- | :---: | :---: | :---: | :---: |")
    for _, r in emb_stats.iterrows():
        f_str = f"{r['faithfulness_mean']:.2f} ± {r['faithfulness_std']:.2f}" if pd.notnull(r['faithfulness_mean']) else "N/A"
        rec_str = f"{r['context_recall_mean']:.2f} ± {r['context_recall_std']:.2f}" if pd.notnull(r['context_recall_mean']) else "N/A"
        c_str = f"{r['answer_correctness_mean']:.2f} ± {r['answer_correctness_std']:.2f}" if pd.notnull(r['answer_correctness_mean']) else "N/A"
        md.append(f"| **{r['embedding_model']}** | {f_str} | {rec_str} | {c_str} | {int(r['valid_count'])} |")
    md.append("\n---\n")

    # Table 2: Generator LLM Comparison (holding retrieval constant)
    md.append("## 2. Generation Performance by Language Model\n")
    gen_stats = df.groupby("generator_llm").agg(
        faithfulness_mean=("faithfulness", "mean"),
        faithfulness_std=("faithfulness", "std"),
        context_recall_mean=("context_recall", "mean"),
        answer_correctness_mean=("answer_correctness", "mean"),
        answer_correctness_std=("answer_correctness", "std"),
        avg_latency=("latency_seconds", "mean"),
        valid_count=("answer_correctness", "count"),
    ).reset_index()

    md.append("| Generator LLM | Faithfulness (Mean ± Std) | Answer Correctness (Mean ± Std) | Avg Latency (s) | Scored Count |")
    md.append("| :--- | :---: | :---: | :---: | :---: |")
    for _, r in gen_stats.iterrows():
        f_str = f"{r['faithfulness_mean']:.2f} ± {r['faithfulness_std']:.2f}" if pd.notnull(r['faithfulness_mean']) else "N/A"
        c_str = f"{r['answer_correctness_mean']:.2f} ± {r['answer_correctness_std']:.2f}" if pd.notnull(r['answer_correctness_mean']) else "N/A"
        lat_str = f"{r['avg_latency']:.2f}s" if pd.notnull(r['avg_latency']) else "N/A"
        md.append(f"| **{r['generator_llm']}** | {f_str} | {c_str} | {lat_str} | {int(r['valid_count'])} |")
    md.append("\n---\n")

    # Table 3: By Question Detail
    md.append("## 3. Metric Breakdown by Benchmark Question\n")
    q_stats = df.groupby("question_id").agg(
        faithfulness_mean=("faithfulness", "mean"),
        context_recall_mean=("context_recall", "mean"),
        answer_correctness_mean=("answer_correctness", "mean"),
    ).reset_index()
    md.append("| Question ID | Faithfulness | Context Recall | Answer Correctness |")
    md.append("| :--- | :---: | :---: | :---: |")
    for _, r in q_stats.iterrows():
        md.append(f"| **{r['question_id']}** | {r['faithfulness_mean']:.2f} | {r['context_recall_mean']:.2f} | {r['answer_correctness_mean']:.2f} |")

    report_content = "\n".join(md)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    return report_content


def generate_visualization(df: pd.DataFrame, output_path: Path = CHART_PNG):
    """Plots clean research-grade grouped bar charts of metrics across models."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    metrics = ["faithfulness", "context_recall", "answer_correctness"]
    labels = ["Faithfulness", "Context Recall", "Answer Correctness"]

    # 1. Embedding Models Comparison
    emb_grp = df.groupby("embedding_model")[metrics].mean()
    emb_grp.plot(kind="bar", ax=axes[0], colormap="viridis", edgecolor="black", alpha=0.85)
    axes[0].set_title("Performance by Embedding Model (Held Generator Constant)", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("Metric Score (0.00 - 100.00)")
    axes[0].set_ylim(0, 105)
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=0)
    axes[0].legend(labels, loc="upper right")
    axes[0].grid(axis="y", linestyle="--", alpha=0.4)

    # 2. Generator LLMs Comparison
    gen_grp = df.groupby("generator_llm")[["faithfulness", "answer_correctness"]].mean()
    gen_grp.plot(kind="bar", ax=axes[1], colormap="magma", edgecolor="black", alpha=0.85)
    axes[1].set_title("Performance by Generator LLM (Held Context Constant)", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("Metric Score (0.00 - 100.00)")
    axes[1].set_ylim(0, 105)
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=0)
    axes[1].legend(["Faithfulness", "Answer Correctness"], loc="upper right")
    axes[1].grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=300)
    plt.close()
    print(f"[Reporter] Visualization saved to {output_path}")


def export_all_reports(
    records: Dict[str, Any],
    results_csv: Path = RESULTS_CSV,
    summary_md: Path = SUMMARY_MD,
    chart_png: Path = CHART_PNG,
    report_title: str = "RAG Research Benchmark — Final Evaluation Report",
):
    """Main reporting function producing CSV, Markdown summary, and figures."""
    df = compile_dataframe(records)
    
    # Save CSV with fallback if locked by Excel
    try:
        df.to_csv(results_csv, index=False, encoding="utf-8")
        print(f"[Reporter] Audit trail CSV saved to {results_csv}")
    except PermissionError:
        fallback_csv = results_csv.with_name(f"{results_csv.stem}_latest.csv")
        df.to_csv(fallback_csv, index=False, encoding="utf-8")
        print(f"[Reporter] Notice: '{results_csv.name}' is currently open in Excel or another app.")
        print(f"[Reporter] Saved updated audit trail to: {fallback_csv}")

    generate_markdown_report(df, summary_md, report_title=report_title)
    print(f"[Reporter] Summary report saved to {summary_md}")

    try:
        generate_visualization(df, chart_png)
    except Exception as e:
        print(f"[Reporter] Warning: Could not generate plot ({e})")
