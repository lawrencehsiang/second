
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Analyze SCRD experiment outputs.

Expected input structure:
- results.jsonl
- trace JSON files in the same directory or a subdirectory, e.g.:
  outputs/
    results.jsonl
    traces/
      gsm8k_0001_trace.json
      gsm8k_0002_trace.json
      ...

This script:
1. Loads sample-level results from results.jsonl
2. Loads trace-level usage / events / final_trace
3. Builds detailed summary tables
4. Generates plots for accuracy, tokens, rounds, stop reasons, and component cost
5. Writes CSV/JSON/PNG outputs to an analysis directory

Run example:
python analyze_scrd_results.py --results outputs/results.jsonl --traces outputs/traces --out outputs/analysis

If --traces is omitted, the script will search:
- sibling directory named "traces"
- the same directory as results.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd


# -----------------------------
# I/O helpers
# -----------------------------
def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSONL at {path}, line {line_no}: {e}") from e
    return rows


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def discover_trace_files(results_path: Path, traces_dir: Path | None) -> dict[str, Path]:
    candidates: list[Path] = []

    if traces_dir is not None and traces_dir.exists():
        candidates.extend(sorted(traces_dir.glob("*_trace.json")))

    sibling_traces = results_path.parent / "traces"
    if sibling_traces.exists():
        candidates.extend(sorted(sibling_traces.glob("*_trace.json")))

    candidates.extend(sorted(results_path.parent.glob("*_trace.json")))

    mapping: dict[str, Path] = {}
    for path in candidates:
        sample_id = path.name.replace("_trace.json", "")
        mapping[sample_id] = path
    return mapping


# -----------------------------
# Metric helpers
# -----------------------------
def safe_mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def safe_median(values: list[float]) -> float:
    return float(statistics.median(values)) if values else 0.0


def bool_int(x: Any) -> int:
    return 1 if bool(x) else 0


def classify_change(scrd_correct: bool, majority_correct: bool) -> str:
    if scrd_correct and not majority_correct:
        return "improved_over_majority"
    if (not scrd_correct) and majority_correct:
        return "degraded_from_majority"
    if scrd_correct and majority_correct:
        return "both_correct"
    return "both_wrong"


def extract_round_usage_summary(usage_records: list[dict[str, Any]]) -> dict[int, int]:
    round_totals: dict[int, int] = defaultdict(int)
    for rec in usage_records:
        round_id = rec.get("round_id")
        total = int(rec.get("total_tokens", 0))
        if round_id is not None:
            round_totals[int(round_id)] += total
    return dict(sorted(round_totals.items(), key=lambda kv: kv[0]))


def accuracy_per_1k_tokens(correct_series: pd.Series, token_series: pd.Series) -> float:
    total_correct = int(correct_series.sum())
    total_tokens = float(token_series.sum())
    if total_tokens <= 0:
        return 0.0
    return total_correct / (total_tokens / 1000.0)


# -----------------------------
# Plot helpers
# -----------------------------
def save_bar(values: pd.Series, title: str, ylabel: str, outpath: Path, rot: int = 0) -> None:
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)
    values.plot(kind="bar", ax=ax)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("")
    plt.xticks(rotation=rot)
    plt.tight_layout()
    fig.savefig(outpath, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_hist(series: pd.Series, title: str, xlabel: str, outpath: Path, bins: int = 20) -> None:
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)
    ax.hist(series.dropna(), bins=bins)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    plt.tight_layout()
    fig.savefig(outpath, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_scatter(
    x: pd.Series,
    y: pd.Series,
    c: pd.Series | None,
    title: str,
    xlabel: str,
    ylabel: str,
    outpath: Path,
) -> None:
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)
    if c is None:
        ax.scatter(x, y)
    else:
        # categorical coloring without explicit colors
        categories = pd.Categorical(c)
        ax.scatter(x, y, c=categories.codes)
        # manual legend labels
        handles = []
        for i, cat in enumerate(categories.categories):
            handles.append(plt.Line2D([0], [0], marker='o', linestyle='', label=str(cat)))
        ax.legend(handles=handles, title="Category")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.tight_layout()
    fig.savefig(outpath, dpi=180, bbox_inches="tight")
    plt.close(fig)


# -----------------------------
# Main analysis
# -----------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze SCRD experiment outputs.")
    parser.add_argument("--results", type=str, required=True, help="Path to results.jsonl")
    parser.add_argument("--traces", type=str, default=None, help="Directory containing *_trace.json files")
    parser.add_argument("--out", type=str, default=None, help="Output directory for analysis artifacts")
    args = parser.parse_args()

    results_path = Path(args.results).resolve()
    traces_dir = Path(args.traces).resolve() if args.traces else None
    out_dir = Path(args.out).resolve() if args.out else results_path.parent / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_jsonl(results_path)
    if not rows:
        raise ValueError(f"No rows found in {results_path}")

    df = pd.DataFrame(rows)

    # Normalize optional columns
    expected_bool_cols = [
        "single_agent_correct",
        "majority_voting_correct",
        "scrd_correct",
    ]
    for col in expected_bool_cols:
        if col in df.columns:
            df[col] = df[col].astype(bool)

    # Attach trace-derived fields
    trace_map = discover_trace_files(results_path, traces_dir)
    trace_rows: list[dict[str, Any]] = []
    for sample_id in df["sample_id"].tolist():
        trace_path = trace_map.get(sample_id)
        if trace_path is None:
            trace_rows.append({
                "sample_id": sample_id,
                "trace_found": False,
                "trace_path": None,
                "usage_records_count": 0,
                "final_trace_rounds": None,
                "execution_events_count": 0,
                "rollback_events": 0,
                "repair_round_events": 0,
                "stable_consensus_events": 0,
                "round_usage_summary_json": "{}",
                "component_token_from_trace_json": "{}",
            })
            continue

        trace = load_json(trace_path)
        usage_records = trace.get("usage_records", [])
        execution_events = trace.get("execution_events", [])
        final_trace = trace.get("final_trace", [])

        rollback_events = sum(1 for e in execution_events if e.get("type") == "rollback_triggered")
        repair_round_events = sum(1 for e in execution_events if e.get("type") == "repair_round_executed")
        stable_consensus_events = sum(1 for e in execution_events if e.get("type") == "stable_consensus_early_stop")

        round_usage_summary = extract_round_usage_summary(usage_records)

        component_counter: dict[str, int] = defaultdict(int)
        for rec in usage_records:
            component_counter[str(rec.get("component"))] += int(rec.get("total_tokens", 0))

        trace_rows.append({
            "sample_id": sample_id,
            "trace_found": True,
            "trace_path": str(trace_path),
            "usage_records_count": len(usage_records),
            "final_trace_rounds": len(final_trace),
            "execution_events_count": len(execution_events),
            "rollback_events": rollback_events,
            "repair_round_events": repair_round_events,
            "stable_consensus_events": stable_consensus_events,
            "round_usage_summary_json": json.dumps(round_usage_summary, ensure_ascii=False),
            "component_token_from_trace_json": json.dumps(component_counter, ensure_ascii=False),
        })

    trace_df = pd.DataFrame(trace_rows)
    df = df.merge(trace_df, on="sample_id", how="left")

    # Derived columns
    if "majority_voting_correct" in df.columns and "scrd_correct" in df.columns:
        df["change_vs_majority"] = [
            classify_change(scrd, maj)
            for scrd, maj in zip(df["scrd_correct"], df["majority_voting_correct"])
        ]
    else:
        df["change_vs_majority"] = "unknown"

    if "single_agent_correct" in df.columns and "scrd_correct" in df.columns:
        df["change_vs_single_agent"] = [
            "improved_over_single"
            if scrd and not single else
            "degraded_from_single"
            if (not scrd) and single else
            "same_outcome"
            for scrd, single in zip(df["scrd_correct"], df["single_agent_correct"])
        ]

    # Fill token columns if absent
    token_cols = [
        "single_agent_total_tokens",
        "majority_vote_total_tokens",
        "scrd_total_tokens",
        "scrd_prompt_tokens",
        "scrd_completion_tokens",
        "agent_total_tokens",
        "recorder_total_tokens",
        "evaluator_total_tokens",
        "repair_brief_total_tokens",
        "repair_evaluator_total_tokens",
        "repair_agent_total_tokens",
        "effective_rounds_used",
        "actual_rounds_executed",
    ]
    for col in token_cols:
        if col not in df.columns:
            df[col] = 0

    # -------------------------
    # Tables
    # -------------------------
    sample_level_path = out_dir / "sample_level_analysis.csv"
    df.to_csv(sample_level_path, index=False, encoding="utf-8-sig")

    overall_summary = {
        "n_samples": int(len(df)),
        "single_agent_accuracy": safe_mean(df["single_agent_correct"].astype(int).tolist()) if "single_agent_correct" in df else None,
        "majority_vote_accuracy": safe_mean(df["majority_voting_correct"].astype(int).tolist()) if "majority_voting_correct" in df else None,
        "scrd_accuracy": safe_mean(df["scrd_correct"].astype(int).tolist()) if "scrd_correct" in df else None,
        "single_agent_total_tokens_mean": safe_mean(df["single_agent_total_tokens"].tolist()),
        "majority_vote_total_tokens_mean": safe_mean(df["majority_vote_total_tokens"].tolist()),
        "scrd_total_tokens_mean": safe_mean(df["scrd_total_tokens"].tolist()),
        "scrd_prompt_tokens_mean": safe_mean(df["scrd_prompt_tokens"].tolist()),
        "scrd_completion_tokens_mean": safe_mean(df["scrd_completion_tokens"].tolist()),
        "effective_rounds_used_mean": safe_mean(df["effective_rounds_used"].tolist()),
        "actual_rounds_executed_mean": safe_mean(df["actual_rounds_executed"].tolist()),
        "scrd_accuracy_per_1k_tokens": accuracy_per_1k_tokens(df["scrd_correct"].astype(int), df["scrd_total_tokens"]),
        "majority_accuracy_per_1k_tokens": accuracy_per_1k_tokens(df["majority_voting_correct"].astype(int), df["majority_vote_total_tokens"]),
        "single_accuracy_per_1k_tokens": accuracy_per_1k_tokens(df["single_agent_correct"].astype(int), df["single_agent_total_tokens"]),
    }

    stop_reason_summary = (
        df.groupby("stop_reason")
        .agg(
            n=("sample_id", "count"),
            scrd_accuracy=("scrd_correct", "mean"),
            mean_scrd_tokens=("scrd_total_tokens", "mean"),
            median_scrd_tokens=("scrd_total_tokens", "median"),
            mean_effective_rounds=("effective_rounds_used", "mean"),
            mean_actual_rounds=("actual_rounds_executed", "mean"),
        )
        .reset_index()
    )

    component_summary = pd.DataFrame(
        {
            "component": [
                "agent_total_tokens",
                "recorder_total_tokens",
                "evaluator_total_tokens",
                "repair_brief_total_tokens",
                "repair_evaluator_total_tokens",
                "repair_agent_total_tokens",
            ],
            "mean_tokens": [
                df["agent_total_tokens"].mean(),
                df["recorder_total_tokens"].mean(),
                df["evaluator_total_tokens"].mean(),
                df["repair_brief_total_tokens"].mean(),
                df["repair_evaluator_total_tokens"].mean(),
                df["repair_agent_total_tokens"].mean(),
            ],
            "sum_tokens": [
                df["agent_total_tokens"].sum(),
                df["recorder_total_tokens"].sum(),
                df["evaluator_total_tokens"].sum(),
                df["repair_brief_total_tokens"].sum(),
                df["repair_evaluator_total_tokens"].sum(),
                df["repair_agent_total_tokens"].sum(),
            ],
        }
    )

    change_vs_majority_summary = (
        df.groupby("change_vs_majority")
        .agg(
            n=("sample_id", "count"),
            mean_scrd_tokens=("scrd_total_tokens", "mean"),
            mean_actual_rounds=("actual_rounds_executed", "mean"),
        )
        .reset_index()
    )

    token_gap_summary = pd.DataFrame(
        {
            "metric": [
                "mean_single_agent_tokens",
                "mean_majority_vote_tokens",
                "mean_scrd_tokens",
                "scrd_vs_majority_ratio",
                "scrd_vs_single_ratio",
            ],
            "value": [
                df["single_agent_total_tokens"].mean(),
                df["majority_vote_total_tokens"].mean(),
                df["scrd_total_tokens"].mean(),
                (df["scrd_total_tokens"].mean() / df["majority_vote_total_tokens"].mean())
                if df["majority_vote_total_tokens"].mean() > 0 else None,
                (df["scrd_total_tokens"].mean() / df["single_agent_total_tokens"].mean())
                if df["single_agent_total_tokens"].mean() > 0 else None,
            ],
        }
    )

    # Save tables
    with (out_dir / "overall_summary.json").open("w", encoding="utf-8") as f:
        json.dump(overall_summary, f, ensure_ascii=False, indent=2)

    stop_reason_summary.to_csv(out_dir / "stop_reason_summary.csv", index=False, encoding="utf-8-sig")
    component_summary.to_csv(out_dir / "component_token_summary.csv", index=False, encoding="utf-8-sig")
    change_vs_majority_summary.to_csv(out_dir / "change_vs_majority_summary.csv", index=False, encoding="utf-8-sig")
    token_gap_summary.to_csv(out_dir / "token_gap_summary.csv", index=False, encoding="utf-8-sig")

    # -------------------------
    # Plots
    # -------------------------
    # 1. Accuracy by method
    acc_series = pd.Series(
        {
            "Single Agent": float(df["single_agent_correct"].mean()),
            "Majority Vote": float(df["majority_voting_correct"].mean()),
            "SCRD": float(df["scrd_correct"].mean()),
        }
    )
    save_bar(acc_series, "Accuracy by Method", "Accuracy", out_dir / "fig_accuracy_by_method.png", rot=0)

    # 2. Mean total tokens by method
    token_series = pd.Series(
        {
            "Single Agent": float(df["single_agent_total_tokens"].mean()),
            "Majority Vote": float(df["majority_vote_total_tokens"].mean()),
            "SCRD": float(df["scrd_total_tokens"].mean()),
        }
    )
    save_bar(token_series, "Mean Total Tokens by Method", "Mean total tokens", out_dir / "fig_tokens_by_method.png", rot=0)

    # 3. SCRD prompt vs completion
    prompt_completion = pd.Series(
        {
            "Prompt": float(df["scrd_prompt_tokens"].mean()),
            "Completion": float(df["scrd_completion_tokens"].mean()),
        }
    )
    save_bar(prompt_completion, "SCRD Mean Prompt vs Completion Tokens", "Mean tokens", out_dir / "fig_scrd_prompt_vs_completion.png", rot=0)

    # 4. Component token breakdown
    comp_plot_series = pd.Series(
        {
            "Agent": float(df["agent_total_tokens"].mean()),
            "Recorder": float(df["recorder_total_tokens"].mean()),
            "Evaluator": float(df["evaluator_total_tokens"].mean()),
            "RepairBrief": float(df["repair_brief_total_tokens"].mean()),
            "RepairEval": float(df["repair_evaluator_total_tokens"].mean()),
            "RepairAgent": float(df["repair_agent_total_tokens"].mean()),
        }
    )
    save_bar(comp_plot_series, "Mean SCRD Tokens by Component", "Mean tokens", out_dir / "fig_component_token_breakdown.png", rot=20)

    # 5. Stop reason distribution
    stop_counts = df["stop_reason"].value_counts()
    save_bar(stop_counts, "Stop Reason Distribution", "Count", out_dir / "fig_stop_reason_distribution.png", rot=20)

    # 6. Mean SCRD tokens by stop reason
    stop_token_means = df.groupby("stop_reason")["scrd_total_tokens"].mean().sort_values(ascending=False)
    save_bar(stop_token_means, "Mean SCRD Tokens by Stop Reason", "Mean tokens", out_dir / "fig_tokens_by_stop_reason.png", rot=20)

    # 7. Actual rounds distribution
    save_hist(df["actual_rounds_executed"], "Actual Rounds Executed Distribution", "Actual rounds executed", out_dir / "fig_actual_rounds_hist.png", bins=10)

    # 8. SCRD tokens distribution
    save_hist(df["scrd_total_tokens"], "SCRD Total Tokens Distribution", "SCRD total tokens", out_dir / "fig_scrd_tokens_hist.png", bins=20)

    # 9. Accuracy-vs-cost scatter (SCRD)
    # y-axis: correctness (0/1), x-axis: scrd_total_tokens
    save_scatter(
        x=df["scrd_total_tokens"],
        y=df["scrd_correct"].astype(int),
        c=df["stop_reason"],
        title="SCRD Correctness vs Total Tokens",
        xlabel="SCRD total tokens",
        ylabel="SCRD correctness (0/1)",
        outpath=out_dir / "fig_scrd_correctness_vs_tokens.png",
    )

    # 10. Change vs majority distribution
    change_counts = df["change_vs_majority"].value_counts()
    save_bar(change_counts, "SCRD vs Majority Outcome Changes", "Count", out_dir / "fig_change_vs_majority.png", rot=20)

    # -------------------------
    # Markdown report
    # -------------------------
    report_path = out_dir / "analysis_report.md"
    lines: list[str] = []
    lines.append("# SCRD Experiment Analysis Report\n")
    lines.append(f"- Samples analyzed: **{overall_summary['n_samples']}**\n")
    lines.append(f"- Single Agent accuracy: **{overall_summary['single_agent_accuracy']:.4f}**\n")
    lines.append(f"- Majority Vote accuracy: **{overall_summary['majority_vote_accuracy']:.4f}**\n")
    lines.append(f"- SCRD accuracy: **{overall_summary['scrd_accuracy']:.4f}**\n")
    lines.append("")
    lines.append("## Cost Overview\n")
    lines.append(f"- Mean single-agent tokens: **{overall_summary['single_agent_total_tokens_mean']:.2f}**\n")
    lines.append(f"- Mean majority-vote tokens: **{overall_summary['majority_vote_total_tokens_mean']:.2f}**\n")
    lines.append(f"- Mean SCRD tokens: **{overall_summary['scrd_total_tokens_mean']:.2f}**\n")
    lines.append(f"- Mean SCRD prompt tokens: **{overall_summary['scrd_prompt_tokens_mean']:.2f}**\n")
    lines.append(f"- Mean SCRD completion tokens: **{overall_summary['scrd_completion_tokens_mean']:.2f}**\n")
    lines.append("")
    if token_gap_summary["metric"].tolist():
        ratio_mv = token_gap_summary.loc[token_gap_summary["metric"] == "scrd_vs_majority_ratio", "value"].values
        ratio_single = token_gap_summary.loc[token_gap_summary["metric"] == "scrd_vs_single_ratio", "value"].values
        if len(ratio_mv) > 0 and ratio_mv[0] is not None:
            lines.append(f"- SCRD / Majority token ratio: **{float(ratio_mv[0]):.2f}x**\n")
        if len(ratio_single) > 0 and ratio_single[0] is not None:
            lines.append(f"- SCRD / Single token ratio: **{float(ratio_single[0]):.2f}x**\n")
    lines.append("")
    lines.append("## Efficiency\n")
    lines.append(f"- Single Agent accuracy per 1k tokens: **{overall_summary['single_accuracy_per_1k_tokens']:.4f}**\n")
    lines.append(f"- Majority Vote accuracy per 1k tokens: **{overall_summary['majority_accuracy_per_1k_tokens']:.4f}**\n")
    lines.append(f"- SCRD accuracy per 1k tokens: **{overall_summary['scrd_accuracy_per_1k_tokens']:.4f}**\n")
    lines.append("")
    lines.append("## Main Takeaways Template\n")
    lines.append("- Check whether SCRD improves over majority vote often enough to justify its extra tokens.\n")
    lines.append("- Check which stop reasons are associated with the highest cost.\n")
    lines.append("- Check whether recorder/evaluator dominate the prompt cost.\n")
    lines.append("- Check whether degraded_from_majority cases are especially expensive.\n")

    report_path.write_text("".join(lines), encoding="utf-8")

    print(f"Analysis completed. Outputs written to: {out_dir}")


if __name__ == "__main__":
    main()
