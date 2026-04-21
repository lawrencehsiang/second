#!/usr/bin/env python3
# -*- coding: utf-8 -*-
### 运行命令： python analyze_scrd_results_revised.py --results outputs/results.jsonl --traces outputs/traces --out outputs/analysis
"""
Analyze SCRD experiment outputs with clearer tables/plots and an extra baseline:
majority voting on round 3 answers (when a third round exists in the trace).

Expected input structure:
- results.jsonl
- trace JSON files in a sibling directory or subdirectory, e.g.:
  outputs/
    results.jsonl
    traces/
      gsm8k_0001_trace.json
      gsm8k_0002_trace.json
      ...

Main outputs:
1. sample_overview.csv                 # readable per-sample table
2. method_comparison.csv               # Single / Round1 Vote / Round3 Vote / SCRD
3. stop_reason_summary.csv             # grouped by stop reason
4. outcome_vs_round1_vote.csv          # improved / degraded / ... vs round-1 majority
5. outcome_vs_round3_vote.csv          # improved / degraded / ... vs round-3 majority
6. round3_vote_samples.csv             # only samples where round-3 majority is available
7. analysis_report.md                  # short written summary
8. several PNG plots

If --traces is omitted, the script searches automatically in:
- sibling directory named "traces"
- the same directory as results.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
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
# Generic helpers
# -----------------------------
def safe_mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if pd.notna(v)]
    return float(sum(vals) / len(vals)) if vals else 0.0



def safe_median(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if pd.notna(v)]
    return float(statistics.median(vals)) if vals else 0.0



def accuracy_per_1k_tokens(correct_series: pd.Series, token_series: pd.Series) -> float:
    valid = token_series.fillna(0) > 0
    total_correct = int(correct_series[valid].sum())
    total_tokens = float(token_series[valid].sum())
    if total_tokens <= 0:
        return 0.0
    return total_correct / (total_tokens / 1000.0)



def normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()



def canonical_numeric(text: str) -> str | None:
    s = str(text).strip().replace(",", "")
    nums = re.findall(r"[-+]?\d*\.?\d+", s)
    if not nums:
        return None
    value = nums[-1]
    try:
        num = float(value)
    except ValueError:
        return None
    if num.is_integer():
        return str(int(num))
    return ("%.10f" % num).rstrip("0").rstrip(".")



def canonical_bool(text: str) -> str | None:
    s = normalize_ws(text).lower()
    positive = {
        "true", "yes", "y", "1", "correct", "supported", "entails"
    }
    negative = {
        "false", "no", "n", "0", "incorrect", "not supported", "does not entail"
    }

    if s in positive:
        return "true"
    if s in negative:
        return "false"

    # looser matching
    if re.fullmatch(r"(answer\s*[:：]\s*)?(true|yes)", s):
        return "true"
    if re.fullmatch(r"(answer\s*[:：]\s*)?(false|no)", s):
        return "false"
    return None



def normalize_prediction(answer: Any, dataset_name: str | None, gold_answer: Any | None = None) -> str:
    if answer is None:
        return ""
    text = normalize_ws(str(answer))
    ds = (dataset_name or "").lower()

    if ds == "strategyqa":
        bool_val = canonical_bool(text)
        return bool_val if bool_val is not None else text.lower()

    if gold_answer is not None:
        gold_num = canonical_numeric(str(gold_answer))
        if gold_num is not None:
            pred_num = canonical_numeric(text)
            if pred_num is not None:
                return pred_num

    bool_val = canonical_bool(text)
    if bool_val is not None:
        return bool_val

    pred_num = canonical_numeric(text)
    if pred_num is not None:
        return pred_num

    return text.lower()



def is_correct(answer: Any, gold_answer: Any, dataset_name: str | None) -> bool:
    return normalize_prediction(answer, dataset_name, gold_answer) == normalize_prediction(gold_answer, dataset_name, gold_answer)



def choose_majority_vote(answers: list[Any], dataset_name: str | None, gold_answer: Any | None = None) -> tuple[str | None, bool]:
    """
    Return (winner_answer_text, strict_majority_exists).
    Uses normalized answers for grouping, and resolves ties by earliest occurrence.
    """
    if not answers:
        return None, False

    normalized = [normalize_prediction(a, dataset_name, gold_answer) for a in answers]
    counts = Counter(normalized)
    max_count = max(counts.values())
    strict_majority = max_count > (len(normalized) // 2)

    winner_norm = None
    for norm in normalized:
        if counts[norm] == max_count:
            winner_norm = norm
            break

    if winner_norm is None:
        return None, False

    for raw, norm in zip(answers, normalized):
        if norm == winner_norm:
            return normalize_ws(str(raw)), strict_majority

    return normalize_ws(str(answers[0])), strict_majority



def classify_change(scrd_correct: Any, baseline_correct: Any, baseline_available: bool = True) -> str:
    if not baseline_available or pd.isna(baseline_correct):
        return "baseline_unavailable"
    if bool(scrd_correct) and not bool(baseline_correct):
        return "improved"
    if (not bool(scrd_correct)) and bool(baseline_correct):
        return "degraded"
    if bool(scrd_correct) and bool(baseline_correct):
        return "both_correct"
    return "both_wrong"



def extract_round_usage_summary(usage_records: list[dict[str, Any]]) -> dict[int, int]:
    round_totals: dict[int, int] = defaultdict(int)
    for rec in usage_records:
        round_id = rec.get("round_id")
        total = int(rec.get("total_tokens", 0))
        if round_id is not None:
            try:
                round_totals[int(round_id)] += total
            except Exception:
                continue
    return dict(sorted(round_totals.items(), key=lambda kv: kv[0]))



def cumulative_tokens_to_round(round_usage_summary: dict[int, int], target_round: int) -> int | None:
    if not round_usage_summary:
        return None
    eligible = [tokens for rnd, tokens in round_usage_summary.items() if rnd <= target_round]
    return int(sum(eligible)) if eligible else None


# -----------------------------
# Trace parsing helpers
# -----------------------------
ROUND_KEYS = ("round_id", "round", "current_round", "round_index")
ANSWER_SCALAR_KEYS = ("current_answer", "answer", "final_answer", "prediction", "response")
ANSWER_LIST_KEYS = ("current_answers", "answers", "round_answers", "agent_answers")
AGENT_CONTAINER_KEYS = ("agent_outputs", "outputs", "responses", "agents")



def coerce_round_id(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        if isinstance(value, str):
            m = re.search(r"(\d+)", value)
            if m:
                return int(m.group(1))
    return None



def extract_answers_direct(obj: Any) -> list[str] | None:
    if isinstance(obj, dict):
        for key in ANSWER_LIST_KEYS:
            if key in obj and isinstance(obj[key], list):
                vals = [normalize_ws(str(x)) for x in obj[key] if x is not None]
                if len(vals) >= 2:
                    return vals

        for key in AGENT_CONTAINER_KEYS:
            if key in obj and isinstance(obj[key], list):
                answers: list[str] = []
                for item in obj[key]:
                    if isinstance(item, dict):
                        for akey in ANSWER_SCALAR_KEYS:
                            if akey in item and item[akey] is not None:
                                answers.append(normalize_ws(str(item[akey])))
                                break
                    elif item is not None:
                        answers.append(normalize_ws(str(item)))
                if len(answers) >= 2:
                    return answers

        # Handle structures like {"A": {...}, "B": {...}, "C": {...}}
        uppercase_keys = [k for k in obj.keys() if isinstance(k, str) and len(k) <= 2 and k.isupper()]
        if len(uppercase_keys) >= 2:
            answers: list[str] = []
            for k in uppercase_keys:
                v = obj[k]
                if isinstance(v, dict):
                    for akey in ANSWER_SCALAR_KEYS:
                        if akey in v and v[akey] is not None:
                            answers.append(normalize_ws(str(v[akey])))
                            break
                elif v is not None:
                    answers.append(normalize_ws(str(v)))
            if len(answers) >= 2:
                return answers

    elif isinstance(obj, list):
        answers: list[str] = []
        for item in obj:
            if isinstance(item, dict):
                for akey in ANSWER_SCALAR_KEYS:
                    if akey in item and item[akey] is not None:
                        answers.append(normalize_ws(str(item[akey])))
                        break
            elif item is not None and not isinstance(item, (list, dict)):
                answers.append(normalize_ws(str(item)))
        if len(answers) >= 2:
            return answers

    return None



def recursively_find_answers(obj: Any, limit: int = 2000) -> list[list[str]]:
    found: list[list[str]] = []
    stack: list[Any] = [obj]
    visited = 0

    while stack and visited < limit:
        current = stack.pop()
        visited += 1

        direct = extract_answers_direct(current)
        if direct is not None:
            found.append(direct)

        if isinstance(current, dict):
            stack.extend(current.values())
        elif isinstance(current, list):
            stack.extend(current)

    return found



def get_round_item_candidates(container: Any, round_id: int) -> list[Any]:
    candidates: list[Any] = []

    if isinstance(container, list):
        # explicit round match
        for item in container:
            if isinstance(item, dict):
                for key in ROUND_KEYS:
                    rid = coerce_round_id(item.get(key))
                    if rid == round_id:
                        candidates.append(item)
                        break
        # fallback: positional assumption
        if round_id - 1 < len(container):
            candidates.append(container[round_id - 1])

    elif isinstance(container, dict):
        # direct numeric keys
        for key, value in container.items():
            rid = coerce_round_id(key)
            if rid == round_id:
                candidates.append(value)

    return candidates



def extract_round_answers(trace: dict[str, Any], round_id: int) -> list[str] | None:
    search_order = [
        "final_trace",
        "rounds",
        "round_results",
        "trace",
        "history",
        "execution_events",
    ]

    # 1) Search top-level named containers first.
    for key in search_order:
        container = trace.get(key)
        if container is None:
            continue
        for item in get_round_item_candidates(container, round_id):
            direct = extract_answers_direct(item)
            if direct is not None:
                return direct
            nested = recursively_find_answers(item)
            if nested:
                nested.sort(key=lambda x: (-len(x), x))
                return nested[0]

    # 2) Search execution events explicitly by round id.
    execution_events = trace.get("execution_events", [])
    if isinstance(execution_events, list):
        for event in execution_events:
            if not isinstance(event, dict):
                continue
            event_round = None
            for key in ROUND_KEYS:
                if key in event:
                    event_round = coerce_round_id(event.get(key))
                    break
            if event_round != round_id:
                continue
            direct = extract_answers_direct(event)
            if direct is not None:
                return direct
            nested = recursively_find_answers(event)
            if nested:
                nested.sort(key=lambda x: (-len(x), x))
                return nested[0]

    # 3) Last resort: search globally for any subtree tagged with this round.
    stack = [trace]
    visited = 0
    while stack and visited < 4000:
        current = stack.pop()
        visited += 1
        if isinstance(current, dict):
            has_match = False
            for key in ROUND_KEYS:
                rid = coerce_round_id(current.get(key))
                if rid == round_id:
                    has_match = True
                    break
            if has_match:
                direct = extract_answers_direct(current)
                if direct is not None:
                    return direct
                nested = recursively_find_answers(current)
                if nested:
                    nested.sort(key=lambda x: (-len(x), x))
                    return nested[0]
            stack.extend(current.values())
        elif isinstance(current, list):
            stack.extend(current)

    return None


# -----------------------------
# Plot helpers
# -----------------------------
def save_bar(values: pd.Series, title: str, ylabel: str, outpath: Path, rot: int = 0) -> None:
    fig = plt.figure(figsize=(8.5, 5.2))
    ax = fig.add_subplot(111)

    x = np.arange(len(values))
    y = values.values.astype(float)
    bars = ax.bar(x, y)
    ax.set_xticks(x)
    ax.set_xticklabels(values.index.tolist(), rotation=rot, ha="right" if rot else "center")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("")

    ymax = max(y) if len(y) else 1.0
    for bar, val in zip(bars, y):
        if abs(val) <= 1.5:
            label = f"{val:.3f}"
        elif abs(val) < 100:
            label = f"{val:.1f}"
        else:
            label = f"{val:.0f}"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + ymax * 0.01, label,
                ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    fig.savefig(outpath, dpi=180, bbox_inches="tight")
    plt.close(fig)



def save_hist(series: pd.Series, title: str, xlabel: str, outpath: Path, bins: int = 20) -> None:
    fig = plt.figure(figsize=(8.5, 5.2))
    ax = fig.add_subplot(111)
    vals = series.dropna()
    ax.hist(vals, bins=bins)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    plt.tight_layout()
    fig.savefig(outpath, dpi=180, bbox_inches="tight")
    plt.close(fig)



def save_scatter_by_category(
    x: pd.Series,
    y: pd.Series,
    categories: pd.Series,
    title: str,
    xlabel: str,
    ylabel: str,
    outpath: Path,
) -> None:
    fig = plt.figure(figsize=(8.5, 5.2))
    ax = fig.add_subplot(111)

    rng = np.random.default_rng(0)
    y_vals = y.astype(float).to_numpy()
    y_jitter = y_vals + rng.normal(loc=0.0, scale=0.03, size=len(y_vals))

    cat_series = categories.fillna("unknown").astype(str)
    cat_order = list(dict.fromkeys(cat_series.tolist()))
    cmap = plt.get_cmap("tab10")
    color_map = {cat: cmap(i % 10) for i, cat in enumerate(cat_order)}

    for cat in cat_order:
        mask = cat_series == cat
        ax.scatter(
            x[mask],
            y_jitter[mask],
            label=cat,
            color=color_map[cat],
            alpha=0.78,
            edgecolors="white",
            linewidths=0.4,
            s=30,
        )

    handles = [
        Line2D([0], [0], marker="o", linestyle="", label=cat,
               markerfacecolor=color_map[cat], markeredgecolor="white", markersize=8)
        for cat in cat_order
    ]
    ax.legend(handles=handles, title="Stop reason")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Wrong", "Correct"])
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

    # Normalize expected columns
    for col in [
        "single_agent_correct",
        "majority_voting_correct",
        "scrd_correct",
        "effective_rounds_used",
        "actual_rounds_executed",
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
    ]:
        if col not in df.columns:
            df[col] = pd.NA

    for col in ["single_agent_correct", "majority_voting_correct", "scrd_correct"]:
        if col in df.columns:
            df[col] = df[col].astype(bool)

    # Basic round-1 structure info from results
    df["round1_unique_answers"] = df["round_1_answers"].apply(lambda x: len({normalize_ws(str(v)) for v in (x or [])}))
    df["round1_all_agree"] = df["round1_unique_answers"] == 1

    # Attach trace-derived fields
    trace_map = discover_trace_files(results_path, traces_dir)
    trace_rows: list[dict[str, Any]] = []

    for _, row in df.iterrows():
        sample_id = row["sample_id"]
        dataset_name = row.get("dataset_name")
        gold_answer = row.get("gold_answer")
        trace_path = trace_map.get(sample_id)

        base_payload = {
            "sample_id": sample_id,
            "trace_found": False,
            "trace_path": None,
            "usage_records_count": 0,
            "final_trace_rounds": pd.NA,
            "execution_events_count": 0,
            "rollback_events": 0,
            "repair_round_events": 0,
            "stable_consensus_events": 0,
            "round_usage_summary_json": "{}",
            "round3_answers_json": "[]",
            "round3_vote_available": False,
            "round3_vote_answer": pd.NA,
            "round3_vote_correct": pd.NA,
            "round3_vote_strict_majority": pd.NA,
            "round3_vote_total_tokens": pd.NA,
        }

        if trace_path is None:
            trace_rows.append(base_payload)
            continue

        trace = load_json(trace_path)
        usage_records = trace.get("usage_records", []) if isinstance(trace.get("usage_records", []), list) else []
        execution_events = trace.get("execution_events", []) if isinstance(trace.get("execution_events", []), list) else []
        final_trace = trace.get("final_trace", []) if isinstance(trace.get("final_trace", []), list) else []

        rollback_events = sum(1 for e in execution_events if isinstance(e, dict) and e.get("type") == "rollback_triggered")
        repair_round_events = sum(1 for e in execution_events if isinstance(e, dict) and e.get("type") == "repair_round_executed")
        stable_consensus_events = sum(1 for e in execution_events if isinstance(e, dict) and e.get("type") == "stable_consensus_early_stop")

        round_usage_summary = extract_round_usage_summary(usage_records)

        round3_answers = extract_round_answers(trace, round_id=3)
        round3_vote_answer = None
        round3_vote_strict_majority = None
        round3_vote_correct = None
        round3_vote_total_tokens = cumulative_tokens_to_round(round_usage_summary, 3)
        round3_vote_available = bool(round3_answers)

        if round3_answers:
            round3_vote_answer, round3_vote_strict_majority = choose_majority_vote(round3_answers, dataset_name, gold_answer)
            round3_vote_correct = is_correct(round3_vote_answer, gold_answer, dataset_name) if round3_vote_answer is not None else False

        payload = {
            "sample_id": sample_id,
            "trace_found": True,
            "trace_path": str(trace_path),
            "usage_records_count": len(usage_records),
            "final_trace_rounds": len(final_trace) if final_trace else pd.NA,
            "execution_events_count": len(execution_events),
            "rollback_events": rollback_events,
            "repair_round_events": repair_round_events,
            "stable_consensus_events": stable_consensus_events,
            "round_usage_summary_json": json.dumps(round_usage_summary, ensure_ascii=False),
            "round3_answers_json": json.dumps(round3_answers or [], ensure_ascii=False),
            "round3_vote_available": round3_vote_available,
            "round3_vote_answer": round3_vote_answer,
            "round3_vote_correct": round3_vote_correct,
            "round3_vote_strict_majority": round3_vote_strict_majority,
            "round3_vote_total_tokens": round3_vote_total_tokens,
        }
        trace_rows.append(payload)

    trace_df = pd.DataFrame(trace_rows)
    df = df.merge(trace_df, on="sample_id", how="left")

    # Outcome categories vs baselines
    df["outcome_vs_round1_vote"] = [
        classify_change(scrd, mv, True)
        for scrd, mv in zip(df["scrd_correct"], df["majority_voting_correct"])
    ]
    df["outcome_vs_round3_vote"] = [
        classify_change(scrd, mv3, bool(avail))
        for scrd, mv3, avail in zip(df["scrd_correct"], df["round3_vote_correct"], df["round3_vote_available"])
    ]

    # Readable per-sample table
    sample_overview = df[[
        "sample_id",
        "dataset_name",
        "gold_answer",
        "stop_reason",
        "round_1_answers",
        "round1_unique_answers",
        "round1_all_agree",
        "single_agent_baseline_answer",
        "majority_voting_baseline_answer",
        "round3_vote_available",
        "round3_vote_answer",
        "scrd_final_answer",
        "single_agent_correct",
        "majority_voting_correct",
        "round3_vote_correct",
        "scrd_correct",
        "outcome_vs_round1_vote",
        "outcome_vs_round3_vote",
        "effective_rounds_used",
        "actual_rounds_executed",
        "single_agent_total_tokens",
        "majority_vote_total_tokens",
        "round3_vote_total_tokens",
        "scrd_total_tokens",
        "rollback_events",
        "repair_round_events",
        "trace_found",
    ]].copy()
    sample_overview.to_csv(out_dir / "sample_overview.csv", index=False, encoding="utf-8-sig")

    # Additional debug table kept separate on purpose, so the main sample table stays readable.
    trace_debug = df[[
        "sample_id",
        "trace_path",
        "usage_records_count",
        "final_trace_rounds",
        "execution_events_count",
        "rollback_events",
        "repair_round_events",
        "stable_consensus_events",
        "round_usage_summary_json",
        "round3_answers_json",
    ]].copy()
    trace_debug.to_csv(out_dir / "trace_debug_details.csv", index=False, encoding="utf-8-sig")

    # Round-3 subset table
    round3_subset = df[df["round3_vote_available"] == True].copy()  # noqa: E712
    round3_vote_samples = round3_subset[[
        "sample_id",
        "gold_answer",
        "stop_reason",
        "round3_vote_answer",
        "round3_vote_correct",
        "scrd_final_answer",
        "scrd_correct",
        "outcome_vs_round3_vote",
        "round3_vote_total_tokens",
        "scrd_total_tokens",
    ]].copy()
    round3_vote_samples.to_csv(out_dir / "round3_vote_samples.csv", index=False, encoding="utf-8-sig")

    # Method comparison table
    method_rows = []
    method_rows.append({
        "method": "Single Agent",
        "coverage_n": int(len(df)),
        "accuracy": float(df["single_agent_correct"].mean()),
        "mean_total_tokens": safe_mean(df["single_agent_total_tokens"]),
        "median_total_tokens": safe_median(df["single_agent_total_tokens"]),
        "accuracy_per_1k_tokens": accuracy_per_1k_tokens(df["single_agent_correct"].astype(int), df["single_agent_total_tokens"]),
        "note": "One-shot single-agent baseline",
    })
    method_rows.append({
        "method": "Round-1 Majority Vote",
        "coverage_n": int(len(df)),
        "accuracy": float(df["majority_voting_correct"].mean()),
        "mean_total_tokens": safe_mean(df["majority_vote_total_tokens"]),
        "median_total_tokens": safe_median(df["majority_vote_total_tokens"]),
        "accuracy_per_1k_tokens": accuracy_per_1k_tokens(df["majority_voting_correct"].astype(int), df["majority_vote_total_tokens"]),
        "note": "Majority vote on round-1 answers",
    })
    method_rows.append({
        "method": "Round-3 Majority Vote",
        "coverage_n": int(len(round3_subset)),
        "accuracy": float(round3_subset["round3_vote_correct"].mean()) if len(round3_subset) > 0 else pd.NA,
        "mean_total_tokens": safe_mean(round3_subset["round3_vote_total_tokens"]) if len(round3_subset) > 0 else pd.NA,
        "median_total_tokens": safe_median(round3_subset["round3_vote_total_tokens"]) if len(round3_subset) > 0 else pd.NA,
        "accuracy_per_1k_tokens": accuracy_per_1k_tokens(round3_subset["round3_vote_correct"].astype(int), round3_subset["round3_vote_total_tokens"]) if len(round3_subset) > 0 else pd.NA,
        "note": "Only samples with an available third round in trace; token cost = cumulative SCRD tokens up to round 3",
    })
    method_rows.append({
        "method": "SCRD Final",
        "coverage_n": int(len(df)),
        "accuracy": float(df["scrd_correct"].mean()),
        "mean_total_tokens": safe_mean(df["scrd_total_tokens"]),
        "median_total_tokens": safe_median(df["scrd_total_tokens"]),
        "accuracy_per_1k_tokens": accuracy_per_1k_tokens(df["scrd_correct"].astype(int), df["scrd_total_tokens"]),
        "note": "Final system output",
    })
    method_comparison = pd.DataFrame(method_rows)
    method_comparison.to_csv(out_dir / "method_comparison.csv", index=False, encoding="utf-8-sig")

    # Stop reason summary
    stop_reason_summary = (
        df.groupby("stop_reason", dropna=False)
        .agg(
            n=("sample_id", "count"),
            scrd_accuracy=("scrd_correct", "mean"),
            round1_vote_accuracy=("majority_voting_correct", "mean"),
            round3_vote_accuracy=("round3_vote_correct", "mean"),
            mean_scrd_tokens=("scrd_total_tokens", "mean"),
            median_scrd_tokens=("scrd_total_tokens", "median"),
            mean_actual_rounds=("actual_rounds_executed", "mean"),
            mean_effective_rounds=("effective_rounds_used", "mean"),
            mean_round3_vote_tokens=("round3_vote_total_tokens", "mean"),
        )
        .reset_index()
        .sort_values(["n", "stop_reason"], ascending=[False, True])
    )
    stop_reason_summary.to_csv(out_dir / "stop_reason_summary.csv", index=False, encoding="utf-8-sig")

    # Outcome summaries vs baselines
    outcome_vs_round1_vote = (
        df.groupby("outcome_vs_round1_vote")
        .agg(
            n=("sample_id", "count"),
            mean_scrd_tokens=("scrd_total_tokens", "mean"),
            mean_actual_rounds=("actual_rounds_executed", "mean"),
        )
        .reset_index()
        .sort_values("n", ascending=False)
    )
    outcome_vs_round1_vote.to_csv(out_dir / "outcome_vs_round1_vote.csv", index=False, encoding="utf-8-sig")

    outcome_vs_round3_vote = (
        df.groupby("outcome_vs_round3_vote")
        .agg(
            n=("sample_id", "count"),
            mean_scrd_tokens=("scrd_total_tokens", "mean"),
            mean_round3_vote_tokens=("round3_vote_total_tokens", "mean"),
            mean_actual_rounds=("actual_rounds_executed", "mean"),
        )
        .reset_index()
        .sort_values("n", ascending=False)
    )
    outcome_vs_round3_vote.to_csv(out_dir / "outcome_vs_round3_vote.csv", index=False, encoding="utf-8-sig")

    # Component token summary
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
                safe_mean(df["agent_total_tokens"]),
                safe_mean(df["recorder_total_tokens"]),
                safe_mean(df["evaluator_total_tokens"]),
                safe_mean(df["repair_brief_total_tokens"]),
                safe_mean(df["repair_evaluator_total_tokens"]),
                safe_mean(df["repair_agent_total_tokens"]),
            ],
            "sum_tokens": [
                df["agent_total_tokens"].fillna(0).sum(),
                df["recorder_total_tokens"].fillna(0).sum(),
                df["evaluator_total_tokens"].fillna(0).sum(),
                df["repair_brief_total_tokens"].fillna(0).sum(),
                df["repair_evaluator_total_tokens"].fillna(0).sum(),
                df["repair_agent_total_tokens"].fillna(0).sum(),
            ],
        }
    )
    component_summary.to_csv(out_dir / "component_token_summary.csv", index=False, encoding="utf-8-sig")

    # Overall summary files
    overall_summary = {
        "n_samples": int(len(df)),
        "round3_vote_available_n": int(len(round3_subset)),
        "single_agent_accuracy": float(df["single_agent_correct"].mean()),
        "round1_majority_accuracy": float(df["majority_voting_correct"].mean()),
        "round3_majority_accuracy": float(round3_subset["round3_vote_correct"].mean()) if len(round3_subset) > 0 else None,
        "scrd_accuracy": float(df["scrd_correct"].mean()),
        "mean_single_agent_tokens": safe_mean(df["single_agent_total_tokens"]),
        "mean_round1_majority_tokens": safe_mean(df["majority_vote_total_tokens"]),
        "mean_round3_majority_tokens": safe_mean(round3_subset["round3_vote_total_tokens"]) if len(round3_subset) > 0 else None,
        "mean_scrd_tokens": safe_mean(df["scrd_total_tokens"]),
    }
    with (out_dir / "overall_summary.json").open("w", encoding="utf-8") as f:
        json.dump(overall_summary, f, ensure_ascii=False, indent=2)
    pd.DataFrame([overall_summary]).to_csv(out_dir / "overall_summary.csv", index=False, encoding="utf-8-sig")

    # -------------------------
    # Plots
    # -------------------------
    acc_dict = {
        "Single\nAgent": float(df["single_agent_correct"].mean()),
        "Round-1\nVote": float(df["majority_voting_correct"].mean()),
        "SCRD\nFinal": float(df["scrd_correct"].mean()),
    }
    if len(round3_subset) > 0:
        acc_dict["Round-3\nVote"] = float(round3_subset["round3_vote_correct"].mean())
    acc_series = pd.Series(acc_dict)
    # keep a more natural order
    acc_series = acc_series.reindex([k for k in ["Single\nAgent", "Round-1\nVote", "Round-3\nVote", "SCRD\nFinal"] if k in acc_series.index])
    save_bar(acc_series, "Accuracy by Method", "Accuracy", out_dir / "fig_accuracy_by_method.png")

    token_dict = {
        "Single\nAgent": safe_mean(df["single_agent_total_tokens"]),
        "Round-1\nVote": safe_mean(df["majority_vote_total_tokens"]),
        "SCRD\nFinal": safe_mean(df["scrd_total_tokens"]),
    }
    if len(round3_subset) > 0:
        token_dict["Round-3\nVote"] = safe_mean(round3_subset["round3_vote_total_tokens"])
    token_series = pd.Series(token_dict)
    token_series = token_series.reindex([k for k in ["Single\nAgent", "Round-1\nVote", "Round-3\nVote", "SCRD\nFinal"] if k in token_series.index])
    save_bar(token_series, "Mean Total Tokens by Method", "Mean total tokens", out_dir / "fig_tokens_by_method.png")

    prompt_completion = pd.Series(
        {
            "Prompt": safe_mean(df["scrd_prompt_tokens"]),
            "Completion": safe_mean(df["scrd_completion_tokens"]),
        }
    )
    save_bar(prompt_completion, "SCRD Mean Prompt vs Completion Tokens", "Mean tokens", out_dir / "fig_scrd_prompt_vs_completion.png")

    comp_plot_series = pd.Series(
        {
            "Agent": safe_mean(df["agent_total_tokens"]),
            "Recorder": safe_mean(df["recorder_total_tokens"]),
            "Evaluator": safe_mean(df["evaluator_total_tokens"]),
            "RepairBrief": safe_mean(df["repair_brief_total_tokens"]),
            "RepairEval": safe_mean(df["repair_evaluator_total_tokens"]),
            "RepairAgent": safe_mean(df["repair_agent_total_tokens"]),
        }
    )
    save_bar(comp_plot_series, "Mean SCRD Tokens by Component", "Mean tokens", out_dir / "fig_component_token_breakdown.png", rot=20)

    stop_counts = df["stop_reason"].fillna("unknown").value_counts()
    save_bar(stop_counts, "Stop Reason Distribution", "Count", out_dir / "fig_stop_reason_distribution.png", rot=20)

    stop_token_means = df.groupby("stop_reason", dropna=False)["scrd_total_tokens"].mean().sort_values(ascending=False)
    stop_token_means.index = stop_token_means.index.fillna("unknown")
    save_bar(stop_token_means, "Mean SCRD Tokens by Stop Reason", "Mean tokens", out_dir / "fig_tokens_by_stop_reason.png", rot=20)

    save_hist(df["actual_rounds_executed"], "Actual Rounds Executed Distribution", "Actual rounds executed", out_dir / "fig_actual_rounds_hist.png", bins=10)
    save_hist(df["scrd_total_tokens"], "SCRD Total Tokens Distribution", "SCRD total tokens", out_dir / "fig_scrd_tokens_hist.png", bins=20)

    save_scatter_by_category(
        x=df["scrd_total_tokens"],
        y=df["scrd_correct"].astype(int),
        categories=df["stop_reason"],
        title="SCRD Correctness vs Total Tokens",
        xlabel="SCRD total tokens",
        ylabel="SCRD result",
        outpath=out_dir / "fig_scrd_correctness_vs_tokens.png",
    )

    change_counts = df["outcome_vs_round1_vote"].value_counts()
    save_bar(change_counts, "SCRD vs Round-1 Vote Outcome Changes", "Count", out_dir / "fig_outcome_vs_round1_vote.png", rot=20)

    if len(round3_subset) > 0:
        change_counts_r3 = df["outcome_vs_round3_vote"].value_counts()
        save_bar(change_counts_r3, "SCRD vs Round-3 Vote Outcome Changes", "Count", out_dir / "fig_outcome_vs_round3_vote.png", rot=20)

    # -------------------------
    # Markdown report
    # -------------------------
    report_path = out_dir / "analysis_report.md"
    lines: list[str] = []
    lines.append("# SCRD Experiment Analysis Report\n\n")
    lines.append(f"- Samples analyzed: **{len(df)}**\n")
    lines.append(f"- Single Agent accuracy: **{float(df['single_agent_correct'].mean()):.4f}**\n")
    lines.append(f"- Round-1 Majority Vote accuracy: **{float(df['majority_voting_correct'].mean()):.4f}**\n")
    if len(round3_subset) > 0:
        lines.append(f"- Round-3 Majority Vote accuracy (subset n={len(round3_subset)}): **{float(round3_subset['round3_vote_correct'].mean()):.4f}**\n")
    else:
        lines.append("- Round-3 Majority Vote accuracy: **N/A** (no usable third-round trace found)\n")
    lines.append(f"- SCRD Final accuracy: **{float(df['scrd_correct'].mean()):.4f}**\n\n")

    lines.append("## Cost Overview\n\n")
    lines.append(f"- Mean Single-Agent tokens: **{safe_mean(df['single_agent_total_tokens']):.2f}**\n")
    lines.append(f"- Mean Round-1 Vote tokens: **{safe_mean(df['majority_vote_total_tokens']):.2f}**\n")
    if len(round3_subset) > 0:
        lines.append(f"- Mean Round-3 Vote tokens (cumulative to round 3, subset): **{safe_mean(round3_subset['round3_vote_total_tokens']):.2f}**\n")
    lines.append(f"- Mean SCRD Final tokens: **{safe_mean(df['scrd_total_tokens']):.2f}**\n\n")

    lines.append("## Efficiency\n\n")
    lines.append(f"- Single-Agent accuracy per 1k tokens: **{accuracy_per_1k_tokens(df['single_agent_correct'].astype(int), df['single_agent_total_tokens']):.4f}**\n")
    lines.append(f"- Round-1 Vote accuracy per 1k tokens: **{accuracy_per_1k_tokens(df['majority_voting_correct'].astype(int), df['majority_vote_total_tokens']):.4f}**\n")
    if len(round3_subset) > 0:
        lines.append(f"- Round-3 Vote accuracy per 1k tokens (subset): **{accuracy_per_1k_tokens(round3_subset['round3_vote_correct'].astype(int), round3_subset['round3_vote_total_tokens']):.4f}**\n")
    lines.append(f"- SCRD Final accuracy per 1k tokens: **{accuracy_per_1k_tokens(df['scrd_correct'].astype(int), df['scrd_total_tokens']):.4f}**\n\n")

    lines.append("## Output Files\n\n")
    lines.append("- `sample_overview.csv`: readable per-sample summary\n")
    lines.append("- `method_comparison.csv`: direct method-level comparison\n")
    lines.append("- `stop_reason_summary.csv`: grouped by stop reason\n")
    lines.append("- `outcome_vs_round1_vote.csv`: how SCRD changes outcomes relative to round-1 vote\n")
    lines.append("- `outcome_vs_round3_vote.csv`: same, but relative to round-3 vote\n")
    lines.append("- `round3_vote_samples.csv`: only samples with third-round vote available\n")
    lines.append("- `trace_debug_details.csv`: trace-heavy debug fields kept separate from the readable sample table\n")

    report_path.write_text("".join(lines), encoding="utf-8")

    print(f"Analysis completed. Outputs written to: {out_dir}")


if __name__ == "__main__":
    main()
