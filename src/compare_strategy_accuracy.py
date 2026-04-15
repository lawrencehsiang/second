#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


def extract_last_number(text: Any) -> str:
    """Match the repository's current normalization logic."""
    if text is None:
        return ""
    matches = re.findall(r"-?\d+(?:\.\d+)?", str(text))
    return matches[-1] if matches else str(text).strip()


normalize_answer = extract_last_number


def is_correct(pred: Any, gold: Any) -> bool:
    return normalize_answer(pred) == normalize_answer(gold)



def majority_vote(answers: list[Any]) -> str:
    if not answers:
        return ""
    normalized_answers = [normalize_answer(a) for a in answers]
    counter = Counter(normalized_answers)
    return counter.most_common(1)[0][0]



def safe_get(items: list[Any], idx: int) -> str:
    return str(items[idx]) if idx < len(items) and items[idx] is not None else ""



def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSONL at {path}:{lineno}: {e}") from e
            if not isinstance(obj, dict):
                raise ValueError(f"Expected a JSON object at {path}:{lineno}, got {type(obj).__name__}")
            rows.append(obj)
    return rows



def load_trace_round_answers(trace_path: Path, target_round: int) -> list[str]:
    if not trace_path.exists():
        return []
    with trace_path.open("r", encoding="utf-8") as f:
        trace = json.load(f)
    if not isinstance(trace, list):
        return []
    for state in trace:
        if not isinstance(state, dict):
            continue
        if state.get("round_id") == target_round:
            answers = state.get("current_answers", [])
            return [str(x) for x in answers] if isinstance(answers, list) else []
    return []



def load_last_trace_answers(trace_path: Path) -> list[str]:
    if not trace_path.exists():
        return []
    with trace_path.open("r", encoding="utf-8") as f:
        trace = json.load(f)
    if not isinstance(trace, list) or not trace:
        return []
    last = trace[-1]
    if not isinstance(last, dict):
        return []
    answers = last.get("current_answers", [])
    return [str(x) for x in answers] if isinstance(answers, list) else []



def make_strategy_table() -> dict[str, dict[str, Any]]:
    return {
        "round1_agent1": {"label": "Round 1 Agent 1", "correct": 0, "valid": 0},
        "round1_agent2": {"label": "Round 1 Agent 2", "correct": 0, "valid": 0},
        "round1_agent3": {"label": "Round 1 Agent 3", "correct": 0, "valid": 0},
        "round1_majority": {"label": "Round 1 Majority Voting", "correct": 0, "valid": 0},
        "round3_majority": {"label": "Round 3 Majority Voting", "correct": 0, "valid": 0},
        "scrd_final": {"label": "SCRD Final Output", "correct": 0, "valid": 0},
    }



def update_scoreboard(scoreboard: dict[str, dict[str, Any]], key: str, pred: str, gold: str) -> bool:
    if pred == "":
        return False
    scoreboard[key]["valid"] += 1
    correct = is_correct(pred, gold)
    scoreboard[key]["correct"] += int(correct)
    return correct



def write_details_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)



def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare the accuracy of 6 answer-selection strategies from SCRD outputs."
    )
    parser.add_argument("--output-dir", default="outputs", help="Directory containing results.jsonl and traces/")
    parser.add_argument("--results-file", default=None, help="Optional explicit path to results.jsonl")
    parser.add_argument("--trace-dir", default=None, help="Optional explicit path to traces directory")
    parser.add_argument("--summary-out", default=None, help="Optional path for summary JSON output")
    parser.add_argument("--details-out", default=None, help="Optional path for per-sample CSV output")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    results_file = Path(args.results_file) if args.results_file else output_dir / "results.jsonl"
    trace_dir = Path(args.trace_dir) if args.trace_dir else output_dir / "traces"
    summary_out = Path(args.summary_out) if args.summary_out else output_dir / "compare_6_strategies_summary.json"
    details_out = Path(args.details_out) if args.details_out else output_dir / "compare_6_strategies_details.csv"

    if not results_file.exists():
        raise FileNotFoundError(f"results.jsonl not found: {results_file}")
    if not trace_dir.exists():
        raise FileNotFoundError(f"trace directory not found: {trace_dir}")

    results = load_jsonl(results_file)
    total_samples = len(results)
    scoreboard = make_strategy_table()
    detail_rows: list[dict[str, Any]] = []
    missing_trace_count = 0
    missing_round3_count = 0

    for row in results:
        sample_id = str(row.get("sample_id", ""))
        gold_answer = str(row.get("gold_answer", ""))
        round1_answers = row.get("round_1_answers", [])
        if not isinstance(round1_answers, list):
            round1_answers = []
        round1_answers = [str(x) for x in round1_answers]

        trace_path = trace_dir / f"{sample_id}_trace.json"
        if not trace_path.exists():
            missing_trace_count += 1

        round3_answers = load_trace_round_answers(trace_path, target_round=3)
        if not round3_answers:
            missing_round3_count += 1

        last_trace_answers = load_last_trace_answers(trace_path)

        predictions = {
            "round1_agent1": safe_get(round1_answers, 0),
            "round1_agent2": safe_get(round1_answers, 1),
            "round1_agent3": safe_get(round1_answers, 2),
            "round1_majority": str(row.get("majority_voting_baseline_answer") or majority_vote(round1_answers)),
            "round3_majority": majority_vote(round3_answers),
            "scrd_final": str(row.get("scrd_final_answer") or majority_vote(last_trace_answers)),
        }

        correctness = {
            key: update_scoreboard(scoreboard, key, pred, gold_answer)
            for key, pred in predictions.items()
        }

        detail_rows.append(
            {
                "sample_id": sample_id,
                "gold_answer": gold_answer,
                "round1_agent1_pred": predictions["round1_agent1"],
                "round1_agent1_correct": correctness["round1_agent1"],
                "round1_agent2_pred": predictions["round1_agent2"],
                "round1_agent2_correct": correctness["round1_agent2"],
                "round1_agent3_pred": predictions["round1_agent3"],
                "round1_agent3_correct": correctness["round1_agent3"],
                "round1_majority_pred": predictions["round1_majority"],
                "round1_majority_correct": correctness["round1_majority"],
                "round3_majority_pred": predictions["round3_majority"],
                "round3_majority_correct": correctness["round3_majority"],
                "scrd_final_pred": predictions["scrd_final"],
                "scrd_final_correct": correctness["scrd_final"],
                "trace_exists": trace_path.exists(),
                "has_round3": bool(round3_answers),
            }
        )

    summary: dict[str, Any] = {
        "total_samples": total_samples,
        "results_file": str(results_file),
        "trace_dir": str(trace_dir),
        "missing_trace_count": missing_trace_count,
        "missing_round3_count": missing_round3_count,
        "strategies": {},
    }

    for key, stats in scoreboard.items():
        valid = stats["valid"]
        correct = stats["correct"]
        summary["strategies"][key] = {
            "label": stats["label"],
            "correct": correct,
            "valid_samples": valid,
            "coverage": (valid / total_samples) if total_samples else 0.0,
            "accuracy": (correct / valid) if valid else None,
        }

    summary_out.parent.mkdir(parents=True, exist_ok=True)
    details_out.parent.mkdir(parents=True, exist_ok=True)

    with summary_out.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    write_details_csv(details_out, detail_rows)

    print("=" * 72)
    print(f"Total samples: {total_samples}")
    print(f"results.jsonl: {results_file}")
    print(f"traces dir   : {trace_dir}")
    print(f"Missing trace files : {missing_trace_count}")
    print(f"Missing round-3 data: {missing_round3_count}")
    print("=" * 72)
    print(f"{'Strategy':30} {'Correct':>10} {'Valid':>10} {'Coverage':>10} {'Accuracy':>10}")
    print("-" * 72)
    for key in [
        "round1_agent1",
        "round1_agent2",
        "round1_agent3",
        "round1_majority",
        "round3_majority",
        "scrd_final",
    ]:
        item = summary["strategies"][key]
        acc = "N/A" if item["accuracy"] is None else f"{item['accuracy']:.4f}"
        print(
            f"{item['label'][:30]:30} "
            f"{item['correct']:>10} "
            f"{item['valid_samples']:>10} "
            f"{item['coverage']:>10.4f} "
            f"{acc:>10}"
        )
    print("-" * 72)
    print(f"Summary JSON saved to: {summary_out}")
    print(f"Details CSV saved to : {details_out}")


if __name__ == "__main__":
    main()
