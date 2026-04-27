from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any


DEFAULT_DATASETS = [
    "gsm8k",
    "math",
    "multiarith",
]


def ensure_repo_imports(repo_root: Path) -> None:
    repo_root = repo_root.resolve()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        raise FileNotFoundError(f"Missing jsonl file: {path}")
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing trace file: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def get_last_effective_answers(trace: dict[str, Any]) -> tuple[int | None, list[str]]:
    """
    Use the same effective trajectory that Full SCRD decision head sees.

    Expected trace format:
      {
        "final_trace": [
          {"round_id": 1, "current_answers": [...]},
          ...
        ]
      }

    For rollback cases, final_trace should already reflect the final effective
    trajectory after rollback cleanup / repair.
    """
    final_trace = trace.get("final_trace") or []
    if not final_trace:
        return None, []

    last_state = final_trace[-1]
    round_id = last_state.get("round_id")
    answers = last_state.get("current_answers") or []
    cleaned = [str(a) for a in answers if a is not None and str(a).strip()]
    return round_id, cleaned


def summarize_dataset(
    *,
    dataset_name: str,
    ablation_rows: list[dict[str, Any]],
    missing_traces: list[str],
) -> dict[str, Any]:
    n = len(ablation_rows)

    full_correct = sum(1 for r in ablation_rows if bool(r.get("full_scrd_correct")))
    ablation_correct = sum(1 for r in ablation_rows if bool(r.get("wo_decision_head_correct")))

    both_correct = sum(
        1
        for r in ablation_rows
        if bool(r.get("full_scrd_correct")) and bool(r.get("wo_decision_head_correct"))
    )
    both_wrong = sum(
        1
        for r in ablation_rows
        if (not bool(r.get("full_scrd_correct")))
        and (not bool(r.get("wo_decision_head_correct")))
    )
    full_wins = sum(
        1
        for r in ablation_rows
        if bool(r.get("full_scrd_correct"))
        and not bool(r.get("wo_decision_head_correct"))
    )
    ablation_wins = sum(
        1
        for r in ablation_rows
        if (not bool(r.get("full_scrd_correct")))
        and bool(r.get("wo_decision_head_correct"))
    )
    answer_changed = sum(1 for r in ablation_rows if bool(r.get("answer_changed")))

    avg_tokens = (
        sum(float(r.get("scrd_total_tokens") or 0) for r in ablation_rows) / n
        if n
        else 0.0
    )

    return {
        "dataset": dataset_name,
        "n": n,
        "missing_trace_count": len(missing_traces),
        "full_scrd_correct": full_correct,
        "full_scrd_acc": full_correct / n if n else None,
        "wo_decision_head_correct": ablation_correct,
        "wo_decision_head_acc": ablation_correct / n if n else None,
        "delta_acc_pp": 100.0 * ((ablation_correct / n) - (full_correct / n)) if n else None,
        "both_correct": both_correct,
        "both_wrong": both_wrong,
        "full_wins": full_wins,
        "ablation_wins": ablation_wins,
        "net_full_minus_ablation": full_wins - ablation_wins,
        "answer_changed_count": answer_changed,
        "answer_changed_pct": answer_changed / n if n else None,
        "avg_tokens_same_as_full_scrd": avg_tokens,
    }


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return

    fieldnames = [
        "dataset",
        "n",
        "missing_trace_count",
        "full_scrd_correct",
        "full_scrd_acc",
        "wo_decision_head_correct",
        "wo_decision_head_acc",
        "delta_acc_pp",
        "both_correct",
        "both_wrong",
        "full_wins",
        "ablation_wins",
        "net_full_minus_ablation",
        "answer_changed_count",
        "answer_changed_pct",
        "avg_tokens_same_as_full_scrd",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def add_overall_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = sum(int(r["n"]) for r in rows)

    def s(key: str) -> int:
        return sum(int(r.get(key) or 0) for r in rows)

    full_correct = s("full_scrd_correct")
    ablation_correct = s("wo_decision_head_correct")
    answer_changed = s("answer_changed_count")
    avg_tokens = (
        sum(float(r.get("avg_tokens_same_as_full_scrd") or 0) * int(r["n"]) for r in rows) / n
        if n
        else 0.0
    )

    return {
        "dataset": "OVERALL",
        "n": n,
        "missing_trace_count": s("missing_trace_count"),
        "full_scrd_correct": full_correct,
        "full_scrd_acc": full_correct / n if n else None,
        "wo_decision_head_correct": ablation_correct,
        "wo_decision_head_acc": ablation_correct / n if n else None,
        "delta_acc_pp": 100.0 * ((ablation_correct / n) - (full_correct / n)) if n else None,
        "both_correct": s("both_correct"),
        "both_wrong": s("both_wrong"),
        "full_wins": s("full_wins"),
        "ablation_wins": s("ablation_wins"),
        "net_full_minus_ablation": s("net_full_minus_ablation"),
        "answer_changed_count": answer_changed,
        "answer_changed_pct": answer_changed / n if n else None,
        "avg_tokens_same_as_full_scrd": avg_tokens,
    }


def format_pct(x: Any) -> str:
    if x is None:
        return "NA"
    return f"{100.0 * float(x):.2f}%"


def write_markdown_report(path: Path, rows_with_overall: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    lines.append("# Ablation: SCRD w/o Decision Head")
    lines.append("")
    lines.append(
        "Definition: reuse the same Full SCRD trajectories, but replace the "
        "trajectory-aware decision head with majority vote over the last effective "
        "state's `current_answers`."
    )
    lines.append("")
    lines.append(
        "This is a post-hoc ablation. It does not call the LLM again; token cost is identical to Full SCRD."
    )
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(
        "| Dataset | N | Full SCRD Acc | w/o Decision Head Acc | Δ Acc | Full wins | Ablation wins | Net Full-Ablation | Answer changed |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")

    for r in rows_with_overall:
        lines.append(
            "| {dataset} | {n} | {full_acc} | {abl_acc} | {delta:.2f} pp | {full_wins} | {ablation_wins} | {net} | {changed} |".format(
                dataset=r["dataset"],
                n=r["n"],
                full_acc=format_pct(r["full_scrd_acc"]),
                abl_acc=format_pct(r["wo_decision_head_acc"]),
                delta=float(r["delta_acc_pp"] or 0.0),
                full_wins=r["full_wins"],
                ablation_wins=r["ablation_wins"],
                net=r["net_full_minus_ablation"],
                changed=r["answer_changed_count"],
            )
        )

    lines.append("")
    lines.append("## Interpretation guide")
    lines.append("")
    lines.append("- `Full wins`: Full SCRD correct, w/o Decision Head wrong.")
    lines.append("- `Ablation wins`: Full SCRD wrong, w/o Decision Head correct.")
    lines.append("- `Net Full-Ablation`: positive means the decision head helps on net.")
    lines.append("- `Answer changed`: final answer differs between Full SCRD and last-round majority.")

    path.write_text("\n".join(lines), encoding="utf-8")


def run_dataset(
    *,
    dataset_name: str,
    input_root: Path,
    output_root: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[str]]:
    from src.utils.result_utils import is_correct, majority_vote

    dataset_dir = input_root / dataset_name
    results_path = dataset_dir / "results.jsonl"
    trace_dir = dataset_dir / "traces"

    full_rows = load_jsonl(results_path)
    ablation_rows: list[dict[str, Any]] = []
    missing_traces: list[str] = []

    for row in full_rows:
        sample_id = str(row.get("sample_id", "")).strip()
        if not sample_id:
            continue

        trace_path = trace_dir / f"{sample_id}_trace.json"
        if not trace_path.exists():
            missing_traces.append(sample_id)
            continue

        trace = load_json(trace_path)
        last_round_id, last_answers = get_last_effective_answers(trace)

        ablation_answer = majority_vote(last_answers, dataset_name=dataset_name) if last_answers else ""
        gold_answer = str(row.get("gold_answer", ""))
        ablation_correct = is_correct(ablation_answer, gold_answer, dataset_name)
        full_answer = str(row.get("scrd_final_answer", ""))
        full_correct = bool(row.get("scrd_correct"))

        out = {
            "sample_id": sample_id,
            "dataset": dataset_name,
            "gold_answer": gold_answer,
            "full_scrd_answer": full_answer,
            "full_scrd_correct": full_correct,
            "wo_decision_head_answer": ablation_answer,
            "wo_decision_head_correct": ablation_correct,
            "answer_changed": ablation_answer != full_answer,
            "last_effective_round_id": last_round_id,
            "last_effective_answers": last_answers,
            "scrd_total_tokens": row.get("scrd_total_tokens"),
            "wo_decision_head_total_tokens": row.get("scrd_total_tokens"),
            "stop_reason": row.get("stop_reason"),
            "effective_rounds_used": row.get("effective_rounds_used"),
            "actual_rounds_executed": row.get("actual_rounds_executed"),
            "single_agent_correct": row.get("single_agent_correct"),
            "majority_voting_correct": row.get("majority_voting_correct"),
        }
        ablation_rows.append(out)

    dataset_out_dir = output_root / dataset_name
    write_jsonl(dataset_out_dir / "results.jsonl", ablation_rows)

    summary = summarize_dataset(
        dataset_name=dataset_name,
        ablation_rows=ablation_rows,
        missing_traces=missing_traces,
    )
    write_json(dataset_out_dir / "summary.json", summary)

    if missing_traces:
        write_json(dataset_out_dir / "missing_traces.json", missing_traces)

    return summary, ablation_rows, missing_traces


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Post-hoc ablation for SCRD w/o Decision Head. "
            "Reuses existing outputs/{dataset}/traces and results.jsonl."
        )
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DEFAULT_DATASETS,
        help="Datasets to process. Default: all 7 current datasets.",
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("outputs"),
        help="Root directory containing Full SCRD outputs.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs") / "ablation" / "wo_decision_head",
        help="Where to write ablation outputs.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path("."),
        help="Repository root for importing src.utils.result_utils.",
    )
    args = parser.parse_args()

    ensure_repo_imports(args.repo_root)
    args.output_root.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []

    for dataset_name in args.datasets:
        print(f"[w/o decision head] processing {dataset_name}...")
        summary, rows, missing = run_dataset(
            dataset_name=dataset_name,
            input_root=args.input_root,
            output_root=args.output_root,
        )
        summaries.append(summary)
        all_rows.extend(rows)

        if missing:
            print(f"  warning: {len(missing)} missing traces for {dataset_name}")

        print(
            "  full={:.2f}% | wo_decision_head={:.2f}% | delta={:+.2f} pp | net={}".format(
                100.0 * float(summary["full_scrd_acc"] or 0.0),
                100.0 * float(summary["wo_decision_head_acc"] or 0.0),
                float(summary["delta_acc_pp"] or 0.0),
                summary["net_full_minus_ablation"],
            )
        )

    overall = add_overall_summary(summaries)
    summaries_with_overall = summaries + [overall]

    write_summary_csv(args.output_root / "summary.csv", summaries_with_overall)
    write_json(args.output_root / "summary.json", summaries_with_overall)
    write_jsonl(args.output_root / "sample_level_results.jsonl", all_rows)
    write_markdown_report(args.output_root / "report.md", summaries_with_overall)

    print("")
    print(f"Saved ablation outputs to: {args.output_root}")
    print(
        "OVERALL full={:.2f}% | wo_decision_head={:.2f}% | delta={:+.2f} pp | net={}".format(
            100.0 * float(overall["full_scrd_acc"] or 0.0),
            100.0 * float(overall["wo_decision_head_acc"] or 0.0),
            float(overall["delta_acc_pp"] or 0.0),
            overall["net_full_minus_ablation"],
        )
    )


if __name__ == "__main__":
    main()
