from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path
from typing import Any


DEFAULT_DATASETS = [
    "addsub",
    "asdiv",
    "gsm8k",
    "math",
    "multiarith",
    "singleeq",
    "svamp",
]


def ensure_repo_imports(repo_root: Path) -> None:
    repo_root = repo_root.resolve()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
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
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def get_last_effective_state(trace: dict[str, Any]) -> dict[str, Any] | None:
    final_trace = trace.get("final_trace") or []
    if not final_trace:
        return None
    return final_trace[-1]


def get_last_effective_answers(trace: dict[str, Any]) -> tuple[int | None, list[str]]:
    state = get_last_effective_state(trace)
    if not state:
        return None, []

    round_id = state.get("round_id")
    answers = state.get("current_answers") or []
    cleaned = [str(a).strip() for a in answers if a is not None and str(a).strip()]
    return round_id, cleaned


def summarize_rows(dataset: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    original_correct = sum(bool(r.get("decision_head_scrd_correct")) for r in rows)
    new_correct = sum(bool(r.get("scrd_correct")) for r in rows)

    decision_head_wins = sum(
        bool(r.get("decision_head_scrd_correct")) and not bool(r.get("scrd_correct"))
        for r in rows
    )
    last_round_mv_wins = sum(
        (not bool(r.get("decision_head_scrd_correct"))) and bool(r.get("scrd_correct"))
        for r in rows
    )
    both_correct = sum(
        bool(r.get("decision_head_scrd_correct")) and bool(r.get("scrd_correct"))
        for r in rows
    )
    both_wrong = sum(
        (not bool(r.get("decision_head_scrd_correct"))) and (not bool(r.get("scrd_correct")))
        for r in rows
    )
    changed = sum(bool(r.get("last_round_majority_changed_answer")) for r in rows)

    return {
        "dataset": dataset,
        "n": n,
        "decision_head_correct": original_correct,
        "decision_head_acc": original_correct / n if n else None,
        "last_round_majority_correct": new_correct,
        "last_round_majority_acc": new_correct / n if n else None,
        "delta_last_round_mv_minus_decision_head_pp": (
            100.0 * ((new_correct / n) - (original_correct / n)) if n else None
        ),
        "both_correct": both_correct,
        "both_wrong": both_wrong,
        "decision_head_wins": decision_head_wins,
        "last_round_mv_wins": last_round_mv_wins,
        "net_last_round_mv_minus_decision_head": last_round_mv_wins - decision_head_wins,
        "answer_changed_count": changed,
        "answer_changed_pct": changed / n if n else None,
    }


def add_overall(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    n = sum(int(r["n"]) for r in summaries)

    def s(key: str) -> int:
        return sum(int(r.get(key) or 0) for r in summaries)

    original_correct = s("decision_head_correct")
    new_correct = s("last_round_majority_correct")
    changed = s("answer_changed_count")

    return {
        "dataset": "OVERALL",
        "n": n,
        "decision_head_correct": original_correct,
        "decision_head_acc": original_correct / n if n else None,
        "last_round_majority_correct": new_correct,
        "last_round_majority_acc": new_correct / n if n else None,
        "delta_last_round_mv_minus_decision_head_pp": (
            100.0 * ((new_correct / n) - (original_correct / n)) if n else None
        ),
        "both_correct": s("both_correct"),
        "both_wrong": s("both_wrong"),
        "decision_head_wins": s("decision_head_wins"),
        "last_round_mv_wins": s("last_round_mv_wins"),
        "net_last_round_mv_minus_decision_head": s("net_last_round_mv_minus_decision_head"),
        "answer_changed_count": changed,
        "answer_changed_pct": changed / n if n else None,
    }


def fmt_pct(x: Any) -> str:
    if x is None:
        return "NA"
    return f"{100.0 * float(x):.2f}%"


def write_report(path: Path, summaries: list[dict[str, Any]]) -> None:
    lines: list[str] = []
    lines.append("# Outputs Recomputed with Last-Round Majority Vote")
    lines.append("")
    lines.append(
        "This report rewrites the Full SCRD outputs by replacing the original "
        "trajectory decision head with majority vote over the last effective "
        "state's `current_answers`. No LLM calls are made."
    )
    lines.append("")
    lines.append("| Dataset | N | Decision Head Acc | Last-Round MV Acc | Δ Acc | DH wins | MV wins | Net MV-DH | Changed |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in summaries:
        lines.append(
            "| {dataset} | {n} | {dh_acc} | {mv_acc} | {delta:.2f} pp | {dh_wins} | {mv_wins} | {net} | {changed} |".format(
                dataset=r["dataset"],
                n=r["n"],
                dh_acc=fmt_pct(r["decision_head_acc"]),
                mv_acc=fmt_pct(r["last_round_majority_acc"]),
                delta=float(r["delta_last_round_mv_minus_decision_head_pp"] or 0.0),
                dh_wins=r["decision_head_wins"],
                mv_wins=r["last_round_mv_wins"],
                net=r["net_last_round_mv_minus_decision_head"],
                changed=r["answer_changed_count"],
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def process_dataset(
    *,
    dataset: str,
    input_root: Path,
    output_root: Path,
    copy_traces: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    from src.utils.result_utils import is_correct, majority_vote

    in_dir = input_root / dataset
    out_dir = output_root / dataset
    results_path = in_dir / "results.jsonl"
    traces_dir = in_dir / "traces"

    if not results_path.exists():
        raise FileNotFoundError(f"Missing {results_path}")
    if not traces_dir.exists():
        raise FileNotFoundError(f"Missing {traces_dir}")

    original_rows = load_jsonl(results_path)
    rewritten_rows: list[dict[str, Any]] = []
    missing_traces: list[str] = []

    for row in original_rows:
        sample_id = str(row.get("sample_id", "")).strip()
        trace_path = traces_dir / f"{sample_id}_trace.json"

        if not trace_path.exists():
            missing_traces.append(sample_id)
            continue

        trace = load_json(trace_path)
        last_round_id, last_answers = get_last_effective_answers(trace)
        last_round_answer = majority_vote(last_answers, dataset_name=dataset) if last_answers else ""

        gold_answer = str(row.get("gold_answer", ""))
        last_round_correct = is_correct(last_round_answer, gold_answer, dataset)

        decision_head_answer = row.get("scrd_final_answer")
        decision_head_correct = bool(row.get("scrd_correct"))

        new_row = dict(row)

        # Preserve original Full SCRD decision-head result.
        new_row["decision_head_scrd_final_answer"] = decision_head_answer
        new_row["decision_head_scrd_correct"] = decision_head_correct

        # Rewrite SCRD result to the new finalizer so existing analysis scripts
        # can be reused against outputs_with_last_round_majority_vote/.
        new_row["scrd_final_answer"] = last_round_answer
        new_row["scrd_correct"] = last_round_correct

        # Extra diagnostics.
        new_row["finalizer"] = "last_effective_round_majority_vote"
        new_row["last_effective_round_id"] = last_round_id
        new_row["last_effective_round_answers"] = last_answers
        new_row["last_round_majority_answer"] = last_round_answer
        new_row["last_round_majority_correct"] = last_round_correct
        new_row["last_round_majority_changed_answer"] = str(last_round_answer) != str(decision_head_answer)

        rewritten_rows.append(new_row)

    write_jsonl(out_dir / "results.jsonl", rewritten_rows)

    if missing_traces:
        write_json(out_dir / "missing_traces.json", missing_traces)

    if copy_traces:
        dst_traces = out_dir / "traces"
        if dst_traces.exists():
            shutil.rmtree(dst_traces)
        shutil.copytree(traces_dir, dst_traces)

    summary = summarize_rows(dataset, rewritten_rows)
    summary["missing_trace_count"] = len(missing_traces)
    write_json(out_dir / "summary.json", summary)

    return summary, rewritten_rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Rewrite existing Full SCRD outputs by replacing the final decision "
            "head with last effective round majority vote. No LLM calls."
        )
    )
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--input-root", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs_with_last_round_majority_vote"),
    )
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--copy-traces", action="store_true")
    args = parser.parse_args()

    ensure_repo_imports(args.repo_root)
    args.output_root.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []

    for dataset in args.datasets:
        print(f"[last-round MV] processing {dataset}...")
        summary, rows = process_dataset(
            dataset=dataset,
            input_root=args.input_root,
            output_root=args.output_root,
            copy_traces=args.copy_traces,
        )
        summaries.append(summary)
        all_rows.extend(rows)
        print(
            "  decision_head={:.2f}% | last_round_mv={:.2f}% | delta={:+.2f} pp | net={}".format(
                100.0 * float(summary["decision_head_acc"] or 0.0),
                100.0 * float(summary["last_round_majority_acc"] or 0.0),
                float(summary["delta_last_round_mv_minus_decision_head_pp"] or 0.0),
                summary["net_last_round_mv_minus_decision_head"],
            )
        )

    overall = add_overall(summaries)
    summaries_with_overall = summaries + [overall]

    write_csv(args.output_root / "summary.csv", summaries_with_overall)
    write_json(args.output_root / "summary.json", summaries_with_overall)
    write_jsonl(args.output_root / "sample_level_results.jsonl", all_rows)
    write_report(args.output_root / "report.md", summaries_with_overall)

    print("")
    print(f"Saved rewritten outputs to: {args.output_root}")
    print(
        "OVERALL decision_head={:.2f}% | last_round_mv={:.2f}% | delta={:+.2f} pp | net={}".format(
            100.0 * float(overall["decision_head_acc"] or 0.0),
            100.0 * float(overall["last_round_majority_acc"] or 0.0),
            float(overall["delta_last_round_mv_minus_decision_head_pp"] or 0.0),
            overall["net_last_round_mv_minus_decision_head"],
        )
    )


if __name__ == "__main__":
    main()
