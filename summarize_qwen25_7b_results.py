from __future__ import annotations

import json
from pathlib import Path
from collections import Counter, defaultdict

import pandas as pd
import os

DATASETS = [
    "addsub",
    "asdiv",
    "gsm8k",
    "math",
    "multiarith",
    "singleeq",
    "svamp",
]
os.environ["HTTP_PROXY"] = ""
os.environ["HTTPS_PROXY"] = ""
os.environ["NO_PROXY"] = "dashscope.aliyuncs.com,localhost,127.0.0.1"


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def pct(x: float) -> float:
    return round(100.0 * x, 2)


def classify(scrd_correct: bool, baseline_correct: bool) -> str:
    if scrd_correct and not baseline_correct:
        return "wrong_to_correct"
    if (not scrd_correct) and baseline_correct:
        return "correct_to_wrong"
    if scrd_correct and baseline_correct:
        return "both_correct"
    return "both_wrong"


def safe_mean(series: pd.Series) -> float:
    series = pd.to_numeric(series, errors="coerce").dropna()
    return float(series.mean()) if len(series) else 0.0


def main() -> None:
    root = Path("outputs")
    out_dir = root / "qwen25_7b_summary"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    all_vanilla_rows = []

    for ds in DATASETS:
        scrd_path = root / ds / "results.jsonl"
        vanilla_path = root / f"{ds}_vanilla_mad7" / "results.jsonl"

        if not scrd_path.exists():
            print(f"[WARN] missing SCRD results: {scrd_path}")
            continue

        scrd_rows = load_jsonl(scrd_path)
        scrd_df = pd.DataFrame(scrd_rows)

        if vanilla_path.exists():
            vanilla_rows = load_jsonl(vanilla_path)
            vanilla_df = pd.DataFrame(vanilla_rows)
            keep_cols = [
                "sample_id",
                "round5_majority_answer",
                "round5_majority_correct",
                "round5_cumulative_total_tokens",
                "round5_cumulative_prompt_tokens",
                "round5_cumulative_completion_tokens",
                "round7_majority_correct",
                "round7_cumulative_total_tokens",
            ]
            vanilla_df = vanilla_df[[c for c in keep_cols if c in vanilla_df.columns]]
        else:
            print(f"[WARN] missing vanilla results: {vanilla_path}")
            vanilla_df = pd.DataFrame({"sample_id": scrd_df["sample_id"]})

        df = scrd_df.merge(vanilla_df, on="sample_id", how="left")
        df["dataset"] = ds

        for c in [
            "single_agent_correct",
            "majority_voting_correct",
            "scrd_correct",
            "round5_majority_correct",
        ]:
            if c not in df.columns:
                df[c] = pd.NA

        df["outcome_vs_round1"] = df.apply(
            lambda r: classify(bool(r["scrd_correct"]), bool(r["majority_voting_correct"])),
            axis=1,
        )

        df["outcome_vs_vanilla5"] = df.apply(
            lambda r: (
                "vanilla_missing"
                if pd.isna(r["round5_majority_correct"])
                else classify(bool(r["scrd_correct"]), bool(r["round5_majority_correct"]))
            ),
            axis=1,
        )

        all_rows.append(df)

    full = pd.concat(all_rows, ignore_index=True)

    summary_rows = []
    for ds, g in full.groupby("dataset", sort=False):
        n = len(g)

        summary_rows.append({
            "dataset": ds,
            "n": n,

            "single_correct": int(g["single_agent_correct"].sum()),
            "single_acc_pct": pct(g["single_agent_correct"].mean()),

            "round1_vote_correct": int(g["majority_voting_correct"].sum()),
            "round1_vote_acc_pct": pct(g["majority_voting_correct"].mean()),

            "vanilla5_correct": int(g["round5_majority_correct"].fillna(False).sum()),
            "vanilla5_acc_pct": pct(g["round5_majority_correct"].dropna().mean())
                if g["round5_majority_correct"].notna().any()
                else None,

            "scrd_correct": int(g["scrd_correct"].sum()),
            "scrd_acc_pct": pct(g["scrd_correct"].mean()),

            "scrd_minus_single_acc_pp": round(
                pct(g["scrd_correct"].mean()) - pct(g["single_agent_correct"].mean()), 2
            ),
            "scrd_minus_round1_acc_pp": round(
                pct(g["scrd_correct"].mean()) - pct(g["majority_voting_correct"].mean()), 2
            ),
            "scrd_minus_vanilla5_acc_pp": (
                round(
                    pct(g["scrd_correct"].mean())
                    - pct(g["round5_majority_correct"].dropna().mean()),
                    2,
                )
                if g["round5_majority_correct"].notna().any()
                else None
            ),

            "avg_single_tokens": round(safe_mean(g["single_agent_total_tokens"]), 1),
            "avg_round1_vote_tokens": round(safe_mean(g["majority_vote_total_tokens"]), 1),
            "avg_vanilla5_tokens": round(safe_mean(g["round5_cumulative_total_tokens"]), 1),
            "avg_scrd_tokens": round(safe_mean(g["scrd_total_tokens"]), 1),

            "avg_scrd_effective_rounds": round(safe_mean(g["effective_rounds_used"]), 2),
            "avg_scrd_actual_rounds": round(safe_mean(g["actual_rounds_executed"]), 2),

            "rollback_count": int((g["stop_reason"] == "rollback").sum()),
            "rollback_pct": pct((g["stop_reason"] == "rollback").mean()),
            "early_stop_count": int((g["stop_reason"] == "early_stop").sum()),
            "early_stop_pct": pct((g["stop_reason"] == "early_stop").mean()),
        })

    method_summary = pd.DataFrame(summary_rows)
    method_summary.to_csv(out_dir / "dataset_method_summary.csv", index=False)

    # Overall weighted summary.
    overall = {
        "dataset": "OVERALL_WEIGHTED",
        "n": len(full),
        "single_correct": int(full["single_agent_correct"].sum()),
        "single_acc_pct": pct(full["single_agent_correct"].mean()),
        "round1_vote_correct": int(full["majority_voting_correct"].sum()),
        "round1_vote_acc_pct": pct(full["majority_voting_correct"].mean()),
        "vanilla5_correct": int(full["round5_majority_correct"].fillna(False).sum()),
        "vanilla5_acc_pct": pct(full["round5_majority_correct"].dropna().mean()),
        "scrd_correct": int(full["scrd_correct"].sum()),
        "scrd_acc_pct": pct(full["scrd_correct"].mean()),
        "avg_single_tokens": round(safe_mean(full["single_agent_total_tokens"]), 1),
        "avg_round1_vote_tokens": round(safe_mean(full["majority_vote_total_tokens"]), 1),
        "avg_vanilla5_tokens": round(safe_mean(full["round5_cumulative_total_tokens"]), 1),
        "avg_scrd_tokens": round(safe_mean(full["scrd_total_tokens"]), 1),
        "avg_scrd_effective_rounds": round(safe_mean(full["effective_rounds_used"]), 2),
        "avg_scrd_actual_rounds": round(safe_mean(full["actual_rounds_executed"]), 2),
        "rollback_count": int((full["stop_reason"] == "rollback").sum()),
        "rollback_pct": pct((full["stop_reason"] == "rollback").mean()),
        "early_stop_count": int((full["stop_reason"] == "early_stop").sum()),
        "early_stop_pct": pct((full["stop_reason"] == "early_stop").mean()),
    }

    method_summary_with_overall = pd.concat(
        [method_summary, pd.DataFrame([overall])],
        ignore_index=True,
    )
    method_summary_with_overall.to_csv(
        out_dir / "dataset_method_summary_with_overall.csv",
        index=False,
    )

    # Outcome transition tables.
    outcome_vs_round1 = (
        full.groupby(["dataset", "outcome_vs_round1"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    outcome_vs_round1.to_csv(out_dir / "outcome_vs_round1.csv", index=False)

    outcome_vs_vanilla5 = (
        full.groupby(["dataset", "outcome_vs_vanilla5"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    outcome_vs_vanilla5.to_csv(out_dir / "outcome_vs_vanilla5.csv", index=False)

    stop_reason_summary = (
        full.groupby(["dataset", "stop_reason"])
        .agg(
            n=("sample_id", "count"),
            scrd_acc=("scrd_correct", "mean"),
            avg_scrd_tokens=("scrd_total_tokens", "mean"),
            avg_effective_rounds=("effective_rounds_used", "mean"),
            avg_actual_rounds=("actual_rounds_executed", "mean"),
        )
        .reset_index()
    )
    stop_reason_summary["scrd_acc_pct"] = stop_reason_summary["scrd_acc"].apply(pct)
    stop_reason_summary.to_csv(out_dir / "stop_reason_summary.csv", index=False)

    component_cols = [
        "agent_total_tokens",
        "recorder_total_tokens",
        "evaluator_total_tokens",
        "repair_brief_total_tokens",
        "repair_evaluator_total_tokens",
        "repair_agent_total_tokens",
        "scrd_total_tokens",
    ]
    component_summary = (
        full.groupby("dataset")[component_cols]
        .mean(numeric_only=True)
        .round(1)
        .reset_index()
    )
    component_summary.to_csv(out_dir / "component_token_summary.csv", index=False)

    # Per-sample debug table.
    sample_cols = [
        "sample_id",
        "dataset",
        "gold_answer",
        "single_agent_baseline_answer",
        "majority_voting_baseline_answer",
        "round5_majority_answer",
        "scrd_final_answer",
        "single_agent_correct",
        "majority_voting_correct",
        "round5_majority_correct",
        "scrd_correct",
        "outcome_vs_round1",
        "outcome_vs_vanilla5",
        "effective_rounds_used",
        "actual_rounds_executed",
        "stop_reason",
        "single_agent_total_tokens",
        "majority_vote_total_tokens",
        "round5_cumulative_total_tokens",
        "scrd_total_tokens",
    ]
    full[[c for c in sample_cols if c in full.columns]].to_csv(
        out_dir / "sample_level_debug.csv",
        index=False,
    )

    # Markdown report.
    report = []
    report.append("# Qwen2.5-7B SCRD vs Baselines Summary\n")
    report.append("## Dataset-level method comparison\n")
    report.append(method_summary_with_overall.to_markdown(index=False))
    report.append("\n\n## SCRD outcome vs Round-1 Majority\n")
    report.append(outcome_vs_round1.to_markdown(index=False))
    report.append("\n\n## SCRD outcome vs Vanilla MAD Round-5\n")
    report.append(outcome_vs_vanilla5.to_markdown(index=False))
    report.append("\n\n## Stop reason summary\n")
    report.append(stop_reason_summary.to_markdown(index=False))
    report.append("\n\n## Component token summary\n")
    report.append(component_summary.to_markdown(index=False))

    (out_dir / "report.md").write_text("\n".join(report), encoding="utf-8")

    print(f"Saved summary to: {out_dir}")
    print(method_summary_with_overall.to_markdown(index=False))


if __name__ == "__main__":
    main()