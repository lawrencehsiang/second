import json
from pathlib import Path

import pandas as pd


DATASET = "gsm8k"
OUTPUT_CSV = Path("outputs/sensitivity/sensitivity_summary.csv")


EXPERIMENTS = [
    # ======================================================
    # max_round sensitivity
    # ======================================================
    {
        "analysis": "max_round",
        "setting": "max_round=3",
        "path": Path("outputs/sensitivity/max_round_3/gsm8k/results.jsonl"),
        "correct_col": "scrd_correct",
        "token_col": "scrd_total_tokens",
    },
    {
        "analysis": "max_round",
        "setting": "max_round=5/default",
        "path": Path("outputs/ablation/wo_decision_head/gsm8k/results.jsonl"),
        "correct_col": "wo_decision_head_correct",
        "token_col": "wo_decision_head_total_tokens",
    },
    {
        "analysis": "max_round",
        "setting": "max_round=7",
        "path": Path("outputs/sensitivity/max_round_7/gsm8k/results.jsonl"),
        "correct_col": "scrd_correct",
        "token_col": "scrd_total_tokens",
    },

    # ======================================================
    # rollback_threshold sensitivity
    # threshold=0: w/o rollback
    # threshold=1: default
    # threshold=2: rollback_2
    # ======================================================
    {
        "analysis": "rollback_threshold",
        "setting": "rollback_threshold=0 / w/o rollback",
        "path": Path("outputs/ablation/wo_rollback_v2/gsm8k/results.jsonl"),
        "correct_col": "wo_rollback_correct",
        "token_col": "wo_rollback_total_tokens",
    },
    {
        "analysis": "rollback_threshold",
        "setting": "rollback_threshold=1 / default",
        "path": Path("outputs/ablation/wo_decision_head/gsm8k/results.jsonl"),
        "correct_col": "wo_decision_head_correct",
        "token_col": "wo_decision_head_total_tokens",
    },
    {
        "analysis": "rollback_threshold",
        "setting": "rollback_threshold=2",
        "path": Path("outputs/sensitivity/rollback_2/gsm8k/results.jsonl"),
        "correct_col": "scrd_correct",
        "token_col": "scrd_total_tokens",
    },
]


def load_jsonl(path: Path) -> pd.DataFrame:
    rows = []

    with path.open("r", encoding="utf-8") as f:
        for line_id, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"JSON decode error: {path}, line {line_id}") from e

    if not rows:
        raise ValueError(f"No rows found in {path}")

    return pd.DataFrame(rows)


def find_col(df: pd.DataFrame, candidates):
    for col in candidates:
        if col in df.columns:
            return col
    return None


def get_required_col(df: pd.DataFrame, preferred_col: str, fallback_candidates, col_type: str):
    if preferred_col in df.columns:
        return preferred_col

    fallback_col = find_col(df, fallback_candidates)

    if fallback_col is not None:
        print(
            f"[Warning] Preferred {col_type} column `{preferred_col}` not found. "
            f"Using fallback column `{fallback_col}`."
        )
        return fallback_col

    raise KeyError(
        f"Cannot find {col_type} column.\n"
        f"Preferred: {preferred_col}\n"
        f"Fallback candidates: {fallback_candidates}\n"
        f"Available columns: {list(df.columns)}"
    )


def mean_or_none(df: pd.DataFrame, col):
    if col is None or col not in df.columns or len(df) == 0:
        return None
    return pd.to_numeric(df[col], errors="coerce").mean()


def acc_or_none(df: pd.DataFrame, correct_col):
    if correct_col is None or correct_col not in df.columns or len(df) == 0:
        return None
    return pd.to_numeric(df[correct_col], errors="coerce").mean() * 100


def aggregate_one(exp):
    path = exp["path"]

    if not path.exists():
        raise FileNotFoundError(
            f"Missing file for {exp['analysis']} / {exp['setting']}:\n"
            f"{path}\n\n"
            f"请确认你是在仓库根目录运行脚本。"
        )

    df = load_jsonl(path)

    correct_col = get_required_col(
        df,
        preferred_col=exp["correct_col"],
        fallback_candidates=[
            "scrd_correct",
            "wo_decision_head_correct",
            "wo_rollback_correct",
            "full_scrd_correct",
            "majority_voting_correct",
            "correct",
        ],
        col_type="correctness",
    )

    token_col = get_required_col(
        df,
        preferred_col=exp["token_col"],
        fallback_candidates=[
            "scrd_total_tokens",
            "wo_decision_head_total_tokens",
            "wo_rollback_total_tokens",
            "total_tokens",
            "tokens",
        ],
        col_type="token",
    )

    effective_round_col = find_col(
        df,
        [
            "effective_rounds_used",
            "effective_rounds",
            "eff_rounds",
        ],
    )

    actual_round_col = find_col(
        df,
        [
            "actual_rounds_executed",
            "actual_rounds",
            "rounds_executed",
        ],
    )

    stop_reason_col = find_col(
        df,
        [
            "stop_reason",
            "final_stop_reason",
        ],
    )

    n = len(df)

    if stop_reason_col is not None:
        rollback_mask = df[stop_reason_col].eq("rollback")
        early_stop_mask = df[stop_reason_col].eq("early_stop")

        rollback_count = int(rollback_mask.sum())
        early_stop_count = int(early_stop_mask.sum())

        rollback_pct = rollback_count / n * 100
        early_stop_pct = early_stop_count / n * 100

        rollback_acc_pct = acc_or_none(df[rollback_mask], correct_col) if rollback_count > 0 else None
        early_stop_acc_pct = acc_or_none(df[early_stop_mask], correct_col) if early_stop_count > 0 else None

        rollback_avg_tokens = mean_or_none(df[rollback_mask], token_col) if rollback_count > 0 else None
        early_stop_avg_tokens = mean_or_none(df[early_stop_mask], token_col) if early_stop_count > 0 else None
    else:
        rollback_count = None
        early_stop_count = None
        rollback_pct = None
        early_stop_pct = None
        rollback_acc_pct = None
        early_stop_acc_pct = None
        rollback_avg_tokens = None
        early_stop_avg_tokens = None

    row = {
        "analysis": exp["analysis"],
        "setting": exp["setting"],
        "dataset": DATASET,
        "n": n,

        "acc_pct": acc_or_none(df, correct_col),
        "avg_tokens": mean_or_none(df, token_col),
        "avg_effective_rounds": mean_or_none(df, effective_round_col),
        "avg_actual_rounds": mean_or_none(df, actual_round_col),

        "rollback_count": rollback_count,
        "rollback_pct": rollback_pct,
        "rollback_acc_pct": rollback_acc_pct,
        "rollback_avg_tokens": rollback_avg_tokens,

        "early_stop_count": early_stop_count,
        "early_stop_pct": early_stop_pct,
        "early_stop_acc_pct": early_stop_acc_pct,
        "early_stop_avg_tokens": early_stop_avg_tokens,

        "used_correct_col": correct_col,
        "used_token_col": token_col,
        "effective_round_col": effective_round_col,
        "actual_round_col": actual_round_col,
        "stop_reason_col": stop_reason_col,

        "source_file": str(path),
    }

    return row


def main():
    rows = []

    for exp in EXPERIMENTS:
        print(f"Processing: {exp['analysis']} / {exp['setting']}")
        row = aggregate_one(exp)
        rows.append(row)

    summary = pd.DataFrame(rows)

    numeric_cols = [
        "acc_pct",
        "avg_tokens",
        "avg_effective_rounds",
        "avg_actual_rounds",
        "rollback_pct",
        "rollback_acc_pct",
        "rollback_avg_tokens",
        "early_stop_pct",
        "early_stop_acc_pct",
        "early_stop_avg_tokens",
    ]

    for col in numeric_cols:
        if col in summary.columns:
            summary[col] = pd.to_numeric(summary[col], errors="coerce").round(2)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print("\nSensitivity summary:")
    print(summary.to_string(index=False))

    print(f"\nSaved CSV to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()