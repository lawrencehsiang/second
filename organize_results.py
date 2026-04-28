import json
from pathlib import Path
from typing import Optional, Dict, Any, List

import pandas as pd


OUT_DIR = Path("outputs_clean")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV = OUT_DIR / "experiment_index_summary.csv"


DATASETS = [
    "addsub",
    "asdiv",
    "gsm8k",
    "math",
    "multiarith",
    "singleeq",
    "svamp",
]


# ============================================================
# 手动登记“已知实验路径”
# 这里只是告诉脚本去哪里找文件，不需要你手动算结果。
# ============================================================

EXPERIMENTS = []


# -----------------------------
# 1. Main SCRD: Decision Head
# outputs/{dataset}/results.jsonl
# -----------------------------
for ds in DATASETS:
    EXPERIMENTS.append({
        "experiment_id": f"main_decision_head_{ds}",
        "category": "main",
        "dataset": ds,
        "setting": "main_scrd_decision_head",
        "max_round": 5,
        "rollback_threshold": 1,
        "rollback_enabled": True,
        "finalizer": "decision_head",
        "source_path": Path(f"outputs/{ds}/results.jsonl"),
        "preferred_correct_col": "scrd_correct",
        "preferred_token_col": "scrd_total_tokens",
        "notes": "Original SCRD result with Decision Head.",
    })


# -----------------------------
# 2. Main SCRD: Last-round Majority Vote
# outputs/ablation/wo_decision_head/{dataset}/results.jsonl
# -----------------------------
for ds in DATASETS:
    EXPERIMENTS.append({
        "experiment_id": f"main_last_round_majority_vote_{ds}",
        "category": "main",
        "dataset": ds,
        "setting": "main_scrd_last_round_majority_vote",
        "max_round": 5,
        "rollback_threshold": 1,
        "rollback_enabled": True,
        "finalizer": "last_round_majority_vote",
        "source_path": Path(f"outputs/ablation/wo_decision_head/{ds}/results.jsonl"),
        "preferred_correct_col": "wo_decision_head_correct",
        "preferred_token_col": "wo_decision_head_total_tokens",
        "notes": "Post-hoc replacement of Decision Head by last-round majority vote.",
    })

# -----------------------------
# 2.5 Baseline: Vanilla MAD-7
# outputs/{dataset}_vanilla_mad7/results.jsonl
# -----------------------------
for ds in DATASETS:
    EXPERIMENTS.append({
        "experiment_id": f"main_vanilla_mad7_{ds}",
        "category": "main",
        "dataset": ds,
        "setting": "vanilla_mad7",
        "max_round": 7,
        "rollback_threshold": None,
        "rollback_enabled": False,
        "finalizer": "vanilla_majority_vote",
        "source_path": Path(f"outputs/{ds}_vanilla_mad7/results.jsonl"),
        "preferred_correct_col": "vanilla_final_correct",
        "preferred_token_col": "vanilla_total_tokens",
        "notes": "Vanilla MAD-7 baseline; final answer is the round-7 majority vote.",
    })
# -----------------------------
# 3. Ablation: w/o Evaluator & Action Mapper
# -----------------------------
for ds in ["math", "gsm8k", "multiarith"]:
    EXPERIMENTS.append({
        "experiment_id": f"ablation_wo_evaluator_action_mapper_{ds}",
        "category": "ablation",
        "dataset": ds,
        "setting": "wo_evaluator_action_mapper_fixed5",
        "max_round": 5,
        "rollback_threshold": None,
        "rollback_enabled": False,
        "finalizer": "unknown",
        "source_path": Path(f"outputs/ablation/wo_evaluator_action_mapper_fixed5/{ds}/results.jsonl"),
        "preferred_correct_col": None,
        "preferred_token_col": None,
        "notes": "Fixed 5 rounds; evaluator/action mapper disabled.",
    })


# -----------------------------
# 4. Ablation: w/o History Filtering
# -----------------------------
for ds in ["math", "gsm8k", "multiarith"]:
    EXPERIMENTS.append({
        "experiment_id": f"ablation_wo_history_filtering_{ds}",
        "category": "ablation",
        "dataset": ds,
        "setting": "wo_history_filtering_raw_full_history",
        "max_round": 5,
        "rollback_threshold": 1,
        "rollback_enabled": True,
        "finalizer": "unknown",
        "source_path": Path(f"outputs/ablation/wo_history_filtering_raw_full_history/{ds}/results.jsonl"),
        "preferred_correct_col": None,
        "preferred_token_col": None,
        "notes": "Raw full history instead of filtered structured history.",
    })


# -----------------------------
# 5. Ablation / Sensitivity: rollback_threshold=0
# w/o rollback
# -----------------------------
for ds in ["math", "gsm8k", "multiarith"]:
    EXPERIMENTS.append({
        "experiment_id": f"rollback_threshold_0_{ds}",
        "category": "sensitivity",
        "dataset": ds,
        "setting": "rollback_threshold=0 / wo_rollback",
        "max_round": 5,
        "rollback_threshold": 0,
        "rollback_enabled": False,
        "finalizer": "unknown",
        "source_path": Path(f"outputs/ablation/wo_rollback_v2/{ds}/results.jsonl"),
        "preferred_correct_col": "wo_rollback_correct",
        "preferred_token_col": "wo_rollback_total_tokens",
        "notes": "Rollback disabled; should be treated as rollback_threshold=0.",
    })


# -----------------------------
# 6. Sensitivity: max_round
# -----------------------------
for mr in [3, 7]:
    EXPERIMENTS.append({
        "experiment_id": f"sensitivity_max_round_{mr}_gsm8k",
        "category": "sensitivity",
        "dataset": "gsm8k",
        "setting": f"max_round={mr}",
        "max_round": mr,
        "rollback_threshold": 1,
        "rollback_enabled": True,
        "finalizer": "decision_head",
        "source_path": Path(f"outputs/sensitivity/max_round_{mr}/gsm8k/results.jsonl"),
        "preferred_correct_col": "scrd_correct",
        "preferred_token_col": "scrd_total_tokens",
        "notes": f"Max round sensitivity: max_round={mr}.",
    })


# -----------------------------
# 7. Sensitivity: rollback_threshold=2
# -----------------------------
EXPERIMENTS.append({
    "experiment_id": "rollback_threshold_2_gsm8k",
    "category": "sensitivity",
    "dataset": "gsm8k",
    "setting": "rollback_threshold=2",
    "max_round": 5,
    "rollback_threshold": 2,
    "rollback_enabled": True,
    "finalizer": "decision_head",
    "source_path": Path("outputs/sensitivity/rollback_2/gsm8k/results.jsonl"),
    "preferred_correct_col": "scrd_correct",
    "preferred_token_col": "scrd_total_tokens",
    "notes": "Rollback threshold sensitivity: threshold=2.",
})


# ============================================================
# 工具函数
# ============================================================

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
                raise ValueError(f"JSON decode error in {path}, line {line_id}: {e}") from e

    return pd.DataFrame(rows)


def find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for col in candidates:
        if col in df.columns:
            return col
    return None

def choose_correct_col(df: pd.DataFrame, preferred: Optional[str]) -> Optional[str]:
    if preferred and preferred in df.columns:
        return preferred

    candidates = [
        "scrd_correct",
        "wo_decision_head_correct",
        "wo_rollback_correct",

        # Vanilla MAD baselines
        "vanilla_final_correct",
        "round7_majority_correct",
        "round5_majority_correct",
        "round3_majority_correct",
        "round1_majority_correct",

        # Other ablations / fallbacks
        "ablation_correct",
        "fixed5_correct",
        "raw_full_history_correct",
        "full_scrd_correct",
        "majority_voting_correct",
        "single_agent_correct",
        "correct",
    ]

    return find_col(df, candidates)

  


def choose_token_col(df: pd.DataFrame, preferred: Optional[str]) -> Optional[str]:
    if preferred and preferred in df.columns:
        return preferred

    candidates = [
        "scrd_total_tokens",
        "wo_decision_head_total_tokens",
        "wo_rollback_total_tokens",

        # Vanilla MAD baselines
        "vanilla_total_tokens",
        "round7_cumulative_total_tokens",
        "round5_cumulative_total_tokens",
        "round3_cumulative_total_tokens",
        "round1_cumulative_total_tokens",

        # Other ablations / fallbacks
        "ablation_total_tokens",
        "fixed5_total_tokens",
        "raw_full_history_total_tokens",
        "total_tokens",
        "tokens",
    ]

    return find_col(df, candidates)


def choose_round_col(df: pd.DataFrame, kind: str) -> Optional[str]:
    if kind == "effective":
        candidates = [
            "effective_rounds_used",
            "effective_rounds",
            "eff_rounds",
        ]
    else:
        candidates = [
            "actual_rounds_executed",
            "actual_rounds",
            "rounds_executed",
        ]

    return find_col(df, candidates)


def choose_stop_reason_col(df: pd.DataFrame) -> Optional[str]:
    return find_col(df, ["stop_reason", "final_stop_reason"])


def mean_numeric(df: pd.DataFrame, col: Optional[str]) -> Optional[float]:
    if col is None or col not in df.columns or len(df) == 0:
        return None
    return float(pd.to_numeric(df[col], errors="coerce").mean())


def acc_numeric(df: pd.DataFrame, col: Optional[str]) -> Optional[float]:
    if col is None or col not in df.columns or len(df) == 0:
        return None
    return float(pd.to_numeric(df[col], errors="coerce").mean() * 100)


def count_stop_reason(df: pd.DataFrame, stop_col: Optional[str], reason: str) -> Optional[int]:
    if stop_col is None or stop_col not in df.columns:
        return None
    return int(df[stop_col].eq(reason).sum())


def infer_can_posthoc_majority_vote(df: pd.DataFrame) -> bool:
    possible_cols = [
        "last_effective_answers",
        "last_round_answers",
        "final_round_answers",
        "round_answers",
    ]
    return any(col in df.columns for col in possible_cols)


def summarize_experiment(exp: Dict[str, Any]) -> Dict[str, Any]:
    path = exp["source_path"]

    base = {
        "experiment_id": exp["experiment_id"],
        "category": exp["category"],
        "dataset": exp["dataset"],
        "setting": exp["setting"],
        "max_round": exp["max_round"],
        "rollback_threshold": exp["rollback_threshold"],
        "rollback_enabled": exp["rollback_enabled"],
        "finalizer": exp["finalizer"],
        "source_path": str(path),
        "file_exists": path.exists(),
        "n": None,
        "acc_pct": None,
        "avg_tokens": None,
        "avg_effective_rounds": None,
        "avg_actual_rounds": None,
        "rollback_count": None,
        "rollback_pct": None,
        "early_stop_count": None,
        "early_stop_pct": None,
        "used_correct_col": None,
        "used_token_col": None,
        "used_effective_round_col": None,
        "used_actual_round_col": None,
        "used_stop_reason_col": None,
        "can_posthoc_majority_vote": None,
        "all_columns": None,
        "notes": exp["notes"],
    }

    if not path.exists():
        base["notes"] += " | FILE MISSING"
        return base

    df = load_jsonl(path)

    correct_col = choose_correct_col(df, exp.get("preferred_correct_col"))
    token_col = choose_token_col(df, exp.get("preferred_token_col"))
    effective_round_col = choose_round_col(df, "effective")
    actual_round_col = choose_round_col(df, "actual")
    stop_col = choose_stop_reason_col(df)

    n = len(df)
    rollback_count = count_stop_reason(df, stop_col, "rollback")
    early_stop_count = count_stop_reason(df, stop_col, "early_stop")

    base.update({
        "n": n,
        "acc_pct": acc_numeric(df, correct_col),
        "avg_tokens": mean_numeric(df, token_col),
        "avg_effective_rounds": mean_numeric(df, effective_round_col),
        "avg_actual_rounds": mean_numeric(df, actual_round_col),
        "rollback_count": rollback_count,
        "rollback_pct": rollback_count / n * 100 if rollback_count is not None and n > 0 else None,
        "early_stop_count": early_stop_count,
        "early_stop_pct": early_stop_count / n * 100 if early_stop_count is not None and n > 0 else None,
        "used_correct_col": correct_col,
        "used_token_col": token_col,
        "used_effective_round_col": effective_round_col,
        "used_actual_round_col": actual_round_col,
        "used_stop_reason_col": stop_col,
        "can_posthoc_majority_vote": infer_can_posthoc_majority_vote(df),
        "all_columns": "|".join(list(df.columns)),
    })

    if correct_col is None:
        base["notes"] += " | WARNING: no correctness column detected"

    if token_col is None:
        base["notes"] += " | WARNING: no token column detected"

    if exp["finalizer"] == "unknown":
        base["notes"] += " | WARNING: finalizer needs manual verification"

    return base


def main():
    rows = []

    for exp in EXPERIMENTS:
        print(f"Scanning: {exp['experiment_id']}")
        rows.append(summarize_experiment(exp))

    df = pd.DataFrame(rows)

    numeric_cols = [
        "acc_pct",
        "avg_tokens",
        "avg_effective_rounds",
        "avg_actual_rounds",
        "rollback_pct",
        "early_stop_pct",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").round(2)

    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print("\nDone.")
    print(f"Saved to: {OUTPUT_CSV}")

    print("\nImportant rows with warnings:")
    warning_df = df[
        (df["file_exists"] == False)
        | (df["used_correct_col"].isna())
        | (df["finalizer"] == "unknown")
    ]

    if len(warning_df) == 0:
        print("No warnings.")
    else:
        print(
            warning_df[
                [
                    "experiment_id",
                    "source_path",
                    "file_exists",
                    "finalizer",
                    "used_correct_col",
                    "used_token_col",
                    "can_posthoc_majority_vote",
                    "notes",
                ]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()