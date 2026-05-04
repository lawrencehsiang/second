from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import os
# 强制禁用代理，直连国内网络
os.environ["HTTP_PROXY"] = ""
os.environ["HTTPS_PROXY"] = ""
os.environ["NO_PROXY"] = "dashscope.aliyuncs.com,localhost,127.0.0.1"
# ============================================================
# Config
# ============================================================

DEFAULT_DATASET = "gsm8k"

# 只统计这一个数据集，因为当前 sensitivity 文件夹主要是 gsm8k
DATASET = DEFAULT_DATASET


# ============================================================
# Normalization / correctness / majority vote
# 与 src/utils/result_utils.py 的数字题口径保持一致
# ============================================================

def normalize_text(text: Any) -> str:
    if text is None:
        return ""
    text = str(text).strip().lower()
    text = text.replace(",", "")
    text = text.replace("$", "")
    text = text.replace("dollars", "")
    text = text.replace("dollar", "")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_last_number(text: Any) -> float | None:
    if text is None:
        return None
    text = normalize_text(text)
    numbers = re.findall(r"-?\d+(?:\.\d+)?", text)
    if not numbers:
        return None
    try:
        return float(numbers[-1])
    except ValueError:
        return None


def is_correct_math_style(pred: Any, gold: Any) -> bool:
    pred_num = extract_last_number(pred)
    gold_num = extract_last_number(gold)
    if pred_num is None or gold_num is None:
        return False
    return int(round(pred_num)) == int(round(gold_num))


def majority_vote_math_style(answers: list[Any]) -> str:
    """
    数学任务多数投票：
    - 优先按最后一个数字投票
    - 如果没有数字，则按归一化文本投票
    """
    if not answers:
        return ""

    vote_keys = []
    key_to_original = {}

    for ans in answers:
        num = extract_last_number(ans)
        if num is not None:
            key = f"num:{int(round(num))}"
        else:
            key = f"text:{normalize_text(ans)}"

        vote_keys.append(key)
        key_to_original.setdefault(key, str(ans))

    majority_key = Counter(vote_keys).most_common(1)[0][0]
    return key_to_original[majority_key]


# ============================================================
# IO helpers
# ============================================================

def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"JSON decode error in {path}, line {line_no}: {e}") from e
    return rows


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing trace file: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def find_trace_file(exp_dir: Path, sample_id: str) -> Path | None:
    """
    默认 trace 文件名：
    traces/gsm8k_0001_trace.json

    如果不存在，也尝试用 glob 容错。
    """
    traces_dir = exp_dir / "traces"
    direct = traces_dir / f"{sample_id}_trace.json"
    if direct.exists():
        return direct

    matches = list(traces_dir.glob(f"{sample_id}*.json"))
    if matches:
        return matches[0]

    return None


# ============================================================
# Core trace statistics
# ============================================================

def get_last_round_answers_from_trace(trace: dict[str, Any]) -> list[str]:
    """
    关键口径：
    取 trace['final_trace'] 最后一条 state 的 current_answers，
    作为 last-round majority vote 的输入。
    """
    final_trace = trace.get("final_trace", [])
    if not final_trace:
        return []

    last_state = final_trace[-1]
    answers = last_state.get("current_answers", [])
    return [str(a) for a in answers if a is not None]


def count_actual_rounds_from_trace(trace: dict[str, Any]) -> int:
    events = trace.get("execution_events", [])
    return sum(
        1
        for e in events
        if e.get("type") in {"normal_round_executed", "repair_round_executed"}
    )


def count_rollback_events_from_trace(trace: dict[str, Any]) -> int:
    events = trace.get("execution_events", [])
    return sum(1 for e in events if e.get("type") == "rollback_triggered")


def summarize_from_result_and_traces(
    exp_dir: Path,
    setting_name: str,
    setting_value: int,
    dataset: str,
) -> dict[str, Any]:
    """
    用于：
    - max_round=3
    - max_round=7
    - rollback=0
    - rollback=2
    - rollback=3

    新口径：
    1. 优先使用 results.jsonl 里已经记录好的 last-round MV 字段
    2. 如果没有 last_round_majority_correct，则用 last_effective_round_answers / last_effective_answers 重新多数投票
    3. 只有这些字段都没有时，才读取 trace
    4. 只有最后 fallback 到 scrd_correct 时才警告，因为 scrd_correct 可能是 decision head
    """
    results_path = exp_dir / "results.jsonl"
    rows = load_jsonl(results_path)

    n = 0
    correct = 0

    total_tokens = []
    prompt_tokens = []
    completion_tokens = []

    effective_rounds = []
    actual_rounds = []

    rollback_event_counts = []
    rollback_sample_flags = []

    missing_trace_count = 0
    fallback_to_scrd_correct_count = 0
    used_result_last_round_mv_count = 0
    used_result_answers_count = 0
    used_trace_count = 0

    for r in rows:
        if r.get("dataset_name", dataset) != dataset:
            continue

        n += 1
        sample_id = r["sample_id"]
        gold = r["gold_answer"]

        # ====================================================
        # 1. 正确性：优先使用 results.jsonl 里的 last-round MV
        # ====================================================
        if "last_round_majority_correct" in r:
            ok = bool(r["last_round_majority_correct"])
            used_result_last_round_mv_count += 1

        elif "last_effective_round_answers" in r and r["last_effective_round_answers"]:
            pred = majority_vote_math_style(r["last_effective_round_answers"])
            ok = is_correct_math_style(pred, gold)
            used_result_answers_count += 1

        elif "last_effective_answers" in r and r["last_effective_answers"]:
            pred = majority_vote_math_style(r["last_effective_answers"])
            ok = is_correct_math_style(pred, gold)
            used_result_answers_count += 1

        else:
            # ====================================================
            # 2. 如果 results.jsonl 没有 last-round 字段，再读 trace
            # ====================================================
            trace_file = find_trace_file(exp_dir, sample_id)

            if trace_file is not None:
                trace = load_json(trace_file)
                answers = get_last_round_answers_from_trace(trace)
                pred = majority_vote_math_style(answers)
                ok = is_correct_math_style(pred, gold)
                used_trace_count += 1
            else:
                # ====================================================
                # 3. 最后兜底：这一步不推荐用于论文，只是防止脚本中断
                # ====================================================
                missing_trace_count += 1
                fallback_to_scrd_correct_count += 1
                ok = bool(r.get("scrd_correct", False))

        correct += int(ok)

        # ====================================================
        # token：优先从 results.jsonl 读
        # ====================================================
        total = (
            r.get("scrd_total_tokens")
            or r.get("wo_rollback_total_tokens")
            or r.get("total_tokens")
        )
        prompt = r.get("scrd_prompt_tokens")
        completion = r.get("scrd_completion_tokens")

        if total is not None:
            total_tokens.append(float(total))
        if prompt is not None:
            prompt_tokens.append(float(prompt))
        if completion is not None:
            completion_tokens.append(float(completion))

        # ====================================================
        # rounds：优先从 results.jsonl 读
        # ====================================================
        eff_round = r.get("effective_rounds_used")
        act_round = r.get("actual_rounds_executed")

        if eff_round is not None:
            effective_rounds.append(float(eff_round))
        if act_round is not None:
            actual_rounds.append(float(act_round))

        # ====================================================
        # rollback 统计
        # - w/o rollback 固定为 0
        # - 其他设置优先看 stop_reason
        # - 如果有 rollback_count / rollback_events 字段，也兼容
        # ====================================================
        if setting_name == "rollback_limit" and setting_value == 0:
            rb_count = 0
        else:
            rb_count = (
                r.get("rollback_count")
                or r.get("rollback_events")
                or (1 if r.get("stop_reason") == "rollback" else 0)
            )

        rollback_event_counts.append(int(rb_count))
        rollback_sample_flags.append(1 if int(rb_count) > 0 else 0)

    if n == 0:
        raise ValueError(f"No rows found for dataset={dataset} in {results_path}")

    return {
        "setting_name": setting_name,
        "setting_value": setting_value,
        "dataset": dataset,
        "n": n,
        "correct": correct,
        "accuracy": correct / n,
        "accuracy_percent": 100 * correct / n,
        "avg_total_tokens": mean_or_none(total_tokens),
        "avg_prompt_tokens": mean_or_none(prompt_tokens),
        "avg_completion_tokens": mean_or_none(completion_tokens),
        "avg_effective_rounds": mean_or_none(effective_rounds),
        "avg_actual_rounds": mean_or_none(actual_rounds),
        "rollback_events": sum(rollback_event_counts),
        "rollback_samples": sum(rollback_sample_flags),
        "rollback_sample_rate": sum(rollback_sample_flags) / n,
        "missing_trace_count": missing_trace_count,
        "fallback_to_scrd_correct_count": fallback_to_scrd_correct_count,
        "used_result_last_round_mv_count": used_result_last_round_mv_count,
        "used_result_answers_count": used_result_answers_count,
        "used_trace_count": used_trace_count,
        "source": str(exp_dir),
    }


def summarize_max_round_1_from_comparison(
    sample_level_path: Path,
    dataset: str,
) -> dict[str, Any]:
    """
    max_round=1：
    使用 SCRD 对比实验里的第一轮多数投票结果。
    这里可以用 majority_voting_baseline_answer / round_1_answers，
    因为这是 SCRD 初始化第一轮的结果，不是 vanilla MAD baseline。
    """
    rows = load_jsonl(sample_level_path)

    n = 0
    correct = 0
    total_tokens = []

    for r in rows:
        if r.get("dataset_name") != dataset:
            continue

        n += 1
        gold = r["gold_answer"]

        if "round_1_answers" in r and r["round_1_answers"]:
            pred = majority_vote_math_style(r["round_1_answers"])
            ok = is_correct_math_style(pred, gold)
        elif "majority_voting_correct" in r:
            ok = bool(r["majority_voting_correct"])
        else:
            pred = r.get("majority_voting_baseline_answer", "")
            ok = is_correct_math_style(pred, gold)

        correct += int(ok)

        if r.get("majority_vote_total_tokens") is not None:
            total_tokens.append(float(r["majority_vote_total_tokens"]))

    if n == 0:
        raise ValueError(f"No rows found for dataset={dataset} in {sample_level_path}")

    return {
        "setting_name": "max_round",
        "setting_value": 1,
        "dataset": dataset,
        "n": n,
        "correct": correct,
        "accuracy": correct / n,
        "accuracy_percent": 100 * correct / n,
        "avg_total_tokens": mean_or_none(total_tokens),
        "avg_prompt_tokens": None,
        "avg_completion_tokens": None,
        "avg_effective_rounds": 1.0,
        "avg_actual_rounds": 1.0,
        "rollback_events": 0,
        "rollback_samples": 0,
        "rollback_sample_rate": 0.0,
        "missing_trace_count": 0,
        "fallback_to_scrd_correct_count": 0,
        "source": str(sample_level_path),
    }


def summarize_default_from_comparison(
    sample_level_path: Path,
    dataset: str,
    setting_name: str,
    setting_value: int,
) -> dict[str, Any]:
    """
    默认 SCRD：
    - max_round=5
    - rollback=1

    使用 outputs_with_last_round_majority_vote/sample_level_results.jsonl
    中已经补好的 last_round_majority_correct。
    """
    rows = load_jsonl(sample_level_path)

    n = 0
    correct = 0

    total_tokens = []
    prompt_tokens = []
    completion_tokens = []

    effective_rounds = []
    actual_rounds = []

    rollback_sample_flags = []

    for r in rows:
        if r.get("dataset_name") != dataset:
            continue

        n += 1

        if "last_round_majority_correct" in r:
            ok = bool(r["last_round_majority_correct"])
        else:
            gold = r["gold_answer"]
            answers = r.get("last_effective_round_answers", [])
            pred = majority_vote_math_style(answers)
            ok = is_correct_math_style(pred, gold)

        correct += int(ok)

        if r.get("scrd_total_tokens") is not None:
            total_tokens.append(float(r["scrd_total_tokens"]))
        if r.get("scrd_prompt_tokens") is not None:
            prompt_tokens.append(float(r["scrd_prompt_tokens"]))
        if r.get("scrd_completion_tokens") is not None:
            completion_tokens.append(float(r["scrd_completion_tokens"]))

        if r.get("effective_rounds_used") is not None:
            effective_rounds.append(float(r["effective_rounds_used"]))
        if r.get("actual_rounds_executed") is not None:
            actual_rounds.append(float(r["actual_rounds_executed"]))

        rb_flag = 1 if r.get("stop_reason") == "rollback" else 0
        rollback_sample_flags.append(rb_flag)

    if n == 0:
        raise ValueError(f"No rows found for dataset={dataset} in {sample_level_path}")

    rollback_samples = sum(rollback_sample_flags)

    return {
        "setting_name": setting_name,
        "setting_value": setting_value,
        "dataset": dataset,
        "n": n,
        "correct": correct,
        "accuracy": correct / n,
        "accuracy_percent": 100 * correct / n,
        "avg_total_tokens": mean_or_none(total_tokens),
        "avg_prompt_tokens": mean_or_none(prompt_tokens),
        "avg_completion_tokens": mean_or_none(completion_tokens),
        "avg_effective_rounds": mean_or_none(effective_rounds),
        "avg_actual_rounds": mean_or_none(actual_rounds),
        "rollback_events": rollback_samples,
        "rollback_samples": rollback_samples,
        "rollback_sample_rate": rollback_samples / n,
        "missing_trace_count": 0,
        "fallback_to_scrd_correct_count": 0,
        "source": str(sample_level_path),
    }


def mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


# ============================================================
# Build two sensitivity tables
# ============================================================

def build_max_round_table(root: Path, dataset: str) -> pd.DataFrame:
    sample_level_path = (
        root
        / "outputs_with_last_round_majority_vote"
        / "sample_level_results.jsonl"
    )

    rows = [
        summarize_max_round_1_from_comparison(sample_level_path, dataset),

        summarize_from_result_and_traces(
            exp_dir=root / "outputs" / "sensitivity" / "max_round_3" / dataset,
            setting_name="max_round",
            setting_value=3,
            dataset=dataset,
        ),

        summarize_default_from_comparison(
            sample_level_path=sample_level_path,
            dataset=dataset,
            setting_name="max_round",
            setting_value=5,
        ),

        summarize_from_result_and_traces(
            exp_dir=root / "outputs" / "sensitivity" / "max_round_7" / dataset,
            setting_name="max_round",
            setting_value=7,
            dataset=dataset,
        ),
    ]

    df = pd.DataFrame(rows)
    df = df.sort_values("setting_value").reset_index(drop=True)
    return df


def build_rollback_table(root: Path, dataset: str) -> pd.DataFrame:
    sample_level_path = (
        root
        / "outputs_with_last_round_majority_vote"
        / "sample_level_results.jsonl"
    )

    rows = [
        summarize_from_result_and_traces(
            exp_dir=root / "outputs" / "ablation" / "wo_rollback_v2" / dataset,
            setting_name="rollback_limit",
            setting_value=0,
            dataset=dataset,
        ),

        summarize_default_from_comparison(
            sample_level_path=sample_level_path,
            dataset=dataset,
            setting_name="rollback_limit",
            setting_value=1,
        ),

        summarize_from_result_and_traces(
            exp_dir=root / "outputs" / "sensitivity" / "rollback_2" / dataset,
            setting_name="rollback_limit",
            setting_value=2,
            dataset=dataset,
        ),

        summarize_from_result_and_traces(
            exp_dir=root / "outputs" / "sensitivity" / "rollback_3" / dataset,
            setting_name="rollback_limit",
            setting_value=3,
            dataset=dataset,
        ),
    ]

    df = pd.DataFrame(rows)
    df = df.sort_values("setting_value").reset_index(drop=True)
    return df


def format_for_paper(df: pd.DataFrame, setting_col_name: str) -> pd.DataFrame:
    """
    生成适合论文查看的简洁表。
    """
    out = pd.DataFrame({
        setting_col_name: df["setting_value"],
        "N": df["n"],
        "Correct": df["correct"],
        "Accuracy (%)": df["accuracy_percent"],
        "Avg Total Tokens": df["avg_total_tokens"],
        "Avg Effective Rounds": df["avg_effective_rounds"],
        "Avg Actual Rounds": df["avg_actual_rounds"],
        "Rollback Samples": df["rollback_samples"],
        "Rollback Rate (%)": df["rollback_sample_rate"] * 100,
        "Fallback to scrd_correct": df["fallback_to_scrd_correct_count"],
    })

    numeric_cols = [
        "Accuracy (%)",
        "Avg Total Tokens",
        "Avg Effective Rounds",
        "Avg Actual Rounds",
        "Rollback Rate (%)",
    ]
    for col in numeric_cols:
        out[col] = out[col].round(2)

    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=str,
        default=".",
        help="Repo root directory. Default: current directory.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=DEFAULT_DATASET,
        help="Dataset name. Default: gsm8k.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="analysis_outputs/sensitivity",
        help="Output directory.",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    dataset = args.dataset
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    max_round_df = build_max_round_table(root, dataset)
    rollback_df = build_rollback_table(root, dataset)

    max_round_paper = format_for_paper(max_round_df, "Max Round")
    rollback_paper = format_for_paper(rollback_df, "Rollback Limit")

    # 保存完整表
    max_round_df.to_csv(out_dir / "sensitivity_max_round_table_full.csv", index=False, encoding="utf-8-sig")
    rollback_df.to_csv(out_dir / "sensitivity_rollback_table_full.csv", index=False, encoding="utf-8-sig")

    # 保存论文简洁表
    max_round_paper.to_csv(out_dir / "sensitivity_max_round_table.csv", index=False, encoding="utf-8-sig")
    rollback_paper.to_csv(out_dir / "sensitivity_rollback_table.csv", index=False, encoding="utf-8-sig")

    # 保存 Excel
    xlsx_path = out_dir / "sensitivity_tables.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        max_round_paper.to_excel(writer, sheet_name="max_round_paper", index=False)
        rollback_paper.to_excel(writer, sheet_name="rollback_paper", index=False)
        max_round_df.to_excel(writer, sheet_name="max_round_full", index=False)
        rollback_df.to_excel(writer, sheet_name="rollback_full", index=False)

    print("\n=== Max Round Sensitivity ===")
    print(max_round_paper.to_string(index=False))

    print("\n=== Rollback Sensitivity ===")
    print(rollback_paper.to_string(index=False))

    print("\nSaved files:")
    print(f"- {out_dir / 'sensitivity_max_round_table.csv'}")
    print(f"- {out_dir / 'sensitivity_rollback_table.csv'}")
    print(f"- {out_dir / 'sensitivity_max_round_table_full.csv'}")
    print(f"- {out_dir / 'sensitivity_rollback_table_full.csv'}")
    print(f"- {xlsx_path}")

    # 安全提醒：如果 Missing Traces > 0，需要回头检查
    total_missing = (
        int(max_round_df["missing_trace_count"].sum())
        + int(rollback_df["missing_trace_count"].sum())
    )
    if total_missing > 0:
        print("\nWARNING:")
        print(f"Found {total_missing} missing trace files.")
        print("Rows with missing traces fall back to scrd_correct, which may be decision-head output.")
        print("Please check missing traces before using the table in the paper.")


if __name__ == "__main__":
    main()