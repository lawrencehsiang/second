import json
import re
from pathlib import Path
from collections import defaultdict
import pandas as pd

DATASETS = ["addsub", "asdiv", "gsm8k", "math", "multiarith", "singleeq", "svamp"]

ROOT = Path(".")
SCRD_SUMMARY = ROOT / "outputs_with_last_round_majority_vote" / "summary.csv"
VANILLA_DIR = ROOT / "outputs"

def norm_num(x):
    if x is None:
        return None
    s = str(x).replace(",", "")
    nums = re.findall(r"[-+]?\d*\.?\d+", s)
    if not nums:
        return str(x).strip().lower()
    try:
        v = float(nums[-1])
        if v.is_integer():
            return str(int(v))
        return str(round(v, 10)).rstrip("0").rstrip(".")
    except Exception:
        return str(x).strip().lower()

def is_correct(pred, gold):
    return norm_num(pred) == norm_num(gold)

def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

scrd_summary = pd.read_csv(SCRD_SUMMARY)
scrd_summary = scrd_summary[scrd_summary["dataset"] != "OVERALL"].set_index("dataset")

rows = []

for ds in DATASETS:
    path = VANILLA_DIR / f"{ds}_vanilla_mad7" / "results.jsonl"
    data = load_jsonl(path)

    n = len(data)

    single_correct = 0
    mv1_correct = 0
    mv3_correct = 0
    mv5_correct = 0

    single_total = []
    single_prompt = []
    single_completion = []

    mv1_total = []
    mv1_prompt = []
    mv1_completion = []

    mv3_total = []
    mv3_prompt = []
    mv3_completion = []

    mv5_total = []
    mv5_prompt = []
    mv5_completion = []

    for r in data:
        gold = r["gold_answer"]

        # 第一轮单个 agent：这里按 Agent A 统计正确性
        a1 = r["vanilla_round_answers"]["1"]["A"]
        single_correct += int(is_correct(a1, gold))

        # token：vanilla 结果里 round1 token 是三个 agent 总和
        # 单 agent 成本按 round1 / 3 估计，这是和单 agent baseline 对齐的口径
        single_total.append(r["round1_cumulative_total_tokens"] / 3)
        single_prompt.append(r["round1_cumulative_prompt_tokens"] / 3)
        single_completion.append(r["round1_cumulative_completion_tokens"] / 3)

        mv1_correct += int(r["round1_majority_correct"])
        mv3_correct += int(r["round3_majority_correct"])
        mv5_correct += int(r["round5_majority_correct"])

        mv1_total.append(r["round1_cumulative_total_tokens"])
        mv1_prompt.append(r["round1_cumulative_prompt_tokens"])
        mv1_completion.append(r["round1_cumulative_completion_tokens"])

        mv3_total.append(r["round3_cumulative_total_tokens"])
        mv3_prompt.append(r["round3_cumulative_prompt_tokens"])
        mv3_completion.append(r["round3_cumulative_completion_tokens"])

        mv5_total.append(r["round5_cumulative_total_tokens"])
        mv5_prompt.append(r["round5_cumulative_prompt_tokens"])
        mv5_completion.append(r["round5_cumulative_completion_tokens"])

    scrd_n = int(scrd_summary.loc[ds, "n"])
    scrd_correct = int(scrd_summary.loc[ds, "last_round_majority_correct"])
    scrd_acc = float(scrd_summary.loc[ds, "last_round_majority_acc"])

    rows.extend([
        {
            "dataset": ds,
            "method": "Single Agent R1-A",
            "n": n,
            "correct": single_correct,
            "accuracy": single_correct / n,
            "avg_total_tokens": sum(single_total) / n,
            "avg_prompt_tokens": sum(single_prompt) / n,
            "avg_completion_tokens": sum(single_completion) / n,
        },
        {
            "dataset": ds,
            "method": "MV@Round1",
            "n": n,
            "correct": mv1_correct,
            "accuracy": mv1_correct / n,
            "avg_total_tokens": sum(mv1_total) / n,
            "avg_prompt_tokens": sum(mv1_prompt) / n,
            "avg_completion_tokens": sum(mv1_completion) / n,
        },
        {
            "dataset": ds,
            "method": "MV@Round3",
            "n": n,
            "correct": mv3_correct,
            "accuracy": mv3_correct / n,
            "avg_total_tokens": sum(mv3_total) / n,
            "avg_prompt_tokens": sum(mv3_prompt) / n,
            "avg_completion_tokens": sum(mv3_completion) / n,
        },
        {
            "dataset": ds,
            "method": "MV@Round5",
            "n": n,
            "correct": mv5_correct,
            "accuracy": mv5_correct / n,
            "avg_total_tokens": sum(mv5_total) / n,
            "avg_prompt_tokens": sum(mv5_prompt) / n,
            "avg_completion_tokens": sum(mv5_completion) / n,
        },
        {
            "dataset": ds,
            "method": "SCRD Last-Round MV",
            "n": scrd_n,
            "correct": scrd_correct,
            "accuracy": scrd_acc,
            "avg_total_tokens": None,
            "avg_prompt_tokens": None,
            "avg_completion_tokens": None,
        },
    ])

df = pd.DataFrame(rows)

# dataset-level accuracy table
acc_table = df.pivot(index="dataset", columns="method", values="accuracy") * 100
print("\n=== Accuracy table (%) ===")
print(acc_table.round(2))

# token table
token_table = df.pivot(index="dataset", columns="method", values="avg_total_tokens")
print("\n=== Avg total tokens ===")
print(token_table.round(1))

# overall table
overall = []
for method, g in df.groupby("method"):
    n_sum = g["n"].sum()
    correct_sum = g["correct"].sum()
    token_valid = g.dropna(subset=["avg_total_tokens"])

    if len(token_valid) > 0:
        avg_total = (token_valid["avg_total_tokens"] * token_valid["n"]).sum() / token_valid["n"].sum()
        avg_prompt = (token_valid["avg_prompt_tokens"] * token_valid["n"]).sum() / token_valid["n"].sum()
        avg_completion = (token_valid["avg_completion_tokens"] * token_valid["n"]).sum() / token_valid["n"].sum()
    else:
        avg_total = avg_prompt = avg_completion = None

    overall.append({
        "method": method,
        "n": n_sum,
        "correct": correct_sum,
        "accuracy": correct_sum / n_sum,
        "avg_total_tokens": avg_total,
        "avg_prompt_tokens": avg_prompt,
        "avg_completion_tokens": avg_completion,
    })

overall_df = pd.DataFrame(overall).sort_values("method")
print("\n=== Overall ===")
print(overall_df.round(4))

df.to_csv("comparison_experiments_anlysis/comparison_200_sample_method_level.csv", index=False, encoding="utf-8-sig")
acc_table.to_csv("comparison_experiments_anlysis/comparison_200_accuracy_table.csv", encoding="utf-8-sig")
token_table.to_csv("comparison_experiments_anlysis/comparison_200_token_table.csv", encoding="utf-8-sig")
overall_df.to_csv("comparison_experiments_anlysis/comparison_200_overall.csv", index=False, encoding="utf-8-sig")