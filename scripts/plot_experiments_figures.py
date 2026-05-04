from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd


DATASETS = ["addsub", "asdiv", "gsm8k", "math", "multiarith", "singleeq", "svamp"]
ROLLBACK_DATASETS = ["math", "gsm8k", "multiarith"]


# ============================================================
# Basic IO
# ============================================================

def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

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


def find_existing_file(root: Path, candidates: list[str]) -> Path:
    for c in candidates:
        p = root / c
        if p.exists():
            return p
    raise FileNotFoundError(
        "Cannot find any of these files:\n" + "\n".join(str(root / c) for c in candidates)
    )


def ensure_out_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_fig(fig: plt.Figure, out_dir: Path, name: str) -> None:
    png_path = out_dir / f"{name}.png"
    pdf_path = out_dir / f"{name}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


# ============================================================
# Math correctness helpers
# Used only when a correctness field is missing.
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
    nums = re.findall(r"-?\d+(?:\.\d+)?", text)
    if not nums:
        return None
    try:
        return float(nums[-1])
    except ValueError:
        return None


def is_correct_math_style(pred: Any, gold: Any) -> bool:
    pred_num = extract_last_number(pred)
    gold_num = extract_last_number(gold)
    if pred_num is None or gold_num is None:
        return False
    return int(round(pred_num)) == int(round(gold_num))


# ============================================================
# Figure 1: Accuracy–Token Trade-off
# ============================================================

def build_tradeoff_from_long_csv(root: Path, scrd_jsonl_path: Path | None) -> pd.DataFrame:
    """
    Preferred input:
    comparison_200_sample_method_level.csv

    Expected columns:
    dataset, method, n, correct, accuracy, avg_total_tokens
    """
    csv_path = find_existing_file(root, [
        "comparison_200_sample_method_level.csv",
        "analysis_outputs/comparison_200_sample_method_level.csv",
        "analysis_outputs/comparison/comparison_200_sample_method_level.csv",
    ])

    df = pd.read_csv(csv_path)

    required = {"method", "n", "correct", "accuracy", "avg_total_tokens"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path} missing columns: {missing}")

    method_name_map = {
        "Single Agent R1-A": "Single",
        "MV@Round1": "MV@R1",
        "MV@Round3": "MV@R3",
        "MV@Round5": "MV@R5",
        "SCRD Last-Round MV": "SCRD",
    }

    df["method_short"] = df["method"].map(method_name_map).fillna(df["method"])

    rows = []
    for method, g in df.groupby("method_short"):
        n = g["n"].sum()
        correct = g["correct"].sum()
        acc = 100 * correct / n

        token_g = g.dropna(subset=["avg_total_tokens"])
        if len(token_g) > 0:
            avg_tok = (token_g["avg_total_tokens"] * token_g["n"]).sum() / token_g["n"].sum()
        else:
            avg_tok = None

        rows.append({
            "method": method,
            "n": n,
            "correct": correct,
            "accuracy_percent": acc,
            "avg_total_tokens": avg_tok,
        })

    out = pd.DataFrame(rows)

    # If SCRD token is missing in the long CSV, fill it from sample_level_results.jsonl
    if "SCRD" in set(out["method"]):
        scrd_idx = out.index[out["method"] == "SCRD"][0]
        if pd.isna(out.loc[scrd_idx, "avg_total_tokens"]) and scrd_jsonl_path is not None:
            scrd_rows = load_jsonl(scrd_jsonl_path)
            scrd_tokens = [float(r["scrd_total_tokens"]) for r in scrd_rows if r.get("scrd_total_tokens") is not None]
            if scrd_tokens:
                out.loc[scrd_idx, "avg_total_tokens"] = sum(scrd_tokens) / len(scrd_tokens)

    return out


def build_tradeoff_from_wide_csv(root: Path) -> pd.DataFrame:
    """
    Fallback input:
    paper_main_comparison_table_200_with_scrd_tokens.csv

    Expected Overall row with columns like:
    Single Acc/Tok, MV@R1 Acc/Tok, MV@R3 Acc/Tok, MV@R5 Acc/Tok, SCRD Acc/Tok
    """
    csv_path = find_existing_file(root, [
        "paper_main_comparison_table_200_with_scrd_tokens.csv",
        "analysis_outputs/paper_main_comparison_table_200_with_scrd_tokens.csv",
        "analysis_outputs/comparison/paper_main_comparison_table_200_with_scrd_tokens.csv",
    ])

    df = pd.read_csv(csv_path)

    dataset_col = None
    for c in df.columns:
        if c.lower() in {"dataset", "指标"}:
            dataset_col = c
            break
    if dataset_col is None:
        raise ValueError(f"Cannot identify dataset column in {csv_path}")

    overall = df[df[dataset_col].astype(str).str.lower().eq("overall")]
    if overall.empty:
        raise ValueError(f"No Overall row found in {csv_path}")

    overall = overall.iloc[0]

    col_map = {
        "Single Acc/Tok": "Single",
        "MV@R1 Acc/Tok": "MV@R1",
        "MV@R3 Acc/Tok": "MV@R3",
        "MV@R5 Acc/Tok": "MV@R5",
        "SCRD Acc/Tok": "SCRD",
    }

    rows = []
    for col, method in col_map.items():
        if col not in df.columns:
            continue
        raw = str(overall[col])
        if "/" not in raw:
            continue
        acc_str, tok_str = raw.split("/", 1)
        rows.append({
            "method": method,
            "accuracy_percent": float(acc_str.strip()),
            "avg_total_tokens": float(tok_str.strip().replace(",", "")),
            "n": None,
            "correct": None,
        })

    if not rows:
        raise ValueError(f"No Acc/Tok columns parsed from {csv_path}")

    return pd.DataFrame(rows)


def build_tradeoff_table(root: Path) -> pd.DataFrame:
    scrd_jsonl_path = None
    possible_scrd = root / "outputs_with_last_round_majority_vote" / "sample_level_results.jsonl"
    if possible_scrd.exists():
        scrd_jsonl_path = possible_scrd

    try:
        df = build_tradeoff_from_long_csv(root, scrd_jsonl_path)
    except FileNotFoundError:
        df = build_tradeoff_from_wide_csv(root)

    order = ["Single", "MV@R1", "MV@R3", "MV@R5", "SCRD"]
    df["order"] = df["method"].map({m: i for i, m in enumerate(order)})
    df = df.sort_values("order").drop(columns=["order"])
    return df


def plot_accuracy_token_tradeoff(tradeoff_df: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 4.6))

    ax.scatter(
        tradeoff_df["avg_total_tokens"],
        tradeoff_df["accuracy_percent"],
        s=80,
    )

    for _, row in tradeoff_df.iterrows():
        ax.annotate(
            row["method"],
            (row["avg_total_tokens"], row["accuracy_percent"]),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=10,
        )

    ax.set_xlabel("Average total tokens per sample")
    ax.set_ylabel("Overall accuracy (%)")
    ax.set_title("Accuracy–Token Trade-off")
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.6)

    save_fig(fig, out_dir, "fig1_accuracy_token_tradeoff")
    plt.close(fig)


# ============================================================
# Figure 2: Sensitivity Token Cost
# ============================================================

def read_sensitivity_full_tables(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    max_path = find_existing_file(root, [
        "sensitivity_max_round_table_full.csv",
        "analysis_outputs/sensitivity/sensitivity_max_round_table_full.csv",
    ])
    rollback_path = find_existing_file(root, [
        "sensitivity_rollback_table_full.csv",
        "analysis_outputs/sensitivity/sensitivity_rollback_table_full.csv",
    ])

    max_df = pd.read_csv(max_path)
    rollback_df = pd.read_csv(rollback_path)

    return max_df, rollback_df


def get_col(df: pd.DataFrame, candidates: list[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"Cannot find any of these columns: {candidates}. Existing columns: {list(df.columns)}")


def plot_sensitivity_token_cost(max_df: pd.DataFrame, rollback_df: pd.DataFrame, out_dir: Path) -> None:
    setting_col_max = get_col(max_df, ["setting_value", "Max Round", "max_round"])
    token_col_max = get_col(max_df, ["avg_total_tokens", "Avg Total Tokens"])

    setting_col_rb = get_col(rollback_df, ["setting_value", "Rollback Limit", "rollback_limit"])
    token_col_rb = get_col(rollback_df, ["avg_total_tokens", "Avg Total Tokens"])

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))

    axes[0].plot(
        max_df[setting_col_max],
        max_df[token_col_max],
        marker="o",
        linewidth=2,
    )
    axes[0].set_xlabel("Max round")
    axes[0].set_ylabel("Average total tokens")
    axes[0].set_title("(a) Max round")
    axes[0].grid(True, linestyle="--", linewidth=0.6, alpha=0.6)

    axes[1].plot(
        rollback_df[setting_col_rb],
        rollback_df[token_col_rb],
        marker="o",
        linewidth=2,
    )
    axes[1].set_xlabel("Rollback limit")
    axes[1].set_ylabel("Average total tokens")
    axes[1].set_title("(b) Rollback limit")
    axes[1].grid(True, linestyle="--", linewidth=0.6, alpha=0.6)

    fig.suptitle("Token Cost under Sensitivity Settings")
    fig.tight_layout()

    save_fig(fig, out_dir, "fig2_sensitivity_token_cost")
    plt.close(fig)


# ============================================================
# Figure 3: Rollback Paired Outcome Analysis
# ============================================================

def build_rollback_paired_table(root: Path) -> pd.DataFrame:
    path = find_existing_file(root, [
        "outputs/ablation/wo_rollback_v2/sample_level_results.jsonl",
        "outputs/ablation/wo_rollback_v2/gsm8k/sample_level_results.jsonl",
        "wo_rollback_v2_sample_level_results.jsonl",
    ])

    rows = load_jsonl(path)

    records = []

    for ds in ROLLBACK_DATASETS:
        ds_rows = [r for r in rows if r.get("dataset_name") == ds]

        both_correct = 0
        both_wrong = 0
        rollback_gain = 0
        rollback_loss = 0

        for r in ds_rows:
            if "reference_full_scrd_correct" in r:
                full_ok = bool(r["reference_full_scrd_correct"])
            elif "last_round_majority_correct" in r:
                full_ok = bool(r["last_round_majority_correct"])
            else:
                full_ok = bool(r.get("scrd_correct", False))

            wo_ok = bool(r.get("wo_rollback_correct", False))

            if full_ok and wo_ok:
                both_correct += 1
            elif (not full_ok) and (not wo_ok):
                both_wrong += 1
            elif full_ok and (not wo_ok):
                rollback_gain += 1
            elif (not full_ok) and wo_ok:
                rollback_loss += 1

        n = len(ds_rows)
        records.append({
            "dataset": ds,
            "n": n,
            "both_correct": both_correct,
            "both_wrong": both_wrong,
            "rollback_gain": rollback_gain,
            "rollback_loss": rollback_loss,
            "net_gain": rollback_gain - rollback_loss,
        })

    overall = {
        "dataset": "overall",
        "n": sum(r["n"] for r in records),
        "both_correct": sum(r["both_correct"] for r in records),
        "both_wrong": sum(r["both_wrong"] for r in records),
        "rollback_gain": sum(r["rollback_gain"] for r in records),
        "rollback_loss": sum(r["rollback_loss"] for r in records),
    }
    overall["net_gain"] = overall["rollback_gain"] - overall["rollback_loss"]
    records.append(overall)

    return pd.DataFrame(records)


def plot_rollback_paired_outcome(rb_df: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.6))

    x = range(len(rb_df))
    labels = rb_df["dataset"].tolist()

    ax.bar(
        x,
        rb_df["rollback_gain"],
        label="Rollback gain: Full correct, w/o rollback wrong",
    )
    ax.bar(
        x,
        -rb_df["rollback_loss"],
        label="Rollback loss: Full wrong, w/o rollback correct",
    )

    for i, row in rb_df.iterrows():
        ax.annotate(
            f"net={int(row['net_gain'])}",
            (i, row["rollback_gain"] if row["net_gain"] >= 0 else -row["rollback_loss"]),
            textcoords="offset points",
            xytext=(0, 8 if row["net_gain"] >= 0 else -14),
            ha="center",
            fontsize=9,
        )

    ax.axhline(0, linewidth=1)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Number of samples")
    ax.set_title("Rollback Paired Outcome Analysis")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.6)

    save_fig(fig, out_dir, "fig3_rollback_paired_outcome")
    plt.close(fig)


# ============================================================
# Figure 4: First-round MV -> Final SCRD Transition
# ============================================================

def build_first_to_final_transition(root: Path) -> pd.DataFrame:
    path = find_existing_file(root, [
        "outputs_with_last_round_majority_vote/sample_level_results.jsonl",
        "sample_level_results.jsonl",
    ])

    rows = load_jsonl(path)

    records = []
    for ds in DATASETS:
        ds_rows = [r for r in rows if r.get("dataset_name") == ds]

        cc = 0  # first correct -> final correct
        cw = 0  # first correct -> final wrong
        wc = 0  # first wrong -> final correct
        ww = 0  # first wrong -> final wrong

        for r in ds_rows:
            if "majority_voting_correct" in r:
                first_ok = bool(r["majority_voting_correct"])
            else:
                first_ok = is_correct_math_style(
                    r.get("majority_voting_baseline_answer"),
                    r.get("gold_answer"),
                )

            if "last_round_majority_correct" in r:
                final_ok = bool(r["last_round_majority_correct"])
            else:
                final_ok = bool(r.get("scrd_correct", False))

            if first_ok and final_ok:
                cc += 1
            elif first_ok and (not final_ok):
                cw += 1
            elif (not first_ok) and final_ok:
                wc += 1
            elif (not first_ok) and (not final_ok):
                ww += 1

        n = len(ds_rows)
        records.append({
            "dataset": ds,
            "n": n,
            "C_to_C": cc,
            "C_to_W": cw,
            "W_to_C": wc,
            "W_to_W": ww,
            "net_correction": wc - cw,
        })

    overall = {
        "dataset": "overall",
        "n": sum(r["n"] for r in records),
        "C_to_C": sum(r["C_to_C"] for r in records),
        "C_to_W": sum(r["C_to_W"] for r in records),
        "W_to_C": sum(r["W_to_C"] for r in records),
        "W_to_W": sum(r["W_to_W"] for r in records),
    }
    overall["net_correction"] = overall["W_to_C"] - overall["C_to_W"]
    records.append(overall)

    return pd.DataFrame(records)


def plot_first_to_final_transition(trans_df: pd.DataFrame, out_dir: Path) -> None:
    """
    Stacked bar chart by dataset.
    Shows how samples move from first-round majority vote to final SCRD.
    """
    plot_df = trans_df.copy()

    categories = ["C_to_C", "W_to_C", "C_to_W", "W_to_W"]
    labels = {
        "C_to_C": "C→C",
        "W_to_C": "W→C",
        "C_to_W": "C→W",
        "W_to_W": "W→W",
    }

    x = range(len(plot_df))
    bottom = [0] * len(plot_df)

    fig, ax = plt.subplots(figsize=(9.5, 4.8))

    for cat in categories:
        values = plot_df[cat].tolist()
        ax.bar(
            x,
            values,
            bottom=bottom,
            label=labels[cat],
        )
        bottom = [b + v for b, v in zip(bottom, values)]

    for i, row in plot_df.iterrows():
        ax.annotate(
            f"net={int(row['net_correction'])}",
            (i, row["n"]),
            textcoords="offset points",
            xytext=(0, 5),
            ha="center",
            fontsize=8,
        )

    ax.set_xticks(list(x))
    ax.set_xticklabels(plot_df["dataset"].tolist(), rotation=25, ha="right")
    ax.set_ylabel("Number of samples")
    ax.set_title("Correctness Transition from First-round MV to Final SCRD")
    ax.legend(title="Transition", fontsize=8, title_fontsize=9, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    ax.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.6)

    fig.tight_layout()

    save_fig(fig, out_dir, "fig4_first_round_to_final_transition")
    plt.close(fig)


# ============================================================
# Save source data for checking
# ============================================================

def save_source_tables(
    out_dir: Path,
    tradeoff_df: pd.DataFrame,
    max_df: pd.DataFrame,
    rollback_sens_df: pd.DataFrame,
    rollback_paired_df: pd.DataFrame,
    transition_df: pd.DataFrame,
) -> None:
    tradeoff_df.to_csv(out_dir / "source_fig1_accuracy_token_tradeoff.csv", index=False, encoding="utf-8-sig")
    max_df.to_csv(out_dir / "source_fig2_max_round_token_cost.csv", index=False, encoding="utf-8-sig")
    rollback_sens_df.to_csv(out_dir / "source_fig2_rollback_token_cost.csv", index=False, encoding="utf-8-sig")
    rollback_paired_df.to_csv(out_dir / "source_fig3_rollback_paired_outcome.csv", index=False, encoding="utf-8-sig")
    transition_df.to_csv(out_dir / "source_fig4_first_to_final_transition.csv", index=False, encoding="utf-8-sig")


# ============================================================
# Main
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=str,
        default=".",
        help="Repository root directory.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="analysis_outputs/figures",
        help="Output directory for figures.",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    out_dir = root / args.out_dir
    ensure_out_dir(out_dir)

    print("Building Figure 1 data...")
    tradeoff_df = build_tradeoff_table(root)
    print(tradeoff_df)

    print("\nBuilding Figure 2 data...")
    max_df, rollback_sens_df = read_sensitivity_full_tables(root)

    print("\nBuilding Figure 3 data...")
    rollback_paired_df = build_rollback_paired_table(root)
    print(rollback_paired_df)

    print("\nBuilding Figure 4 data...")
    transition_df = build_first_to_final_transition(root)
    print(transition_df)

    print("\nPlotting figures...")
    plot_accuracy_token_tradeoff(tradeoff_df, out_dir)
    plot_sensitivity_token_cost(max_df, rollback_sens_df, out_dir)
    plot_rollback_paired_outcome(rollback_paired_df, out_dir)
    plot_first_to_final_transition(transition_df, out_dir)

    save_source_tables(
        out_dir=out_dir,
        tradeoff_df=tradeoff_df,
        max_df=max_df,
        rollback_sens_df=rollback_sens_df,
        rollback_paired_df=rollback_paired_df,
        transition_df=transition_df,
    )

    print("\nDone.")
    print(f"Figures and source tables saved to: {out_dir}")


if __name__ == "__main__":
    main()