from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Config
# ============================================================

DATASETS = ["addsub", "asdiv", "gsm8k", "math", "multiarith", "singleeq", "svamp"]
ROLLBACK_DATASETS = ["math", "gsm8k", "multiarith"]

METHOD_ORDER = ["Single", "MV@R1", "MV@R3", "MV@R5", "SCRD"]

METHOD_MAP = {
    "Single Agent R1-A": "Single",
    "MV@Round1": "MV@R1",
    "MV@Round3": "MV@R3",
    "MV@Round5": "MV@R5",
    "SCRD Last-Round MV": "SCRD",
}

COLORS = {
    "scrd": "#C00000",
    "frontier": "#1F77B4",
    "dominated": "#9E9E9E",
    "bar_main": "#4C78A8",
    "bar_alt": "#F58518",
    "line": "#C00000",
    "positive": "#2CA02C",
    "negative": "#D62728",
    "light_blue": "#DCEAF7",
    "light_red": "#F9D6D5",
    "grid": "#BFBFBF",
}


# ============================================================
# IO helpers
# ============================================================

def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    rows = []
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


def find_existing(root: Path, candidates: list[str]) -> Path:
    for rel in candidates:
        path = root / rel
        if path.exists():
            return path

    msg = "\n".join(str(root / c) for c in candidates)
    raise FileNotFoundError(f"Cannot find any of these files:\n{msg}")


def save_fig(fig: plt.Figure, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    png = out_dir / f"{name}.png"
    pdf = out_dir / f"{name}.pdf"

    fig.savefig(png, dpi=450, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")

    print(f"[saved] {png}")
    print(f"[saved] {pdf}")


def set_paper_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "figure.titlesize": 14,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.linestyle": "--",
        "grid.linewidth": 0.6,
        "grid.alpha": 0.45,
    })


# ============================================================
# Data builders
# ============================================================

def build_tradeoff_table(root: Path) -> pd.DataFrame:
    """
    Sources:
    - comparison_experiments_anlysis/comparison_200_sample_method_level.csv
    - outputs_with_last_round_majority_vote/sample_level_results.jsonl

    The comparison CSV may not contain SCRD token, so SCRD token is filled from
    sample_level_results.jsonl.
    """
    comparison_path = find_existing(root, [
        "comparison_experiments_anlysis/comparison_200_sample_method_level.csv",
        "comparison_experiment_analysis/comparison_200_sample_method_level.csv",
        "analysis_outputs/comparison_200_sample_method_level.csv",
    ])

    scrd_sample_path = find_existing(root, [
        "outputs_with_last_round_majority_vote/sample_level_results.jsonl",
    ])

    df = pd.read_csv(comparison_path)
    df["method_short"] = df["method"].map(METHOD_MAP)

    records = []

    for method in METHOD_ORDER:
        g = df[df["method_short"] == method].copy()

        if g.empty:
            raise ValueError(f"Method not found in comparison CSV: {method}")

        n = int(g["n"].sum())
        correct = int(g["correct"].sum())
        acc = 100 * correct / n

        token_g = g.dropna(subset=["avg_total_tokens"])
        if len(token_g):
            avg_tokens = float((token_g["avg_total_tokens"] * token_g["n"]).sum() / token_g["n"].sum())
        else:
            avg_tokens = np.nan

        records.append({
            "method": method,
            "n": n,
            "correct": correct,
            "accuracy_percent": acc,
            "avg_total_tokens": avg_tokens,
        })

    out = pd.DataFrame(records)

    # Fill SCRD token from sample-level results.
    scrd_rows = load_jsonl(scrd_sample_path)
    scrd_tokens = [
        float(r["scrd_total_tokens"])
        for r in scrd_rows
        if r.get("scrd_total_tokens") is not None
    ]
    if scrd_tokens:
        out.loc[out["method"] == "SCRD", "avg_total_tokens"] = sum(scrd_tokens) / len(scrd_tokens)

    return out


def build_sensitivity_tables(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    max_path = find_existing(root, [
        "analysis_outputs/sensitivity/sensitivity_max_round_table_full.csv",
        "sensitivity_max_round_table_full.csv",
    ])

    rb_path = find_existing(root, [
        "analysis_outputs/sensitivity/sensitivity_rollback_table_full.csv",
        "sensitivity_rollback_table_full.csv",
    ])

    max_df = pd.read_csv(max_path).sort_values("setting_value")
    rb_df = pd.read_csv(rb_path).sort_values("setting_value")

    return max_df, rb_df


def build_rollback_effect_table(root: Path) -> pd.DataFrame:
    """
    Prefer corrected v3 results.

    Required fields:
    - reference_full_scrd_correct
    - wo_rollback_correct
    - rerun_required
    """
    sample_path = find_existing(root, [
        "outputs/ablation/wo_rollback_v3/sample_level_results.jsonl",
        "outputs/ablation/wo_rollback_v2/sample_level_results.jsonl",
    ])

    rows = load_jsonl(sample_path)

    records = []

    for ds in ROLLBACK_DATASETS:
        ds_rows = [r for r in rows if r.get("dataset_name") == ds]
        if not ds_rows:
            raise ValueError(f"No rows found for dataset={ds} in {sample_path}")

        triggered_rows = [r for r in ds_rows if bool(r.get("rerun_required"))]

        full_correct = sum(bool(r.get("reference_full_scrd_correct")) for r in ds_rows)
        wo_correct = sum(bool(r.get("wo_rollback_correct")) for r in ds_rows)

        trig_full_correct = sum(bool(r.get("reference_full_scrd_correct")) for r in triggered_rows)
        trig_wo_correct = sum(bool(r.get("wo_rollback_correct")) for r in triggered_rows)

        records.append({
            "dataset": ds,
            "n": len(ds_rows),
            "triggered_n": len(triggered_rows),

            "full_overall_correct": full_correct,
            "wo_overall_correct": wo_correct,
            "full_overall_acc": 100 * full_correct / len(ds_rows),
            "wo_overall_acc": 100 * wo_correct / len(ds_rows),

            "full_triggered_correct": trig_full_correct,
            "wo_triggered_correct": trig_wo_correct,
            "full_triggered_acc": 100 * trig_full_correct / len(triggered_rows) if triggered_rows else np.nan,
            "wo_triggered_acc": 100 * trig_wo_correct / len(triggered_rows) if triggered_rows else np.nan,

            "overall_gain_pp": 100 * (full_correct - wo_correct) / len(ds_rows),
            "triggered_gain_pp": (
                100 * (trig_full_correct - trig_wo_correct) / len(triggered_rows)
                if triggered_rows else np.nan
            ),
        })

    total_n = sum(r["n"] for r in records)
    total_triggered_n = sum(r["triggered_n"] for r in records)

    total_full = sum(r["full_overall_correct"] for r in records)
    total_wo = sum(r["wo_overall_correct"] for r in records)
    total_trig_full = sum(r["full_triggered_correct"] for r in records)
    total_trig_wo = sum(r["wo_triggered_correct"] for r in records)

    records.append({
        "dataset": "overall",
        "n": total_n,
        "triggered_n": total_triggered_n,

        "full_overall_correct": total_full,
        "wo_overall_correct": total_wo,
        "full_overall_acc": 100 * total_full / total_n,
        "wo_overall_acc": 100 * total_wo / total_n,

        "full_triggered_correct": total_trig_full,
        "wo_triggered_correct": total_trig_wo,
        "full_triggered_acc": 100 * total_trig_full / total_triggered_n,
        "wo_triggered_acc": 100 * total_trig_wo / total_triggered_n,

        "overall_gain_pp": 100 * (total_full - total_wo) / total_n,
        "triggered_gain_pp": 100 * (total_trig_full - total_trig_wo) / total_triggered_n,
    })

    out = pd.DataFrame(records)

    # Check whether the file is corrected.
    if "reference_correct_source" in rows[0]:
        bad_sources = [
            r.get("reference_correct_source")
            for r in rows
            if str(r.get("reference_correct_source", "")).startswith("FALLBACK")
        ]
        if bad_sources:
            print("[warning] Some rows use fallback reference fields. Please verify reference correctness.")

    return out


def build_correction_degradation_table(root: Path) -> pd.DataFrame:
    """
    Compare first-round majority vote vs final SCRD last-round majority vote.

    We only keep:
    - W->C: first wrong, final correct
    - C->W: first correct, final wrong
    """
    sample_path = find_existing(root, [
        "outputs_with_last_round_majority_vote/sample_level_results.jsonl",
    ])

    rows = load_jsonl(sample_path)

    records = []

    for ds in DATASETS:
        ds_rows = [r for r in rows if r.get("dataset_name") == ds]
        if not ds_rows:
            raise ValueError(f"No rows found for dataset={ds} in {sample_path}")

        w_to_c = 0
        c_to_w = 0
        c_to_c = 0
        w_to_w = 0

        for r in ds_rows:
            first_ok = bool(r.get("majority_voting_correct"))
            final_ok = bool(r.get("last_round_majority_correct"))

            if first_ok and final_ok:
                c_to_c += 1
            elif first_ok and not final_ok:
                c_to_w += 1
            elif not first_ok and final_ok:
                w_to_c += 1
            else:
                w_to_w += 1

        n = len(ds_rows)

        records.append({
            "dataset": ds,
            "n": n,
            "W_to_C": w_to_c,
            "C_to_W": c_to_w,
            "C_to_C": c_to_c,
            "W_to_W": w_to_w,
            "W_to_C_rate": 100 * w_to_c / n,
            "C_to_W_rate": 100 * c_to_w / n,
            "net_correction": w_to_c - c_to_w,
            "net_correction_rate": 100 * (w_to_c - c_to_w) / n,
        })

    total_n = sum(r["n"] for r in records)
    total_w_to_c = sum(r["W_to_C"] for r in records)
    total_c_to_w = sum(r["C_to_W"] for r in records)
    total_c_to_c = sum(r["C_to_C"] for r in records)
    total_w_to_w = sum(r["W_to_W"] for r in records)

    records.append({
        "dataset": "overall",
        "n": total_n,
        "W_to_C": total_w_to_c,
        "C_to_W": total_c_to_w,
        "C_to_C": total_c_to_c,
        "W_to_W": total_w_to_w,
        "W_to_C_rate": 100 * total_w_to_c / total_n,
        "C_to_W_rate": 100 * total_c_to_w / total_n,
        "net_correction": total_w_to_c - total_c_to_w,
        "net_correction_rate": 100 * (total_w_to_c - total_c_to_w) / total_n,
    })

    return pd.DataFrame(records)


# ============================================================
# Figure 1: Efficiency frontier
# ============================================================

def is_dominated(df: pd.DataFrame, method: str) -> bool:
    row = df[df["method"] == method].iloc[0]
    for _, other in df.iterrows():
        if other["method"] == method:
            continue
        if (
            other["avg_total_tokens"] <= row["avg_total_tokens"]
            and other["accuracy_percent"] >= row["accuracy_percent"]
            and (
                other["avg_total_tokens"] < row["avg_total_tokens"]
                or other["accuracy_percent"] > row["accuracy_percent"]
            )
        ):
            return True
    return False


def plot_efficiency_frontier(df: pd.DataFrame, out_dir: Path) -> None:
    df = df.copy()
    df["dominated"] = df["method"].apply(lambda m: is_dominated(df, m))

    frontier = df[~df["dominated"]].sort_values("avg_total_tokens")
    dominated = df[df["dominated"]]

    fig, ax = plt.subplots(figsize=(7.2, 4.9))

    # Dominated methods: grey, hollow.
    if not dominated.empty:
        ax.scatter(
            dominated["avg_total_tokens"],
            dominated["accuracy_percent"],
            s=95,
            facecolors="white",
            edgecolors=COLORS["dominated"],
            linewidths=2.0,
            label="Dominated vanilla setting",
            zorder=3,
        )

    # Frontier methods.
    ax.scatter(
        frontier["avg_total_tokens"],
        frontier["accuracy_percent"],
        s=115,
        color=COLORS["frontier"],
        label="Efficiency frontier",
        zorder=4,
    )

    # SCRD highlight.
    scrd = df[df["method"] == "SCRD"].iloc[0]
    ax.scatter(
        [scrd["avg_total_tokens"]],
        [scrd["accuracy_percent"]],
        s=155,
        color=COLORS["scrd"],
        edgecolors="black",
        linewidths=0.8,
        label="SCRD",
        zorder=5,
    )

    # Frontier line.
    ax.plot(
        frontier["avg_total_tokens"],
        frontier["accuracy_percent"],
        color=COLORS["frontier"],
        linewidth=2.0,
        alpha=0.75,
        zorder=2,
    )

    # Labels.
    offsets = {
        "Single": (8, -10),
        "MV@R1": (8, 6),
        "MV@R3": (8, 6),
        "MV@R5": (8, -14),
        "SCRD": (8, 8),
    }

    for _, r in df.iterrows():
        dx, dy = offsets.get(r["method"], (6, 6))
        ax.annotate(
            r["method"],
            (r["avg_total_tokens"], r["accuracy_percent"]),
            textcoords="offset points",
            xytext=(dx, dy),
            fontsize=10,
            weight="bold" if r["method"] == "SCRD" else "normal",
        )

    # Show dominance relation for vanilla multi-round.
    ax.annotate(
        "More tokens but lower accuracy\nthan earlier vanilla baselines",
        xy=(df.loc[df["method"] == "MV@R5", "avg_total_tokens"].iloc[0],
            df.loc[df["method"] == "MV@R5", "accuracy_percent"].iloc[0]),
        xytext=(3800, 70.3),
        arrowprops=dict(arrowstyle="->", linewidth=1.2, color="black"),
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#AAAAAA", alpha=0.9),
    )

    ax.set_xscale("log")
    ax.set_xlabel("Average total tokens per sample, log scale")
    ax.set_ylabel("Overall accuracy (%)")
    ax.set_title("Accuracy–Cost Efficiency Frontier")

    ax.set_ylim(66.5, 82.4)
    ax.legend(loc="lower left", frameon=True)
    ax.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.45)

    fig.tight_layout()
    save_fig(fig, out_dir, "fig1_accuracy_cost_efficiency_frontier")
    plt.close(fig)


# ============================================================
# Figure 2: Sensitivity accuracy + token
# ============================================================

def plot_sensitivity_accuracy_token(max_df: pd.DataFrame, rb_df: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.6))

    panels = [
        (axes[0], max_df, "Max round", "(a) Maximum reasoning rounds", 5),
        (axes[1], rb_df, "Rollback limit", "(b) Rollback limit", 1),
    ]

    for ax, df, xlabel, title, best_value in panels:
        x = np.arange(len(df))
        settings = df["setting_value"].astype(int).tolist()
        acc = df["accuracy_percent"].astype(float).tolist()
        tokens = df["avg_total_tokens"].astype(float).tolist()

        bar_colors = [
            COLORS["scrd"] if s == best_value else COLORS["bar_main"]
            for s in settings
        ]

        bars = ax.bar(
            x,
            acc,
            color=bar_colors,
            width=0.58,
            alpha=0.9,
            label="Accuracy",
            zorder=3,
        )

        ax2 = ax.twinx()
        ax2.plot(
            x,
            tokens,
            color=COLORS["line"],
            marker="o",
            linewidth=2.2,
            markersize=6,
            label="Avg tokens",
            zorder=4,
        )

        for i, (b, a) in enumerate(zip(bars, acc)):
            ax.text(
                b.get_x() + b.get_width() / 2,
                b.get_height() + 0.7,
                f"{a:.1f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        for i, t in enumerate(tokens):
            ax2.text(
                i,
                t + max(tokens) * 0.025,
                f"{t/1000:.1f}k",
                ha="center",
                va="bottom",
                fontsize=9,
                color=COLORS["line"],
            )

        ax.set_xticks(x)
        ax.set_xticklabels(settings)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Accuracy (%)")
        ax2.set_ylabel("Average total tokens")

        ax.set_title(title)

        # Accuracy y-limits set to make differences visible but not deceptive.
        ymin = max(0, min(acc) - 6)
        ymax = max(acc) + 7
        ax.set_ylim(ymin, ymax)

        ax2.set_ylim(0, max(tokens) * 1.22)

        ax.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.45)
        ax2.grid(False)

        # Merge legends.
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc="lower right", frameon=True, fontsize=8)

        ax.axvline(
            settings.index(best_value),
            color="#333333",
            linestyle=":",
            linewidth=1.0,
            alpha=0.7,
        )

    fig.suptitle("Sensitivity Analysis: Accuracy and Token Cost", y=1.02)
    fig.tight_layout()
    save_fig(fig, out_dir, "fig2_sensitivity_accuracy_token")
    plt.close(fig)


# ============================================================
# Figure 3: Rollback effect
# ============================================================

def plot_rollback_effect(df: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8), sharey=False)

    labels = df["dataset"].tolist()
    x = np.arange(len(labels))
    width = 0.34

    # Panel A: overall.
    ax = axes[0]
    ax.bar(
        x - width / 2,
        df["wo_overall_acc"],
        width,
        color=COLORS["bar_alt"],
        label="w/o Rollback",
    )
    ax.bar(
        x + width / 2,
        df["full_overall_acc"],
        width,
        color=COLORS["scrd"],
        label="Full SCRD",
    )

    for i, r in df.iterrows():
        gain = r["overall_gain_pp"]
        ax.text(
            i,
            max(r["wo_overall_acc"], r["full_overall_acc"]) + 1.0,
            f"{gain:+.1f} pp",
            ha="center",
            va="bottom",
            fontsize=9,
            color=COLORS["positive"] if gain >= 0 else COLORS["negative"],
            weight="bold",
        )

    ax.set_title("(a) Overall samples")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(45, 95)
    ax.legend(frameon=True, loc="lower right")
    ax.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.45)

    # Panel B: triggered subset.
    ax = axes[1]
    ax.bar(
        x - width / 2,
        df["wo_triggered_acc"],
        width,
        color=COLORS["bar_alt"],
        label="w/o Rollback",
    )
    ax.bar(
        x + width / 2,
        df["full_triggered_acc"],
        width,
        color=COLORS["scrd"],
        label="Full SCRD",
    )

    for i, r in df.iterrows():
        gain = r["triggered_gain_pp"]
        ax.text(
            i,
            max(r["wo_triggered_acc"], r["full_triggered_acc"]) + 2.0,
            f"{gain:+.1f} pp\nn={int(r['triggered_n'])}",
            ha="center",
            va="bottom",
            fontsize=9,
            color=COLORS["positive"] if gain >= 0 else COLORS["negative"],
            weight="bold",
        )

    ax.set_title("(b) Rollback-triggered subset")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(20, 90)
    ax.legend(frameon=True, loc="lower right")
    ax.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.45)

    fig.suptitle("Effect of Rollback: Overall vs. Triggered Samples", y=1.02)
    fig.tight_layout()
    save_fig(fig, out_dir, "fig3_rollback_effect_overall_triggered")
    plt.close(fig)


# ============================================================
# Figure 4: Correction vs degradation
# ============================================================

def plot_correction_degradation(df: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.6, 4.9))

    labels = df["dataset"].tolist()
    x = np.arange(len(labels))

    pos = df["W_to_C_rate"].astype(float).to_numpy()
    neg = -df["C_to_W_rate"].astype(float).to_numpy()

    ax.bar(
        x,
        pos,
        color=COLORS["positive"],
        width=0.62,
        label="Corrected: first-round MV wrong → final SCRD correct",
        zorder=3,
    )
    ax.bar(
        x,
        neg,
        color=COLORS["negative"],
        width=0.62,
        label="Degraded: first-round MV correct → final SCRD wrong",
        zorder=3,
    )

    ax.axhline(0, color="black", linewidth=0.9)

    for i, r in df.iterrows():
        net = r["net_correction_rate"]
        y = pos[i] + 0.45 if net >= 0 else neg[i] - 0.75
        ax.text(
            i,
            y,
            f"net {net:+.1f} pp\n({int(r['net_correction']):+d})",
            ha="center",
            va="bottom" if net >= 0 else "top",
            fontsize=8.5,
            color=COLORS["positive"] if net >= 0 else COLORS["negative"],
            weight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Sample rate (%)")
    ax.set_title("Correction vs. Degradation from First-round MV to Final SCRD")

    ymax = max(pos) + 3
    ymin = min(neg) - 3
    ax.set_ylim(ymin, ymax)

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=1,
        frameon=True,
        fontsize=9,
    )
    ax.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.45)

    fig.tight_layout()
    save_fig(fig, out_dir, "fig4_correction_vs_degradation")
    plt.close(fig)


# ============================================================
# Main
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--out-dir", type=Path, default=Path("analysis_outputs/paper_figures_v2"))
    args = parser.parse_args()

    root = args.root.resolve()
    out_dir = (root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    set_paper_style()

    print("[1/4] Building accuracy-cost frontier...")
    tradeoff_df = build_tradeoff_table(root)
    tradeoff_df.to_csv(out_dir / "source_fig1_accuracy_cost_frontier.csv", index=False, encoding="utf-8-sig")
    print(tradeoff_df)
    plot_efficiency_frontier(tradeoff_df, out_dir)

    print("\n[2/4] Building sensitivity accuracy-token figure...")
    max_df, rb_df = build_sensitivity_tables(root)
    max_df.to_csv(out_dir / "source_fig2_max_round.csv", index=False, encoding="utf-8-sig")
    rb_df.to_csv(out_dir / "source_fig2_rollback_limit.csv", index=False, encoding="utf-8-sig")
    print(max_df[["setting_value", "accuracy_percent", "avg_total_tokens"]])
    print(rb_df[["setting_value", "accuracy_percent", "avg_total_tokens"]])
    plot_sensitivity_accuracy_token(max_df, rb_df, out_dir)

    print("\n[3/4] Building rollback effect figure...")
    rollback_df = build_rollback_effect_table(root)
    rollback_df.to_csv(out_dir / "source_fig3_rollback_effect.csv", index=False, encoding="utf-8-sig")
    print(rollback_df)
    plot_rollback_effect(rollback_df, out_dir)

    print("\n[4/4] Building correction vs degradation figure...")
    transition_df = build_correction_degradation_table(root)
    transition_df.to_csv(out_dir / "source_fig4_correction_degradation.csv", index=False, encoding="utf-8-sig")
    print(transition_df[["dataset", "n", "W_to_C", "C_to_W", "net_correction", "net_correction_rate"]])
    plot_correction_degradation(transition_df, out_dir)

    print("\nDone.")
    print(f"Figures saved to: {out_dir}")


if __name__ == "__main__":
    main()