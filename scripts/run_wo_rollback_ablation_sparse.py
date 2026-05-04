from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any


DEFAULT_DATASETS = ["multiarith"]
ALL_SEVEN_DATASETS = [
    "addsub",
    "asdiv",
    "gsm8k",
    "math",
    "multiarith",
    "singleeq",
    "svamp",
]
AGENT_IDS = ["A", "B", "C"]
AGENT_ROLES: dict[str, str] = {
    "A": "parser",
    "B": "planner",
    "C": "verifier",
}
MAX_ROUND = 5


# Match the main experiment's direct-network setup.
os.environ["HTTP_PROXY"] = ""
os.environ["HTTPS_PROXY"] = ""
os.environ["NO_PROXY"] = "dashscope.aliyuncs.com,localhost,127.0.0.1"


def ensure_repo_imports(repo_root: Path) -> None:
    repo_root = repo_root.resolve()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
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


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def safe_bool(x: Any) -> bool:
    return bool(x) if x is not None else False


def pct(x: float | None) -> str:
    if x is None:
        return "NA"
    return f"{100.0 * x:.2f}%"


def build_reused_row(full_row: dict[str, Any]) -> dict[str, Any]:
    """
    For samples that did not trigger rollback in Full SCRD, w/o rollback is
    identical by definition. Keep a full 80-row table without paying extra API cost.
    """
    return {
        "sample_id": full_row.get("sample_id"),
        "dataset_name": full_row.get("dataset_name") or full_row.get("dataset"),
        "question": full_row.get("question"),
        "gold_answer": full_row.get("gold_answer"),
        "agent_roles": full_row.get("agent_roles", AGENT_ROLES),
        "round_1_answers": full_row.get("round_1_answers"),
        "single_agent_baseline_answer": full_row.get("single_agent_baseline_answer"),
        "majority_voting_baseline_answer": full_row.get("majority_voting_baseline_answer"),
        "wo_rollback_final_answer": full_row.get("scrd_final_answer"),
        "wo_rollback_correct": full_row.get("scrd_correct"),
        "single_agent_correct": full_row.get("single_agent_correct"),
        "majority_voting_correct": full_row.get("majority_voting_correct"),
        "effective_rounds_used": full_row.get("effective_rounds_used"),
        "actual_rounds_executed": full_row.get("actual_rounds_executed"),
        "stop_reason": full_row.get("stop_reason"),
        "wo_rollback_stop_reason": full_row.get("stop_reason"),
        "would_trigger_rollback": False,
        "rerun_required": False,
        "reused_from_full_scrd": True,
        "rollback_suppressed": False,
        "full_scrd_final_answer": full_row.get("scrd_final_answer"),
        "full_scrd_correct": full_row.get("scrd_correct"),
        "answer_changed_vs_full_scrd": False,
        "wo_rollback_total_tokens": full_row.get("scrd_total_tokens"),
        "full_scrd_total_tokens": full_row.get("scrd_total_tokens"),
        "token_delta_wo_minus_full": 0,
        "single_agent_total_tokens": full_row.get("single_agent_total_tokens"),
        "majority_vote_total_tokens": full_row.get("majority_vote_total_tokens"),
        "scrd_prompt_tokens": full_row.get("scrd_prompt_tokens"),
        "scrd_completion_tokens": full_row.get("scrd_completion_tokens"),
        "agent_total_tokens": full_row.get("agent_total_tokens"),
        "recorder_total_tokens": full_row.get("recorder_total_tokens"),
        "evaluator_total_tokens": full_row.get("evaluator_total_tokens"),
        "repair_brief_total_tokens": 0,
        "repair_evaluator_total_tokens": 0,
        "repair_agent_total_tokens": 0,
    }


def run_normal_mode_without_rollback(
    *,
    llm_client: Any,
    question: str,
    gold_answer: str,
    sample_id: str,
    dataset_name: str,
    decision_rollback_context_policy: str = "none",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Re-run normal SCRD, but suppress rollback/repair.

    Important definition:
    - evaluator/action mapper are still active;
    - if the orchestrator detects rollback, we DO NOT call repair mode;
    - we directly finalize the current normal trajectory.

    decision_rollback_context_policy:
    - "none"      : pass rollback_context=None to decision head. Recommended.
                    This avoids giving post-anchor repair bonuses when no repair happened.
    - "detected"  : pass the detected rollback_context to decision head. Useful as a diagnostic.
    """
    from src.components.action_mapper import ActionMapper
    from src.components.agent_runner import AgentRunner
    from src.components.decision_head import ConservativeTrajectoryDecisionHead
    from src.components.evaluator import Evaluator
    from src.components.history_manager import HistoryManager
    from src.components.recorder import Recorder
    from src.components.rollback_controller import RollbackController
    from src.components.state_store import StateStore
    from src.components.usage_logger import UsageLogger
    from src.pipeline.debate_orchestrator import DebateOrchestrator, DebateOrchestratorConfig
    from src.pipeline.normal_round_executor import NormalRoundExecutor, NormalRoundExecutorConfig
    from src.utils.result_utils import (
        build_trace_bundle,
        build_usage_summary,
        get_actual_rounds_executed,
        get_effective_rounds_used,
        get_round_1_answers,
        get_stop_reason,
        is_correct,
        majority_vote,
    )

    state_store = StateStore()
    usage_logger = UsageLogger()

    agent_runner = AgentRunner(
        llm_client=llm_client,
        usage_logger=usage_logger,
        sample_id=sample_id,
        dataset_name=dataset_name,
        role_by_agent_id=AGENT_ROLES,
    )
    history_manager = HistoryManager()
    recorder = Recorder(
        llm_client=llm_client,
        usage_logger=usage_logger,
        sample_id=sample_id,
    )
    evaluator = Evaluator(
        llm_client=llm_client,
        usage_logger=usage_logger,
        sample_id=sample_id,
    )

    normal_round_executor = NormalRoundExecutor(
        config=NormalRoundExecutorConfig(
            question=question,
            agent_ids=AGENT_IDS,
            max_round=MAX_ROUND,
            sample_id=sample_id,
        ),
        agent_runner=agent_runner,
        state_store=state_store,
        history_manager=history_manager,
        recorder=recorder,
        evaluator=evaluator,
        action_mapper=ActionMapper(),
        rollback_controller=RollbackController(),
    )
    debate_orchestrator = DebateOrchestrator(
        config=DebateOrchestratorConfig(
            question=question,
            agent_ids=AGENT_IDS,
            max_round=MAX_ROUND,
        ),
        state_store=state_store,
        normal_round_executor=normal_round_executor,
    )

    print(f"\n===== w/o rollback: running {sample_id} =====")
    debate_result = debate_orchestrator.run_debate()
    rollback_context = debate_result["rollback_context"]
    early_stopped = debate_result["early_stopped"]
    rollback_suppressed = rollback_context is not None

    if rollback_suppressed:
        state_store.add_event(
            {
                "type": "rollback_suppressed_ablation",
                "mode": "normal",
                "sample_id": sample_id,
                "trigger_round": rollback_context.get("trigger_round"),
                "anchor_round": rollback_context.get("anchor_round"),
                "note": "w/o rollback ablation: repair mode intentionally skipped",
            }
        )
        print(
            "Rollback detected but suppressed. "
            "Skipping repair and finalizing current normal trajectory."
        )

    if decision_rollback_context_policy == "detected":
        decision_context = rollback_context
    elif decision_rollback_context_policy == "none":
        decision_context = None
    else:
        raise ValueError(
            "decision_rollback_context_policy must be either 'none' or 'detected'"
        )

    round_1_answers = get_round_1_answers(state_store)
    single_agent_baseline_answer = round_1_answers[0] if round_1_answers else ""
    majority_voting_baseline_answer = majority_vote(
        round_1_answers,
        dataset_name=dataset_name,
    )

    decision_head = ConservativeTrajectoryDecisionHead()
    wo_rollback_final_answer = decision_head.select_final_answer(
        state_store=state_store,
        rollback_context=decision_context,
        dataset_name=dataset_name,
        agent_roles=AGENT_ROLES,
    )

    usage_summary = build_usage_summary(usage_logger)
    stop_reason = get_stop_reason(rollback_context, early_stopped)
    if rollback_suppressed:
        wo_rollback_stop_reason = "rollback_suppressed_finalize"
    else:
        wo_rollback_stop_reason = stop_reason

    result = {
        "sample_id": sample_id,
        "dataset_name": dataset_name,
        "question": question,
        "gold_answer": gold_answer,
        "agent_roles": AGENT_ROLES,
        "round_1_answers": round_1_answers,
        "single_agent_baseline_answer": single_agent_baseline_answer,
        "majority_voting_baseline_answer": majority_voting_baseline_answer,
        "wo_rollback_final_answer": wo_rollback_final_answer,
        "wo_rollback_correct": is_correct(
            wo_rollback_final_answer,
            gold_answer,
            dataset_name=dataset_name,
        ),
        "single_agent_correct": is_correct(
            single_agent_baseline_answer,
            gold_answer,
            dataset_name=dataset_name,
        ),
        "majority_voting_correct": is_correct(
            majority_voting_baseline_answer,
            gold_answer,
            dataset_name=dataset_name,
        ),
        "effective_rounds_used": get_effective_rounds_used(state_store),
        "actual_rounds_executed": get_actual_rounds_executed(state_store),
        "stop_reason": stop_reason,
        "wo_rollback_stop_reason": wo_rollback_stop_reason,
        "would_trigger_rollback": rollback_suppressed,
        "rerun_required": True,
        "reused_from_full_scrd": False,
        "rollback_suppressed": rollback_suppressed,
        "suppressed_rollback_context": _json_safe_rollback_context(rollback_context),
        "decision_rollback_context_policy": decision_rollback_context_policy,
        "wo_rollback_total_tokens": usage_summary["scrd_total_tokens"],
        "single_agent_total_tokens": usage_summary["single_agent_total_tokens"],
        "majority_vote_total_tokens": usage_summary["majority_vote_total_tokens"],
        "scrd_prompt_tokens": usage_summary["scrd_prompt_tokens"],
        "scrd_completion_tokens": usage_summary["scrd_completion_tokens"],
        "agent_total_tokens": usage_summary["agent_total_tokens"],
        "recorder_total_tokens": usage_summary["recorder_total_tokens"],
        "evaluator_total_tokens": usage_summary["evaluator_total_tokens"],
        "repair_brief_total_tokens": usage_summary["repair_brief_total_tokens"],
        "repair_evaluator_total_tokens": usage_summary["repair_evaluator_total_tokens"],
        "repair_agent_total_tokens": usage_summary["repair_agent_total_tokens"],
    }

    trace = build_trace_bundle(state_store, usage_logger)
    trace["ablation"] = {
        "name": "wo_rollback",
        "definition": "suppress rollback/repair and finalize current normal trajectory",
        "decision_rollback_context_policy": decision_rollback_context_policy,
        "rollback_suppressed": rollback_suppressed,
        "suppressed_rollback_context": _json_safe_rollback_context(rollback_context),
    }
    return result, trace


def _json_safe_rollback_context(rollback_context: dict[str, Any] | None) -> dict[str, Any] | None:
    if not rollback_context:
        return None

    def dump_state(x: Any) -> Any:
        if x is None:
            return None
        if hasattr(x, "model_dump"):
            return x.model_dump()
        return x

    return {
        "trigger_round": rollback_context.get("trigger_round"),
        "anchor_round": rollback_context.get("anchor_round"),
        "anchor_state": dump_state(rollback_context.get("anchor_state")),
        "failed_suffix_state_records": [
            dump_state(s) for s in rollback_context.get("failed_suffix_state_records", [])
        ],
    }


def augment_rerun_row_with_full_comparison(
    row: dict[str, Any],
    full_row: dict[str, Any],
) -> dict[str, Any]:
    row = dict(row)
    row["full_scrd_final_answer"] = full_row.get("scrd_final_answer")
    row["full_scrd_correct"] = full_row.get("scrd_correct")
    row["answer_changed_vs_full_scrd"] = (
        str(row.get("wo_rollback_final_answer")) != str(full_row.get("scrd_final_answer"))
    )
    row["full_scrd_total_tokens"] = full_row.get("scrd_total_tokens")
    row["token_delta_wo_minus_full"] = (
        (row.get("wo_rollback_total_tokens") or 0)
        - (full_row.get("scrd_total_tokens") or 0)
    )
    # Preserve Full SCRD context for sample-level paired analysis.
    row["full_scrd_stop_reason"] = full_row.get("stop_reason")
    row["full_scrd_effective_rounds_used"] = full_row.get("effective_rounds_used")
    row["full_scrd_actual_rounds_executed"] = full_row.get("actual_rounds_executed")
    return row


def summarize_dataset(dataset_name: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    if n == 0:
        return {
            "dataset": dataset_name,
            "n": 0,
        }

    full_correct = sum(1 for r in rows if safe_bool(r.get("full_scrd_correct")))
    wo_correct = sum(1 for r in rows if safe_bool(r.get("wo_rollback_correct")))
    full_wins = sum(
        1
        for r in rows
        if safe_bool(r.get("full_scrd_correct"))
        and not safe_bool(r.get("wo_rollback_correct"))
    )
    ablation_wins = sum(
        1
        for r in rows
        if (not safe_bool(r.get("full_scrd_correct")))
        and safe_bool(r.get("wo_rollback_correct"))
    )
    both_correct = sum(
        1
        for r in rows
        if safe_bool(r.get("full_scrd_correct"))
        and safe_bool(r.get("wo_rollback_correct"))
    )
    both_wrong = sum(
        1
        for r in rows
        if (not safe_bool(r.get("full_scrd_correct")))
        and (not safe_bool(r.get("wo_rollback_correct")))
    )
    changed = sum(1 for r in rows if safe_bool(r.get("answer_changed_vs_full_scrd")))
    rerun = sum(1 for r in rows if safe_bool(r.get("rerun_required")))
    reused = sum(1 for r in rows if safe_bool(r.get("reused_from_full_scrd")))
    suppressed = sum(1 for r in rows if safe_bool(r.get("rollback_suppressed")))

    avg_wo_tokens = sum(float(r.get("wo_rollback_total_tokens") or 0) for r in rows) / n
    avg_full_tokens = sum(float(r.get("full_scrd_total_tokens") or 0) for r in rows) / n
    avg_token_delta = avg_wo_tokens - avg_full_tokens

    rollback_rows = [r for r in rows if safe_bool(r.get("rerun_required"))]
    rn = len(rollback_rows)
    if rn:
        rollback_full_correct = sum(
            1 for r in rollback_rows if safe_bool(r.get("full_scrd_correct"))
        )
        rollback_wo_correct = sum(
            1 for r in rollback_rows if safe_bool(r.get("wo_rollback_correct"))
        )
        rollback_full_acc = rollback_full_correct / rn
        rollback_wo_acc = rollback_wo_correct / rn
        rollback_avg_wo_tokens = sum(
            float(r.get("wo_rollback_total_tokens") or 0) for r in rollback_rows
        ) / rn
        rollback_avg_full_tokens = sum(
            float(r.get("full_scrd_total_tokens") or 0) for r in rollback_rows
        ) / rn
    else:
        rollback_full_acc = None
        rollback_wo_acc = None
        rollback_avg_wo_tokens = None
        rollback_avg_full_tokens = None

    return {
        "dataset": dataset_name,
        "n": n,
        "rerun_count": rerun,
        "reused_count": reused,
        "rollback_suppressed_count": suppressed,
        "full_scrd_correct": full_correct,
        "full_scrd_acc": full_correct / n,
        "wo_rollback_correct": wo_correct,
        "wo_rollback_acc": wo_correct / n,
        "delta_wo_minus_full_pp": 100.0 * ((wo_correct / n) - (full_correct / n)),
        "both_correct": both_correct,
        "both_wrong": both_wrong,
        "full_wins": full_wins,
        "ablation_wins": ablation_wins,
        "net_full_minus_ablation": full_wins - ablation_wins,
        "answer_changed_count": changed,
        "answer_changed_pct": changed / n,
        "avg_full_scrd_tokens": avg_full_tokens,
        "avg_wo_rollback_tokens": avg_wo_tokens,
        "avg_token_delta_wo_minus_full": avg_token_delta,
        "rollback_subset_n": rn,
        "rollback_subset_full_acc": rollback_full_acc,
        "rollback_subset_wo_rollback_acc": rollback_wo_acc,
        "rollback_subset_avg_full_tokens": rollback_avg_full_tokens,
        "rollback_subset_avg_wo_tokens": rollback_avg_wo_tokens,
    }


def add_overall_summary(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [s for s in summaries if int(s.get("n", 0)) > 0]
    n = sum(int(s["n"]) for s in rows)
    if n == 0:
        return {"dataset": "OVERALL", "n": 0}

    def s(key: str) -> float:
        return sum(float(r.get(key) or 0) for r in rows)

    full_correct = s("full_scrd_correct")
    wo_correct = s("wo_rollback_correct")

    avg_full_tokens = sum(
        float(r.get("avg_full_scrd_tokens") or 0) * int(r["n"]) for r in rows
    ) / n
    avg_wo_tokens = sum(
        float(r.get("avg_wo_rollback_tokens") or 0) * int(r["n"]) for r in rows
    ) / n

    rn = int(s("rollback_subset_n"))
    if rn:
        rollback_full_correct = 0.0
        rollback_wo_correct = 0.0
        rollback_full_tokens = 0.0
        rollback_wo_tokens = 0.0
        for r in rows:
            cur_n = int(r.get("rollback_subset_n") or 0)
            if cur_n == 0:
                continue
            rollback_full_correct += float(r.get("rollback_subset_full_acc") or 0) * cur_n
            rollback_wo_correct += float(r.get("rollback_subset_wo_rollback_acc") or 0) * cur_n
            rollback_full_tokens += float(r.get("rollback_subset_avg_full_tokens") or 0) * cur_n
            rollback_wo_tokens += float(r.get("rollback_subset_avg_wo_tokens") or 0) * cur_n
        rollback_full_acc = rollback_full_correct / rn
        rollback_wo_acc = rollback_wo_correct / rn
        rollback_avg_full_tokens = rollback_full_tokens / rn
        rollback_avg_wo_tokens = rollback_wo_tokens / rn
    else:
        rollback_full_acc = None
        rollback_wo_acc = None
        rollback_avg_full_tokens = None
        rollback_avg_wo_tokens = None

    return {
        "dataset": "OVERALL",
        "n": n,
        "rerun_count": int(s("rerun_count")),
        "reused_count": int(s("reused_count")),
        "rollback_suppressed_count": int(s("rollback_suppressed_count")),
        "full_scrd_correct": int(full_correct),
        "full_scrd_acc": full_correct / n,
        "wo_rollback_correct": int(wo_correct),
        "wo_rollback_acc": wo_correct / n,
        "delta_wo_minus_full_pp": 100.0 * ((wo_correct / n) - (full_correct / n)),
        "both_correct": int(s("both_correct")),
        "both_wrong": int(s("both_wrong")),
        "full_wins": int(s("full_wins")),
        "ablation_wins": int(s("ablation_wins")),
        "net_full_minus_ablation": int(s("net_full_minus_ablation")),
        "answer_changed_count": int(s("answer_changed_count")),
        "answer_changed_pct": s("answer_changed_count") / n,
        "avg_full_scrd_tokens": avg_full_tokens,
        "avg_wo_rollback_tokens": avg_wo_tokens,
        "avg_token_delta_wo_minus_full": avg_wo_tokens - avg_full_tokens,
        "rollback_subset_n": rn,
        "rollback_subset_full_acc": rollback_full_acc,
        "rollback_subset_wo_rollback_acc": rollback_wo_acc,
        "rollback_subset_avg_full_tokens": rollback_avg_full_tokens,
        "rollback_subset_avg_wo_tokens": rollback_avg_wo_tokens,
    }


def write_report(path: Path, summaries: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# Ablation: SCRD w/o Rollback / Repair")
    lines.append("")
    lines.append(
        "Definition: keep the normal SCRD evaluator/action mapper active, but when rollback would be triggered, suppress rollback and repair, and directly finalize the current normal trajectory."
    )
    lines.append("")
    lines.append(
        "Efficiency note: early-stop samples are reused from Full SCRD because removing rollback cannot affect samples that never triggered rollback. Only rollback-triggered samples are rerun."
    )
    lines.append("")
    lines.append("## Full dataset summary")
    lines.append("")
    lines.append(
        "| Dataset | N | Rerun | Full Acc | w/o Rollback Acc | Δ Acc | Full wins | Ablation wins | Net Full-Ablation | Avg Full Tok | Avg w/o Tok | Δ Tok |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in summaries:
        lines.append(
            "| {dataset} | {n} | {rerun} | {full_acc} | {wo_acc} | {delta:+.2f} pp | {full_wins} | {abl_wins} | {net} | {full_tok:.1f} | {wo_tok:.1f} | {tok_delta:+.1f} |".format(
                dataset=r.get("dataset"),
                n=int(r.get("n") or 0),
                rerun=int(r.get("rerun_count") or 0),
                full_acc=pct(r.get("full_scrd_acc")),
                wo_acc=pct(r.get("wo_rollback_acc")),
                delta=float(r.get("delta_wo_minus_full_pp") or 0.0),
                full_wins=int(r.get("full_wins") or 0),
                abl_wins=int(r.get("ablation_wins") or 0),
                net=int(r.get("net_full_minus_ablation") or 0),
                full_tok=float(r.get("avg_full_scrd_tokens") or 0.0),
                wo_tok=float(r.get("avg_wo_rollback_tokens") or 0.0),
                tok_delta=float(r.get("avg_token_delta_wo_minus_full") or 0.0),
            )
        )

    lines.append("")
    lines.append("## Rollback-subset summary")
    lines.append("")
    lines.append(
        "| Dataset | Rollback N | Full Acc on rollback subset | w/o Rollback Acc | Full Tok | w/o Tok |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|")
    for r in summaries:
        lines.append(
            "| {dataset} | {rn} | {full_acc} | {wo_acc} | {full_tok} | {wo_tok} |".format(
                dataset=r.get("dataset"),
                rn=int(r.get("rollback_subset_n") or 0),
                full_acc=pct(r.get("rollback_subset_full_acc")),
                wo_acc=pct(r.get("rollback_subset_wo_rollback_acc")),
                full_tok=(
                    f"{float(r.get('rollback_subset_avg_full_tokens')):.1f}"
                    if r.get("rollback_subset_avg_full_tokens") is not None
                    else "NA"
                ),
                wo_tok=(
                    f"{float(r.get('rollback_subset_avg_wo_tokens')):.1f}"
                    if r.get("rollback_subset_avg_wo_tokens") is not None
                    else "NA"
                ),
            )
        )

    lines.append("")
    lines.append("## Interpretation guide")
    lines.append("")
    lines.append("- `Full wins`: Full SCRD correct, w/o rollback wrong.")
    lines.append("- `Ablation wins`: Full SCRD wrong, w/o rollback correct.")
    lines.append("- Positive `Net Full-Ablation` means rollback/repair helps on net.")
    lines.append("- Rollback-subset metrics are the most important numbers for this ablation.")
    path.write_text("\n".join(lines), encoding="utf-8")


def process_dataset(
    *,
    dataset_name: str,
    input_root: Path,
    output_root: Path,
    full_samples_limit: int,
    llm_client: Any,
    overwrite: bool,
    decision_rollback_context_policy: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    from src.main import load_samples

    full_results_path = input_root / dataset_name / "results.jsonl"
    full_rows = load_jsonl(full_results_path)
    full_by_id = {str(r.get("sample_id")): r for r in full_rows}

    dataset_out_dir = output_root / dataset_name
    results_path = dataset_out_dir / "results.jsonl"
    trace_dir = dataset_out_dir / "traces"
    error_path = dataset_out_dir / "errors.jsonl"

    if results_path.exists() and not overwrite:
        print(f"[reuse existing ablation] {results_path}")
        rows = load_jsonl(results_path)
        return summarize_dataset(dataset_name, rows), rows

    rollback_ids = {
        str(r.get("sample_id"))
        for r in full_rows
        if str(r.get("stop_reason")) == "rollback"
    }
    print(
        f"[{dataset_name}] Full rows={len(full_rows)}, rollback rows to rerun={len(rollback_ids)}"
    )

    samples = load_samples(dataset_name, limit=full_samples_limit)
    sample_by_id = {sample_id: (sample_id, question, gold_answer) for sample_id, question, gold_answer in samples}

    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    for full_row in full_rows:
        sample_id = str(full_row.get("sample_id"))
        if sample_id not in rollback_ids:
            rows.append(build_reused_row(full_row))
            continue

        sample = sample_by_id.get(sample_id)
        if sample is None:
            err = {
                "sample_id": sample_id,
                "dataset_name": dataset_name,
                "error": "sample_id not found from load_samples(); check sample_id alignment and --limit",
            }
            errors.append(err)
            print(f"  ERROR: {err}")
            continue

        _, question, gold_answer = sample
        try:
            rerun_row, trace = run_normal_mode_without_rollback(
                llm_client=llm_client,
                question=question,
                gold_answer=gold_answer,
                sample_id=sample_id,
                dataset_name=dataset_name,
                decision_rollback_context_policy=decision_rollback_context_policy,
            )
            rerun_row = augment_rerun_row_with_full_comparison(rerun_row, full_row)
            rows.append(rerun_row)
            write_json(trace_dir / f"{sample_id}_trace.json", trace)
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            err = {
                "sample_id": sample_id,
                "dataset_name": dataset_name,
                "question": question,
                "gold_answer": gold_answer,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
            errors.append(err)
            print(f"  ERROR running {sample_id}: {exc}")

    # Preserve original Full SCRD row order.
    order = {str(r.get("sample_id")): i for i, r in enumerate(full_rows)}
    rows.sort(key=lambda r: order.get(str(r.get("sample_id")), 10**9))

    write_jsonl(results_path, rows)
    if errors:
        write_jsonl(error_path, errors)

    summary = summarize_dataset(dataset_name, rows)
    write_json(dataset_out_dir / "summary.json", summary)
    return summary, rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Sparse w/o rollback/repair ablation. Reuses Full SCRD outputs for early-stop samples "
            "and reruns only samples whose Full SCRD stop_reason == rollback."
        )
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DEFAULT_DATASETS,
        help="Datasets to process. Default: math gsm8k multiarith.",
    )
    parser.add_argument(
        "--all-seven",
        action="store_true",
        help="Process all seven current datasets instead of the default three.",
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
        default=Path("outputs") / "ablation" / "wo_rollback_beixuan",
        help="Where to write ablation outputs.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=200,
        help="How many samples to load from each dataset. Should match the Full SCRD run.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path("."),
        help="Repository root for importing src modules.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing ablation results instead of reusing them.",
    )
    parser.add_argument(
        "--decision-rollback-context-policy",
        choices=["none", "detected"],
        default="none",
        help=(
            "Whether to pass detected rollback_context into the decision head after suppressing repair. "
            "Default 'none' is recommended because no post-anchor repair happened."
        ),
    )
    args = parser.parse_args()

    ensure_repo_imports(args.repo_root)

    from src.main import build_llm_client

    datasets = ALL_SEVEN_DATASETS if args.all_seven else args.datasets
    args.output_root.mkdir(parents=True, exist_ok=True)

    print("Building LLM client...")
    llm_client = build_llm_client()

    summaries: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []

    for dataset_name in datasets:
        summary, rows = process_dataset(
            dataset_name=dataset_name,
            input_root=args.input_root,
            output_root=args.output_root,
            full_samples_limit=args.limit,
            llm_client=llm_client,
            overwrite=args.overwrite,
            decision_rollback_context_policy=args.decision_rollback_context_policy,
        )
        summaries.append(summary)
        all_rows.extend(rows)
        print(
            "[{dataset}] full={full} | wo={wo} | delta={delta:+.2f} pp | rerun={rerun} | net={net}".format(
                dataset=dataset_name,
                full=pct(summary.get("full_scrd_acc")),
                wo=pct(summary.get("wo_rollback_acc")),
                delta=float(summary.get("delta_wo_minus_full_pp") or 0.0),
                rerun=summary.get("rerun_count"),
                net=summary.get("net_full_minus_ablation"),
            )
        )

    overall = add_overall_summary(summaries)
    summaries_with_overall = summaries + [overall]

    write_csv(args.output_root / "summary.csv", summaries_with_overall)
    write_json(args.output_root / "summary.json", summaries_with_overall)
    write_jsonl(args.output_root / "sample_level_results.jsonl", all_rows)
    write_report(args.output_root / "report.md", summaries_with_overall)

    print("\nSaved outputs to:", args.output_root)
    print(
        "OVERALL full={full} | wo={wo} | delta={delta:+.2f} pp | rerun={rerun} | net={net}".format(
            full=pct(overall.get("full_scrd_acc")),
            wo=pct(overall.get("wo_rollback_acc")),
            delta=float(overall.get("delta_wo_minus_full_pp") or 0.0),
            rerun=overall.get("rerun_count"),
            net=overall.get("net_full_minus_ablation"),
        )
    )


if __name__ == "__main__":
    main()
