from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
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


# ============================================================
# Basic IO
# ============================================================

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


def safe_load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return load_jsonl(path)


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
    if not rows:
        return

    path.parent.mkdir(parents=True, exist_ok=True)

    # Use union of all keys so later rows do not lose fields.
    fieldnames: list[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)

    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def rows_by_sample_id(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        sample_id = str(row.get("sample_id", "")).strip()
        if sample_id:
            out[sample_id] = row
    return out


def order_rows_by_reference(
    rows: list[dict[str, Any]],
    reference_ids: list[str],
) -> list[dict[str, Any]]:
    by_id = rows_by_sample_id(rows)
    return [by_id[sid] for sid in reference_ids if sid in by_id]


def save_dataset_outputs(
    *,
    dataset_out: Path,
    reference_ids: list[str],
    out_rows: list[dict[str, Any]],
    error_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    ordered_rows = order_rows_by_reference(out_rows, reference_ids)
    write_jsonl(dataset_out / "results.jsonl", ordered_rows)

    if error_rows:
        write_jsonl(dataset_out / "errors.jsonl", error_rows)

    return ordered_rows


# ============================================================
# Correct Full SCRD reference fields
# ============================================================

def get_reference_fields(
    reference_row: dict[str, Any],
    *,
    dataset_name: str,
    allow_reference_fallback: bool = False,
) -> dict[str, Any]:
    """
    Correct paper-level reference for Full SCRD.

    IMPORTANT:
    The paper uses last-round majority vote as Full SCRD finalizer.
    Therefore reference correctness must come from:
        last_round_majority_correct
    not:
        scrd_correct

    This function refuses to silently fall back to scrd_correct unless
    --allow-reference-fallback is explicitly enabled.
    """
    from src.utils.result_utils import is_correct, majority_vote

    gold = reference_row.get("gold_answer")

    answer_source = None
    correct_source = None

    # 1) Preferred answer field.
    if reference_row.get("last_round_majority_answer") is not None:
        ref_answer = reference_row.get("last_round_majority_answer")
        answer_source = "last_round_majority_answer"

    # 2) Some files may only store last effective answers.
    elif reference_row.get("last_effective_answers"):
        ref_answer = majority_vote(
            reference_row["last_effective_answers"],
            dataset_name=dataset_name,
        )
        answer_source = "majority_vote(last_effective_answers)"

    elif reference_row.get("last_effective_round_answers"):
        ref_answer = majority_vote(
            reference_row["last_effective_round_answers"],
            dataset_name=dataset_name,
        )
        answer_source = "majority_vote(last_effective_round_answers)"

    else:
        if not allow_reference_fallback:
            raise KeyError(
                "Reference row has no last-round MV answer field. "
                f"sample_id={reference_row.get('sample_id')} dataset={dataset_name}. "
                "Expected last_round_majority_answer or last_effective_answers. "
                "Do not use scrd_final_answer unless you explicitly pass "
                "--allow-reference-fallback."
            )

        ref_answer = reference_row.get("scrd_final_answer")
        answer_source = "FALLBACK:scrd_final_answer"

    # 3) Preferred correctness field.
    if "last_round_majority_correct" in reference_row:
        ref_correct = bool(reference_row["last_round_majority_correct"])
        correct_source = "last_round_majority_correct"

    # 4) If no correctness field but answer exists, recompute correctness.
    elif ref_answer is not None and gold is not None:
        ref_correct = bool(is_correct(ref_answer, gold, dataset_name))
        correct_source = f"recomputed_from:{answer_source}"

    else:
        if not allow_reference_fallback:
            raise KeyError(
                "Reference row has no last-round MV correctness field. "
                f"sample_id={reference_row.get('sample_id')} dataset={dataset_name}. "
                "Expected last_round_majority_correct. "
                "Do not use scrd_correct unless you explicitly pass "
                "--allow-reference-fallback."
            )

        ref_correct = bool(reference_row.get("scrd_correct", False))
        correct_source = "FALLBACK:scrd_correct"

    ref_tokens = float(reference_row.get("scrd_total_tokens") or 0)

    return {
        "reference_full_scrd_answer": ref_answer,
        "reference_full_scrd_correct": ref_correct,
        "reference_full_scrd_total_tokens": ref_tokens,
        "reference_answer_source": answer_source,
        "reference_correct_source": correct_source,
    }


# ============================================================
# w/o rollback implementation
# ============================================================

def get_last_effective_answers_from_state_store(state_store: Any) -> list[str]:
    latest = state_store.get_latest_state_record()
    if latest is None:
        return []

    return [
        str(a).strip()
        for a in latest.current_answers
        if a is not None and str(a).strip()
    ]


class NoRollbackActionMapper:
    """
    Wrapper around the original ActionMapper.

    Key definition:
    - evaluator/action mapper are still active;
    - rollback is never available;
    - therefore a degraded transition maps to early_stop instead of rollback;
    - improved/plateau can still continue until early_stop or max_round.
    """

    def __init__(self) -> None:
        from src.components.action_mapper import ActionMapper

        self.base_mapper = ActionMapper()

    def map_action(
        self,
        evaluation: Any,
        *,
        round_id: int,
        max_round: int,
        rollback_available: bool,
    ) -> Any:
        del rollback_available

        return self.base_mapper.map_action(
            evaluation,
            round_id=round_id,
            max_round=max_round,
            rollback_available=False,
        )


def build_llm_client_from_env() -> Any:
    from src.components.qianfan_client import QianfanClient

    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise ValueError("Missing DASHSCOPE_API_KEY. Please set it in your .env file.")

    return QianfanClient(
        api_key=api_key,
        model=os.getenv("DASHSCOPE_MODEL", "qwen2.5-7b-instruct-1m"),
        base_url=os.getenv(
            "DASHSCOPE_BASE_URL",
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
        ),
    )


def run_no_rollback_mode(
    *,
    llm_client: Any,
    question: str,
    gold_answer: str,
    sample_id: str,
    dataset_name: str,
    max_round: int,
    agent_ids: list[str],
    agent_roles: dict[str, str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    from src.components.agent_runner import AgentRunner
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
        role_by_agent_id=agent_roles,
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
            agent_ids=agent_ids,
            max_round=max_round,
            sample_id=sample_id,
        ),
        agent_runner=agent_runner,
        state_store=state_store,
        history_manager=history_manager,
        recorder=recorder,
        evaluator=evaluator,
        action_mapper=NoRollbackActionMapper(),
        rollback_controller=RollbackController(max_rollbacks=0),
    )

    debate_orchestrator = DebateOrchestrator(
        config=DebateOrchestratorConfig(
            question=question,
            agent_ids=agent_ids,
            max_round=max_round,
        ),
        state_store=state_store,
        normal_round_executor=normal_round_executor,
    )

    print(f"\n===== Running w/o rollback {sample_id} =====")
    print("Dataset:", dataset_name)
    print("Gold answer:", gold_answer)
    print(
        "Policy: rollback disabled; degraded transitions early-stop; "
        "otherwise continue until early_stop or max_round."
    )

    debate_result = debate_orchestrator.run_debate()
    rollback_context = debate_result["rollback_context"]
    early_stopped = debate_result["early_stopped"]

    if rollback_context:
        state_store.add_event(
            {
                "type": "unexpected_rollback_context_in_wo_rollback",
                "rollback_context": str(rollback_context),
            }
        )

    round_1_answers = get_round_1_answers(state_store)
    single_agent_baseline_answer = round_1_answers[0] if round_1_answers else ""
    majority_voting_baseline_answer = majority_vote(
        round_1_answers,
        dataset_name=dataset_name,
    )

    last_answers = get_last_effective_answers_from_state_store(state_store)
    wo_rollback_final_answer = (
        majority_vote(last_answers, dataset_name=dataset_name)
        if last_answers
        else ""
    )

    usage_summary = build_usage_summary(usage_logger)

    result = {
        "sample_id": sample_id,
        "dataset_name": dataset_name,
        "question": question,
        "gold_answer": gold_answer,
        "ablation": "wo_rollback",
        "ablation_policy": "rollback_disabled_degraded_to_early_stop",
        "finalizer": "last_effective_round_majority_vote",
        "agent_roles": agent_roles,
        "round_1_answers": round_1_answers,
        "single_agent_baseline_answer": single_agent_baseline_answer,
        "majority_voting_baseline_answer": majority_voting_baseline_answer,

        # In this ablation run, scrd_final_answer is the w/o rollback final answer.
        "scrd_final_answer": wo_rollback_final_answer,
        "wo_rollback_final_answer": wo_rollback_final_answer,

        "single_agent_correct": is_correct(
            single_agent_baseline_answer,
            gold_answer,
            dataset_name,
        ),
        "majority_voting_correct": is_correct(
            majority_voting_baseline_answer,
            gold_answer,
            dataset_name,
        ),

        # In this ablation run, scrd_correct is also w/o rollback correctness.
        "scrd_correct": is_correct(
            wo_rollback_final_answer,
            gold_answer,
            dataset_name,
        ),
        "wo_rollback_correct": is_correct(
            wo_rollback_final_answer,
            gold_answer,
            dataset_name,
        ),

        "last_effective_answers": last_answers,
        "effective_rounds_used": get_effective_rounds_used(state_store),
        "actual_rounds_executed": get_actual_rounds_executed(state_store),
        "stop_reason": get_stop_reason(None, early_stopped),
        "original_rollback_disabled": True,

        "single_agent_total_tokens": usage_summary["single_agent_total_tokens"],
        "majority_vote_total_tokens": usage_summary["majority_vote_total_tokens"],
        "scrd_total_tokens": usage_summary["scrd_total_tokens"],
        "wo_rollback_total_tokens": usage_summary["scrd_total_tokens"],
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
    trace["ablation"] = "wo_rollback"
    trace["ablation_policy"] = "rollback_disabled_degraded_to_early_stop"
    trace["finalizer"] = "last_effective_round_majority_vote"

    return result, trace


# ============================================================
# Row construction / repair
# ============================================================

def make_reused_row(
    reference_row: dict[str, Any],
    *,
    dataset_name: str,
    allow_reference_fallback: bool = False,
) -> dict[str, Any]:
    """
    For non-rollback samples, w/o rollback is identical to the Full SCRD reference.

    Correct paper-level reference:
    Full SCRD = last-round majority vote.
    """
    ref = get_reference_fields(
        reference_row,
        dataset_name=dataset_name,
        allow_reference_fallback=allow_reference_fallback,
    )

    row = dict(reference_row)

    row["ablation"] = "wo_rollback"
    row["ablation_policy"] = "rollback_disabled_degraded_to_early_stop"
    row["finalizer"] = "last_effective_round_majority_vote"

    row["reused_from_reference_full_scrd"] = True
    row["rerun_required"] = False

    row["reference_stop_reason"] = reference_row.get("stop_reason")
    row["reference_full_scrd_answer"] = ref["reference_full_scrd_answer"]
    row["reference_full_scrd_correct"] = ref["reference_full_scrd_correct"]
    row["reference_full_scrd_total_tokens"] = ref["reference_full_scrd_total_tokens"]
    row["reference_answer_source"] = ref["reference_answer_source"]
    row["reference_correct_source"] = ref["reference_correct_source"]

    # Since this row is reused, w/o rollback equals the Full SCRD reference.
    row["wo_rollback_final_answer"] = ref["reference_full_scrd_answer"]
    row["wo_rollback_correct"] = ref["reference_full_scrd_correct"]
    row["wo_rollback_total_tokens"] = ref["reference_full_scrd_total_tokens"]

    row["answer_changed_vs_reference"] = False
    row["correct_changed_vs_reference"] = False
    row["token_delta_wo_minus_reference"] = 0.0

    return row


def make_rerun_row(
    reference_row: dict[str, Any],
    rerun_result: dict[str, Any],
    *,
    dataset_name: str,
    allow_reference_fallback: bool = False,
) -> dict[str, Any]:
    """
    For rollback samples, keep actual w/o rollback rerun result,
    but compare it against the correct Full SCRD reference:
    last-round majority vote.
    """
    ref = get_reference_fields(
        reference_row,
        dataset_name=dataset_name,
        allow_reference_fallback=allow_reference_fallback,
    )

    row = dict(rerun_result)

    row["reused_from_reference_full_scrd"] = False
    row["rerun_required"] = True

    row["reference_stop_reason"] = reference_row.get("stop_reason")
    row["reference_full_scrd_answer"] = ref["reference_full_scrd_answer"]
    row["reference_full_scrd_correct"] = ref["reference_full_scrd_correct"]
    row["reference_full_scrd_total_tokens"] = ref["reference_full_scrd_total_tokens"]
    row["reference_answer_source"] = ref["reference_answer_source"]
    row["reference_correct_source"] = ref["reference_correct_source"]

    row["answer_changed_vs_reference"] = (
        str(row.get("wo_rollback_final_answer"))
        != str(ref["reference_full_scrd_answer"])
    )
    row["correct_changed_vs_reference"] = (
        bool(row.get("wo_rollback_correct"))
        != bool(ref["reference_full_scrd_correct"])
    )
    row["token_delta_wo_minus_reference"] = (
        float(row.get("wo_rollback_total_tokens") or 0)
        - float(ref["reference_full_scrd_total_tokens"] or 0)
    )

    return row


def repair_existing_row_against_reference(
    existing_row: dict[str, Any],
    reference_row: dict[str, Any],
    *,
    dataset_name: str,
    allow_reference_fallback: bool = False,
) -> dict[str, Any]:
    """
    If an old wrong results.jsonl already exists, this function repairs its
    reference fields without rerunning the model.

    - If reused_from_reference_full_scrd=True, w/o rollback result must also
      be reset to correct Full SCRD last-round MV.
    - If rerun_required=True, preserve actual w/o rollback rerun result,
      but update reference comparison fields.
    """
    ref = get_reference_fields(
        reference_row,
        dataset_name=dataset_name,
        allow_reference_fallback=allow_reference_fallback,
    )

    row = dict(existing_row)

    row["ablation"] = "wo_rollback"
    row["ablation_policy"] = "rollback_disabled_degraded_to_early_stop"
    row["finalizer"] = row.get("finalizer", "last_effective_round_majority_vote")

    row["reference_stop_reason"] = reference_row.get("stop_reason")
    row["reference_full_scrd_answer"] = ref["reference_full_scrd_answer"]
    row["reference_full_scrd_correct"] = ref["reference_full_scrd_correct"]
    row["reference_full_scrd_total_tokens"] = ref["reference_full_scrd_total_tokens"]
    row["reference_answer_source"] = ref["reference_answer_source"]
    row["reference_correct_source"] = ref["reference_correct_source"]

    reused = bool(row.get("reused_from_reference_full_scrd")) and not bool(row.get("rerun_required"))

    if reused:
        row["wo_rollback_final_answer"] = ref["reference_full_scrd_answer"]
        row["wo_rollback_correct"] = ref["reference_full_scrd_correct"]
        row["wo_rollback_total_tokens"] = ref["reference_full_scrd_total_tokens"]

        row["answer_changed_vs_reference"] = False
        row["correct_changed_vs_reference"] = False
        row["token_delta_wo_minus_reference"] = 0.0
    else:
        row["answer_changed_vs_reference"] = (
            str(row.get("wo_rollback_final_answer"))
            != str(ref["reference_full_scrd_answer"])
        )
        row["correct_changed_vs_reference"] = (
            bool(row.get("wo_rollback_correct"))
            != bool(ref["reference_full_scrd_correct"])
        )
        row["token_delta_wo_minus_reference"] = (
            float(row.get("wo_rollback_total_tokens") or 0)
            - float(ref["reference_full_scrd_total_tokens"] or 0)
        )

    return row


# ============================================================
# Summaries
# ============================================================

def summarize_dataset(dataset: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)

    ref_correct = sum(bool(r.get("reference_full_scrd_correct")) for r in rows)
    wo_correct = sum(bool(r.get("wo_rollback_correct")) for r in rows)

    full_wins = sum(
        bool(r.get("reference_full_scrd_correct"))
        and not bool(r.get("wo_rollback_correct"))
        for r in rows
    )

    ablation_wins = sum(
        not bool(r.get("reference_full_scrd_correct"))
        and bool(r.get("wo_rollback_correct"))
        for r in rows
    )

    rerun_rows = [r for r in rows if r.get("rerun_required")]
    rb_n = len(rerun_rows)

    rb_ref_correct = sum(bool(r.get("reference_full_scrd_correct")) for r in rerun_rows)
    rb_wo_correct = sum(bool(r.get("wo_rollback_correct")) for r in rerun_rows)

    avg_ref_tokens = (
        sum(float(r.get("reference_full_scrd_total_tokens") or 0) for r in rows) / n
        if n
        else 0
    )

    avg_wo_tokens = (
        sum(float(r.get("wo_rollback_total_tokens") or 0) for r in rows) / n
        if n
        else 0
    )

    rb_avg_ref_tokens = (
        sum(float(r.get("reference_full_scrd_total_tokens") or 0) for r in rerun_rows)
        / rb_n
        if rb_n
        else 0
    )

    rb_avg_wo_tokens = (
        sum(float(r.get("wo_rollback_total_tokens") or 0) for r in rerun_rows)
        / rb_n
        if rb_n
        else 0
    )

    reference_fallback_count = sum(
        str(r.get("reference_correct_source", "")).startswith("FALLBACK")
        or str(r.get("reference_answer_source", "")).startswith("FALLBACK")
        for r in rows
    )

    return {
        "dataset": dataset,
        "n": n,
        "rerun_count": rb_n,
        "reused_count": n - rb_n,

        "reference_full_scrd_correct": ref_correct,
        "reference_full_scrd_acc": ref_correct / n if n else None,

        "wo_rollback_correct": wo_correct,
        "wo_rollback_acc": wo_correct / n if n else None,

        "delta_wo_minus_reference_pp": (
            100 * ((wo_correct / n) - (ref_correct / n))
            if n
            else None
        ),

        "full_wins": full_wins,
        "ablation_wins": ablation_wins,
        "net_full_minus_ablation": full_wins - ablation_wins,

        "avg_reference_tokens": avg_ref_tokens,
        "avg_wo_rollback_tokens": avg_wo_tokens,
        "avg_token_delta_wo_minus_reference": avg_wo_tokens - avg_ref_tokens,

        "rollback_subset_n": rb_n,
        "rollback_subset_reference_correct": rb_ref_correct,
        "rollback_subset_reference_acc": rb_ref_correct / rb_n if rb_n else None,
        "rollback_subset_wo_correct": rb_wo_correct,
        "rollback_subset_wo_acc": rb_wo_correct / rb_n if rb_n else None,
        "rollback_subset_avg_reference_tokens": rb_avg_ref_tokens,
        "rollback_subset_avg_wo_tokens": rb_avg_wo_tokens,
        "rollback_subset_avg_token_delta_wo_minus_reference": (
            rb_avg_wo_tokens - rb_avg_ref_tokens if rb_n else 0
        ),

        "reference_fallback_count": reference_fallback_count,
    }


def add_overall(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    n = sum(int(s["n"]) for s in summaries)

    def sum_key(key: str) -> int:
        return sum(int(s.get(key) or 0) for s in summaries)

    def weighted_avg(key: str) -> float:
        if not n:
            return 0.0
        return (
            sum(float(s.get(key) or 0) * int(s["n"]) for s in summaries)
            / n
        )

    rb_n = sum_key("rollback_subset_n")

    def rb_weighted_avg(key: str) -> float:
        if not rb_n:
            return 0.0
        return (
            sum(
                float(s.get(key) or 0) * int(s.get("rollback_subset_n") or 0)
                for s in summaries
            )
            / rb_n
        )

    ref_correct = sum_key("reference_full_scrd_correct")
    wo_correct = sum_key("wo_rollback_correct")
    rb_ref_correct = sum_key("rollback_subset_reference_correct")
    rb_wo_correct = sum_key("rollback_subset_wo_correct")

    return {
        "dataset": "OVERALL",
        "n": n,
        "rerun_count": sum_key("rerun_count"),
        "reused_count": sum_key("reused_count"),

        "reference_full_scrd_correct": ref_correct,
        "reference_full_scrd_acc": ref_correct / n if n else None,

        "wo_rollback_correct": wo_correct,
        "wo_rollback_acc": wo_correct / n if n else None,

        "delta_wo_minus_reference_pp": (
            100 * ((wo_correct / n) - (ref_correct / n))
            if n
            else None
        ),

        "full_wins": sum_key("full_wins"),
        "ablation_wins": sum_key("ablation_wins"),
        "net_full_minus_ablation": sum_key("net_full_minus_ablation"),

        "avg_reference_tokens": weighted_avg("avg_reference_tokens"),
        "avg_wo_rollback_tokens": weighted_avg("avg_wo_rollback_tokens"),
        "avg_token_delta_wo_minus_reference": weighted_avg(
            "avg_token_delta_wo_minus_reference"
        ),

        "rollback_subset_n": rb_n,
        "rollback_subset_reference_correct": rb_ref_correct,
        "rollback_subset_reference_acc": rb_ref_correct / rb_n if rb_n else None,
        "rollback_subset_wo_correct": rb_wo_correct,
        "rollback_subset_wo_acc": rb_wo_correct / rb_n if rb_n else None,
        "rollback_subset_avg_reference_tokens": rb_weighted_avg(
            "rollback_subset_avg_reference_tokens"
        ),
        "rollback_subset_avg_wo_tokens": rb_weighted_avg(
            "rollback_subset_avg_wo_tokens"
        ),
        "rollback_subset_avg_token_delta_wo_minus_reference": rb_weighted_avg(
            "rollback_subset_avg_token_delta_wo_minus_reference"
        ),

        "reference_fallback_count": sum_key("reference_fallback_count"),
    }


def write_report(path: Path, summaries: list[dict[str, Any]]) -> None:
    def pct(x: Any) -> str:
        return "NA" if x is None else f"{100 * float(x):.2f}%"

    lines = []
    lines.append("# Ablation: w/o Rollback / Repair")
    lines.append("")
    lines.append(
        "Policy: rollback is disabled. The evaluator and action mapper are kept. "
        "Degraded transitions map to early_stop; otherwise the debate may continue "
        "until early_stop or max_round."
    )
    lines.append("")
    lines.append(
        "Reference Full SCRD is evaluated using last-round majority voting "
        "`last_round_majority_correct`, not decision-head `scrd_correct`."
    )
    lines.append("")
    lines.append(
        "Only samples whose reference Full SCRD run stopped by rollback are rerun. "
        "All other samples are reused from the reference outputs."
    )
    lines.append("")
    lines.append(
        "| Dataset | N | Rerun | Reference Acc | w/o Rollback Acc | Δ Acc | "
        "Full wins | Ablation wins | Net Full-Ablation | Ref tokens | w/o tokens | Fallback |"
    )
    lines.append(
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    )

    for s in summaries:
        lines.append(
            "| {dataset} | {n} | {rerun} | {ref_acc} | {wo_acc} | "
            "{delta:.2f} pp | {full_wins} | {abl_wins} | {net} | "
            "{ref_tok:.1f} | {wo_tok:.1f} | {fallback} |".format(
                dataset=s["dataset"],
                n=s["n"],
                rerun=s["rerun_count"],
                ref_acc=pct(s["reference_full_scrd_acc"]),
                wo_acc=pct(s["wo_rollback_acc"]),
                delta=float(s["delta_wo_minus_reference_pp"] or 0),
                full_wins=s["full_wins"],
                abl_wins=s["ablation_wins"],
                net=s["net_full_minus_ablation"],
                ref_tok=float(s["avg_reference_tokens"] or 0),
                wo_tok=float(s["avg_wo_rollback_tokens"] or 0),
                fallback=s.get("reference_fallback_count", 0),
            )
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


# ============================================================
# Main
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Sparse rerun ablation for w/o rollback. "
            "It reruns only reference samples whose stop_reason == rollback, "
            "while reusing all other rows. "
            "Reference Full SCRD is last-round majority vote."
        )
    )

    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--all-seven", action="store_true")
    parser.add_argument("--limit", type=int, default=200)

    parser.add_argument(
        "--reference-root",
        type=Path,
        default=Path("outputs_with_last_round_majority_vote"),
        help=(
            "Reference Full SCRD outputs. Must contain per-dataset results.jsonl "
            "with last_round_majority_correct."
        ),
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/ablation/wo_rollback_v3"),
    )

    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--max-round", type=int, default=5)

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete each dataset output directory and rerun from scratch.",
    )

    parser.add_argument(
        "--allow-reference-fallback",
        action="store_true",
        help=(
            "Allow fallback to scrd_final_answer/scrd_correct if last-round MV fields "
            "are missing. Not recommended for paper results."
        ),
    )

    args = parser.parse_args()

    ensure_repo_imports(args.repo_root)

    from dotenv import load_dotenv
    from src.main import AGENT_IDS, AGENT_ROLES, load_samples

    os.environ["HTTP_PROXY"] = ""
    os.environ["HTTPS_PROXY"] = ""
    os.environ["NO_PROXY"] = "dashscope.aliyuncs.com,localhost,127.0.0.1"

    load_dotenv()

    datasets = ALL_SEVEN_DATASETS if args.all_seven else args.datasets
    args.output_root.mkdir(parents=True, exist_ok=True)

    llm_client = build_llm_client_from_env()

    all_summaries: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []

    for dataset in datasets:
        print(f"\n========== Dataset: {dataset} ==========")

        reference_path = args.reference_root / dataset / "results.jsonl"
        reference_rows = load_jsonl(reference_path)

        reference_ids = [
            str(r.get("sample_id", "")).strip()
            for r in reference_rows
            if str(r.get("sample_id", "")).strip()
        ]

        dataset_out = args.output_root / dataset

        if args.overwrite and dataset_out.exists():
            print(f"[overwrite] Removing old output directory: {dataset_out}")
            shutil.rmtree(dataset_out)

        trace_out = dataset_out / "traces"
        result_path = dataset_out / "results.jsonl"
        error_path = dataset_out / "errors.jsonl"

        existing_rows = [] if args.overwrite else safe_load_jsonl(result_path)
        existing_errors = [] if args.overwrite else safe_load_jsonl(error_path)

        existing_by_id = rows_by_sample_id(existing_rows)

        error_rows: list[dict[str, Any]] = list(existing_errors)
        out_rows: list[dict[str, Any]] = []

        if existing_rows and not args.overwrite:
            print(
                f"[resume] Found {len(existing_rows)} existing rows in {result_path}. "
                "Completed sample_ids will be repaired against last-round MV reference and skipped."
            )

        samples = load_samples(dataset, limit=args.limit)
        sample_by_id = {
            str(sample_id): (question, gold)
            for sample_id, question, gold in samples
        }

        interrupted = False

        for ref_row in reference_rows:
            sample_id = str(ref_row.get("sample_id", "")).strip()
            if not sample_id:
                continue

            # If an old row exists, repair its reference fields and do not rerun.
            if sample_id in existing_by_id and not args.overwrite:
                fixed_existing = repair_existing_row_against_reference(
                    existing_by_id[sample_id],
                    ref_row,
                    dataset_name=dataset,
                    allow_reference_fallback=args.allow_reference_fallback,
                )
                out_rows.append(fixed_existing)
                print(
                    "[skip+repair] sample={} | ref_correct={} | wo_correct={} | source={}".format(
                        sample_id,
                        fixed_existing.get("reference_full_scrd_correct"),
                        fixed_existing.get("wo_rollback_correct"),
                        fixed_existing.get("reference_correct_source"),
                    )
                )
                continue

            # Non-rollback reference rows are unaffected by removing rollback.
            if ref_row.get("stop_reason") != "rollback":
                reused_row = make_reused_row(
                    ref_row,
                    dataset_name=dataset,
                    allow_reference_fallback=args.allow_reference_fallback,
                )
                out_rows.append(reused_row)

                ordered_rows = save_dataset_outputs(
                    dataset_out=dataset_out,
                    reference_ids=reference_ids,
                    out_rows=out_rows,
                    error_rows=error_rows,
                )

                print(
                    "[reuse] sample={} | stop_reason={} | ref_correct={} | source={} | saved_rows={}".format(
                        sample_id,
                        ref_row.get("stop_reason"),
                        reused_row.get("reference_full_scrd_correct"),
                        reused_row.get("reference_correct_source"),
                        len(ordered_rows),
                    )
                )
                continue

            if sample_id not in sample_by_id:
                err = {
                    "sample_id": sample_id,
                    "dataset": dataset,
                    "error": "sample_id not found in load_samples output",
                }
                error_rows.append(err)

                save_dataset_outputs(
                    dataset_out=dataset_out,
                    reference_ids=reference_ids,
                    out_rows=out_rows,
                    error_rows=error_rows,
                )

                print(f"[error] {sample_id}: sample_id not found in load_samples output")
                continue

            question, gold_answer = sample_by_id[sample_id]

            try:
                rerun_result, trace = run_no_rollback_mode(
                    llm_client=llm_client,
                    question=question,
                    gold_answer=gold_answer,
                    sample_id=sample_id,
                    dataset_name=dataset,
                    max_round=args.max_round,
                    agent_ids=AGENT_IDS,
                    agent_roles=AGENT_ROLES,
                )

                merged_row = make_rerun_row(
                    ref_row,
                    rerun_result,
                    dataset_name=dataset,
                    allow_reference_fallback=args.allow_reference_fallback,
                )
                out_rows.append(merged_row)

                write_json(trace_out / f"{sample_id}_trace.json", trace)

                ordered_rows = save_dataset_outputs(
                    dataset_out=dataset_out,
                    reference_ids=reference_ids,
                    out_rows=out_rows,
                    error_rows=error_rows,
                )

                print(
                    "[saved] sample={} | ref_correct={} | wo_correct={} | ref_source={} | stop={} | saved_rows={}".format(
                        sample_id,
                        merged_row.get("reference_full_scrd_correct"),
                        rerun_result.get("wo_rollback_correct"),
                        merged_row.get("reference_correct_source"),
                        rerun_result.get("stop_reason"),
                        len(ordered_rows),
                    )
                )

            except KeyboardInterrupt:
                print("\n[interrupt] Interrupted by user. Saving partial outputs...")
                interrupted = True

                save_dataset_outputs(
                    dataset_out=dataset_out,
                    reference_ids=reference_ids,
                    out_rows=out_rows,
                    error_rows=error_rows,
                )

                break

            except Exception as exc:
                tb = traceback.format_exc()

                error_rows.append(
                    {
                        "sample_id": sample_id,
                        "dataset": dataset,
                        "question": question,
                        "gold_answer": gold_answer,
                        "error": str(exc),
                        "traceback": tb,
                    }
                )

                save_dataset_outputs(
                    dataset_out=dataset_out,
                    reference_ids=reference_ids,
                    out_rows=out_rows,
                    error_rows=error_rows,
                )

                print(f"[failed] {sample_id}: {exc}")

        ordered_rows = save_dataset_outputs(
            dataset_out=dataset_out,
            reference_ids=reference_ids,
            out_rows=out_rows,
            error_rows=error_rows,
        )

        summary = summarize_dataset(dataset, ordered_rows)
        write_json(dataset_out / "summary.json", summary)

        all_summaries.append(summary)
        all_rows.extend(ordered_rows)

        print(
            "SUMMARY {}: ref={:.2f}% | wo={:.2f}% | delta={:+.2f} pp | rerun={} | fallback={} | saved={}/{}".format(
                dataset,
                100 * float(summary["reference_full_scrd_acc"] or 0),
                100 * float(summary["wo_rollback_acc"] or 0),
                float(summary["delta_wo_minus_reference_pp"] or 0),
                summary["rerun_count"],
                summary["reference_fallback_count"],
                len(ordered_rows),
                len(reference_rows),
            )
        )

        if interrupted:
            print(
                f"[interrupt] Stop after dataset={dataset}. "
                "Re-run the same command to continue from saved results."
            )
            break

    if all_summaries:
        overall = add_overall(all_summaries)
        summaries_with_overall = all_summaries + [overall]

        write_csv(args.output_root / "summary.csv", summaries_with_overall)
        write_json(args.output_root / "summary.json", summaries_with_overall)
        write_jsonl(args.output_root / "sample_level_results.jsonl", all_rows)
        write_report(args.output_root / "report.md", summaries_with_overall)

        print("")
        print(f"Saved outputs to: {args.output_root}")
        print(
            "OVERALL ref={:.2f}% | wo={:.2f}% | delta={:+.2f} pp | rerun={} | fallback={}".format(
                100 * float(overall["reference_full_scrd_acc"] or 0),
                100 * float(overall["wo_rollback_acc"] or 0),
                float(overall["delta_wo_minus_reference_pp"] or 0),
                overall["rerun_count"],
                overall["reference_fallback_count"],
            )
        )

        if int(overall.get("reference_fallback_count") or 0) > 0:
            print(
                "\nWARNING: Some rows used fallback reference fields. "
                "For paper results, fallback should ideally be 0."
            )
    else:
        print("No datasets were completed. No overall summary was written.")


if __name__ == "__main__":
    main()