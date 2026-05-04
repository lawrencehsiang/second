from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import traceback
from pathlib import Path
from typing import Any


DEFAULT_DATASETS = ["math", "gsm8k", "multiarith"]
ALL_SEVEN_DATASETS = [
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
    if not path.exists():
        raise FileNotFoundError(f"Missing JSONL: {path}")

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


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


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


def sample_index(sample_id: str) -> int | None:
    m = re.search(r"_(\d+)$", str(sample_id))
    if not m:
        return None
    return int(m.group(1))


def get_sample_by_id_or_index(
    *,
    sample_id: str,
    sample_by_id: dict[str, tuple[str, str, str]],
    sample_by_index: dict[int, tuple[str, str, str]],
) -> tuple[str, str, str] | None:
    if sample_id in sample_by_id:
        return sample_by_id[sample_id]

    idx = sample_index(sample_id)
    if idx is not None and idx in sample_by_index:
        return sample_by_index[idx]

    return None


class RawFullHistoryManager:
    """
    w/o History Filtering ablation.

    This class intentionally replaces the normal HistoryManager's selective
    top-k structured history units with a single unfiltered raw-history unit.

    Important:
    - We still keep the State Recorder internally because evaluator/action mapper
      and rollback depend on StateRecord objects.
    - What is ablated is the history FILTERING seen by agents.
    - By default, agents see all previous rounds' state-level history.
    """

    def __init__(self, *, scope: str = "all", max_chars: int | None = None) -> None:
        if scope not in {"all", "last"}:
            raise ValueError("scope must be one of: all, last")
        self.scope = scope
        self.max_chars = max_chars

    def build_history_units(
        self,
        question: str,
        current_round_id: int,
        state_store: Any,
    ) -> list[Any]:
        from src.schemas import HistoryUnit

        del question

        if current_round_id <= 1:
            return []

        if self.scope == "last":
            round_ids = [current_round_id - 1]
        else:
            round_ids = list(range(1, current_round_id))

        sections: list[str] = []

        for round_id in round_ids:
            state = state_store.get_state_record(round_id)
            if state is None:
                continue
            sections.append(self._format_state_record(state))

        if not sections:
            return []

        raw_history = "\n\n".join(sections)
        if self.max_chars is not None and len(raw_history) > self.max_chars:
            # Keep the most recent content if truncation is necessary.
            raw_history = raw_history[-self.max_chars :]

        return [
            HistoryUnit(
                type="core_unresolved_conflict",
                conflict="Unfiltered raw debate history from previous rounds.",
                why_still_open=(
                    "This ablation disables selective history filtering. "
                    "The agent receives raw previous-round state history."
                ),
                snippet=raw_history,
                source_round=current_round_id - 1,
            )
        ]

    def _format_state_record(self, state: Any) -> str:
        lines: list[str] = []
        lines.append(f"===== Round {state.round_id} raw state history =====")

        answers = getattr(state, "current_answers", []) or []
        for i, answer in enumerate(answers, start=1):
            lines.append(f"Agent {i} answer: {answer}")

        claims = getattr(state, "newly_added_claims", []) or []
        if claims:
            lines.append("Newly added claims:")
            for claim in claims:
                text = getattr(claim, "text", str(claim))
                claim_type = getattr(claim, "claim_type", None)
                related_answer = getattr(claim, "related_answer", None)
                lines.append(
                    f"- [{claim_type}] {text}"
                    + (f" | related_answer={related_answer}" if related_answer else "")
                )

        conflicts = getattr(state, "unresolved_conflicts", []) or []
        if conflicts:
            lines.append("Unresolved conflicts:")
            for conflict in conflicts:
                conflict_text = getattr(conflict, "conflict", str(conflict))
                why = getattr(conflict, "why_still_open", "")
                involved = getattr(conflict, "involved_answers", [])
                lines.append(
                    f"- {conflict_text}"
                    + (f" | why_still_open={why}" if why else "")
                    + (f" | involved_answers={involved}" if involved else "")
                )

        snippets = getattr(state, "key_raw_snippets", []) or []
        if snippets:
            lines.append("Key raw snippets:")
            for snippet in snippets:
                lines.append(f"- {snippet}")

        return "\n".join(lines)


def get_latest_answers(state_store: Any) -> list[str]:
    latest = state_store.get_latest_state_record()
    if latest is None:
        return []
    return [
        str(a).strip()
        for a in (latest.current_answers or [])
        if a is not None and str(a).strip()
    ]


def run_repair_mode_with_history_manager(
    *,
    llm_client: Any,
    question: str,
    rollback_context: dict[str, Any],
    state_store: Any,
    history_manager: Any,
    usage_logger: Any,
    sample_id: str,
    dataset_name: str,
    max_round: int,
) -> None:
    from src.components.recorder import Recorder
    from src.components.repair_action_mapper import RepairActionMapper
    from src.components.repair_agent_runner import RepairAgentRunner
    from src.components.repair_brief_generator import RepairBriefGenerator
    from src.components.repair_evaluator import RepairEvaluator
    from src.main import AGENT_IDS, AGENT_ROLES
    from src.pipeline.repair_orchestrator import (
        RepairOrchestrator,
        RepairOrchestratorConfig,
    )
    from src.pipeline.repair_round_executor import (
        RepairRoundExecutor,
        RepairRoundExecutorConfig,
    )

    repair_action_mapper = RepairActionMapper()

    recorder = Recorder(
        llm_client=llm_client,
        usage_logger=usage_logger,
        sample_id=sample_id,
    )
    repair_brief_generator = RepairBriefGenerator(
        llm_client=llm_client,
        usage_logger=usage_logger,
        sample_id=sample_id,
    )
    repair_evaluator = RepairEvaluator(
        llm_client=llm_client,
        usage_logger=usage_logger,
        sample_id=sample_id,
    )
    repair_agent_runner = RepairAgentRunner(
        llm_client=llm_client,
        usage_logger=usage_logger,
        sample_id=sample_id,
        dataset_name=dataset_name,
        role_by_agent_id=AGENT_ROLES,
    )

    anchor_round = rollback_context["anchor_round"]
    anchor_state = rollback_context["anchor_state"]
    failed_suffix_state_records = rollback_context["failed_suffix_state_records"]

    repair_brief = repair_brief_generator.generate_brief_from_parts(
        question=question,
        anchor_state=anchor_state,
        failed_suffix_state_records=failed_suffix_state_records,
    )

    state_store.remove_rounds_after(anchor_round)

    repair_round_executor = RepairRoundExecutor(
        config=RepairRoundExecutorConfig(
            question=question,
            agent_ids=AGENT_IDS,
            max_round=max_round,
            sample_id=sample_id,
        ),
        repair_agent_runner=repair_agent_runner,
        state_store=state_store,
        repair_brief_generator=repair_brief_generator,
        recorder=recorder,
        repair_evaluator=repair_evaluator,
        repair_action_mapper=repair_action_mapper,
        history_manager=history_manager,
    )

    repair_orchestrator = RepairOrchestrator(
        config=RepairOrchestratorConfig(
            question=question,
            agent_ids=AGENT_IDS,
            max_round=max_round,
        ),
        state_store=state_store,
        repair_round_executor=repair_round_executor,
    )

    print("Starting repair mode under raw-history ablation...")
    repair_orchestrator.run_repair(
        rollback_context={
            **rollback_context,
            "repair_brief": repair_brief,
        }
    )


def run_one_sample(
    *,
    llm_client: Any,
    question: str,
    gold_answer: str,
    sample_id: str,
    dataset_name: str,
    max_round: int,
    history_scope: str,
    max_history_chars: int | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    from src.components.action_mapper import ActionMapper
    from src.components.agent_runner import AgentRunner
    from src.components.evaluator import Evaluator
    from src.components.recorder import Recorder
    from src.components.rollback_controller import RollbackController
    from src.components.state_store import StateStore
    from src.components.usage_logger import UsageLogger
    from src.main import AGENT_IDS, AGENT_ROLES
    from src.pipeline.debate_orchestrator import (
        DebateOrchestrator,
        DebateOrchestratorConfig,
    )
    from src.pipeline.normal_round_executor import (
        NormalRoundExecutor,
        NormalRoundExecutorConfig,
    )
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
    history_manager = RawFullHistoryManager(
        scope=history_scope,
        max_chars=max_history_chars,
    )
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
            max_round=max_round,
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
            max_round=max_round,
        ),
        state_store=state_store,
        normal_round_executor=normal_round_executor,
    )

    print(f"\n===== w/o history filtering: {sample_id} =====")
    print("Dataset:", dataset_name)
    print("Gold answer:", gold_answer)
    print(f"History scope: {history_scope}; max_history_chars={max_history_chars}")

    debate_result = debate_orchestrator.run_debate()
    rollback_context = debate_result["rollback_context"]
    early_stopped = debate_result["early_stopped"]

    if rollback_context:
        anchor_round = rollback_context.get("anchor_round")
        anchor_state = rollback_context.get("anchor_state")
        if anchor_round is not None and anchor_state is not None:
            print("Rollback detected; running repair with raw-history manager...")
            run_repair_mode_with_history_manager(
                llm_client=llm_client,
                question=question,
                rollback_context=rollback_context,
                state_store=state_store,
                history_manager=history_manager,
                usage_logger=usage_logger,
                sample_id=sample_id,
                dataset_name=dataset_name,
                max_round=max_round,
            )
        else:
            print("Rollback detected, but no valid anchor is available. Skip repair mode.")

    round_1_answers = get_round_1_answers(state_store)
    single_agent_baseline_answer = round_1_answers[0] if round_1_answers else ""
    majority_voting_baseline_answer = majority_vote(
        round_1_answers,
        dataset_name=dataset_name,
    )

    # New default finalizer.
    last_answers = get_latest_answers(state_store)
    final_answer = majority_vote(last_answers, dataset_name=dataset_name)

    usage_summary = build_usage_summary(usage_logger)

    result = {
        "sample_id": sample_id,
        "dataset_name": dataset_name,
        "question": question,
        "gold_answer": gold_answer,
        "ablation_name": "wo_history_filtering_raw_full_history",
        "history_scope": history_scope,
        "max_history_chars": max_history_chars,
        "state_recorder_internal_only": True,
        "history_manager_replaced": True,
        "evaluator_preserved": True,
        "action_mapper_preserved": True,
        "rollback_repair_preserved": True,
        "finalizer": "last_effective_round_majority_vote",
        "agent_roles": AGENT_ROLES,
        "round_1_answers": round_1_answers,
        "single_agent_baseline_answer": single_agent_baseline_answer,
        "majority_voting_baseline_answer": majority_voting_baseline_answer,
        "last_effective_round_answers": last_answers,
        "scrd_final_answer": final_answer,
        "wo_history_filtering_final_answer": final_answer,
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
        "scrd_correct": is_correct(final_answer, gold_answer, dataset_name=dataset_name),
        "wo_history_filtering_correct": is_correct(
            final_answer,
            gold_answer,
            dataset_name=dataset_name,
        ),
        "effective_rounds_used": get_effective_rounds_used(state_store),
        "actual_rounds_executed": get_actual_rounds_executed(state_store),
        "stop_reason": get_stop_reason(rollback_context, early_stopped),
        "single_agent_total_tokens": usage_summary["single_agent_total_tokens"],
        "majority_vote_total_tokens": usage_summary["majority_vote_total_tokens"],
        "scrd_total_tokens": usage_summary["scrd_total_tokens"],
        "wo_history_filtering_total_tokens": usage_summary["scrd_total_tokens"],
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
    trace["ablation_name"] = "wo_history_filtering_raw_full_history"
    trace["history_scope"] = history_scope
    trace["max_history_chars"] = max_history_chars
    trace["state_recorder_internal_only"] = True
    trace["history_manager_replaced"] = True
    trace["evaluator_preserved"] = True
    trace["action_mapper_preserved"] = True
    trace["rollback_repair_preserved"] = True
    trace["finalizer"] = "last_effective_round_majority_vote"

    return result, trace


def summarize_dataset(dataset: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)

    full_correct = sum(bool(r.get("full_scrd_correct")) for r in rows)
    ablation_correct = sum(bool(r.get("wo_history_filtering_correct")) for r in rows)

    full_wins = sum(
        bool(r.get("full_scrd_correct")) and not bool(r.get("wo_history_filtering_correct"))
        for r in rows
    )
    ablation_wins = sum(
        (not bool(r.get("full_scrd_correct"))) and bool(r.get("wo_history_filtering_correct"))
        for r in rows
    )
    both_correct = sum(
        bool(r.get("full_scrd_correct")) and bool(r.get("wo_history_filtering_correct"))
        for r in rows
    )
    both_wrong = sum(
        (not bool(r.get("full_scrd_correct")))
        and (not bool(r.get("wo_history_filtering_correct")))
        for r in rows
    )

    full_tokens = [
        float(r.get("full_scrd_total_tokens") or 0)
        for r in rows
        if r.get("full_scrd_total_tokens") is not None
    ]
    ablation_tokens = [
        float(r.get("wo_history_filtering_total_tokens") or 0)
        for r in rows
        if r.get("wo_history_filtering_total_tokens") is not None
    ]

    full_rounds = [
        float(r.get("full_actual_rounds_executed") or 0)
        for r in rows
        if r.get("full_actual_rounds_executed") is not None
    ]
    ablation_rounds = [
        float(r.get("actual_rounds_executed") or 0)
        for r in rows
        if r.get("actual_rounds_executed") is not None
    ]

    rollback_count = sum(r.get("stop_reason") == "rollback" for r in rows)
    early_stop_count = sum(r.get("stop_reason") == "early_stop" for r in rows)

    return {
        "dataset": dataset,
        "n": n,
        "full_scrd_correct": full_correct,
        "full_scrd_acc": full_correct / n if n else None,
        "wo_history_filtering_correct": ablation_correct,
        "wo_history_filtering_acc": ablation_correct / n if n else None,
        "delta_ablation_minus_full_pp": (
            100.0 * ((ablation_correct / n) - (full_correct / n)) if n else None
        ),
        "both_correct": both_correct,
        "both_wrong": both_wrong,
        "full_wins": full_wins,
        "ablation_wins": ablation_wins,
        "net_full_minus_ablation": full_wins - ablation_wins,
        "avg_full_scrd_tokens": (
            sum(full_tokens) / len(full_tokens) if full_tokens else None
        ),
        "avg_ablation_tokens": (
            sum(ablation_tokens) / len(ablation_tokens) if ablation_tokens else None
        ),
        "avg_token_delta_ablation_minus_full": (
            (sum(ablation_tokens) / len(ablation_tokens))
            - (sum(full_tokens) / len(full_tokens))
            if ablation_tokens and full_tokens
            else None
        ),
        "avg_full_actual_rounds": (
            sum(full_rounds) / len(full_rounds) if full_rounds else None
        ),
        "avg_ablation_rounds": (
            sum(ablation_rounds) / len(ablation_rounds) if ablation_rounds else None
        ),
        "ablation_rollback_count": rollback_count,
        "ablation_rollback_pct": rollback_count / n if n else None,
        "ablation_early_stop_count": early_stop_count,
        "ablation_early_stop_pct": early_stop_count / n if n else None,
    }


def add_overall(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    n = sum(int(s["n"]) for s in summaries)

    def sum_key(key: str) -> int:
        return sum(int(s.get(key) or 0) for s in summaries)

    def weighted_avg(key: str) -> float | None:
        if not n:
            return None
        values = [
            (float(s.get(key) or 0), int(s["n"]))
            for s in summaries
            if s.get(key) is not None
        ]
        denom = sum(w for _, w in values)
        if not denom:
            return None
        return sum(v * w for v, w in values) / denom

    full_correct = sum_key("full_scrd_correct")
    ablation_correct = sum_key("wo_history_filtering_correct")

    return {
        "dataset": "OVERALL",
        "n": n,
        "full_scrd_correct": full_correct,
        "full_scrd_acc": full_correct / n if n else None,
        "wo_history_filtering_correct": ablation_correct,
        "wo_history_filtering_acc": ablation_correct / n if n else None,
        "delta_ablation_minus_full_pp": (
            100.0 * ((ablation_correct / n) - (full_correct / n)) if n else None
        ),
        "both_correct": sum_key("both_correct"),
        "both_wrong": sum_key("both_wrong"),
        "full_wins": sum_key("full_wins"),
        "ablation_wins": sum_key("ablation_wins"),
        "net_full_minus_ablation": sum_key("net_full_minus_ablation"),
        "avg_full_scrd_tokens": weighted_avg("avg_full_scrd_tokens"),
        "avg_ablation_tokens": weighted_avg("avg_ablation_tokens"),
        "avg_token_delta_ablation_minus_full": weighted_avg(
            "avg_token_delta_ablation_minus_full"
        ),
        "avg_full_actual_rounds": weighted_avg("avg_full_actual_rounds"),
        "avg_ablation_rounds": weighted_avg("avg_ablation_rounds"),
        "ablation_rollback_count": sum_key("ablation_rollback_count"),
        "ablation_rollback_pct": sum_key("ablation_rollback_count") / n if n else None,
        "ablation_early_stop_count": sum_key("ablation_early_stop_count"),
        "ablation_early_stop_pct": sum_key("ablation_early_stop_count") / n if n else None,
    }


def fmt_pct(x: Any) -> str:
    if x is None:
        return "NA"
    return f"{100.0 * float(x):.2f}%"


def write_report(path: Path, summaries: list[dict[str, Any]]) -> None:
    lines: list[str] = []
    lines.append("# Ablation: w/o History Filtering / Raw Full History")
    lines.append("")
    lines.append(
        "Definition: preserve the internal State Recorder, evaluator, action mapper, "
        "rollback/repair, and last-round majority finalizer. Replace the normal "
        "filtered HistoryManager with an unfiltered raw full-history manager. "
        "Agents receive raw previous-round state history instead of selected top-k "
        "structured history units."
    )
    lines.append("")
    lines.append("| Dataset | N | Full Acc | Raw-History Acc | Δ Acc | Full wins | Ablation wins | Net Full-Ablation | Full tok | Raw-History tok |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

    for s in summaries:
        lines.append(
            "| {dataset} | {n} | {full_acc} | {abl_acc} | {delta:.2f} pp | {full_wins} | {abl_wins} | {net} | {full_tok:.1f} | {abl_tok:.1f} |".format(
                dataset=s["dataset"],
                n=s["n"],
                full_acc=fmt_pct(s["full_scrd_acc"]),
                abl_acc=fmt_pct(s["wo_history_filtering_acc"]),
                delta=float(s["delta_ablation_minus_full_pp"] or 0.0),
                full_wins=s["full_wins"],
                abl_wins=s["ablation_wins"],
                net=s["net_full_minus_ablation"],
                full_tok=float(s.get("avg_full_scrd_tokens") or 0.0),
                abl_tok=float(s.get("avg_ablation_tokens") or 0.0),
            )
        )

    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- This is not a full removal of the State Recorder. The recorder is retained internally because evaluator/action mapper and rollback require StateRecord objects.")
    lines.append("- The ablation removes selective history filtering from the agent input.")
    lines.append("- Default history scope is all previous rounds. Use `--history-scope last` for a cheaper last-round-only variant.")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Ablation runner: w/o History Filtering / Raw Full History. "
            "Preserves evaluator/action mapper/rollback/repair, but replaces "
            "filtered HistoryManager with raw full previous-round state history."
        )
    )
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--all-seven", action="store_true")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--max-round", type=int, default=5)
    parser.add_argument(
        "--history-scope",
        choices=["all", "last"],
        default="all",
        help="Use all previous rounds or only the immediately previous round.",
    )
    parser.add_argument(
        "--max-history-chars",
        type=int,
        default=None,
        help="Optional truncation cap for raw history text. Default: no truncation.",
    )
    parser.add_argument(
        "--full-root",
        type=Path,
        default=Path("outputs_with_last_round_majority_vote"),
        help="Reference Full SCRD outputs, preferably after last-round MV finalizer rewrite.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/ablation/wo_history_filtering_raw_full_history"),
    )
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.all_seven:
        args.datasets = ALL_SEVEN_DATASETS

    ensure_repo_imports(args.repo_root)

    os.environ["HTTP_PROXY"] = ""
    os.environ["HTTPS_PROXY"] = ""
    os.environ["NO_PROXY"] = "dashscope.aliyuncs.com,localhost,127.0.0.1"

    from dotenv import load_dotenv
    from src.main import build_llm_client, load_samples

    load_dotenv()
    llm_client = build_llm_client()

    args.output_root.mkdir(parents=True, exist_ok=True)

    all_summaries: list[dict[str, Any]] = []
    all_output_rows: list[dict[str, Any]] = []

    for dataset in args.datasets:
        print(f"\n######## Dataset: {dataset} ########")

        full_path = args.full_root / dataset / "results.jsonl"
        full_rows = load_jsonl(full_path)
        full_by_id = {str(r["sample_id"]): r for r in full_rows}

        samples = load_samples(dataset, limit=args.limit)
        sample_by_id = {str(sid): (str(sid), q, a) for sid, q, a in samples}
        sample_by_index = {
            idx: (str(sid), q, a)
            for idx, (sid, q, a) in enumerate(samples, start=1)
        }

        out_dir = args.output_root / dataset
        result_path = out_dir / "results.jsonl"
        error_path = out_dir / "errors.jsonl"
        trace_dir = out_dir / "traces"

        if args.overwrite:
            if result_path.exists():
                result_path.unlink()
            if error_path.exists():
                error_path.unlink()

        existing_ids: set[str] = set()
        if result_path.exists() and not args.overwrite:
            existing_ids = {str(r["sample_id"]) for r in load_jsonl(result_path)}

        for full_row in full_rows:
            sample_id = str(full_row["sample_id"])
            if sample_id in existing_ids:
                continue

            sample_tuple = get_sample_by_id_or_index(
                sample_id=sample_id,
                sample_by_id=sample_by_id,
                sample_by_index=sample_by_index,
            )

            if sample_tuple is None:
                append_jsonl(
                    error_path,
                    {
                        "sample_id": sample_id,
                        "dataset": dataset,
                        "error": "Sample not found by sample_id or numeric index.",
                    },
                )
                continue

            _, question, gold_answer = sample_tuple

            try:
                result, trace = run_one_sample(
                    llm_client=llm_client,
                    question=question,
                    gold_answer=gold_answer,
                    sample_id=sample_id,
                    dataset_name=dataset,
                    max_round=args.max_round,
                    history_scope=args.history_scope,
                    max_history_chars=args.max_history_chars,
                )

                result["full_scrd_final_answer"] = full_row.get("scrd_final_answer")
                result["full_scrd_correct"] = full_row.get("scrd_correct")
                result["full_scrd_total_tokens"] = full_row.get("scrd_total_tokens")
                result["full_effective_rounds_used"] = full_row.get("effective_rounds_used")
                result["full_actual_rounds_executed"] = full_row.get("actual_rounds_executed")
                result["full_stop_reason"] = full_row.get("stop_reason")
                result["answer_changed_vs_full_scrd"] = (
                    str(result.get("wo_history_filtering_final_answer"))
                    != str(full_row.get("scrd_final_answer"))
                )

                append_jsonl(result_path, result)
                write_json(trace_dir / f"{sample_id}_trace.json", trace)

                print(
                    "Saved {} | full_correct={} | raw_history_correct={} | answer={}".format(
                        sample_id,
                        full_row.get("scrd_correct"),
                        result.get("wo_history_filtering_correct"),
                        result.get("wo_history_filtering_final_answer"),
                    )
                )

            except KeyboardInterrupt:
                print("\nInterrupted by user. Partial progress saved.")
                raise
            except Exception as exc:
                append_jsonl(
                    error_path,
                    {
                        "sample_id": sample_id,
                        "dataset": dataset,
                        "question": question,
                        "gold_answer": gold_answer,
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    },
                )
                print(f"Failed {sample_id}: {exc}. Continuing...")

        dataset_rows = load_jsonl(result_path)
        summary = summarize_dataset(dataset, dataset_rows)
        write_json(out_dir / "summary.json", summary)

        all_summaries.append(summary)
        all_output_rows.extend(dataset_rows)

        print(
            "SUMMARY {}: full={:.2f}% | raw_history={:.2f}% | delta={:+.2f} pp | tokens raw-full={:+.1f}".format(
                dataset,
                100.0 * float(summary["full_scrd_acc"] or 0.0),
                100.0 * float(summary["wo_history_filtering_acc"] or 0.0),
                float(summary["delta_ablation_minus_full_pp"] or 0.0),
                float(summary["avg_token_delta_ablation_minus_full"] or 0.0),
            )
        )

    overall = add_overall(all_summaries)
    summaries_with_overall = all_summaries + [overall]

    write_csv(args.output_root / "summary.csv", summaries_with_overall)
    write_json(args.output_root / "summary.json", summaries_with_overall)
    write_jsonl(args.output_root / "sample_level_results.jsonl", all_output_rows)
    write_report(args.output_root / "report.md", summaries_with_overall)

    print("")
    print(f"Saved outputs to: {args.output_root}")
    print(
        "OVERALL full={:.2f}% | raw_history={:.2f}% | delta={:+.2f} pp | tokens raw-full={:+.1f}".format(
            100.0 * float(overall["full_scrd_acc"] or 0.0),
            100.0 * float(overall["wo_history_filtering_acc"] or 0.0),
            float(overall["delta_ablation_minus_full_pp"] or 0.0),
            float(overall["avg_token_delta_ablation_minus_full"] or 0.0),
        )
    )


if __name__ == "__main__":
    main()
