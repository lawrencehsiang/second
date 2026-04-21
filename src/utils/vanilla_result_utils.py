# src/utils/vanilla_result_utils.py
from __future__ import annotations

from typing import Any

from src.utils.result_utils import majority_vote, is_correct


# ------------------------------------------------------------------
# Trace builders
# ------------------------------------------------------------------
def build_vanilla_trace(
    outputs_by_round: dict[int, dict[str, dict[str, Any]]],
    *,
    agent_ids: list[str],
    dataset_name: str,
) -> list[dict[str, Any]]:
    """
    Build a simple vanilla trace:
    [
      {
        "round_id": 1,
        "current_answers": [...],
        "agent_outputs": {...},
        "majority_answer": "..."
      },
      ...
    ]
    """
    trace: list[dict[str, Any]] = []

    for round_id in sorted(outputs_by_round.keys()):
        round_outputs = outputs_by_round[round_id]

        current_answers = [
            str(round_outputs[agent_id].get("current_answer", ""))
            for agent_id in agent_ids
        ]
        majority_answer = majority_vote(current_answers, dataset_name)

        trace.append(
            {
                "round_id": round_id,
                "current_answers": current_answers,
                "agent_outputs": round_outputs,
                "majority_answer": majority_answer,
            }
        )

    return trace


def build_vanilla_trace_bundle(
    outputs_by_round: dict[int, dict[str, dict[str, Any]]],
    *,
    agent_ids: list[str],
    dataset_name: str,
    usage_logger,
    sample_id: str,
    max_round: int,
) -> dict[str, Any]:
    usage_summary = build_vanilla_usage_summary(
        usage_logger=usage_logger,
        sample_id=sample_id,
        max_round=max_round,
    )
    return {
        "final_trace": build_vanilla_trace(
            outputs_by_round,
            agent_ids=agent_ids,
            dataset_name=dataset_name,
        ),
        "usage_records": usage_summary["usage_records"],
        "usage_summary": {k: v for k, v in usage_summary.items() if k != "usage_records"},
    }


# ------------------------------------------------------------------
# Usage summary
# ------------------------------------------------------------------
def build_vanilla_usage_summary(
    *,
    usage_logger,
    sample_id: str,
    max_round: int,
    relevant_components: set[str] | None = None,
) -> dict[str, Any]:
    """
    Build vanilla-only token summary.

    By default, only count:
    - round 1: component == "agent_round_1"
    - round >= 2: component == "agent_vanilla"

    This intentionally excludes recorder / evaluator / repair components.
    """
    if relevant_components is None:
        relevant_components = {"agent_round_1", "agent_vanilla"}

    all_records = usage_logger.list_records()

    usage_records = [
        record
        for record in all_records
        if record.get("sample_id") == sample_id
        and record.get("component") in relevant_components
    ]

    vanilla_total_tokens = sum(int(r.get("total_tokens", 0)) for r in usage_records)
    vanilla_prompt_tokens = sum(int(r.get("prompt_tokens", 0)) for r in usage_records)
    vanilla_completion_tokens = sum(int(r.get("completion_tokens", 0)) for r in usage_records)

    vanilla_round_total_tokens: dict[str, int] = {}
    vanilla_round_prompt_tokens: dict[str, int] = {}
    vanilla_round_completion_tokens: dict[str, int] = {}

    for round_id in range(1, max_round + 1):
        round_records = [
            r for r in usage_records if int(r.get("round_id") or 0) == round_id
        ]
        key = str(round_id)

        vanilla_round_total_tokens[key] = sum(
            int(r.get("total_tokens", 0)) for r in round_records
        )
        vanilla_round_prompt_tokens[key] = sum(
            int(r.get("prompt_tokens", 0)) for r in round_records
        )
        vanilla_round_completion_tokens[key] = sum(
            int(r.get("completion_tokens", 0)) for r in round_records
        )

    vanilla_cumulative_total_tokens: dict[str, int] = {}
    vanilla_cumulative_prompt_tokens: dict[str, int] = {}
    vanilla_cumulative_completion_tokens: dict[str, int] = {}

    running_total = 0
    running_prompt = 0
    running_completion = 0

    for round_id in range(1, max_round + 1):
        key = str(round_id)

        running_total += vanilla_round_total_tokens[key]
        running_prompt += vanilla_round_prompt_tokens[key]
        running_completion += vanilla_round_completion_tokens[key]

        vanilla_cumulative_total_tokens[key] = running_total
        vanilla_cumulative_prompt_tokens[key] = running_prompt
        vanilla_cumulative_completion_tokens[key] = running_completion

    vanilla_component_total_tokens: dict[str, int] = {}
    for record in usage_records:
        component = str(record.get("component") or "unknown")
        vanilla_component_total_tokens[component] = (
            vanilla_component_total_tokens.get(component, 0)
            + int(record.get("total_tokens", 0))
        )

    return {
        "usage_records": usage_records,
        "vanilla_total_tokens": vanilla_total_tokens,
        "vanilla_prompt_tokens": vanilla_prompt_tokens,
        "vanilla_completion_tokens": vanilla_completion_tokens,
        "vanilla_round_total_tokens": vanilla_round_total_tokens,
        "vanilla_round_prompt_tokens": vanilla_round_prompt_tokens,
        "vanilla_round_completion_tokens": vanilla_round_completion_tokens,
        "vanilla_cumulative_total_tokens": vanilla_cumulative_total_tokens,
        "vanilla_cumulative_prompt_tokens": vanilla_cumulative_prompt_tokens,
        "vanilla_cumulative_completion_tokens": vanilla_cumulative_completion_tokens,
        "vanilla_component_total_tokens": vanilla_component_total_tokens,
    }


# ------------------------------------------------------------------
# Result record
# ------------------------------------------------------------------
def build_vanilla_result_record(
    *,
    sample_id: str,
    dataset_name: str,
    question: str,
    gold_answer: str,
    outputs_by_round: dict[int, dict[str, dict[str, Any]]],
    agent_ids: list[str],
    max_round: int,
    usage_summary: dict[str, Any],
) -> dict[str, Any]:
    """
    Build one JSONL-ready result record for a vanilla MAD sample.
    """
    vanilla_round_answers: dict[str, dict[str, str]] = {}
    vanilla_majority_answers: dict[str, str] = {}

    for round_id in sorted(outputs_by_round.keys()):
        round_outputs = outputs_by_round[round_id]
        current_answers = [
            str(round_outputs[agent_id].get("current_answer", ""))
            for agent_id in agent_ids
        ]
        vanilla_majority_answers[str(round_id)] = majority_vote(
            current_answers,
            dataset_name,
        )
        vanilla_round_answers[str(round_id)] = {
            agent_id: str(round_outputs[agent_id].get("current_answer", ""))
            for agent_id in agent_ids
        }

    vanilla_final_answer = vanilla_majority_answers[str(max_round)]
    vanilla_final_correct = is_correct(
        vanilla_final_answer,
        gold_answer,
        dataset_name,
    )

    result: dict[str, Any] = {
        "sample_id": sample_id,
        "dataset_name": dataset_name,
        "question": question,
        "gold_answer": gold_answer,
        "max_round": max_round,
        "agent_ids": agent_ids,
        "vanilla_round_answers": vanilla_round_answers,
        "vanilla_majority_answers": vanilla_majority_answers,
        "vanilla_final_answer": vanilla_final_answer,
        "vanilla_final_correct": vanilla_final_correct,
        **usage_summary,
    }

    # Flat keys for downstream analysis convenience
    for target_round in (1, 3, 5, 7):
        if target_round <= max_round and str(target_round) in vanilla_majority_answers:
            answer = vanilla_majority_answers[str(target_round)]
            result[f"round{target_round}_majority_answer"] = answer
            result[f"round{target_round}_majority_correct"] = is_correct(
                answer,
                gold_answer,
                dataset_name,
            )
            result[f"round{target_round}_cumulative_total_tokens"] = (
                usage_summary["vanilla_cumulative_total_tokens"][str(target_round)]
            )
            result[f"round{target_round}_cumulative_prompt_tokens"] = (
                usage_summary["vanilla_cumulative_prompt_tokens"][str(target_round)]
            )
            result[f"round{target_round}_cumulative_completion_tokens"] = (
                usage_summary["vanilla_cumulative_completion_tokens"][str(target_round)]
            )

    return result