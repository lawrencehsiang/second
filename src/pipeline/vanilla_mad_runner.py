# src/pipeline/vanilla_mad_runner.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.schemas import AgentInputRound1
from src.utils.vanilla_result_utils import (
    build_vanilla_result_record,
    build_vanilla_trace_bundle,
)


@dataclass
class VanillaMADRunnerConfig:
    question: str
    gold_answer: str
    dataset_name: str
    sample_id: str
    agent_ids: list[str] = field(default_factory=lambda: ["A", "B", "C"])
    max_round: int = 7


class VanillaMADRunner:
    """
    Run a fixed-round vanilla MAD baseline.

    Design goals:
    - Reuse existing AgentRunner.run_round_1(...)
    - Use AgentRunner.run_vanilla_round(...)
    - Do NOT use recorder / evaluator / rollback / early stop
    - Keep debate logic minimal and explicit
    - Delegate trace / usage summary / result record to vanilla_result_utils
    """

    def __init__(
        self,
        *,
        config: VanillaMADRunnerConfig,
        agent_runner: Any,
        usage_logger: Any,
    ) -> None:
        self.config = config
        self.agent_runner = agent_runner
        self.usage_logger = usage_logger

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(self) -> tuple[dict[str, Any], dict[str, Any]]:
        self._validate_config()

        outputs_by_round: dict[int, dict[str, dict[str, Any]]] = {}

        # ---------------------------
        # Round 1: independent answers
        # ---------------------------
        round1_outputs: dict[str, dict[str, Any]] = {}
        for agent_id in self.config.agent_ids:
            out = self.agent_runner.run_round_1(
                agent_id=agent_id,
                agent_input=AgentInputRound1(question=self.config.question),
                round_id=1,
                sample_id=self.config.sample_id,
            )
            round1_outputs[agent_id] = self._coerce_output_dict(out, round_id=1)

        outputs_by_round[1] = round1_outputs

        # ---------------------------
        # Round 2 ~ max_round: vanilla debate
        # ---------------------------
        for round_id in range(2, self.config.max_round + 1):
            previous_round_outputs = outputs_by_round[round_id - 1]
            current_round_outputs: dict[str, dict[str, Any]] = {}

            for agent_id in self.config.agent_ids:
                own_previous_answer = previous_round_outputs[agent_id]["current_answer"]
                peer_previous_answers = {
                    other_id: previous_round_outputs[other_id]["current_answer"]
                    for other_id in self.config.agent_ids
                    if other_id != agent_id
                }

                out = self.agent_runner.run_vanilla_round(
                    question=self.config.question,
                    agent_id=agent_id,
                    round_id=round_id,
                    sample_id=self.config.sample_id,
                    own_previous_answer=own_previous_answer,
                    peer_previous_answers=peer_previous_answers,
                )
                current_round_outputs[agent_id] = self._coerce_output_dict(
                    out,
                    round_id=round_id,
                )

            outputs_by_round[round_id] = current_round_outputs

        usage_summary = self._build_usage_summary()
        result = self._build_result_record(
            outputs_by_round=outputs_by_round,
            usage_summary=usage_summary,
        )
        trace_bundle = self._build_trace_bundle(
            outputs_by_round=outputs_by_round,
        )

        return result, trace_bundle

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def _validate_config(self) -> None:
        if not self.config.question or not self.config.question.strip():
            raise ValueError("config.question must be a non-empty string.")
        if not self.config.gold_answer or not str(self.config.gold_answer).strip():
            raise ValueError("config.gold_answer must be a non-empty string.")
        if self.config.max_round < 1:
            raise ValueError("config.max_round must be >= 1.")
        if len(self.config.agent_ids) < 2:
            raise ValueError("config.agent_ids must contain at least 2 agents.")

    # ------------------------------------------------------------------
    # Output coercion
    # ------------------------------------------------------------------
    def _coerce_output_dict(
        self,
        output: Any,
        *,
        round_id: int,
    ) -> dict[str, Any]:
        """
        Accept either:
        - Pydantic model with model_dump()
        - plain dict
        - any object with agent_id/current_answer/brief_reason attrs
        """
        if hasattr(output, "model_dump"):
            data = output.model_dump()
        elif isinstance(output, dict):
            data = dict(output)
        else:
            data = {
                "agent_id": getattr(output, "agent_id", None),
                "current_answer": getattr(output, "current_answer", None),
                "brief_reason": getattr(output, "brief_reason", None),
            }

        agent_id = str(data.get("agent_id") or "").strip()
        current_answer = str(data.get("current_answer") or "").strip()
        brief_reason = str(data.get("brief_reason") or "").strip()

        return {
            "agent_id": agent_id,
            "current_answer": current_answer,
            "brief_reason": brief_reason,
            "round_id": round_id,
        }

    # ------------------------------------------------------------------
    # Delegated builders
    # ------------------------------------------------------------------
    def _build_usage_summary(self) -> dict[str, Any]:
        summary = build_vanilla_usage_summary_wrapper(
            usage_logger=self.usage_logger,
            sample_id=self.config.sample_id,
            max_round=self.config.max_round,
        )
        return summary

    def _build_result_record(
        self,
        *,
        outputs_by_round: dict[int, dict[str, dict[str, Any]]],
        usage_summary: dict[str, Any],
    ) -> dict[str, Any]:
        return build_vanilla_result_record(
            sample_id=self.config.sample_id,
            dataset_name=self.config.dataset_name,
            question=self.config.question,
            gold_answer=self.config.gold_answer,
            outputs_by_round=outputs_by_round,
            agent_ids=self.config.agent_ids,
            max_round=self.config.max_round,
            usage_summary=usage_summary,
        )

    def _build_trace_bundle(
        self,
        *,
        outputs_by_round: dict[int, dict[str, dict[str, Any]]],
    ) -> dict[str, Any]:
        return build_vanilla_trace_bundle(
            outputs_by_round=outputs_by_round,
            agent_ids=self.config.agent_ids,
            dataset_name=self.config.dataset_name,
            usage_logger=self.usage_logger,
            sample_id=self.config.sample_id,
            max_round=self.config.max_round,
        )


# ----------------------------------------------------------------------
# Thin wrapper to keep runner code readable
# ----------------------------------------------------------------------
def build_vanilla_usage_summary_wrapper(
    *,
    usage_logger,
    sample_id: str,
    max_round: int,
) -> dict[str, Any]:
    from src.utils.vanilla_result_utils import build_vanilla_usage_summary

    usage_summary = build_vanilla_usage_summary(
        usage_logger=usage_logger,
        sample_id=sample_id,
        max_round=max_round,
    )

    # result record 不需要把 raw usage_records 整个塞进去
    return {k: v for k, v in usage_summary.items() if k != "usage_records"}