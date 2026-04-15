from __future__ import annotations

import json
import re
from typing import Any, Protocol

from src.schemas import RepairBrief, RepairScores, StateRecord


class LLMClientProtocol(Protocol):
    def generate(self, prompt: str) -> str:
        ...

    def generate_with_usage(self, prompt: str) -> dict[str, Any]:
        ...


class RepairEvaluator:
    """
    LLM-based evaluator for repair mode.

    Logic:
    - First repair round: evaluate anchor -> current, with repair_brief
    - Later repair rounds: evaluate previous_repair -> current
    """

    def __init__(
        self,
        llm_client: LLMClientProtocol,
        max_retries: int = 1,
        fallback_evaluator: Any | None = None,
        usage_logger: Any | None = None,
        sample_id: str | None = None,
    ) -> None:
        self.llm_client = llm_client
        self.max_retries = max_retries
        self.fallback_evaluator = fallback_evaluator
        self.usage_logger = usage_logger
        self.sample_id = sample_id

    def evaluate_repair(
        self,
        question: str,
        anchor_state: StateRecord,
        repair_brief: RepairBrief | None,
        current_state_record: StateRecord,
        previous_repair_state_record: StateRecord | None = None,
        round_id: int | None = None,
        sample_id: str | None = None,
        mode: str | None = "repair",
    ) -> RepairScores:
        prompt = self._build_prompt(
            question=question,
            anchor_state=anchor_state,
            repair_brief=repair_brief,
            current_state_record=current_state_record,
            previous_repair_state_record=previous_repair_state_record,
        )

        last_error: Exception | None = None

        for _ in range(self.max_retries + 1):
            try:
                raw_text, usage = self._generate_with_optional_usage(prompt)

                self._log_usage(
                    sample_id=sample_id or self.sample_id,
                    round_id=round_id,
                    mode=mode,
                    component="repair_evaluator",
                    agent_id=None,
                    usage=usage,
                )

                data = self._extract_json(raw_text)
                return self._parse_scores(data)

            except Exception as exc:
                last_error = exc

        if self.fallback_evaluator is not None:
            return self.fallback_evaluator.evaluate_repair(
                question=question,
                anchor_state=anchor_state,
                repair_brief=repair_brief,
                current_state_record=current_state_record,
                previous_repair_state_record=previous_repair_state_record,
            )

        raise RuntimeError(
            f"RepairEvaluator failed to generate scores after retries. "
            f"Last error: {last_error}"
        ) from last_error

    # ------------------------------------------------------------------
    # Prompt
    # ------------------------------------------------------------------
    def _build_prompt(
        self,
        question: str,
        anchor_state: StateRecord,
        repair_brief: RepairBrief | None,
        current_state_record: StateRecord,
        previous_repair_state_record: StateRecord | None,
    ) -> str:
        current_payload = self._build_state_view(current_state_record)

        # First repair round: anchor -> current
        if previous_repair_state_record is None:
            anchor_payload = self._build_state_view(anchor_state)
            repair_brief_payload = (
                repair_brief.model_dump() if repair_brief is not None else None
            )

            prompt = f"""
                You are a repair-stage evaluator.

                This is the FIRST repair round after rollback.
                Evaluate whether the current repair state improves over the anchor state
                and whether it addresses the repair brief.

                Return JSON only. No markdown. No extra text.

                Schema:
                {{
                "progress_score": 1-5,
                "information_quality_score": 1-5,
                "completion_readiness_score": 1-5,
                "rationale": "string"
                }}

                Scoring:
                - progress_score:
                1 = little or no repair progress
                3 = some repair progress
                5 = clear substantial repair progress

                - information_quality_score:
                1 = vague, repetitive, low-quality, or not useful
                3 = mixed quality
                5 = clear, coherent, specific, and useful

                - completion_readiness_score:
                1 = not ready to finalize
                3 = partly mature but still needs another repair round
                5 = ready to finalize

                Focus on:
                - whether the current state improves over the anchor state
                - whether the repair brief's remaining conflicts are being addressed
                - whether the current state is clear, stable, and useful
                - whether repair can stop now

                Use integer scores only.
                Keep rationale short (1-2 sentences).

                Question:
                {question}

                Anchor StateRecord:
                {json.dumps(anchor_payload, ensure_ascii=False, indent=2)}

                Repair Brief:
                {json.dumps(repair_brief_payload, ensure_ascii=False, indent=2)}

                Current Repair StateRecord:
                {json.dumps(current_payload, ensure_ascii=False, indent=2)}

                Return JSON only.
                """.strip()
            return prompt

        # Later repair rounds: previous_repair -> current
        previous_payload = self._build_state_view(previous_repair_state_record)

        prompt = f"""
            You are a repair-stage evaluator.

            This is a LATER repair round.
            Compare the PREVIOUS repair state and the CURRENT repair state.

            Return JSON only. No markdown. No extra text.

            Schema:
            {{
            "progress_score": 1-5,
            "information_quality_score": 1-5,
            "completion_readiness_score": 1-5,
            "rationale": "string"
            }}

            Scoring:
            - progress_score:
            1 = little or no progress
            3 = some progress
            5 = clear substantial progress

            - information_quality_score:
            1 = vague, repetitive, low-quality, or not useful
            3 = mixed quality
            5 = clear, coherent, specific, and useful

            - completion_readiness_score:
            1 = not ready to finalize
            3 = partly mature but still needs another repair round
            5 = ready to finalize

            Focus on:
            - whether the current repair state improves over the previous repair state
            - whether conflicts are becoming fewer, clearer, or more actionable
            - whether the current repair state is more stable and useful
            - whether another repair round is still necessary

            Use integer scores only.
            Keep rationale short (1-2 sentences).

            Question:
            {question}

            Previous Repair StateRecord:
            {json.dumps(previous_payload, ensure_ascii=False, indent=2)}

            Current Repair StateRecord:
            {json.dumps(current_payload, ensure_ascii=False, indent=2)}

            Return JSON only.
            """.strip()

        return prompt

    def _build_state_view(self, state_record: StateRecord) -> dict:
        return {
            "round_id": state_record.round_id,
            "current_answers": state_record.current_answers,
            "newly_added_claims": [
                claim.model_dump() for claim in state_record.newly_added_claims
            ],
            "unresolved_conflicts": [
                conflict.model_dump() for conflict in state_record.unresolved_conflicts
            ],
        }

    # ------------------------------------------------------------------
    # LLM helpers
    # ------------------------------------------------------------------
    def _generate_with_optional_usage(
        self,
        prompt: str,
    ) -> tuple[str, dict[str, int] | None]:
        if hasattr(self.llm_client, "generate_with_usage"):
            resp = self.llm_client.generate_with_usage(prompt)
            return resp["content"], resp.get("usage")

        raw_text = self.llm_client.generate(prompt)
        return raw_text, None

    def _log_usage(
        self,
        *,
        sample_id: str | None,
        round_id: int | None,
        mode: str | None,
        component: str,
        agent_id: str | None,
        usage: dict[str, Any] | None,
    ) -> None:
        if self.usage_logger is None:
            return

        self.usage_logger.log(
            sample_id=sample_id,
            round_id=round_id,
            mode=mode,
            component=component,
            agent_id=agent_id,
            usage=usage,
        )

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------
    def _extract_json(self, raw_text: str) -> dict[str, Any]:
        raw_text = raw_text.strip()

        try:
            data = json.loads(raw_text)
            if not isinstance(data, dict):
                raise ValueError("Top-level JSON is not an object.")
            return data
        except json.JSONDecodeError:
            pass

        match = re.search(r"\{.*\}", raw_text, re.DOTALL)
        if not match:
            raise ValueError("No JSON object found in model output.")

        json_block = match.group(0)
        data = json.loads(json_block)

        if not isinstance(data, dict):
            raise ValueError("Extracted JSON is not an object.")

        return data

    def _parse_scores(self, data: dict[str, Any]) -> RepairScores:
        progress = self._sanitize_score(data.get("progress_score"))
        info = self._sanitize_score(data.get("information_quality_score"))
        readiness = self._sanitize_score(data.get("completion_readiness_score"))
        rationale = self._sanitize_optional_string(data.get("rationale"))

        print("repair阶段打分结果: ", progress, info, readiness, rationale)

        return RepairScores(
            progress_score=progress,
            information_quality_score=info,
            completion_readiness_score=readiness,
            rationale=rationale,
        )

    # ------------------------------------------------------------------
    # Sanitizers
    # ------------------------------------------------------------------
    def _sanitize_score(self, value: Any) -> int:
        if isinstance(value, bool):
            value = int(value)
        elif isinstance(value, str):
            value = value.strip()
            if not value:
                value = 3

        try:
            numeric = float(value)
        except (TypeError, ValueError):
            numeric = 3.0

        rounded = int(round(numeric))
        return min(5, max(1, rounded))

    def _sanitize_optional_string(self, value: Any) -> str | None:
        if isinstance(value, str):
            value = value.strip()
            return value if value else None
        return None