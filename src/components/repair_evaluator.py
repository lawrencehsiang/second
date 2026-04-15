from __future__ import annotations

import json
import re
from typing import Any, Protocol

from src.schemas import RepairBrief, RepairScores, StateRecord


class LLMClientProtocol(Protocol):
    """
    Minimal protocol expected by RepairEvaluator.
    """

    def generate(self, prompt: str) -> str:
        """Generate raw text from the model."""
        ...

    def generate_with_usage(self, prompt: str) -> dict[str, Any]:
        """
        Optional richer interface:
        {
            "content": str,
            "usage": {
                "prompt_tokens": int,
                "completion_tokens": int,
                "total_tokens": int
            },
            "raw_response": dict
        }
        """
        ...


class RepairEvaluator:
    """
    LLM-based evaluator for repair mode.

    Responsibilities:
    1. Evaluate the current repair-round state in the context of:
       - anchor_state
       - repair_brief
       - optional previous repair state
    2. Output three scores:
       - progress_score
       - information_quality_score
       - completion_readiness_score
    3. Parse and validate JSON into RepairScores
    4. Optionally log token usage

    Notes:
    - This is the primary implementation.
    - A fallback evaluator can be injected if desired.
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
        repair_brief: RepairBrief,
        current_state_record: StateRecord,
        previous_repair_state_record: StateRecord | None = None,
        round_id: int | None = None,
        sample_id: str | None = None,
        mode: str | None = "repair",
    ) -> RepairScores:
        """
        Evaluate one repair-mode state and return RepairScores.

        Args:
            question: Original question.
            anchor_state: The healthy anchor state selected after rollback.
            repair_brief: Compact repair brief generated from the failed suffix.
            current_state_record: Current repair-round StateRecord.
            previous_repair_state_record: Previous repair-round StateRecord, if any.
            round_id: Optional round id for usage logging.
            sample_id: Optional sample id for usage logging.
            mode: Optional mode tag, default "repair".

        Returns:
            RepairScores
        """
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
        repair_brief: RepairBrief,
        current_state_record: StateRecord,
        previous_repair_state_record: StateRecord | None,
    ) -> str:
        """
        Build the repair evaluator prompt.
        """
        anchor_payload = anchor_state.model_dump()
        repair_brief_payload = repair_brief.model_dump()
        current_payload = current_state_record.model_dump()
        previous_repair_payload = (
            previous_repair_state_record.model_dump()
            if previous_repair_state_record is not None
            else None
        )

        prompt = f"""
            You are a repair-stage evaluator for a multi-agent debate system.

            Evaluate the CURRENT repair state using:
            - anchor_state
            - repair_brief
            - previous_repair_state_record (if any)
            - current_state_record

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
            - whether remaining conflicts in the repair brief are being addressed
            - whether the current repair state avoids repeating the failure pattern
            - whether the repair state is becoming clearer, more stable, and more useful
            - whether another repair round is still necessary

            Use integer scores only.
            Keep rationale short (1-2 sentences).

            Question:
            {question}

            Anchor StateRecord:
            {json.dumps(anchor_payload, ensure_ascii=False, indent=2)}

            Repair Brief:
            {json.dumps(repair_brief_payload, ensure_ascii=False, indent=2)}

            Previous Repair StateRecord (may be null):
            {json.dumps(previous_repair_payload, ensure_ascii=False, indent=2)}

            Current Repair StateRecord:
            {json.dumps(current_payload, ensure_ascii=False, indent=2)}

            Return JSON only.
            """.strip()

        return prompt

    # ------------------------------------------------------------------
    # LLM helpers
    # ------------------------------------------------------------------
    def _generate_with_optional_usage(
        self,
        prompt: str,
    ) -> tuple[str, dict[str, int] | None]:
        """
        Prefer generate_with_usage if available; otherwise fall back to generate.
        """
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
        """
        Extract a JSON object from raw model text.

        Strategy:
        1. Try direct json.loads
        2. If that fails, extract first {...} block
        """
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
        """
        Convert raw dict into validated RepairScores.
        """
        progress = self._sanitize_score(data.get("progress_score"))
        info = self._sanitize_score(data.get("information_quality_score"))
        readiness = self._sanitize_score(data.get("completion_readiness_score"))
        rationale = self._sanitize_optional_string(data.get("rationale"))

        print("rollback阶段打分结果: ", progress, info, readiness, rationale)

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