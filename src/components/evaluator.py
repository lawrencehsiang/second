from __future__ import annotations

import json
import re
from typing import Any, Protocol

from src.schemas import EvaluatorScores, StateRecord


class LLMClientProtocol(Protocol):
    """
    Minimal protocol expected by Evaluator.
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


class Evaluator:
    """
    LLM-based Evaluator.

    Responsibilities:
    1. Compare previous and current StateRecord.
    2. Output three scores:
       - progress_score
       - information_quality_score
       - future_utility_score
    3. Parse and validate JSON into EvaluatorScores.
    4. Optionally log token usage.

    Notes:
    - This is the primary Evaluator implementation.
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

    def evaluate_state(
        self,
        question: str,
        previous_state_record: StateRecord,
        current_state_record: StateRecord,
        round_id: int | None = None,
        sample_id: str | None = None,
        mode: str | None = "normal",
    ) -> EvaluatorScores:
        """
        Compare two adjacent StateRecords and return EvaluatorScores.

        Strategy:
        1. Build a structured evaluation prompt.
        2. Ask LLM to return JSON only.
        3. Parse + sanitize.
        4. Retry if parsing fails.
        5. Fall back to fallback_evaluator if available.
        """
        prompt = self._build_prompt(
            question=question,
            previous_state_record=previous_state_record,
            current_state_record=current_state_record,
        )

        last_error: Exception | None = None

        for _ in range(self.max_retries + 1):
            try:
                raw_text, usage = self._generate_with_optional_usage(prompt)

                self._log_usage(
                    sample_id=sample_id or self.sample_id,
                    round_id=round_id,
                    mode=mode,
                    component="evaluator",
                    agent_id=None,
                    usage=usage,
                )

                data = self._extract_json(raw_text)
                return self._parse_scores(data)

            except Exception as exc:
                last_error = exc

        if self.fallback_evaluator is not None:
            return self.fallback_evaluator.evaluate_state(
                question=question,
                previous_state_record=previous_state_record,
                current_state_record=current_state_record,
            )

        raise RuntimeError(
            f"Evaluator failed to generate scores after retries. "
            f"Last error: {last_error}"
        ) from last_error

    # ------------------------------------------------------------------
    # Prompt
    # ------------------------------------------------------------------
    def _build_prompt(
        self,
        question: str,
        previous_state_record: StateRecord,
        current_state_record: StateRecord,
    ) -> str:
        """
        Build the Evaluator prompt.
        """
        previous_payload = self._build_state_view(previous_state_record)
        current_payload = self._build_state_view(current_state_record)

        prompt = f"""
            You are a debate state evaluator.

            Compare the PREVIOUS state and the CURRENT state.
            Return JSON only. No markdown. No extra text.

            Schema:
            {{
            "progress_score": 1-5,
            "information_quality_score": 1-5,
            "future_utility_score": 1-5,
            "rationale": "string"
            }}

            Scoring:
            - progress_score:
            1 = little or no progress / regression
            3 = some progress
            5 = clear substantial progress

            - information_quality_score:
            1 = low-quality, vague, noisy, or contradictory
            3 = mixed quality
            5 = clear, coherent, specific, useful

            - future_utility_score:
            1 = low value for next round
            3 = somewhat useful
            5 = highly useful for continuation

            Focus on:
            - whether answers become more stable, informative, or justified
            - whether conflicts become fewer, clearer, or more actionable
            - whether newly added claims are useful and non-redundant
            - whether the current state is worth continuing from

            Use integer scores only.
            Keep rationale short (1-2 sentences).

            Question:
            {question}

            Previous StateRecord:
            {json.dumps(previous_payload, ensure_ascii=False, indent=2)}

            Current StateRecord:
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


    def _build_state_view(self, state_record: StateRecord) -> dict:
        return {
            "round_id": state_record.round_id,
            "current_answers": state_record.current_answers,
            "newly_added_claims": [claim.model_dump() for claim in state_record.newly_added_claims],
            "unresolved_conflicts": [conflict.model_dump() for conflict in state_record.unresolved_conflicts],
        }

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

    def _parse_scores(self, data: dict[str, Any]) -> EvaluatorScores:
        """
        Convert raw dict into validated EvaluatorScores.
        """
        progress = self._sanitize_score(data.get("progress_score"))
        info = self._sanitize_score(data.get("information_quality_score"))
        future = self._sanitize_score(data.get("future_utility_score"))
        rationale = self._sanitize_optional_string(data.get("rationale"))

        print("打分结果: ", progress, info, future, rationale)

        return EvaluatorScores(
            progress_score=progress,
            information_quality_score=info,
            future_utility_score=future,
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