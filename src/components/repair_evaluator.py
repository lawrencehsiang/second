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
        """
        Generate raw text from the model.
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

    Notes:
    - This is the primary implementation.
    - A fallback evaluator can be injected if desired.
    """

    def __init__(
        self,
        llm_client: LLMClientProtocol,
        max_retries: int = 1,
        fallback_evaluator: Any | None = None,
    ) -> None:
        self.llm_client = llm_client
        self.max_retries = max_retries
        self.fallback_evaluator = fallback_evaluator

    def evaluate_repair(
        self,
        question: str,
        anchor_state: StateRecord,
        repair_brief: RepairBrief,
        current_state_record: StateRecord,
        previous_repair_state_record: StateRecord | None = None,
    ) -> RepairScores:
        """
        Evaluate one repair-mode state and return RepairScores.

        Args:
            question: Original question.
            anchor_state: The healthy anchor state selected after rollback.
            repair_brief: Compact repair brief generated from the failed suffix.
            current_state_record: Current repair-round StateRecord.
            previous_repair_state_record: Previous repair-round StateRecord, if any.

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
                raw_text = self.llm_client.generate(prompt)
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
            f"RepairEvaluator failed to generate scores after retries. Last error: {last_error}"
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
You are a structured evaluator for the repair stage of a multi-agent debate system.

Your task is to evaluate the CURRENT repair-round state and output ONE JSON object only.

Do not output markdown.
Do not output explanation outside JSON.

JSON schema:
{{
  "progress_score": <int 1-5>,
  "information_quality_score": <int 1-5>,
  "completion_readiness_score": <int 1-5>,
  "rationale": <string>
}}

Repair-stage context:
- anchor_state: the last healthy state before rollback
- repair_brief: compact summary of what went wrong in the failed suffix and what conflicts remain
- previous_repair_state_record: the previous repair-round state if available
- current_state_record: the current repair-round state to evaluate

Scoring rubric:

1) progress_score
- 1: the current repair round makes almost no meaningful progress relative to the anchor/previous repair state
- 3: the current repair round makes moderate progress, clarifies some issues, or partially advances repair
- 5: the current repair round makes clear and substantial progress in addressing the repair brief

2) information_quality_score
- 1: the current repair state is vague, repetitive, low-quality, or not useful for repair
- 3: the current repair state is partially useful but still mixed in quality
- 5: the current repair state is clear, coherent, specific, and highly useful for resolving the remaining conflicts

3) completion_readiness_score
- 1: the current repair state is clearly not ready to finalize
- 3: the repair state is somewhat mature but still needs another repair round
- 5: the repair state is ready to finalize because the remaining conflicts are largely resolved or the answer state is sufficiently stable and justified

Evaluation guidance:
- Use anchor_state as the healthy rollback point.
- Use repair_brief as the core repair objective.
- If previous_repair_state_record is provided, compare whether the current repair round improves over it.
- Focus on:
  - whether the current state addresses the remaining_conflicts in the repair_brief
  - whether the current state avoids repeating the same failure pattern described in failure_summary
  - whether the current state is becoming clearer, more stable, and more decision-useful
  - whether another repair round is still necessary
- Keep rationale concise (1-3 sentences).
- Return integers only for the three scores.

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