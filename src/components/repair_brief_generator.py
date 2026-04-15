from __future__ import annotations

import json
import re
from typing import Any, Protocol

from pydantic import ValidationError

from src.schemas import (
    RemainingConflict,
    RepairBrief,
    RepairBriefGeneratorInput,
    StateRecord,
)


class LLMClientProtocol(Protocol):
    """
    Minimal protocol expected by RepairBriefGenerator.
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


class RepairBriefGenerator:
    """
    LLM-based RepairBriefGenerator.

    Responsibilities:
    1. Compress old failed suffix (t+1 ... k) into a compact repair brief.
    2. Extract only:
       - remaining_conflicts
       - failure_summary
    3. Return validated RepairBrief.
    4. Optionally log token usage.

    Notes:
    - This is the primary implementation.
    - A fallback generator can be injected if desired.
    """

    def __init__(
        self,
        llm_client: LLMClientProtocol,
        max_remaining_conflicts: int = 2,
        max_retries: int = 2,
        fallback_generator: Any | None = None,
        usage_logger: Any | None = None,
        sample_id: str | None = None,
    ) -> None:
        self.llm_client = llm_client
        self.max_remaining_conflicts = max_remaining_conflicts
        self.max_retries = max_retries
        self.fallback_generator = fallback_generator
        self.usage_logger = usage_logger
        self.sample_id = sample_id

    def generate_brief(
        self,
        generator_input: RepairBriefGeneratorInput,
        round_id: int | None = None,
        sample_id: str | None = None,
        mode: str | None = "repair",
    ) -> RepairBrief:
        """
        Generate RepairBrief from structured input object.
        """
        prompt = self._build_prompt(generator_input)
        last_error: Exception | None = None

        for _ in range(self.max_retries + 1):
            try:
                raw_text, usage = self._generate_with_optional_usage(prompt)

                self._log_usage(
                    sample_id=sample_id or self.sample_id,
                    round_id=round_id,
                    mode=mode,
                    component="repair_brief_generator",
                    agent_id=None,
                    usage=usage,
                )

                data = self._extract_json(raw_text)
                return self._parse_repair_brief(data)

            except Exception as exc:
                last_error = exc

        if self.fallback_generator is not None:
            return self.fallback_generator.generate_brief(generator_input)

        raise RuntimeError(
            f"RepairBriefGenerator failed after retries. "
            f"Last error: {last_error}"
        ) from last_error

    def generate_brief_from_parts(
        self,
        question: str,
        anchor_state: StateRecord,
        failed_suffix_state_records: list[StateRecord],
        round_id: int | None = None,
        sample_id: str | None = None,
        mode: str | None = "repair",
    ) -> RepairBrief:
        """
        Convenience wrapper when caller does not want to manually
        construct RepairBriefGeneratorInput.
        """
        generator_input = RepairBriefGeneratorInput(
            question=question,
            anchor_state=anchor_state,
            failed_suffix_state_records=failed_suffix_state_records,
        )
        return self.generate_brief(
            generator_input=generator_input,
            round_id=round_id,
            sample_id=sample_id,
            mode=mode,
        )

    # ------------------------------------------------------------------
    # Prompt
    # ------------------------------------------------------------------
    def _build_prompt(
        self,
        generator_input: RepairBriefGeneratorInput,
    ) -> str:
        payload = generator_input.model_dump()

        prompt = f"""
            You are a repair-brief generator for a multi-agent debate system.

            The anchor_state is the last healthy state before rollback.
            The failed_suffix_state_records are the later states that led to rollback.

            Your job is to output ONE compact JSON object with only:
            1. remaining_conflicts
            2. failure_summary

            Return JSON only. No markdown. No extra text.

            Schema:
            {{
            "remaining_conflicts": [
                {{
                "conflict": "string",
                "why_still_open": "string"
                }}
            ],
            "failure_summary": "string"
            }}

            Rules:
            - remaining_conflicts: keep only the most important unresolved conflicts still worth focusing on in repair mode.
            - Keep at most {self.max_remaining_conflicts} conflicts.
            - Merge duplicates if they refer to the same issue.
            - failure_summary: briefly explain why the failed suffix did not progress well.
            - Focus on patterns like stagnation, repeated support without progress, unresolved core conflict, or circular disagreement.
            - Keep the output compact, faithful, and non-redundant.
            - Do not hallucinate.
            - Do not output extra fields.

            Input:
            {json.dumps(payload, ensure_ascii=False, indent=2)}

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

    def _parse_repair_brief(self, data: dict[str, Any]) -> RepairBrief:
        remaining_conflicts = self._parse_remaining_conflicts(
            data.get("remaining_conflicts", [])
        )
        failure_summary = self._sanitize_optional_string(data.get("failure_summary"))

        if not failure_summary:
            failure_summary = (
                "The previous suffix did not resolve the core conflict "
                "and failed to produce stable progress."
            )

        return RepairBrief(
            remaining_conflicts=remaining_conflicts[: self.max_remaining_conflicts],
            failure_summary=failure_summary,
        )

    def _parse_remaining_conflicts(
        self,
        value: Any,
    ) -> list[RemainingConflict]:
        if not isinstance(value, list):
            return []

        results: list[RemainingConflict] = []

        for item in value:
            if not isinstance(item, dict):
                continue

            conflict = self._sanitize_optional_string(item.get("conflict"))
            why_still_open = self._sanitize_optional_string(item.get("why_still_open"))

            if not conflict:
                continue

            if not why_still_open:
                why_still_open = "The conflict remains unresolved."

            try:
                rc = RemainingConflict(
                    conflict=conflict,
                    why_still_open=why_still_open,
                )
                results.append(rc)
            except ValidationError:
                continue

        return self._deduplicate_remaining_conflicts(results)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _sanitize_optional_string(self, value: Any) -> str | None:
        if isinstance(value, str):
            value = value.strip()
            return value if value else None
        return None

    def _deduplicate_remaining_conflicts(
        self,
        conflicts: list[RemainingConflict],
    ) -> list[RemainingConflict]:
        seen = set()
        results: list[RemainingConflict] = []

        for item in conflicts:
            key = self._normalize_text(item.conflict)
            if key in seen:
                continue
            seen.add(key)
            results.append(item)

        return results

    def _normalize_text(self, text: str) -> str:
        return " ".join(text.strip().lower().split())