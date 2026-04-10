from __future__ import annotations

import json
import re
from typing import Any, Protocol

from pydantic import ValidationError

from src.schemas import (
    RepairAgentInput,
    AgentOutputNormal,
    ConflictResponse,
)


class LLMClientProtocol(Protocol):
    """
    Minimal protocol expected by RepairAgentRunner.
    """

    def generate(self, prompt: str) -> str:
        """
        Generate raw text from the model.
        """
        ...


class RepairAgentRunner:
    """
    LLM-based runner for repair-mode agents.

    Responsibilities:
    1. Build prompts for repair-mode agents.
    2. Ask the LLM to return strict JSON.
    3. Parse and validate the JSON.
    4. Return AgentOutputNormal, even in repair mode.
    """

    def __init__(
        self,
        llm_client: LLMClientProtocol,
        max_retries: int = 2,
    ) -> None:
        self.llm_client = llm_client
        self.max_retries = max_retries

    def run_repair_round(
        self,
        agent_id: str,
        repair_agent_input: RepairAgentInput,
    ) -> AgentOutputNormal:
        """
        Execute one repair-mode round for an agent.
        """
        prompt = self._build_repair_prompt(
            agent_id=agent_id,
            repair_agent_input=repair_agent_input,
        )

        last_error: Exception | None = None

        for _ in range(self.max_retries + 1):
            try:
                raw_text = self.llm_client.generate(prompt)
                data = self._extract_json(raw_text)
                return self._parse_repair_output(
                    agent_id=agent_id,
                    data=data,
                )
            except Exception as exc:
                last_error = exc

        raise RuntimeError(
            f"RepairAgentRunner repair round failed after retries for agent {agent_id}. Last error: {last_error}"
        ) from last_error

    # ------------------------------------------------------------------
    # Prompt builders
    # ------------------------------------------------------------------

    def _build_repair_prompt(
        self,
        agent_id: str,
        repair_agent_input: RepairAgentInput,
    ) -> str:
        payload = repair_agent_input.model_dump()

        prompt = f"""
You are agent {agent_id} in a repair mode round (after rollback) of a multi-agent debate system.

Your task is to:
1. Analyze the provided repair_brief, which contains unresolved conflicts.
2. Respond to each conflict and provide a coherent answer to the question.
3. Explain your reasoning for each conflict response.

JSON output format:
{{
  "agent_id": "{agent_id}",
  "current_answer": <string>,   # Your answer to the question
  "response_to_conflicts": [
    {{
      "conflict": <string>,        # The conflict being responded to
      "response": <string>,        # Your response to this conflict
      "status": "resolved" | "partially_resolved" | "still_open"
    }}
  ],
  "brief_reason": <string>      # Brief reasoning for your answer
}}

Input:
{json.dumps(payload, ensure_ascii=False, indent=2)}

Return JSON only.
""".strip()

        return prompt

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

    def _parse_repair_output(
        self,
        agent_id: str,
        data: dict[str, Any],
    ) -> AgentOutputNormal:
        current_answer = self._sanitize_required_string(
            data.get("current_answer"),
            fallback="UNKNOWN",
        )
        brief_reason = self._sanitize_required_string(
            data.get("brief_reason"),
            fallback="No brief reason provided.",
        )

        response_to_conflicts = self._parse_conflict_responses(
            value=data.get("response_to_conflicts", []),
        )

        return AgentOutputNormal(
            agent_id=agent_id,
            current_answer=current_answer,
            response_to_conflicts=response_to_conflicts,
            brief_reason=brief_reason,
        )

    def _parse_conflict_responses(
        self,
        value: Any,
    ) -> list[ConflictResponse]:
        if not isinstance(value, list):
            return []

        results: list[ConflictResponse] = []

        for item in value:
            if not isinstance(item, dict):
                continue

            conflict = self._sanitize_optional_string(item.get("conflict"))
            response = self._sanitize_optional_string(item.get("response"))
            status = self._sanitize_conflict_status(item.get("status"))

            if not conflict or not response:
                continue

            try:
                results.append(
                    ConflictResponse(
                        conflict=conflict,
                        response=response,
                        status=status,
                    )
                )
            except ValidationError:
                continue

        return results

    # ------------------------------------------------------------------
    # Sanitizers
    # ------------------------------------------------------------------

    def _sanitize_required_string(
        self,
        value: Any,
        fallback: str,
    ) -> str:
        if isinstance(value, str):
            value = value.strip()
            if value:
                return value
        return fallback

    def _sanitize_optional_string(
        self,
        value: Any,
    ) -> str | None:
        if isinstance(value, str):
            value = value.strip()
            return value if value else None
        return None

    def _sanitize_conflict_status(
        self,
        value: Any,
    ) -> str:
        valid = {"resolved", "partially_resolved", "still_open"}
        if isinstance(value, str):
            value = value.strip().lower()
            if value in valid:
                return value
        return "still_open"