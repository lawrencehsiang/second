from __future__ import annotations

import json
import re
from typing import Any, Protocol

from pydantic import ValidationError

from src.schemas import (
    AgentInputNormal,
    AgentInputRound1,
    AgentOutputNormal,
    AgentOutputRound1,
    ConflictResponse,
)


class LLMClientProtocol(Protocol):
    """
    Minimal protocol expected by AgentRunner.
    """

    def generate(self, prompt: str) -> str:
        """
        Generate raw text from the model.
        """
        ...


class AgentRunner:
    """
    LLM-based runner for normal-stage agents.

    Responsibilities:
    1. Build prompts for round 1 and normal rounds.
    2. Ask the LLM to return strict JSON.
    3. Parse and validate the JSON.
    4. Return AgentOutputRound1 / AgentOutputNormal.
    """

    def __init__(
        self,
        llm_client: LLMClientProtocol,
        max_retries: int = 2,
    ) -> None:
        self.llm_client = llm_client
        self.max_retries = max_retries

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_round_1(
        self,
        agent_id: str,
        agent_input: AgentInputRound1,
    ) -> AgentOutputRound1:
        prompt = self._build_round_1_prompt(
            agent_id=agent_id,
            agent_input=agent_input,
        )

        last_error: Exception | None = None

        for _ in range(self.max_retries + 1):
            try:
                raw_text = self.llm_client.generate(prompt)
                data = self._extract_json(raw_text)
                return self._parse_round_1_output(
                    agent_id=agent_id,
                    data=data,
                )
            except Exception as exc:
                last_error = exc

        raise RuntimeError(
            f"AgentRunner round 1 failed after retries for agent {agent_id}. Last error: {last_error}"
        ) from last_error

    def run_normal_round(
        self,
        agent_id: str,
        agent_input: AgentInputNormal,
    ) -> AgentOutputNormal:
        prompt = self._build_normal_round_prompt(
            agent_id=agent_id,
            agent_input=agent_input,
        )

        last_error: Exception | None = None

        for _ in range(self.max_retries + 1):
            try:
                raw_text = self.llm_client.generate(prompt)
                data = self._extract_json(raw_text)
                return self._parse_normal_round_output(
                    agent_id=agent_id,
                    data=data,
                    agent_input=agent_input,
                )
            except Exception as exc:
                last_error = exc

        raise RuntimeError(
            f"AgentRunner normal round failed after retries for agent {agent_id}. Last error: {last_error}"
        ) from last_error

    # ------------------------------------------------------------------
    # Prompt builders
    # ------------------------------------------------------------------

    def _build_round_1_prompt(
        self,
        agent_id: str,
        agent_input: AgentInputRound1,
    ) -> str:
        payload = agent_input.model_dump()

        prompt = f"""
You are agent {agent_id} in round 1 of a multi-agent debate system.

This is the independent initialization round.
You must answer the question independently.
Do not assume access to other agents' outputs.

Return JSON only.
Do not output markdown.
Do not output explanation outside JSON.

JSON schema:
{{
  "agent_id": "{agent_id}",
  "current_answer": <string>,
  "brief_reason": <string>
}}

Requirements:
- current_answer: your current best answer to the question
- brief_reason: a short reason explaining your answer
- Keep the answer concise but meaningful
- Be faithful to the question only

Input:
{json.dumps(payload, ensure_ascii=False, indent=2)}

Return JSON only.
""".strip()

        return prompt

    def _build_normal_round_prompt(
        self,
        agent_id: str,
        agent_input: AgentInputNormal,
    ) -> str:
        payload = agent_input.model_dump()

        prompt = f"""
You are agent {agent_id} in a normal round (t >= 2) of a multi-agent debate system.

You are given:
- the original question
- your own previous-round answer
- structured historical information selected by the system

You must:
1. produce your current answer
2. respond to the unresolved conflicts contained in the input history_units
3. provide a short reason

Important behavioral rules:
- You are NOT fixed to your previous answer; you may keep it or change it
- You do NOT see other agents' new outputs from this same round
- You should use only the provided structured information
- Focus on unresolved conflicts and useful progress
- Do not output judge-like meta-evaluation
- Keep your response concise and structured

Return JSON only.
Do not output markdown.
Do not output explanation outside JSON.

JSON schema:
{{
  "agent_id": "{agent_id}",
  "current_answer": <string>,
  "response_to_conflicts": [
    {{
      "conflict": <string>,
      "response": <string>,
      "status": "resolved" | "partially_resolved" | "still_open"
    }}
  ],
  "brief_reason": <string>
}}

Rules for response_to_conflicts:
- You should respond to the unresolved conflicts represented in history_units of type "core_unresolved_conflict"
- If no such history unit exists, response_to_conflicts may be an empty list
- status meanings:
  - resolved: you think this conflict is now fully resolved
  - partially_resolved: some progress is made but not fully solved
  - still_open: the conflict remains unresolved

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

    def _parse_round_1_output(
        self,
        agent_id: str,
        data: dict[str, Any],
    ) -> AgentOutputRound1:
        current_answer = self._sanitize_required_string(
            data.get("current_answer"),
            fallback="UNKNOWN",
        )
        brief_reason = self._sanitize_required_string(
            data.get("brief_reason"),
            fallback="No brief reason provided.",
        )

        return AgentOutputRound1(
            agent_id=agent_id,
            current_answer=current_answer,
            brief_reason=brief_reason,
        )

    def _parse_normal_round_output(
        self,
        agent_id: str,
        data: dict[str, Any],
        agent_input: AgentInputNormal,
    ) -> AgentOutputNormal:
        current_answer = self._sanitize_required_string(
            data.get("current_answer"),
            fallback=agent_input.own_previous_answer,
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