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
    """Minimal protocol expected by AgentRunner."""

    def generate(self, prompt: str) -> str:
        """Generate raw text from the model."""
        ...


class AgentRunner:
    """
    LLM-based runner for normal-stage agents.

    Responsibilities:
    1. Build prompts for round 1 and normal rounds
    2. Ask the LLM to return strict JSON
    3. Parse basic JSON fields into schema objects
    4. Keep protocol simple and robust

    Note:
    - This version intentionally does NOT implement semantic consistency parsing.
    - It relies mainly on prompt design + output order to reduce inconsistency.
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
                # print(f"Raw model output for round 1 agent {agent_id}: {raw_text}")

                data = self._extract_json(raw_text)
                output = self._parse_round_1_output(
                    agent_id=agent_id,
                    data=data,
                )
                # print(f"Round 1 output for agent {agent_id}: {output.model_dump()}")
                return output
            except Exception as exc:
                last_error = exc

        raise RuntimeError(
            f"AgentRunner round 1 failed after retries for agent {agent_id}. "
            f"Last error: {last_error}"
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
                # print(f"Raw model output for agent {agent_id}: {raw_text}")

                data = self._extract_json(raw_text)
                output = self._parse_normal_round_output(
                    agent_id=agent_id,
                    data=data,
                )
                # print(f"Normal round output for agent {agent_id}: {output.model_dump()}")
                return output
            except Exception as exc:
                last_error = exc

        raise RuntimeError(
            f"AgentRunner normal round failed after retries for agent {agent_id}. "
            f"Last error: {last_error}"
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
You do not have access to any other agents' answers.

Return JSON only.
Do not output markdown.
Do not output any explanation outside JSON.

Output JSON schema:
{{
  "agent_id": "{agent_id}",
  "brief_reason": "string",
  "current_answer": "string"
}}

IMPORTANT RULES:
1. First decide your reasoning briefly.
2. Then output your final answer in current_answer.
3. current_answer is the final answer for this round.
4. Keep brief_reason short but informative.
5. Do not include extra fields.

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
- your own previous answer
- structured historical information selected by the system

You do NOT see other agents' new outputs from this same round.
You may keep your previous answer or revise it.

Return JSON only.
Do not output markdown.
Do not output any explanation outside JSON.

Output JSON schema:
{{
  "agent_id": "{agent_id}",
  "response_to_conflicts": [
    {{
      "conflict": "string",
      "response": "string",
      "status": "resolved|partially_resolved|still_open"
    }}
  ],
  "brief_reason": "string",
  "current_answer": "string"
}}

VERY IMPORTANT RULES:
1. You must complete the fields in this order:
   (a) response_to_conflicts
   (b) brief_reason
   (c) current_answer

2. current_answer is the FINAL answer for this round.

3. If your reasoning changes anywhere in response_to_conflicts or brief_reason,
   you MUST update current_answer so that it matches your final view.

4. current_answer is the single source of truth used by the system.

5. Do NOT let current_answer disagree with response_to_conflicts or brief_reason.

6. If you revise your answer during reasoning, the final revised answer must appear in current_answer.

7. Keep response_to_conflicts concise and directly tied to the structured conflicts in the input.

8. If there is no true unresolved conflict to respond to, response_to_conflicts may be [].

Field instructions:
- response_to_conflicts:
  Respond only to the unresolved conflicts represented in the structured input.
  Each item should contain:
  - conflict: the conflict text
  - response: your direct response to that conflict
  - status:
      resolved = fully addressed
      partially_resolved = some progress but not fully solved
      still_open = remains unresolved

- brief_reason:
  A short summary of why you keep or revise your answer.

- current_answer:
  The final answer after considering all conflict responses and reasoning above.

Input:
{json.dumps(payload, ensure_ascii=False, indent=2)}

Return JSON only.
        """.strip()
        return prompt

    # ------------------------------------------------------------------
    # JSON extraction
    # ------------------------------------------------------------------
    def _extract_json(self, raw_text: str) -> dict[str, Any]:
        raw_text = raw_text.strip()

        # Case 1: pure JSON
        try:
            data = json.loads(raw_text)
            if not isinstance(data, dict):
                raise ValueError("Top-level JSON is not an object.")
            return data
        except json.JSONDecodeError:
            pass

        # Case 2: JSON embedded in text
        match = re.search(r"\{.*\}", raw_text, re.DOTALL)
        if not match:
            raise ValueError("No JSON object found in model output.")

        json_block = match.group(0)
        data = json.loads(json_block)
        if not isinstance(data, dict):
            raise ValueError("Extracted JSON is not an object.")
        return data

    # ------------------------------------------------------------------
    # Output parsing
    # ------------------------------------------------------------------
    def _parse_round_1_output(
        self,
        agent_id: str,
        data: dict[str, Any],
    ) -> AgentOutputRound1:
        brief_reason = self._sanitize_required_string(
            data.get("brief_reason"),
            fallback="No brief reason provided.",
        )
        current_answer = self._sanitize_required_string(
            data.get("current_answer"),
            fallback="UNKNOWN",
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
    ) -> AgentOutputNormal:
        response_to_conflicts = self._parse_conflict_responses(
            data.get("response_to_conflicts", [])
        )
        brief_reason = self._sanitize_required_string(
            data.get("brief_reason"),
            fallback="No brief reason provided.",
        )
        current_answer = self._sanitize_required_string(
            data.get("current_answer"),
            fallback="UNKNOWN",
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
            cleaned = value.strip()
            if cleaned:
                return cleaned
        return fallback

    def _sanitize_optional_string(
        self,
        value: Any,
    ) -> str | None:
        if isinstance(value, str):
            cleaned = value.strip()
            if cleaned:
                return cleaned
        return None

    def _sanitize_conflict_status(
        self,
        value: Any,
    ) -> str:
        valid = {"resolved", "partially_resolved", "still_open"}
        if isinstance(value, str):
            cleaned = value.strip().lower()
            if cleaned in valid:
                return cleaned
        return "still_open"