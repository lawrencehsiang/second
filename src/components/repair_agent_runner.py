from __future__ import annotations

import json
import re
from typing import Any, Protocol

from pydantic import ValidationError

from src.schemas import (
    AgentOutputNormal,
    ConflictResponse,
    RepairAgentInput,
)


class LLMClientProtocol(Protocol):
    def generate(self, prompt: str) -> str:
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


class RepairAgentRunner:
    """
    LLM-based runner for repair-stage agents.

    Responsibilities:
    1. Build prompts for repair rounds
    2. Ask the LLM to return strict JSON
    3. Parse JSON fields into AgentOutputNormal
    4. Optionally log token usage

    Notes:
    - This version keeps your existing style:
      repair agent returns the same schema family as normal agents.
    - It only adds optional usage logging support.
    """

    def __init__(
        self,
        llm_client: LLMClientProtocol,
        max_retries: int = 2,
        usage_logger: Any | None = None,
        sample_id: str | None = None,
    ) -> None:
        self.llm_client = llm_client
        self.max_retries = max_retries
        self.usage_logger = usage_logger
        self.sample_id = sample_id

    def run_repair_round(
        self,
        agent_id: str,
        repair_agent_input: RepairAgentInput,
        round_id: int | None = None,
        sample_id: str | None = None,
    ) -> AgentOutputNormal:
        prompt = self._build_repair_prompt(
            agent_id=agent_id,
            repair_agent_input=repair_agent_input,
        )

        last_error: Exception | None = None

        for _ in range(self.max_retries + 1):
            try:
                raw_text, usage = self._generate_with_optional_usage(prompt)

                self._log_usage(
                    sample_id=sample_id or self.sample_id,
                    round_id=round_id,
                    mode="repair",
                    component="repair_agent",
                    agent_id=agent_id,
                    usage=usage,
                )

                # print(f"Raw repair model output for agent {agent_id}: {raw_text}")
                data = self._extract_json(raw_text)
                output = self._parse_repair_output(
                    agent_id=agent_id,
                    data=data,
                )
                # print(f"Repair output for agent {agent_id}: {output.model_dump()}")
                return output

            except Exception as exc:
                last_error = exc

        raise RuntimeError(
            f"RepairAgentRunner failed after retries for agent {agent_id}. "
            f"Last error: {last_error}"
        ) from last_error

    # ------------------------------------------------------------------
    # Prompt builder
    # ------------------------------------------------------------------
    def _build_repair_prompt(
        self,
        agent_id: str,
        repair_agent_input: RepairAgentInput,
    ) -> str:
        payload = repair_agent_input.model_dump()

        if repair_agent_input.repair_brief is not None:
            prompt = f"""
                You are agent {agent_id} in the FIRST repair round after rollback.

                You are given:
                - the original question
                - anchor-derived history units
                - a repair brief summarizing the failed suffix

                Use the anchor-derived history as the stable base.
                Use the repair brief to understand what went wrong and what still needs repair.

                Return JSON only. No markdown. No extra text.
                Do NOT use LaTeX.
                Do NOT use backslashes.
                Do NOT write things like \\( \\) or \\[ \\].

                Schema:
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

                Rules:
                - response_to_conflicts should address the conflicts in the repair brief.
                - brief_reason should explain why your repaired answer is justified.
                - current_answer is your FINAL repaired answer for this round.
                - If your reasoning changes, current_answer must match your final view.
                - If there is no real remaining conflict, response_to_conflicts may be [].
                - Do not output extra fields.

                Input:
                {json.dumps(payload, ensure_ascii=False, indent=2)}

                Return JSON only.
                """.strip()
        else:
            prompt = f"""
                You are agent {agent_id} in a later repair round.

                You are given:
                - the original question
                - structured history from the repaired trajectory so far

                Continue the repaired discussion normally.
                Use the history to decide whether to keep or revise your answer.

                Return JSON only. No markdown. No extra text.
                Do NOT use LaTeX.
                Do NOT use backslashes.
                Do NOT write things like \\( \\) or \\[ \\].

                Schema:
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

                Rules:
                - respond to any unresolved conflicts present in the structured history
                - brief_reason should explain why you keep or revise your answer
                - current_answer is your FINAL answer for this round
                - if your reasoning changes, current_answer must match your final view
                - if there is no real unresolved conflict, response_to_conflicts may be []
                - do not output extra fields

                Input:
                {json.dumps(payload, ensure_ascii=False, indent=2)}

                Return JSON only.
                """.strip()
        return prompt

    # ------------------------------------------------------------------
    # LLM call helpers
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
    # JSON extraction
    # ------------------------------------------------------------------
    def _repair_invalid_backslashes(self, text: str) -> str:
        """
        Repair invalid backslash escapes that often appear in model-generated pseudo-JSON.

        Example:
        - "\\("  -> "\\\\("
        - "\\*"  -> "\\\\*"

        Valid JSON escapes are:
        \", \\, \/, \b, \f, \n, \r, \t, \\uXXXX

        Any backslash not followed by one of the valid escape chars
        is converted into a double backslash.
        """
        return re.sub(r'\\(?!["\\/bfnrtu])', r"\\\\", text)
    def _extract_json(self, raw_text: str) -> dict[str, Any]:
        raw_text = raw_text.strip()

        # Case 1: direct parse
        try:
            data = json.loads(raw_text)
            if not isinstance(data, dict):
                raise ValueError("Top-level JSON is not an object.")
            return data
        except json.JSONDecodeError:
            pass

        # Case 2: repair invalid backslashes, then retry direct parse
        repaired_text = self._repair_invalid_backslashes(raw_text)
        try:
            data = json.loads(repaired_text)
            if not isinstance(data, dict):
                raise ValueError("Top-level JSON is not an object.")
            return data
        except json.JSONDecodeError:
            pass

        # Case 3: extract first {...} block
        match = re.search(r"\{.*\}", raw_text, re.DOTALL)
        if not match:
            raise ValueError(f"No JSON object found in model output:\n{raw_text}")

        json_block = match.group(0)

        # Try original block
        try:
            data = json.loads(json_block)
            if not isinstance(data, dict):
                raise ValueError("Extracted JSON is not an object.")
            return data
        except json.JSONDecodeError:
            pass

        # Try repaired block
        repaired_block = self._repair_invalid_backslashes(json_block)
        data = json.loads(repaired_block)
        if not isinstance(data, dict):
            raise ValueError("Extracted repaired JSON is not an object.")

        return data

    # ------------------------------------------------------------------
    # Output parsing
    # ------------------------------------------------------------------
    def _parse_repair_output(
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