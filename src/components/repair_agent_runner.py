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
    """

    def __init__(
        self,
        llm_client: LLMClientProtocol,
        max_retries: int = 2,
        usage_logger: Any | None = None,
        sample_id: str | None = None,
        dataset_name: str = "gsm8k",
    ) -> None:
        self.llm_client = llm_client
        self.max_retries = max_retries
        self.usage_logger = usage_logger
        self.sample_id = sample_id
        self.dataset_name = dataset_name

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
                data = self._extract_json(raw_text)
                output = self._parse_repair_output(
                    agent_id=agent_id,
                    data=data,
                )
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
    def _dataset_instruction(self) -> str:
        if self.dataset_name == "strategyqa":
            return (
                'Task type: boolean question answering.\n'
                '- current_answer must be exactly "true" or "false".\n'
                '- Do not output yes/no.\n'
                '- Do not output explanations inside current_answer.\n'
            )

        if self.dataset_name in {"gsm8k", "svamp"}:
            return (
                "Task type: math reasoning.\n"
                "- current_answer should be the final numeric answer for this round.\n"
                "- Keep brief_reason short, but make sure current_answer matches your final computation.\n"
            )
        
        if self.dataset_name in {"aime2025", "aime2026"}:
            return (
                "Task type: AIME-style math reasoning.\n"
                "- current_answer must be a bare final numeric answer only.\n"
                "- Do not include units, degree symbols, words, commas, or explanations inside current_answer.\n"
                "- If you derive 336 degrees, current_answer should be \"336\", not \"336^circ\".\n"
                "- Keep brief_reason short, but make sure current_answer matches your final computation.\n"
            )
        
        if self.dataset_name in {"mmlu", "mmlu_pro"}:
            return (
                "Task type: multiple-choice question answering.\n"
                '- current_answer must be exactly one option label such as "A", "B", "C".\n'
                "- Do not output the full option text.\n"
                "- Do not output explanations inside current_answer.\n"
                '- Example valid outputs: "A", "D", "J".\n'
            )

        return ""

    def _build_repair_prompt(
        self,
        agent_id: str,
        repair_agent_input: RepairAgentInput,
    ) -> str:
        payload = repair_agent_input.model_dump()
        dataset_instruction = self._dataset_instruction()

        if repair_agent_input.repair_brief is not None:
            prompt = f"""
                You are agent {agent_id} in the FIRST repair round after rollback.

                You are given:
                - the original question
                - anchor-derived history units
                - a repair brief summarizing the failed suffix

                Use the anchor-derived history as the stable base.
                Use the repair brief to understand what went wrong and what still needs repair.

                {dataset_instruction}

                Return JSON only.
                No markdown.
                No extra text.
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

                {dataset_instruction}

                Return JSON only.
                No markdown.
                No extra text.
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
        return re.sub(r'\\(?!["\\/bfnrtu])', r"\\\\", text)

    def _extract_json(self, raw_text: str) -> dict[str, Any]:
        raw_text = raw_text.strip()

        try:
            data = json.loads(raw_text)
            if not isinstance(data, dict):
                raise ValueError("Top-level JSON is not an object.")
            return data
        except json.JSONDecodeError:
            pass

        repaired_text = self._repair_invalid_backslashes(raw_text)
        try:
            data = json.loads(repaired_text)
            if not isinstance(data, dict):
                raise ValueError("Top-level JSON is not an object.")
            return data
        except json.JSONDecodeError:
            pass

        match = re.search(r"\{.*\}", raw_text, re.DOTALL)
        if not match:
            raise ValueError(f"No JSON object found in model output:\n{raw_text}")

        json_block = match.group(0)

        try:
            data = json.loads(json_block)
            if not isinstance(data, dict):
                raise ValueError("Extracted JSON is not an object.")
            return data
        except json.JSONDecodeError:
            pass

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
        current_answer = self._normalize_answer_for_dataset(current_answer)

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
    

    def _normalize_multiple_choice_label(self, text: str) -> str:
        if not text:
            return ""

        s = str(text).strip().upper()

        if re.fullmatch(r"[A-Z]", s):
            return s

        patterns = [
            r"\bOPTION\s*([A-Z])\b",
            r"\bANSWER\s*(?:IS|:)?\s*([A-Z])\b",
            r"\bI\s+CHOOSE\s+([A-Z])\b",
            r"^\(?([A-Z])\)?[\.:\s]*$",
            r"\b([A-Z])\b",
        ]

        for pattern in patterns:
            m = re.search(pattern, s)
            if m:
                return m.group(1)

        return s


    def _normalize_numeric_answer(self, text: str) -> str:
        if not text:
            return ""

        s = str(text).strip()

        # 先去掉逗号
        s = s.replace(",", "")

        # 提取最后一个数字，兼容 336^\circ / answer is 42 / $1,234
        matches = re.findall(r"-?\d+(?:\.\d+)?", s)
        if matches:
            return matches[-1]

        return s


    def _normalize_answer_for_dataset(self, answer: str) -> str:
        if self.dataset_name == "strategyqa":
            s = str(answer).strip().lower()
            if s in {"yes", "true"}:
                return "true"
            if s in {"no", "false"}:
                return "false"
            return s

        if self.dataset_name in {"gsm8k", "aime2025", "aime2026","svamp"}:
            return self._normalize_numeric_answer(answer)

        if self.dataset_name in {"mmlu", "mmlu_pro"}:
            return self._normalize_multiple_choice_label(answer)

        return str(answer).strip()