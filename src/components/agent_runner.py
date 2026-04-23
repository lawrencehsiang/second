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

    def generate_with_usage(self, prompt: str) -> dict[str, Any]:
        """
        Optional richer interface:
        {
            "content": str,
            "usage": {
                "prompt_tokens": int,
                "completion_tokens": int,
                "total_tokens": int,
            },
            "raw_response": dict
        }
        """
        ...


class AgentRunner:
    """
    LLM-based runner for normal-stage agents.

    Diversity-Lite version:
    - same model
    - same output schema
    - same same-round parallel / cross-round update topology
    - different role prompts by agent_id
    """

    def __init__(
        self,
        llm_client: LLMClientProtocol,
        max_retries: int = 2,
        usage_logger: Any | None = None,
        sample_id: str | None = None,
        dataset_name: str = "gsm8k",
        role_by_agent_id: dict[str, str] | None = None,
    ) -> None:
        self.llm_client = llm_client
        self.max_retries = max_retries
        self.usage_logger = usage_logger
        self.sample_id = sample_id
        self.dataset_name = dataset_name
        self.role_by_agent_id = role_by_agent_id or {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run_round_1(
        self,
        agent_id: str,
        agent_input: AgentInputRound1,
        round_id: int = 1,
        sample_id: str | None = None,
    ) -> AgentOutputRound1:
        prompt = self._build_round_1_prompt(
            agent_id=agent_id,
            agent_input=agent_input,
        )

        last_error: Exception | None = None
        for _ in range(self.max_retries + 1):
            try:
                raw_text, usage = self._generate_with_optional_usage(prompt)
                self._log_usage(
                    sample_id=sample_id or self.sample_id,
                    round_id=round_id,
                    mode="normal",
                    component="agent_round_1",
                    agent_id=agent_id,
                    usage=usage,
                )
                data = self._extract_json(raw_text)
                output = self._parse_round_1_output(
                    agent_id=agent_id,
                    data=data,
                )
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
        round_id: int | None = None,
        sample_id: str | None = None,
    ) -> AgentOutputNormal:
        prompt = self._build_normal_round_prompt(
            agent_id=agent_id,
            agent_input=agent_input,
        )

        last_error: Exception | None = None
        for _ in range(self.max_retries + 1):
            try:
                raw_text, usage = self._generate_with_optional_usage(prompt)
                self._log_usage(
                    sample_id=sample_id or self.sample_id,
                    round_id=round_id,
                    mode="normal",
                    component="agent_normal",
                    agent_id=agent_id,
                    usage=usage,
                )
                data = self._extract_json(raw_text)
                output = self._parse_normal_round_output(
                    agent_id=agent_id,
                    data=data,
                )
                return output
            except Exception as exc:
                last_error = exc

        raise RuntimeError(
            f"AgentRunner normal round failed after retries for agent {agent_id}. "
            f"Last error: {last_error}"
        ) from last_error

    def run_vanilla_round(
        self,
        *,
        question: str,
        agent_id: str,
        round_id: int,
        sample_id: str | None = None,
        own_previous_answer: str,
        peer_previous_answers: dict[str, str],
    ) -> dict[str, Any]:
        """
        Run one vanilla MAD round (typically round >= 2).

        Returns a simple dict:
        {
            "agent_id": "A",
            "current_answer": "...",
            "brief_reason": "...",
            "round_id": 3
        }
        """
        if round_id < 2:
            raise ValueError("run_vanilla_round is intended for round_id >= 2.")
        if not question or not str(question).strip():
            raise ValueError("question must be a non-empty string.")
        if not isinstance(peer_previous_answers, dict) or not peer_previous_answers:
            raise ValueError("peer_previous_answers must be a non-empty dict.")

        prompt = self._build_vanilla_round_prompt(
            question=question,
            agent_id=agent_id,
            own_previous_answer=own_previous_answer,
            peer_previous_answers=peer_previous_answers,
            round_id=round_id,
        )

        last_error: Exception | None = None
        for _ in range(self.max_retries + 1):
            try:
                raw_text, usage = self._generate_with_optional_usage(prompt)
                self._log_usage(
                    sample_id=sample_id or self.sample_id,
                    round_id=round_id,
                    mode="vanilla",
                    component="agent_vanilla",
                    agent_id=agent_id,
                    usage=usage,
                )
                data = self._extract_json(raw_text)
                output = self._parse_vanilla_round_output(
                    raw_output=data,
                    agent_id=agent_id,
                    round_id=round_id,
                )
                return output
            except Exception as exc:
                last_error = exc

        raise RuntimeError(
            f"AgentRunner vanilla round failed after retries for agent {agent_id}. "
            f"Last error: {last_error}"
        ) from last_error

    # ------------------------------------------------------------------
    # Prompt builders
    # ------------------------------------------------------------------
    def _dataset_instruction(self) -> str:
        if self.dataset_name == "strategyqa":
            return (
                'Task type: boolean question answering.\n'
                '- current_answer must be exactly "true" or "false".\n'
                '- Do not output yes/no.\n'
                '- Do not output explanations inside current_answer.\n'
            )

        if self.dataset_name in {"gsm8k", "svamp", "multiarith","addsub","asdiv","math","singleeq"}:
            return (
                "Task type: math reasoning.\n"
                "- current_answer should be the final numeric answer for this round.\n"
                "- Keep brief_reason short, but make sure current_answer matches your final computation.\n"
            )

        if self.dataset_name in {"aime2025", "aime2026"}:
            return (
                "Task type: AIME-style math reasoning.\n"
                '- current_answer must be a bare final numeric answer only.\n'
                '- Do not include units, degree symbols, words, commas, or explanations inside current_answer.\n'
                '- If you derive 336 degrees, current_answer should be "336", not "336^circ".\n'
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

    def _role_name(self, agent_id: str) -> str:
        return self.role_by_agent_id.get(agent_id, "general_solver").strip().lower()

    def _role_instruction(self, agent_id: str, *, round_type: str) -> str:
        role = self._role_name(agent_id)

        common = (
            "You must still solve the whole problem and output a complete final answer for this round.\n"
            "Your role changes your reasoning emphasis, not the JSON schema.\n"
            "Do not pretend to be unable to solve the problem just because your role emphasizes one lens.\n"
        )

        if role == "parser":
            if round_type == "round_1":
                return (
                    "Role: Parser.\n"
                    "Primary lens: understand the problem statement precisely before computing.\n"
                    "Focus on extracting quantities, entities, units, comparisons, and what is actually being asked.\n"
                    "Be careful about quantity-to-entity mapping and hidden assumptions.\n"
                    "If the problem is simple, solve it after parsing cleanly.\n"
                    + common
                )
            return (
                "Role: Parser.\n"
                "Primary lens in this round: re-check whether the current debate state is built on the correct reading of the problem.\n"
                "Pay special attention to parsing conflicts, quantity/entity mismatches, unit mismatches, or misread conditions.\n"
                "Revise your answer mainly when the interpretation of the problem should change.\n"
                + common
            )

        if role == "planner":
            if round_type == "round_1":
                return (
                    "Role: Planner.\n"
                    "Primary lens: construct the most coherent solution path.\n"
                    "Focus on the operation chain, equation setup, sub-step order, and how intermediate quantities combine.\n"
                    "Prefer a concise but structurally sound plan.\n"
                    + common
                )
            return (
                "Role: Planner.\n"
                "Primary lens in this round: compare competing solution paths and choose the most coherent one.\n"
                "Pay special attention to operation order, equation structure, missing intermediate steps, and whether the current path actually reaches the asked quantity.\n"
                "Revise your answer mainly when the solution plan itself should change.\n"
                + common
            )

        if role == "verifier":
            if round_type == "round_1":
                return (
                    "Role: Verifier.\n"
                    "Primary lens: solve the problem while actively checking whether the answer is numerically and logically plausible.\n"
                    "Use sanity checks, reverse checks, boundary checks, or quick substitution when helpful.\n"
                    "Prefer answers that survive verification, not just answers with long reasoning.\n"
                    + common
                )
            return (
                "Role: Verifier.\n"
                "Primary lens in this round: test whether candidate answers actually hold up.\n"
                "Pay special attention to reverse verification, sanity checks, unit consistency, magnitude checks, and whether a proposed answer contradicts the problem conditions.\n"
                "Do not follow the majority unless the answer also passes verification.\n"
                "Revise your answer mainly when a candidate fails or passes a concrete check.\n"
                + common
            )

        # fallback
        return (
            "Role: General solver.\n"
            "Solve the problem carefully and output a complete final answer for this round.\n"
            + common
        )

    def _build_round_1_prompt(
        self,
        agent_id: str,
        agent_input: AgentInputRound1,
    ) -> str:
        payload = agent_input.model_dump()
        dataset_instruction = self._dataset_instruction()
        role_instruction = self._role_instruction(agent_id, round_type="round_1")

        prompt = f"""
            You are agent {agent_id} in round 1 of a multi-agent debate system.

            This is the independent initialization round.
            You do not have access to other agents' answers.

            {role_instruction}

            {dataset_instruction}

            Return JSON only. No markdown. No extra text.
            Do NOT use LaTeX. Do NOT use backslashes.
            Do NOT write things like \\( \\) or \\[ \\].

            Schema:
            {{
            "agent_id": "{agent_id}",
            "brief_reason": "string",
            "current_answer": "string"
            }}

            Rules:
            - brief_reason: short reasoning
            - current_answer: your final answer for this round
            - do not output extra fields

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
        dataset_instruction = self._dataset_instruction()
        role_instruction = self._role_instruction(agent_id, round_type="normal")

        prompt = f"""
            You are agent {agent_id} in a normal debate round (t >= 2).

            You are given:
            - the original question
            - your own previous answer
            - structured history selected by the system

            You do NOT see other agents' new outputs from this same round.
            You may keep or revise your answer.

            {role_instruction}

            {dataset_instruction}

            Return JSON only. No markdown. No extra text.
            Do NOT use LaTeX. Do NOT use backslashes.
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
            - Fill fields in this order:
            1. response_to_conflicts
            2. brief_reason
            3. current_answer
            - current_answer is your FINAL answer for this round.
            - If your reasoning changes, current_answer must match your final view.
            - Do not let current_answer contradict brief_reason or response_to_conflicts.
            - If there is no real unresolved conflict, response_to_conflicts may be [].
            - Do not output extra fields.

            Input:
            {json.dumps(payload, ensure_ascii=False, indent=2)}

            Return JSON only.
            """.strip()
        return prompt

    def _build_vanilla_round_prompt(
        self,
        *,
        question: str,
        agent_id: str,
        own_previous_answer: str,
        peer_previous_answers: dict[str, str],
        round_id: int,
    ) -> str:
        dataset_instruction = self._dataset_instruction()
        role_instruction = self._role_instruction(agent_id, round_type="normal")

        payload = {
            "question": question,
            "own_previous_answer": own_previous_answer,
            "peer_previous_answers": peer_previous_answers,
            "round_id": round_id,
        }

        prompt = f"""
            You are agent {agent_id} in round {round_id} of a vanilla multi-agent debate.

            You are given:
            1. The original question
            2. Your own answer from the previous round
            3. Other agents' answers from the previous round

            Your task:
            - Re-evaluate the problem carefully
            - Consider whether the other agents exposed mistakes in your previous answer
            - Give your updated current answer and a brief reason
            - You may keep your previous answer or revise it

            Important:
            - Focus on correctness, not agreement
            - Do not copy others blindly
            - If your previous answer was wrong, correct it
            - If you still believe your answer is correct, keep it and explain briefly

            {role_instruction}

            {dataset_instruction}

            Return JSON only. No markdown. No extra text.
            Do NOT use LaTeX. Do NOT use backslashes.
            Do NOT write things like \\( \\) or \\[ \\].

            Schema:
            {{
            "agent_id": "{agent_id}",
            "current_answer": "string",
            "brief_reason": "string"
            }}

            Rules:
            - current_answer must be your FINAL answer for this round
            - brief_reason should be short
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
        current_answer = self._normalize_answer_for_dataset(current_answer)

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

    def _parse_vanilla_round_output(
        self,
        raw_output: dict[str, Any],
        *,
        agent_id: str,
        round_id: int,
    ) -> dict[str, Any]:
        current_answer = self._sanitize_required_string(
            raw_output.get("current_answer"),
            fallback="UNKNOWN",
        )
        brief_reason = self._sanitize_required_string(
            raw_output.get("brief_reason"),
            fallback="",
        )
        current_answer = self._normalize_answer_for_dataset(current_answer)

        return {
            "agent_id": agent_id,
            "current_answer": current_answer,
            "brief_reason": brief_reason,
            "round_id": round_id,
        }

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

    @staticmethod
    def _normalize_multiple_choice_label(text: str) -> str:
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

    def _normalize_answer_for_dataset(self, answer: str) -> str:
        """
        Lightweight post-processing for dataset-specific answer format.

        StrategyQA:
        - map yes/no -> true/false

        MMLU / MMLU-Pro:
        - normalize common multiple-choice forms to a single option label

        Numeric datasets:
        - keep as-is for now
        - later correctness code already normalizes answers
        """
        cleaned = answer.strip()

        if self.dataset_name == "strategyqa":
            lowered = cleaned.lower()
            if lowered == "yes":
                return "true"
            if lowered == "no":
                return "false"
            if lowered in {"true", "false"}:
                return lowered
            return lowered

        if self.dataset_name in {"mmlu", "mmlu_pro"}:
            return self._normalize_multiple_choice_label(answer)

        return cleaned