from __future__ import annotations

import json
import re
from typing import Any, Protocol

from pydantic import ValidationError

from src.schemas import (
    AgentOutputNormal,
    AgentOutputRound1,
    Claim,
    StateRecord,
    UnresolvedConflict,
)


class LLMClientProtocol(Protocol):
    """
    Minimal protocol expected by Recorder.
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


class Recorder:
    """
    LLM-based Recorder.

    Responsibilities:
    1. Summarize one round of agent outputs into a compact StateRecord.
    2. Ask the LLM to produce strictly structured JSON for:
       - newly_added_claims
       - unresolved_conflicts
       - key_raw_snippets
    3. Build current_answers programmatically from agent_outputs
       (instead of trusting the LLM to preserve one entry per agent).
    4. Parse and validate the LLM-generated fields into StateRecord.
    5. Optionally log token usage.

    Notes:
    - current_answers is always deterministic.
    - fallback_recorder can still be injected.
    """

    def __init__(
        self,
        llm_client: LLMClientProtocol,
        max_snippets: int = 5,
        max_retries: int = 2,
        fallback_recorder: Any | None = None,
        usage_logger: Any | None = None,
        sample_id: str | None = None,
    ) -> None:
        self.llm_client = llm_client
        self.max_snippets = max_snippets
        self.max_retries = max_retries
        self.fallback_recorder = fallback_recorder
        self.usage_logger = usage_logger
        self.sample_id = sample_id

    def build_state_record(
        self,
        round_id: int,
        agent_outputs: list[AgentOutputRound1] | list[AgentOutputNormal],
        previous_state_record: StateRecord | None = None,
        sample_id: str | None = None,
        mode: str | None = None,
    ) -> StateRecord:
        """
        Build a StateRecord for the given round.

        Strategy:
        1. Build fixed_current_answers programmatically from agent_outputs
        2. Build a structured prompt for the remaining 3 fields
        3. Ask LLM to return JSON only
        4. Parse + validate
        5. Merge deterministic current_answers with LLM-generated summary fields
        6. Retry if parsing fails
        7. Fall back to fallback_recorder if available
        """
        fixed_current_answers = self._build_current_answers(agent_outputs)

        prompt = self._build_prompt(
            round_id=round_id,
            agent_outputs=agent_outputs,
            previous_state_record=previous_state_record,
            fixed_current_answers=fixed_current_answers,
        )

        last_error: Exception | None = None

        for _ in range(self.max_retries + 1):
            try:
                raw_text, usage = self._generate_with_optional_usage(prompt)

                self._log_usage(
                    sample_id=sample_id or self.sample_id,
                    round_id=round_id,
                    mode=mode,
                    component="recorder",
                    agent_id=None,
                    usage=usage,
                )

                data = self._extract_json(raw_text)
                state_record = self._parse_state_record(
                    data=data,
                    round_id=round_id,
                    fixed_current_answers=fixed_current_answers,
                )
                return state_record

            except Exception as exc:
                last_error = exc

        if self.fallback_recorder is not None:
            return self.fallback_recorder.build_state_record(
                round_id=round_id,
                agent_outputs=agent_outputs,
                previous_state_record=previous_state_record,
            )

        raise RuntimeError(
            f"Recorder failed to build StateRecord after retries. "
            f"Last error: {last_error}"
        ) from last_error

    # ------------------------------------------------------------------
    # Deterministic current_answers
    # ------------------------------------------------------------------
    def _build_current_answers(
        self,
        agent_outputs: list[AgentOutputRound1] | list[AgentOutputNormal],
    ) -> list[str]:
        """
        Build current_answers deterministically from the agent outputs.

        Important:
        - Preserve one entry per agent output
        - Preserve order exactly as provided by the caller
        - Do not deduplicate
        """
        results: list[str] = []

        for output in agent_outputs:
            answer = getattr(output, "current_answer", None)
            if isinstance(answer, str) and answer.strip():
                results.append(answer.strip())
            else:
                results.append("UNKNOWN")

        return results

    # ------------------------------------------------------------------
    # Prompt
    # ------------------------------------------------------------------
    def _build_prompt(
        self,
        round_id: int,
        agent_outputs: list[AgentOutputRound1] | list[AgentOutputNormal],
        previous_state_record: StateRecord | None,
        fixed_current_answers: list[str],
    ) -> str:
        """
        Build the Recorder prompt.

        Key idea:
        - current_answers is already fixed by the system and should NOT be regenerated
        - the LLM only needs to summarize:
          1. newly_added_claims
          2. unresolved_conflicts
          3. key_raw_snippets
        """
        agent_outputs_payload = [output.model_dump() for output in agent_outputs]
        previous_state_payload = (
            previous_state_record.model_dump()
            if previous_state_record is not None
            else None
        )

        prompt = f"""
You are a structured debate state recorder.

Your task is to summarize the current round's agent outputs into ONE compact JSON object.

IMPORTANT:
- The system has already fixed the field `current_answers`.
- You must NOT regenerate, compress, merge, or modify `current_answers`.
- You only need to produce the following three fields:
  1. newly_added_claims
  2. unresolved_conflicts
  3. key_raw_snippets

You must output JSON only.
Do not output markdown.
Do not output explanation.

The JSON schema is:
{{
  "newly_added_claims": [
    {{
      "text": "string",
      "claim_type": "support" | "rebuttal" | "constraint" | "explanation",
      "related_answer": "string or null"
    }}
  ],
  "unresolved_conflicts": [
    {{
      "conflict": "string",
      "why_still_open": "string",
      "involved_answers": ["string", ...]
    }}
  ],
  "key_raw_snippets": ["string", ...]
}}

Definitions:
1. newly_added_claims:
- Include only the key claims newly surfaced or meaningfully expressed in this round.
- A claim should be useful for future debate continuation.
- Do NOT include every sentence.
- Prefer concise, non-redundant claims.
- claim_type meanings:
  - support: supports a conclusion/answer
  - rebuttal: challenges another answer/reasoning
  - constraint: states a requirement/condition/limitation
  - explanation: explains how a conclusion is derived
- related_answer should be the answer this claim mainly supports or targets, if clear; otherwise null.

2. unresolved_conflicts:
- Include only conflicts that remain unresolved after considering all agent outputs in this round.
- If a conflict is only partially resolved, it should still remain here.
- why_still_open should summarize why the conflict remains open.
- involved_answers should list the answers directly involved in the conflict, if clear.

3. key_raw_snippets:
- Preserve a few short raw snippets from this round that are especially informative.
- Maximum {self.max_snippets} snippets.
- Prefer snippets that are useful for future debate.
- Avoid redundancy.

Additional requirements:
- Be faithful to the agent outputs.
- Keep the output compact.
- Avoid duplicate claims and duplicate conflicts.
- Do not hallucinate information not present in the inputs.

Previous StateRecord (for reference, may be null):
{json.dumps(previous_state_payload, ensure_ascii=False, indent=2)}

Current round_id:
{round_id}

System-fixed current_answers (for reference only; DO NOT output them):
{json.dumps(fixed_current_answers, ensure_ascii=False, indent=2)}

Current round agent_outputs:
{json.dumps(agent_outputs_payload, ensure_ascii=False, indent=2)}

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

        # direct parse
        try:
            data = json.loads(raw_text)
            if not isinstance(data, dict):
                raise ValueError("Top-level JSON is not an object.")
            return data
        except json.JSONDecodeError:
            pass

        # fallback: find first JSON object block
        match = re.search(r"\{.*\}", raw_text, re.DOTALL)
        if not match:
            raise ValueError("No JSON object found in model output.")

        json_block = match.group(0)
        data = json.loads(json_block)

        if not isinstance(data, dict):
            raise ValueError("Extracted JSON is not an object.")

        return data

    def _parse_state_record(
        self,
        data: dict[str, Any],
        round_id: int,
        fixed_current_answers: list[str],
    ) -> StateRecord:
        """
        Convert raw dict into validated StateRecord.

        Important:
        - current_answers always comes from fixed_current_answers
        - even if the LLM outputs a current_answers field, we ignore it
        """
        newly_added_claims = self._parse_claims(data.get("newly_added_claims", []))
        unresolved_conflicts = self._parse_conflicts(data.get("unresolved_conflicts", []))
        key_raw_snippets = self._parse_snippets(data.get("key_raw_snippets", []))

        return StateRecord(
            round_id=round_id,
            current_answers=fixed_current_answers,
            newly_added_claims=newly_added_claims,
            unresolved_conflicts=unresolved_conflicts,
            key_raw_snippets=key_raw_snippets[: self.max_snippets],
        )

    def _parse_claims(self, value: Any) -> list[Claim]:
        if not isinstance(value, list):
            return []

        results: list[Claim] = []

        for item in value:
            if not isinstance(item, dict):
                continue

            try:
                claim = Claim(
                    text=str(item.get("text", "")).strip(),
                    claim_type=self._sanitize_claim_type(item.get("claim_type")),
                    related_answer=self._sanitize_optional_string(item.get("related_answer")),
                )
                if claim.text:
                    results.append(claim)
            except ValidationError:
                continue

        return self._deduplicate_claims(results)

    def _parse_conflicts(self, value: Any) -> list[UnresolvedConflict]:
        if not isinstance(value, list):
            return []

        results: list[UnresolvedConflict] = []

        for item in value:
            if not isinstance(item, dict):
                continue

            conflict_text = self._sanitize_optional_string(item.get("conflict"))
            why_still_open = self._sanitize_optional_string(item.get("why_still_open"))

            involved_answers_raw = item.get("involved_answers", [])
            involved_answers: list[str] = []

            if isinstance(involved_answers_raw, list):
                for ans in involved_answers_raw:
                    if isinstance(ans, str) and ans.strip():
                        involved_answers.append(ans.strip())

            if not conflict_text:
                continue

            if not why_still_open:
                why_still_open = "Conflict remains unresolved."

            try:
                conflict = UnresolvedConflict(
                    conflict=conflict_text,
                    why_still_open=why_still_open,
                    involved_answers=self._deduplicate_strings(involved_answers),
                )
                results.append(conflict)
            except ValidationError:
                continue

        return self._deduplicate_conflicts(results)

    def _parse_snippets(self, value: Any) -> list[str]:
        if not isinstance(value, list):
            return []

        results: list[str] = []
        for item in value:
            if isinstance(item, str) and item.strip():
                results.append(item.strip())

        return self._deduplicate_strings(results)

    # ------------------------------------------------------------------
    # Sanitizers
    # ------------------------------------------------------------------
    def _sanitize_claim_type(self, value: Any) -> str:
        valid_types = {"support", "rebuttal", "constraint", "explanation"}

        if isinstance(value, str):
            value = value.strip().lower()
            if value in valid_types:
                return value

        return "support"

    def _sanitize_optional_string(self, value: Any) -> str | None:
        if isinstance(value, str):
            value = value.strip()
            return value if value else None
        return None

    # ------------------------------------------------------------------
    # Dedup
    # ------------------------------------------------------------------
    def _deduplicate_claims(self, claims: list[Claim]) -> list[Claim]:
        seen = set()
        results: list[Claim] = []

        for claim in claims:
            key = (
                self._normalize_text(claim.text),
                claim.claim_type,
                claim.related_answer,
            )
            if key in seen:
                continue
            seen.add(key)
            results.append(claim)

        return results

    def _deduplicate_conflicts(
        self,
        conflicts: list[UnresolvedConflict],
    ) -> list[UnresolvedConflict]:
        seen = set()
        results: list[UnresolvedConflict] = []

        for conflict in conflicts:
            key = self._normalize_text(conflict.conflict)
            if key in seen:
                continue
            seen.add(key)
            results.append(conflict)

        return results

    def _deduplicate_strings(self, items: list[str]) -> list[str]:
        seen = set()
        results: list[str] = []

        for item in items:
            key = self._normalize_text(item)
            if key in seen:
                continue
            seen.add(key)
            results.append(item)

        return results

    def _normalize_text(self, text: str) -> str:
        return " ".join(text.strip().lower().split())