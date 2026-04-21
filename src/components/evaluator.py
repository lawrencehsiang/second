from __future__ import annotations

import json
import re
from typing import Any, Protocol

from src.components.transition_extractor import TransitionExtractor
from src.schemas.evaluation import TransitionEvaluation
from src.schemas.state import StateRecord
from src.schemas.transition import TransitionDigest


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

    New design:
    1. Primary input is TransitionDigest.
    2. Output is unified TransitionEvaluation.
    3. Parse and validate JSON into TransitionEvaluation.
    4. Optionally log token usage.
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
        self.transition_extractor = TransitionExtractor()

    # ------------------------------------------------------------------
    # New primary API
    # ------------------------------------------------------------------

    def evaluate_transition(
        self,
        question: str,
        transition_digest: TransitionDigest,
        round_id: int | None = None,
        sample_id: str | None = None,
        mode: str | None = "normal",
    ) -> TransitionEvaluation:
        """
        Evaluate a TransitionDigest and return TransitionEvaluation.
        """
        prompt = self._build_prompt(
            question=question,
            transition_digest=transition_digest,
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
                return self._parse_evaluation(data)

            except Exception as exc:
                last_error = exc

        if self.fallback_evaluator is not None:
            if hasattr(self.fallback_evaluator, "evaluate_transition"):
                return self.fallback_evaluator.evaluate_transition(
                    question=question,
                    transition_digest=transition_digest,
                    round_id=round_id,
                    sample_id=sample_id,
                    mode=mode,
                )

        raise RuntimeError(
            "Evaluator failed to generate transition evaluation after retries. "
            f"Last error: {last_error}"
        ) from last_error

    # ------------------------------------------------------------------
    # Prompt
    # ------------------------------------------------------------------

    def _build_prompt(
        self,
        question: str,
        transition_digest: TransitionDigest,
    ) -> str:
        """
        Build the Evaluator prompt based on TransitionDigest.
        """
        digest_payload = transition_digest.model_dump()

        prompt = f"""
            You are a debate transition evaluator.

            Your job is to evaluate whether the CURRENT round transition improved, plateaued, or degraded the debate state.

            Return JSON only. No markdown. No extra text.

            Schema:
            {{
            "transition_judgement": "improved|plateau|degraded",
            "continue_value": "high|medium|low",
            "reason": "string"
            }}

            Decision criteria:
            - "improved":
            The new round clearly makes the debate healthier.
            Examples:
            - answers become more informative or better aligned with useful evidence
            - important conflicts are resolved
            - new claims are useful, specific, and non-redundant
            - the state looks worth continuing from

            - "plateau":
            The new round does not clearly improve the state, but it does not obviously make it worse.
            Examples:
            - little real progress
            - mostly repetition
            - some value remains, but limited

            - "degraded":
            The new round makes the debate state worse.
            Examples:
            - answer changes look unhelpful or destabilizing
            - more unresolved conflict is introduced without useful clarification
            - new claims are noisy, weak, or distracting
            - continuing from here seems less promising than before

            For continue_value:
            - "high": continuing is likely worthwhile
            - "medium": continuing may still be useful
            - "low": continuing is unlikely to help much

            Be conservative and transition-focused.
            Do NOT restate the full problem.
            Do NOT solve the question directly.
            Keep reason short: 1-2 sentences.

            Question:
            {question}

            Transition Digest:
            {json.dumps(digest_payload, ensure_ascii=False, indent=2)}

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

        This helper tolerates both:
        - generate_with_usage(prompt)
        - generate_with_usage(user_prompt=prompt)
        """
        if hasattr(self.llm_client, "generate_with_usage"):
            try:
                resp = self.llm_client.generate_with_usage(user_prompt=prompt)
            except TypeError:
                resp = self.llm_client.generate_with_usage(prompt)
            return resp["content"], resp.get("usage")

        try:
            raw_text = self.llm_client.generate(user_prompt=prompt)
        except TypeError:
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

    def _parse_evaluation(self, data: dict[str, Any]) -> TransitionEvaluation:
        """
        Convert raw dict into validated TransitionEvaluation.
        """
        judgement = self._sanitize_transition_judgement(
            data.get("transition_judgement")
        )
        continue_value = self._sanitize_continue_value(
            data.get("continue_value")
        )
        reason = self._sanitize_reason(data.get("reason"))

        print("状态评估结果:", judgement, continue_value, reason)

        return TransitionEvaluation(
            transition_judgement=judgement,
            continue_value=continue_value,
            reason=reason,
        )

    # ------------------------------------------------------------------
    # Sanitizers
    # ------------------------------------------------------------------

    def _sanitize_transition_judgement(self, value: Any) -> str:
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"improved", "plateau", "degraded"}:
                return normalized

            # tolerate common variants
            if normalized in {"better", "improve", "improving"}:
                return "improved"
            if normalized in {"same", "flat", "stalled", "neutral"}:
                return "plateau"
            if normalized in {"worse", "regressed", "regression"}:
                return "degraded"

        return "plateau"

    def _sanitize_continue_value(self, value: Any) -> str:
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"high", "medium", "low"}:
                return normalized

            # tolerate common variants
            if normalized in {"strong", "very high"}:
                return "high"
            if normalized in {"moderate", "mid", "middle"}:
                return "medium"
            if normalized in {"weak", "very low"}:
                return "low"

        return "medium"

    def _sanitize_reason(self, value: Any) -> str:
        if isinstance(value, str):
            cleaned = value.strip()
            if cleaned:
                return cleaned
        return "No reason provided."