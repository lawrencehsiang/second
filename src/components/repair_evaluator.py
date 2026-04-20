from __future__ import annotations

import json
import re
from typing import Any, Protocol

from src.components.transition_extractor import TransitionExtractor
from src.schemas.evaluation import TransitionEvaluation
from src.schemas.repair import RepairBrief
from src.schemas.state import StateRecord
from src.schemas.transition import TransitionDigest


class LLMClientProtocol(Protocol):
    def generate(self, prompt: str) -> str:
        ...

    def generate_with_usage(self, prompt: str) -> dict[str, Any]:
        ...


class RepairEvaluator:
    """
    LLM-based evaluator for repair mode.

    New design:
    - First repair round after rollback:
        repair_brief + transition_digest(anchor -> current)
    - Later repair rounds:
        transition_digest(previous_repair -> current)
    - Unified output:
        TransitionEvaluation
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
    # New primary APIs
    # ------------------------------------------------------------------

    def evaluate_first_repair_transition(
        self,
        question: str,
        transition_digest: TransitionDigest,
        repair_brief: RepairBrief | None,
        round_id: int | None = None,
        sample_id: str | None = None,
        mode: str | None = "repair",
    ) -> TransitionEvaluation:
        """
        First repair round after rollback:
        evaluate with repair_brief + transition_digest.
        """
        prompt = self._build_first_repair_prompt(
            question=question,
            transition_digest=transition_digest,
            repair_brief=repair_brief,
        )
        return self._run_llm_and_parse(
            prompt=prompt,
            round_id=round_id,
            sample_id=sample_id,
            mode=mode,
            component="repair_evaluator",
        )

    def evaluate_later_repair_transition(
        self,
        question: str,
        transition_digest: TransitionDigest,
        round_id: int | None = None,
        sample_id: str | None = None,
        mode: str | None = "repair",
    ) -> TransitionEvaluation:
        """
        Later repair rounds:
        evaluate with transition_digest only.
        """
        prompt = self._build_later_repair_prompt(
            question=question,
            transition_digest=transition_digest,
        )
        return self._run_llm_and_parse(
            prompt=prompt,
            round_id=round_id,
            sample_id=sample_id,
            mode=mode,
            component="repair_evaluator",
        )

    # ------------------------------------------------------------------
    # Compatibility wrapper
    # ------------------------------------------------------------------

    def evaluate_repair(
        self,
        question: str,
        anchor_state: StateRecord,
        repair_brief: RepairBrief | None,
        current_state_record: StateRecord,
        previous_repair_state_record: StateRecord | None = None,
        round_id: int | None = None,
        sample_id: str | None = None,
        mode: str | None = "repair",
    ) -> TransitionEvaluation:
        """
        Compatibility wrapper for existing call sites.

        - first repair round:
            digest(anchor_state -> current_state_record) + repair_brief
        - later repair rounds:
            digest(previous_repair_state_record -> current_state_record)
        """
        if previous_repair_state_record is None:
            transition_digest = self.transition_extractor.extract(
                previous_state_record=anchor_state,
                current_state_record=current_state_record,
            )
            return self.evaluate_first_repair_transition(
                question=question,
                transition_digest=transition_digest,
                repair_brief=repair_brief,
                round_id=round_id,
                sample_id=sample_id,
                mode=mode,
            )

        transition_digest = self.transition_extractor.extract(
            previous_state_record=previous_repair_state_record,
            current_state_record=current_state_record,
        )
        return self.evaluate_later_repair_transition(
            question=question,
            transition_digest=transition_digest,
            round_id=round_id,
            sample_id=sample_id,
            mode=mode,
        )

    # ------------------------------------------------------------------
    # Prompt builders
    # ------------------------------------------------------------------

    def _build_first_repair_prompt(
        self,
        question: str,
        transition_digest: TransitionDigest,
        repair_brief: RepairBrief | None,
    ) -> str:
        """
        First repair round after rollback:
        use repair_brief + transition_digest(anchor -> current).
        """
        digest_payload = transition_digest.model_dump()
        repair_brief_payload = (
            repair_brief.model_dump() if repair_brief is not None else None
        )

        prompt = f"""
            You are a repair-stage evaluator.

            This is the FIRST repair round after rollback.
            Evaluate whether the current repair transition improves over the rollback anchor transition
            and whether it addresses the repair brief.

            Return JSON only. No markdown. No extra text.

            Schema:
            {{
            "transition_judgement": "improved|plateau|degraded",
            "continue_value": "high|medium|low",
            "reason": "string"
            }}

            Decision criteria:
            - "improved":
            The repair transition clearly makes the repaired debate state healthier.
            Examples:
            - answers become more stable, clearer, or better aligned
            - important conflicts are reduced or clarified
            - the repair seems to be correcting earlier bad drift
            - another repair round may still be worthwhile

            - "plateau":
            The repair transition does not clearly improve the state, but it does not obviously make it worse.
            Examples:
            - limited repair progress
            - mostly repetition
            - some residual value may remain, but not much

            - "degraded":
            The repair transition makes the repaired state worse.
            Examples:
            - answer changes look unhelpful or destabilizing
            - the repair brief's remaining conflicts are not being meaningfully addressed
            - new claims are weak, repetitive, or distracting
            - continued repair looks unlikely to help

            For continue_value:
            - "high": another repair round is likely worthwhile
            - "medium": another repair round may still be useful
            - "low": further repair is unlikely to help much

            Be conservative and transition-focused.
            Do NOT solve the question directly.
            Keep reason short: 1-2 sentences.

            Question:
            {question}

            Repair Brief:
            {json.dumps(repair_brief_payload, ensure_ascii=False, indent=2)}

            Transition Digest (anchor -> current):
            {json.dumps(digest_payload, ensure_ascii=False, indent=2)}

            Return JSON only.
            """.strip()
        return prompt

    def _build_later_repair_prompt(
        self,
        question: str,
        transition_digest: TransitionDigest,
    ) -> str:
        """
        Later repair rounds:
        evaluate with transition_digest only.
        """
        digest_payload = transition_digest.model_dump()

        prompt = f"""
            You are a repair-stage evaluator.

            This is a LATER repair round.
            Compare the PREVIOUS repair state and the CURRENT repair state through the transition digest below.

            Return JSON only. No markdown. No extra text.

            Schema:
            {{
            "transition_judgement": "improved|plateau|degraded",
            "continue_value": "high|medium|low",
            "reason": "string"
            }}

            Decision criteria:
            - "improved":
            The repair transition clearly improves over the previous repair state.
            Examples:
            - answers become more stable, clearer, or better aligned
            - important conflicts are reduced or clarified
            - another repair round may still be worthwhile

            - "plateau":
            The repair transition does not clearly improve the state, but it does not obviously make it worse.
            Examples:
            - little real progress
            - mostly repetition
            - some residual value remains, but limited

            - "degraded":
            The repair transition makes the repaired state worse.
            Examples:
            - answer changes look unhelpful or destabilizing
            - conflicts remain messy or become noisier
            - new claims are weak, repetitive, or distracting
            - continuing repair looks less promising than before

            For continue_value:
            - "high": another repair round is likely worthwhile
            - "medium": another repair round may still be useful
            - "low": further repair is unlikely to help much

            Be conservative and transition-focused.
            Do NOT solve the question directly.
            Keep reason short: 1-2 sentences.

            Question:
            {question}

            Transition Digest (previous repair -> current repair):
            {json.dumps(digest_payload, ensure_ascii=False, indent=2)}

            Return JSON only.
            """.strip()
        return prompt

    # ------------------------------------------------------------------
    # LLM helpers
    # ------------------------------------------------------------------

    def _run_llm_and_parse(
        self,
        *,
        prompt: str,
        round_id: int | None,
        sample_id: str | None,
        mode: str | None,
        component: str,
    ) -> TransitionEvaluation:
        last_error: Exception | None = None

        for _ in range(self.max_retries + 1):
            try:
                raw_text, usage = self._generate_with_optional_usage(prompt)

                self._log_usage(
                    sample_id=sample_id or self.sample_id,
                    round_id=round_id,
                    mode=mode,
                    component=component,
                    agent_id=None,
                    usage=usage,
                )

                data = self._extract_json(raw_text)
                return self._parse_evaluation(data)

            except Exception as exc:
                last_error = exc

        if self.fallback_evaluator is not None:
            if hasattr(self.fallback_evaluator, "evaluate_repair"):
                raise RuntimeError(
                    "Fallback repair evaluator still uses the old interface. "
                    "Please migrate it explicitly."
                ) from last_error

        raise RuntimeError(
            f"RepairEvaluator failed to generate transition evaluation after retries. "
            f"Last error: {last_error}"
        ) from last_error

    def _generate_with_optional_usage(
        self,
        prompt: str,
    ) -> tuple[str, dict[str, int] | None]:
        """
        Prefer generate_with_usage if available; otherwise fall back to generate.

        Tolerate both:
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
        judgement = self._sanitize_transition_judgement(
            data.get("transition_judgement")
        )
        continue_value = self._sanitize_continue_value(
            data.get("continue_value")
        )
        reason = self._sanitize_reason(data.get("reason"))

        print("repair状态评估结果:", judgement, continue_value, reason)

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