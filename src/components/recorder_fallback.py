from __future__ import annotations

from collections import defaultdict

from src.schemas import (
    AgentOutputNormal,
    AgentOutputRound1,
    Claim,
    StateRecord,
    UnresolvedConflict,
)


class Recorder:
    """
    Rule-based Recorder.

    Responsibilities:
    1. Convert one round of agent outputs into a compact StateRecord.
    2. Keep only the core structural information needed by downstream modules.

    Current design:
    - No LLM summarization yet.
    - Use lightweight heuristics to construct claims and unresolved conflicts.
    """

    def __init__(self, max_snippets: int = 5) -> None:
        self.max_snippets = max_snippets

    def build_state_record(
        self,
        round_id: int,
        agent_outputs: list[AgentOutputRound1] | list[AgentOutputNormal],
        previous_state_record: StateRecord | None = None,
    ) -> StateRecord:
        """
        Build a StateRecord for the given round.

        Args:
            round_id: Current round id.
            agent_outputs: Outputs from all agents in the round.
            previous_state_record: Previous round StateRecord, optional.

        Returns:
            A StateRecord.
        """
        current_answers = self._extract_current_answers(agent_outputs)
        newly_added_claims = self._extract_new_claims(agent_outputs)
        unresolved_conflicts = self._extract_unresolved_conflicts(agent_outputs)
        key_raw_snippets = self._extract_key_raw_snippets(agent_outputs)

        return StateRecord(
            round_id=round_id,
            current_answers=current_answers,
            newly_added_claims=newly_added_claims,
            unresolved_conflicts=unresolved_conflicts,
            key_raw_snippets=key_raw_snippets,
        )

    # ------------------------------------------------------------------
    # Core extraction methods
    # ------------------------------------------------------------------

    def _extract_current_answers(
        self,
        agent_outputs: list[AgentOutputRound1] | list[AgentOutputNormal],
    ) -> list[str]:
        return [output.current_answer for output in agent_outputs]

    def _extract_new_claims(
        self,
        agent_outputs: list[AgentOutputRound1] | list[AgentOutputNormal],
    ) -> list[Claim]:
        claims: list[Claim] = []

        for output in agent_outputs:
            # brief_reason -> one claim
            if output.brief_reason and output.brief_reason.strip():
                claims.append(
                    Claim(
                        text=output.brief_reason.strip(),
                        claim_type=self._infer_claim_type(output.brief_reason),
                        related_answer=output.current_answer,
                    )
                )

            # normal round: responses to conflicts -> more claims
            if isinstance(output, AgentOutputNormal):
                for item in output.response_to_conflicts:
                    if item.response and item.response.strip():
                        claims.append(
                            Claim(
                                text=item.response.strip(),
                                claim_type=self._infer_claim_type(item.response),
                                related_answer=output.current_answer,
                            )
                        )

        return self._deduplicate_claims(claims)

    def _extract_unresolved_conflicts(
        self,
        agent_outputs: list[AgentOutputRound1] | list[AgentOutputNormal],
    ) -> list[UnresolvedConflict]:
        """
        Merge unresolved conflicts across agent responses.

        Rules:
        - Round 1 has none.
        - For normal rounds:
          - 'still_open' => unresolved
          - 'partially_resolved' => also keep as unresolved for now
        - Merge by conflict text.
        """
        if not agent_outputs:
            return []

        if not isinstance(agent_outputs[0], AgentOutputNormal):
            return []

        grouped: dict[str, dict] = defaultdict(
            lambda: {
                "why_still_open": [],
                "involved_answers": set(),
            }
        )

        for output in agent_outputs:
            if not isinstance(output, AgentOutputNormal):
                continue

            for item in output.response_to_conflicts:
                if item.status not in {"still_open", "partially_resolved"}:
                    continue

                grouped[item.conflict]["why_still_open"].append(item.response.strip())
                grouped[item.conflict]["involved_answers"].add(output.current_answer)

        results: list[UnresolvedConflict] = []
        for conflict_text, payload in grouped.items():
            reasons = [x for x in payload["why_still_open"] if x]
            why_still_open = " | ".join(reasons[:2]) if reasons else "Conflict remains unresolved."
            involved_answers = sorted(payload["involved_answers"])

            results.append(
                UnresolvedConflict(
                    conflict=conflict_text,
                    why_still_open=why_still_open,
                    involved_answers=involved_answers,
                )
            )

        return results

    def _extract_key_raw_snippets(
        self,
        agent_outputs: list[AgentOutputRound1] | list[AgentOutputNormal],
    ) -> list[str]:
        """
        Preserve a small number of raw snippets from the round.
        """
        snippets: list[str] = []

        for output in agent_outputs:
            if output.brief_reason and output.brief_reason.strip():
                snippets.append(output.brief_reason.strip())

            if isinstance(output, AgentOutputNormal):
                for item in output.response_to_conflicts:
                    if item.response and item.response.strip():
                        snippets.append(item.response.strip())

        snippets = self._deduplicate_strings(snippets)
        return snippets[: self.max_snippets]

    # ------------------------------------------------------------------
    # Heuristics
    # ------------------------------------------------------------------

    def _infer_claim_type(self, text: str) -> str:
        """
        Lightweight heuristic for claim type inference.

        Priority order:
        1. rebuttal
        2. constraint
        3. explanation
        4. support
        """
        t = text.lower()

        rebuttal_markers = [
            "disagree",
            "incorrect",
            "wrong",
            "however",
            "but",
            "not valid",
            "does not follow",
            "flawed",
        ]
        constraint_markers = [
            "must",
            "should",
            "need to",
            "requires",
            "constraint",
            "condition",
            "depends on",
        ]
        explanation_markers = [
            "because",
            "therefore",
            "so",
            "thus",
            "which means",
            "this means",
        ]

        if any(marker in t for marker in rebuttal_markers):
            return "rebuttal"
        if any(marker in t for marker in constraint_markers):
            return "constraint"
        if any(marker in t for marker in explanation_markers):
            return "explanation"
        return "support"

    # ------------------------------------------------------------------
    # Dedup helpers
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