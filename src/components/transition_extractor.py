from __future__ import annotations

from collections import OrderedDict

from src.schemas.state import Claim, StateRecord, UnresolvedConflict
from src.schemas.transition import (
    AnswerTransition,
    ClaimTransition,
    ClaimsByAnswer,
    ConflictTransition,
    TransitionDigest,
)


class TransitionExtractor:
    """
    Build a compact transition digest from two adjacent StateRecords.

    Responsibilities:
    1. Compare previous/current answers directly.
    2. Compare unresolved conflicts by conflict text.
    3. Group current round newly_added_claims by related_answer.
    4. Return a minimal TransitionDigest for downstream evaluator use.
    """

    def extract(
        self,
        previous_state_record: StateRecord,
        current_state_record: StateRecord,
    ) -> TransitionDigest:
        self._validate_inputs(
            previous_state_record=previous_state_record,
            current_state_record=current_state_record,
        )

        answer_transition = self._build_answer_transition(
            previous_state_record=previous_state_record,
            current_state_record=current_state_record,
        )
        conflict_transition = self._build_conflict_transition(
            previous_state_record=previous_state_record,
            current_state_record=current_state_record,
        )
        claim_transition = self._build_claim_transition(
            current_state_record=current_state_record,
        )

        return TransitionDigest(
            answer_transition=answer_transition,
            conflict_transition=conflict_transition,
            claim_transition=claim_transition,
        )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_inputs(
        self,
        *,
        previous_state_record: StateRecord,
        current_state_record: StateRecord,
    ) -> None:
        prev_round = previous_state_record.round_id
        curr_round = current_state_record.round_id

        if curr_round <= prev_round:
            raise ValueError(
                "current_state_record.round_id must be greater than "
                "previous_state_record.round_id."
            )

    # ------------------------------------------------------------------
    # Answer transition
    # ------------------------------------------------------------------

    def _build_answer_transition(
        self,
        *,
        previous_state_record: StateRecord,
        current_state_record: StateRecord,
    ) -> AnswerTransition:
        return AnswerTransition(
            answers_prev=list(previous_state_record.current_answers),
            answers_curr=list(current_state_record.current_answers),
        )

    # ------------------------------------------------------------------
    # Conflict transition
    # ------------------------------------------------------------------

    def _build_conflict_transition(
        self,
        *,
        previous_state_record: StateRecord,
        current_state_record: StateRecord,
    ) -> ConflictTransition:
        prev_conflicts = previous_state_record.unresolved_conflicts
        curr_conflicts = current_state_record.unresolved_conflicts

        prev_map = self._build_conflict_map(prev_conflicts)
        curr_map = self._build_conflict_map(curr_conflicts)

        persistent_conflicts: list[str] = []
        resolved_conflicts: list[str] = []
        new_conflicts: list[str] = []

        # previous-round order
        for item in prev_conflicts:
            key = self._conflict_key(item)
            if key in curr_map:
                persistent_conflicts.append(item.conflict.strip())
            else:
                resolved_conflicts.append(item.conflict.strip())

        # current-round order
        for item in curr_conflicts:
            key = self._conflict_key(item)
            if key not in prev_map:
                new_conflicts.append(item.conflict.strip())

        return ConflictTransition(
            persistent_conflicts=self._dedupe_keep_order(persistent_conflicts),
            resolved_conflicts=self._dedupe_keep_order(resolved_conflicts),
            new_conflicts=self._dedupe_keep_order(new_conflicts),
        )

    def _build_conflict_map(
        self,
        conflicts: list[UnresolvedConflict],
    ) -> dict[str, UnresolvedConflict]:
        result: dict[str, UnresolvedConflict] = {}
        for item in conflicts:
            key = self._conflict_key(item)
            if key not in result:
                result[key] = item
        return result

    def _conflict_key(self, conflict: UnresolvedConflict) -> str:
        return self._normalize_text(conflict.conflict)

    # ------------------------------------------------------------------
    # Claim transition
    # ------------------------------------------------------------------

    def _build_claim_transition(
        self,
        *,
        current_state_record: StateRecord,
    ) -> ClaimTransition:
        """
        Group current round newly_added_claims by related_answer.

        Mapping rule:
        - rebuttal -> rebuttal_claims
        - support / explanation / constraint -> support_claims

        If related_answer is missing, place it in a reserved bucket so
        no information is silently lost.
        """
        grouped: OrderedDict[str, dict[str, list[str]]] = OrderedDict()

        for claim in current_state_record.newly_added_claims:
            answer_key = self._claim_answer_key(claim)

            if answer_key not in grouped:
                grouped[answer_key] = {
                    "support_claims": [],
                    "rebuttal_claims": [],
                }

            claim_text = claim.text.strip()
            if not claim_text:
                continue

            if claim.claim_type == "rebuttal":
                grouped[answer_key]["rebuttal_claims"].append(claim_text)
            else:
                grouped[answer_key]["support_claims"].append(claim_text)

        new_claims_by_answer: list[ClaimsByAnswer] = []
        for answer, payload in grouped.items():
            support_claims = self._dedupe_keep_order(payload["support_claims"])
            rebuttal_claims = self._dedupe_keep_order(payload["rebuttal_claims"])

            if not support_claims and not rebuttal_claims:
                continue

            new_claims_by_answer.append(
                ClaimsByAnswer(
                    answer=answer,
                    support_claims=support_claims,
                    rebuttal_claims=rebuttal_claims,
                )
            )

        return ClaimTransition(new_claims_by_answer=new_claims_by_answer)

    def _claim_answer_key(self, claim: Claim) -> str:
        if claim.related_answer and claim.related_answer.strip():
            return claim.related_answer.strip()
        return "__UNSCOPED__"

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _normalize_text(self, text: str) -> str:
        return " ".join(text.strip().lower().split())

    def _dedupe_keep_order(self, items: list[str]) -> list[str]:
        seen: set[str] = set()
        results: list[str] = []

        for item in items:
            cleaned = item.strip()
            if not cleaned:
                continue
            key = self._normalize_text(cleaned)
            if key in seen:
                continue
            seen.add(key)
            results.append(cleaned)

        return results