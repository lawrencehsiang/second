from __future__ import annotations

from collections import OrderedDict

from src.components.semantic_matcher import SemanticMatcher
from src.schemas.state import Claim, StateRecord
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

    Main responsibilities:
    1. Compare previous/current answers directly.
    2. Align unresolved conflicts by semantic similarity.
    3. Group current-round newly_added_claims by related_answer.
    4. Semantically deduplicate claims inside each answer bucket.
    5. Return a TransitionDigest for downstream evaluator use.
    """

    def __init__(
        self,
        semantic_matcher: SemanticMatcher | None = None,
        conflict_match_threshold: float = 0.85,
        claim_dedup_threshold: float = 0.89,
    ) -> None:
        self.semantic_matcher = semantic_matcher or SemanticMatcher()
        self.conflict_match_threshold = conflict_match_threshold
        self.claim_dedup_threshold = claim_dedup_threshold

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

        if not prev_conflicts and not curr_conflicts:
            return ConflictTransition(
                persistent_conflicts=[],
                resolved_conflicts=[],
                new_conflicts=[],
            )

        match_result = self.semantic_matcher.greedy_match_items(
            prev_items=prev_conflicts,
            curr_items=curr_conflicts,
            text_getter=lambda c: c.conflict,
            threshold=self.conflict_match_threshold,
        )

        persistent_conflicts: list[str] = []
        resolved_conflicts: list[str] = []
        new_conflicts: list[str] = []

        # Prefer current-round wording for persistent conflicts,
        # because it reflects the latest formulation.
        for match in match_result.matches:
            curr_item = curr_conflicts[match.curr_index]
            text = self._clean_text(curr_item.conflict)
            if text:
                persistent_conflicts.append(text)

        # Previous unmatched -> resolved
        for idx in match_result.unmatched_prev_indices:
            prev_item = prev_conflicts[idx]
            text = self._clean_text(prev_item.conflict)
            if text:
                resolved_conflicts.append(text)

        # Current unmatched -> new
        for idx in match_result.unmatched_curr_indices:
            curr_item = curr_conflicts[idx]
            text = self._clean_text(curr_item.conflict)
            if text:
                new_conflicts.append(text)

        return ConflictTransition(
            persistent_conflicts=self._dedupe_exact_keep_order(persistent_conflicts),
            resolved_conflicts=self._dedupe_exact_keep_order(resolved_conflicts),
            new_conflicts=self._dedupe_exact_keep_order(new_conflicts),
        )

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
        - support / explanation / constraint / others -> support_claims

        If related_answer is missing, place it in a reserved bucket so that
        information is not silently lost.
        """
        grouped: OrderedDict[str, dict[str, list[str]]] = OrderedDict()

        for claim in current_state_record.newly_added_claims:
            answer_key = self._claim_answer_key(claim)

            if answer_key not in grouped:
                grouped[answer_key] = {
                    "support_claims": [],
                    "rebuttal_claims": [],
                }

            claim_text = self._clean_text(claim.text)
            if not claim_text:
                continue

            if claim.claim_type == "rebuttal":
                grouped[answer_key]["rebuttal_claims"].append(claim_text)
            else:
                grouped[answer_key]["support_claims"].append(claim_text)

        new_claims_by_answer: list[ClaimsByAnswer] = []

        for answer, payload in grouped.items():
            support_claims = self._semantic_dedupe_keep_order(
                payload["support_claims"],
                threshold=self.claim_dedup_threshold,
            )
            rebuttal_claims = self._semantic_dedupe_keep_order(
                payload["rebuttal_claims"],
                threshold=self.claim_dedup_threshold,
            )

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

    def _semantic_dedupe_keep_order(
        self,
        items: list[str],
        threshold: float,
    ) -> list[str]:
        """
        Keep-order semantic dedupe for free-form texts.

        Strategy:
        1. Remove trivial exact duplicates first.
        2. Iterate in original order.
        3. Keep the first occurrence.
        4. For each next item, compare it semantically against already-kept items.
        5. If max similarity >= threshold, treat it as a rephrasing and skip it.
        """
        cleaned_items = self._dedupe_exact_keep_order(items)
        if len(cleaned_items) <= 1:
            return cleaned_items

        kept: list[str] = []

        for text in cleaned_items:
            if not kept:
                kept.append(text)
                continue

            sim = self.semantic_matcher.pairwise_similarity([text], kept)
            max_score = float(sim.max()) if sim.size > 0 else 0.0

            if max_score >= threshold:
                continue

            kept.append(text)

        return kept

    def _dedupe_exact_keep_order(self, items: list[str]) -> list[str]:
        """
        Cheap first-pass dedupe:
        only remove literally identical items after light cleaning.
        """
        seen: set[str] = set()
        results: list[str] = []

        for item in items:
            cleaned = self._clean_text(item)
            if not cleaned:
                continue
            lowered = cleaned.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            results.append(cleaned)

        return results

    def _clean_text(self, text: str | None) -> str:
        if text is None:
            return ""
        return " ".join(str(text).strip().split())