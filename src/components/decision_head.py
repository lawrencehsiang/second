from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.utils.result_utils import (
    extract_last_number,
    get_final_answers,
    majority_vote,
    normalize_bool_answer,
    normalize_multiple_choice_answer,
    normalize_text,
)


@dataclass
class CandidateStats:
    key: str
    output: str
    round1_support: int = 0
    final_round_support: int = 0
    rounds_present: set[int] = field(default_factory=set)
    total_occurrences: int = 0
    role_support: set[str] = field(default_factory=set)
    first_seen_round: int = 10**9
    last_seen_round: int = -1
    survives_post_anchor: bool = False
    survives_across_anchor: bool = False


class ConservativeTrajectoryDecisionHead:
    """
    Decision Head v1:
    - conservative
    - trajectory-aware
    - rule-based
    - uses only the effective final state trajectory in state_store
    """

    def __init__(self) -> None:
        # You can tune these later after ablations.
        self.w_round1_support = 4.0
        self.w_round_presence = 2.5
        self.w_role_coverage = 1.5
        self.w_final_support = 1.5
        self.w_total_occurrence = 0.5

        self.bonus_multi_round1 = 2.0
        self.bonus_multi_role = 1.0
        self.bonus_stable = 1.5
        self.bonus_survive_across_anchor = 2.5
        self.bonus_post_anchor = 1.0

        self.penalty_late_singleton = 3.0
        self.penalty_singleton = 1.0
        self.penalty_segment_break = 1.5

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def select_final_answer(
        self,
        *,
        state_store: Any,
        rollback_context: dict | None,
        dataset_name: str,
        agent_roles: dict[str, str],
    ) -> str:
        states = state_store.list_state_records()
        if not states:
            return ""

        final_answers = get_final_answers(state_store)
        if not final_answers:
            return ""

        anchor_round = None
        if rollback_context:
            anchor_round = rollback_context.get("anchor_round")

        ordered_agent_ids = list(agent_roles.keys())
        candidates = self._collect_candidates(
            states=states,
            dataset_name=dataset_name,
            agent_roles=agent_roles,
            ordered_agent_ids=ordered_agent_ids,
            anchor_round=anchor_round,
        )

        if not candidates:
            return majority_vote(final_answers, dataset_name=dataset_name)

        scored = []
        for candidate in candidates.values():
            score = self._score_candidate(candidate, last_round=states[-1].round_id)
            scored.append((score, self._tie_break_tuple(candidate), candidate))

        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
        best = scored[0][2]
        return best.output

    # ------------------------------------------------------------------
    # Candidate collection
    # ------------------------------------------------------------------
    def _collect_candidates(
        self,
        *,
        states: list[Any],
        dataset_name: str,
        agent_roles: dict[str, str],
        ordered_agent_ids: list[str],
        anchor_round: int | None,
    ) -> dict[str, CandidateStats]:
        candidates: dict[str, CandidateStats] = {}

        for state in states:
            round_id = state.round_id
            answers = list(state.current_answers)

            for idx, answer in enumerate(answers):
                key, canonical_output = self._canonicalize_answer(
                    answer=answer,
                    dataset_name=dataset_name,
                )
                if not key:
                    continue

                if key not in candidates:
                    candidates[key] = CandidateStats(
                        key=key,
                        output=canonical_output,
                    )

                c = candidates[key]
                c.rounds_present.add(round_id)
                c.total_occurrences += 1
                c.first_seen_round = min(c.first_seen_round, round_id)
                c.last_seen_round = max(c.last_seen_round, round_id)

                if round_id == 1:
                    c.round1_support += 1

                if round_id == states[-1].round_id:
                    c.final_round_support += 1

                if idx < len(ordered_agent_ids):
                    agent_id = ordered_agent_ids[idx]
                    role_name = agent_roles.get(agent_id, agent_id)
                else:
                    role_name = f"agent_{idx}"
                c.role_support.add(role_name)

        # rollback-aware trajectory flags
        if anchor_round is not None:
            for c in candidates.values():
                has_pre_or_anchor = any(r <= anchor_round for r in c.rounds_present)
                has_post_anchor = any(r > anchor_round for r in c.rounds_present)

                c.survives_post_anchor = has_post_anchor
                c.survives_across_anchor = has_pre_or_anchor and has_post_anchor

        return candidates

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------
    def _score_candidate(
        self,
        candidate: CandidateStats,
        *,
        last_round: int,
    ) -> float:
        rounds_present_count = len(candidate.rounds_present)
        role_coverage = len(candidate.role_support)
        segments = self._count_segments(candidate.rounds_present)

        score = 0.0
        score += self.w_round1_support * candidate.round1_support
        score += self.w_round_presence * rounds_present_count
        score += self.w_role_coverage * role_coverage
        score += self.w_final_support * candidate.final_round_support
        score += self.w_total_occurrence * candidate.total_occurrences

        if candidate.round1_support >= 2:
            score += self.bonus_multi_round1

        if role_coverage >= 2:
            score += self.bonus_multi_role

        if rounds_present_count >= 2:
            score += self.bonus_stable

        if candidate.survives_across_anchor:
            score += self.bonus_survive_across_anchor
        elif candidate.survives_post_anchor:
            score += self.bonus_post_anchor

        late_singleton = (
            candidate.first_seen_round == last_round and candidate.total_occurrences == 1
        )
        if late_singleton:
            score -= self.penalty_late_singleton

        if candidate.total_occurrences == 1:
            score -= self.penalty_singleton

        if segments > 1:
            score -= self.penalty_segment_break * (segments - 1)

        return score

    def _tie_break_tuple(self, candidate: CandidateStats) -> tuple:
        """
        Conservative tie-break:
        1. more round-1 support
        2. more rounds present
        3. more role coverage
        4. more final support
        5. earlier first appearance
        """
        return (
            candidate.round1_support,
            len(candidate.rounds_present),
            len(candidate.role_support),
            candidate.final_round_support,
            -candidate.first_seen_round,
            candidate.total_occurrences,
        )

    # ------------------------------------------------------------------
    # Answer canonicalization
    # ------------------------------------------------------------------
    def _canonicalize_answer(
        self,
        *,
        answer: str,
        dataset_name: str,
    ) -> tuple[str, str]:
        if answer is None:
            return "", ""

        if dataset_name == "strategyqa":
            norm = normalize_bool_answer(answer)
            if not norm:
                return "", ""
            return f"bool:{norm}", norm

        if dataset_name in {"mmlu", "mmlu_pro"}:
            norm = normalize_multiple_choice_answer(answer)
            if not norm:
                return "", ""
            return f"mcq:{norm}", norm

        if dataset_name in {"gsm8k", "svamp", "multiarith", "aime2025", "aime2026","addsub","asdiv","math","singleeq"}:
            num = extract_last_number(answer)
            if num is not None:
                rendered = self._render_number(num)
                return f"num:{rendered}", rendered

            # fallback for malformed numeric outputs
            norm_text = normalize_text(answer)
            if not norm_text:
                return "", ""
            return f"text:{norm_text}", norm_text

        # generic fallback
        norm_text = normalize_text(answer)
        if not norm_text:
            return "", ""
        return f"text:{norm_text}", norm_text

    @staticmethod
    def _render_number(num: float) -> str:
        if abs(num - round(num)) < 1e-9:
            return str(int(round(num)))

        rendered = f"{num:.10f}".rstrip("0").rstrip(".")
        return rendered

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _count_segments(rounds_present: set[int]) -> int:
        """
        Example:
        {1,2,3} -> 1 segment
        {1,2,4,5} -> 2 segments
        {1,3,5} -> 3 segments
        """
        if not rounds_present:
            return 0

        rounds = sorted(rounds_present)
        segments = 1
        for i in range(1, len(rounds)):
            if rounds[i] != rounds[i - 1] + 1:
                segments += 1
        return segments