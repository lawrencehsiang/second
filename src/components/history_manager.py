from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass

from src.components.state_store import StateStore
from src.schemas import Claim, HistoryUnit, StateRecord, UnresolvedConflict


@dataclass
class _CandidateUnit:
    unit: HistoryUnit
    score: float


class HistoryManager:
    """
    Improved rule-based HistoryManager.

    Design principles of this version:
    1. Use a sliding window over recent StateRecords.
    2. Compute mainstream/minority answers from the whole window with recency weighting.
    3. Build candidate units first, then score and select top-1 per slot.
    4. Keep score amplitudes relatively small and interpretable.
    5. Use claim=claim.text and snippet=None for claim-derived units for now.
    """

    def __init__(
        self,
        history_window_rounds: int = 3,
        normal_mode_history_unit_count: int = 4,
    ) -> None:
        self.history_window_rounds = history_window_rounds
        self.normal_mode_history_unit_count = normal_mode_history_unit_count

    def build_history_units(
        self,
        question: str,
        current_round_id: int,
        state_store: StateStore,
    ) -> list[HistoryUnit]:
        """
        Build structured history units for current round (t >= 2).
        """
        if current_round_id <= 1:
            return []

        window_states = self._get_window_states(
            current_round_id=current_round_id,
            state_store=state_store,
        )
        if not window_states:
            return []

        answer_stats = self._compute_answer_window_stats(window_states)
        mainstream_answer = self._select_mainstream_answer(answer_stats)
        minority_answer = self._select_minority_answer(answer_stats, mainstream_answer)

        claims = self._collect_claims(window_states)
        snippets = self._collect_snippets(window_states)
        conflicts = self._collect_conflicts(window_states)

        units: list[HistoryUnit] = []

        mainstream_unit = self._select_mainstream_support_unit(
            mainstream_answer=mainstream_answer,
            minority_answer=minority_answer,
            claims=claims,
            snippets=snippets,
            window_states=window_states,
        )
        if mainstream_unit is not None:
            units.append(mainstream_unit)

        rebuttal_unit = self._select_key_rebuttal_unit(
            mainstream_answer=mainstream_answer,
            minority_answer=minority_answer,
            claims=claims,
        )
        if rebuttal_unit is not None and not self._duplicate_unit(units, rebuttal_unit):
            units.append(rebuttal_unit)

        minority_unit = self._select_minority_objection_unit(
            minority_answer=minority_answer,
            claims=claims,
            window_states=window_states,
            answer_stats=answer_stats,
        )
        if minority_unit is not None and not self._duplicate_unit(units, minority_unit):
            units.append(minority_unit)

        conflict_unit = self._select_core_unresolved_conflict_unit(
            conflicts=conflicts,
            mainstream_answer=mainstream_answer,
            minority_answer=minority_answer,
        )
        if conflict_unit is not None and not self._duplicate_unit(units, conflict_unit):
            units.append(conflict_unit)

        return units[: self.normal_mode_history_unit_count]

    # ------------------------------------------------------------------
    # Window selection
    # ------------------------------------------------------------------

    def _get_window_states(
        self,
        current_round_id: int,
        state_store: StateStore,
    ) -> list[StateRecord]:
        """
        Select StateRecords in:
        [current_round_id - history_window_rounds, current_round_id - 1]
        """
        start_round = max(1, current_round_id - self.history_window_rounds)
        end_round = current_round_id - 1

        selected: list[StateRecord] = []
        for round_id in range(start_round, end_round + 1):
            state = state_store.get_state_record(round_id)
            if state is not None:
                selected.append(state)
        return selected

    # ------------------------------------------------------------------
    # Window-level answer statistics
    # ------------------------------------------------------------------

    def _compute_answer_window_stats(
        self,
        states: list[StateRecord],
    ) -> dict[str, dict]:
        """
        Compute weighted answer statistics over the whole window.

        For each answer, collect:
        - weighted_score: recency-weighted count across rounds
        - raw_count: total occurrences
        - rounds: set of round_ids where it appears
        - latest_round: latest round_id where it appears
        """
        if not states:
            return {}

        latest_round_id = states[-1].round_id
        stats: dict[str, dict] = defaultdict(
            lambda: {
                "weighted_score": 0.0,
                "raw_count": 0,
                "rounds": set(),
                "latest_round": -1,
            }
        )

        for state in states:
            round_weight = self._recency_weight(
                round_id=state.round_id,
                latest_round_id=latest_round_id,
            )
            for answer in state.current_answers:
                stats[answer]["weighted_score"] += round_weight
                stats[answer]["raw_count"] += 1
                stats[answer]["rounds"].add(state.round_id)
                stats[answer]["latest_round"] = max(stats[answer]["latest_round"], state.round_id)

        return dict(stats)

    def _select_mainstream_answer(self, answer_stats: dict[str, dict]) -> str | None:
        """
        Select mainstream answer by:
        1. higher weighted_score
        2. higher raw_count
        3. later latest_round
        4. lexical order
        """
        if not answer_stats:
            return None

        ranked = sorted(
            answer_stats.items(),
            key=lambda x: (
                x[1]["weighted_score"],
                x[1]["raw_count"],
                x[1]["latest_round"],
                x[0],
            ),
            reverse=True,
        )
        return ranked[0][0]

    def _select_minority_answer(
        self,
        answer_stats: dict[str, dict],
        mainstream_answer: str | None,
    ) -> str | None:
        """
        Select minority answer.

        Current heuristic (kept as requested):
        - exclude mainstream
        - require: round_count >= 2 OR raw_count >= 2
        - among remaining candidates, choose the strongest persistent non-mainstream answer
        """
        if not answer_stats or mainstream_answer is None:
            return None

        candidates = []
        for answer, stat in answer_stats.items():
            if answer == mainstream_answer:
                continue

            round_count = len(stat["rounds"])
            raw_count = stat["raw_count"]

            if round_count >= 2 or raw_count >= 2:
                candidates.append((answer, stat))

        if not candidates:
            return None

        ranked = sorted(
            candidates,
            key=lambda x: (
                x[1]["weighted_score"],
                x[1]["raw_count"],
                x[1]["latest_round"],
                x[0],
            ),
            reverse=True,
        )
        return ranked[0][0]

    # ------------------------------------------------------------------
    # Collectors
    # ------------------------------------------------------------------

    def _collect_claims(self, states: list[StateRecord]) -> list[tuple[int, Claim]]:
        result: list[tuple[int, Claim]] = []
        for state in states:
            for claim in state.newly_added_claims:
                result.append((state.round_id, claim))
        return result

    def _collect_snippets(self, states: list[StateRecord]) -> list[tuple[int, str]]:
        result: list[tuple[int, str]] = []
        for state in states:
            for snippet in state.key_raw_snippets:
                result.append((state.round_id, snippet))
        return result

    def _collect_conflicts(
        self,
        states: list[StateRecord],
    ) -> list[tuple[int, UnresolvedConflict]]:
        result: list[tuple[int, UnresolvedConflict]] = []
        for state in states:
            for conflict in state.unresolved_conflicts:
                result.append((state.round_id, conflict))
        return result

    # ------------------------------------------------------------------
    # Unit selection
    # ------------------------------------------------------------------

    def _select_mainstream_support_unit(
        self,
        mainstream_answer: str | None,
        minority_answer: str | None,
        claims: list[tuple[int, Claim]],
        snippets: list[tuple[int, str]],
        window_states: list[StateRecord],
    ) -> HistoryUnit | None:
        if mainstream_answer is None:
            return None

        latest_round_id = window_states[-1].round_id
        candidates: list[_CandidateUnit] = []

        for round_id, claim in claims:
            if claim.claim_type not in {"support", "constraint", "explanation"}:
                continue

            score = 0.0

            # 一级：时间
            score += self._recency_score(round_id, latest_round_id)

            # 一级：和 mainstream 的直接相关性
            if claim.related_answer == mainstream_answer:
                score += 2.0
            elif claim.related_answer == minority_answer and minority_answer is not None:
                score -= 0.5
            elif claim.related_answer is None:
                score += 0.0
            else:
                score += 0.5

            # 三级：claim type 轻微偏好
            if claim.claim_type == "support":
                score += 1.0
            elif claim.claim_type == "constraint":
                score += 0.75
            elif claim.claim_type == "explanation":
                score += 0.5

            unit = HistoryUnit(
                type="mainstream_support",
                answer=mainstream_answer,
                claim=claim.text,
                snippet=None,
                source_round=round_id,
            )
            candidates.append(_CandidateUnit(unit=unit, score=score))

        if candidates:
            candidates.sort(key=lambda x: x.score, reverse=True)
            return candidates[0].unit

        # Fallback: if no claim candidate, use a recent raw snippet
        if snippets:
            round_id, snippet = max(snippets, key=lambda x: (x[0], len(x[1])))
            return HistoryUnit(
                type="mainstream_support",
                answer=mainstream_answer,
                claim=None,
                snippet=snippet,
                source_round=round_id,
            )

        return HistoryUnit(
            type="mainstream_support",
            answer=mainstream_answer,
            claim=f"Current mainstream answer in the recent window: {mainstream_answer}",
            snippet=None,
            source_round=latest_round_id,
        )

    def _select_key_rebuttal_unit(
        self,
        mainstream_answer: str | None,
        minority_answer: str | None,
        claims: list[tuple[int, Claim]],
    ) -> HistoryUnit | None:
        if mainstream_answer is None:
            return None

        latest_round_id = self._get_latest_round_id_from_claims(claims)
        if latest_round_id is None:
            return None

        candidates: list[_CandidateUnit] = []

        for round_id, claim in claims:
            if claim.claim_type != "rebuttal":
                continue

            score = 0.0

            # 一级：时间
            score += self._recency_score(round_id, latest_round_id)

            # 二级：rebuttal 本身就是这个槽位的核心候选
            score += 2.0

            # 一级/二级：打向谁
            if claim.related_answer == minority_answer and minority_answer is not None:
                score += 2.0
            elif claim.related_answer is not None and claim.related_answer != mainstream_answer:
                score += 1.5
            elif claim.related_answer == mainstream_answer:
                score += 0.5
            else:
                score += 1.0

            unit = HistoryUnit(
                type="key_rebuttal",
                target_answer=claim.related_answer or minority_answer or mainstream_answer,
                claim=claim.text,
                snippet=None,
                source_round=round_id,
            )
            candidates.append(_CandidateUnit(unit=unit, score=score))

        if not candidates:
            return None

        candidates.sort(key=lambda x: x.score, reverse=True)
        return candidates[0].unit

    def _select_minority_objection_unit(
        self,
        minority_answer: str | None,
        claims: list[tuple[int, Claim]],
        window_states: list[StateRecord],
        answer_stats: dict[str, dict],
    ) -> HistoryUnit | None:
        if minority_answer is None:
            return None

        latest_round_id = window_states[-1].round_id
        minority_stat = answer_stats.get(minority_answer, None)

        persistence_bonus = 0.0
        if minority_stat is not None:
            round_count = len(minority_stat["rounds"])
            raw_count = minority_stat["raw_count"]

            if round_count >= 2:
                persistence_bonus += 1.5
            elif raw_count >= 2:
                persistence_bonus += 0.75

        candidates: list[_CandidateUnit] = []

        for round_id, claim in claims:
            if claim.related_answer != minority_answer:
                continue

            score = 0.0

            # 一级：时间
            score += self._recency_score(round_id, latest_round_id)

            # 一级：这是 minority answer 的直接材料
            score += 2.0

            # 二级：持续性
            score += persistence_bonus

            # 三级：claim type 轻微偏好
            if claim.claim_type == "rebuttal":
                score += 1.0
            elif claim.claim_type == "support":
                score += 0.75
            elif claim.claim_type == "constraint":
                score += 0.75
            else:
                score += 0.5

            unit = HistoryUnit(
                type="minority_objection",
                minority_answer=minority_answer,
                claim=claim.text,
                snippet=None,
                why_unresolved=(
                    f"A minority answer remains in the recent window: {minority_answer}"
                ),
                source_round=round_id,
            )
            candidates.append(_CandidateUnit(unit=unit, score=score))

        if candidates:
            candidates.sort(key=lambda x: x.score, reverse=True)
            return candidates[0].unit

        return HistoryUnit(
            type="minority_objection",
            minority_answer=minority_answer,
            claim=f"A minority answer remains: {minority_answer}",
            snippet=None,
            why_unresolved=f"A minority answer remains in the recent window: {minority_answer}",
            source_round=latest_round_id,
        )

    def _select_core_unresolved_conflict_unit(
        self,
        conflicts: list[tuple[int, UnresolvedConflict]],
        mainstream_answer: str | None,
        minority_answer: str | None,
    ) -> HistoryUnit | None:
        if not conflicts:
            return None

        latest_round_id = max(round_id for round_id, _ in conflicts)
        conflict_text_freq = Counter(conflict.conflict for _, conflict in conflicts)

        candidates: list[_CandidateUnit] = []

        for round_id, conflict in conflicts:
            score = 0.0

            # 一级：时间
            score += self._recency_score(round_id, latest_round_id)

            # 二级：持续性（同一 conflict 文本多次出现）
            same_conflict_count = conflict_text_freq[conflict.conflict]
            if same_conflict_count >= 2:
                score += 1.5
            elif same_conflict_count == 1:
                score += 0.5

            involved = set(conflict.involved_answers)

            # 一级：与当前主流/少数派答案直接相关
            if mainstream_answer is not None and mainstream_answer in involved:
                score += 1.5
            if minority_answer is not None and minority_answer in involved:
                score += 1.5
            if (
                mainstream_answer is not None
                and minority_answer is not None
                and mainstream_answer in involved
                and minority_answer in involved
            ):
                score += 1.0

            unit = HistoryUnit(
                type="core_unresolved_conflict",
                conflict=conflict.conflict,
                why_still_open=conflict.why_still_open,
                source_round=round_id,
            )
            candidates.append(_CandidateUnit(unit=unit, score=score))

        if not candidates:
            return None

        candidates.sort(key=lambda x: x.score, reverse=True)
        return candidates[0].unit

    # ------------------------------------------------------------------
    # Scoring helpers
    # ------------------------------------------------------------------

    def _recency_weight(self, round_id: int, latest_round_id: int) -> float:
        """
        Used for answer window statistics.
        Example in a 3-round window:
        latest -> 3
        one step older -> 2
        two steps older -> 1
        """
        delta = latest_round_id - round_id
        return max(1.0, float(self.history_window_rounds - delta))

    def _recency_score(self, round_id: int, latest_round_id: int) -> float:
        """
        Used for candidate scoring.
        Lower amplitude than _recency_weight, to avoid overpowering other signals.
        Example in a 3-round window:
        latest -> 2.0
        one step older -> 1.0
        two steps older -> 0.5
        """
        delta = latest_round_id - round_id
        if delta <= 0:
            return 2.0
        if delta == 1:
            return 1.0
        return 0.5

    def _get_latest_round_id_from_claims(
        self,
        claims: list[tuple[int, Claim]],
    ) -> int | None:
        if not claims:
            return None
        return max(round_id for round_id, _ in claims)

    # ------------------------------------------------------------------
    # Duplicate control
    # ------------------------------------------------------------------

    def _duplicate_unit(self, existing_units: list[HistoryUnit], new_unit: HistoryUnit) -> bool:
        new_key = (
            new_unit.type,
            new_unit.claim,
            new_unit.snippet,
            new_unit.conflict,
            new_unit.answer,
            new_unit.target_answer,
            new_unit.minority_answer,
        )
        for unit in existing_units:
            old_key = (
                unit.type,
                unit.claim,
                unit.snippet,
                unit.conflict,
                unit.answer,
                unit.target_answer,
                unit.minority_answer,
            )
            if old_key == new_key:
                return True
        return False