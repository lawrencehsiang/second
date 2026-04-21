from __future__ import annotations

from src.components.state_store import StateStore
from src.schemas import RollbackDecision, RoundAction


class RollbackController:
    """
    Rule-based rollback controller aligned with the current normal-mode action space.

    Responsibilities:
    1. Decide whether a mapper-suggested rollback should be formally triggered.
    2. Enforce rollback budget constraints.
    3. Locate the latest healthy anchor round (latest 'continue' round before current round).
    """

    def __init__(
        self,
        max_rollbacks: int = 1,
        round_1_not_rollback_target: bool = False,
    ) -> None:
        self.max_rollbacks = max_rollbacks
        self.round_1_not_rollback_target = round_1_not_rollback_target

    def decide_rollback(
        self,
        current_round_id: int,
        current_round_action: RoundAction,
        previous_round_action: RoundAction | None,
        has_used_rollback: bool,
        state_store: StateStore | None = None,
    ) -> RollbackDecision:
        """
        Decide whether rollback should be formally triggered for the current round.

        Current protocol:
        - Only an explicit mapper action == "rollback" can trigger rollback.
        - "continue" and "early_stop" never trigger rollback.
        - Round 1 never triggers rollback.
        - If rollback is triggered, try to find the latest valid anchor round.

        Args:
            current_round_id: Current round index.
            current_round_action: Mapper output for the current round.
            previous_round_action: Previous round action. Kept for interface compatibility.
            has_used_rollback: Whether rollback budget has already been consumed.
            state_store: Optional StateStore for locating the rollback anchor.

        Returns:
            RollbackDecision
        """
        del previous_round_action  # compatibility only; no cumulative logic in current protocol

        if has_used_rollback:
            return RollbackDecision(
                trigger_rollback=False,
                rollback_to_round=None,
                reason="none: rollback budget already used.",
            )

        if current_round_id <= 1:
            return RollbackDecision(
                trigger_rollback=False,
                rollback_to_round=None,
                reason="none: round 1 does not trigger rollback.",
            )

        if current_round_action != "rollback":
            return RollbackDecision(
                trigger_rollback=False,
                rollback_to_round=None,
                reason="none: current round action is not rollback.",
            )

        rollback_to_round = self._find_anchor_round(
            current_round_id=current_round_id,
            state_store=state_store,
        )

        if rollback_to_round is None:
            return RollbackDecision(
                trigger_rollback=True,
                rollback_to_round=None,
                reason=(
                    "immediate: rollback is triggered, "
                    "but no valid continue anchor round is available."
                ),
            )

        return RollbackDecision(
            trigger_rollback=True,
            rollback_to_round=rollback_to_round,
            reason=(
                f"immediate: rollback is triggered and will return to "
                f"round {rollback_to_round}."
            ),
        )

    def decide_rollback_from_store(
        self,
        current_round_id: int,
        current_round_action: RoundAction,
        state_store: StateStore,
        used_rollback_count: int = 0,
    ) -> RollbackDecision:
        """
        Convenience wrapper:
        - derives previous_round_action from StateStore for interface compatibility
        - checks rollback budget
        - delegates to decide_rollback()

        Args:
            current_round_id: Current round index.
            current_round_action: Mapper output for the current round.
            state_store: StateStore containing action history.
            used_rollback_count: Number of rollbacks already used.

        Returns:
            RollbackDecision
        """
        previous_round_action = state_store.get_round_action(current_round_id - 1)
        has_used_rollback = used_rollback_count >= self.max_rollbacks

        return self.decide_rollback(
            current_round_id=current_round_id,
            current_round_action=current_round_action,
            previous_round_action=previous_round_action,
            has_used_rollback=has_used_rollback,
            state_store=state_store,
        )

    # ------------------------------------------------------------------
    # Internal logic
    # ------------------------------------------------------------------

    def _find_anchor_round(
        self,
        current_round_id: int,
        state_store: StateStore | None,
    ) -> int | None:
        """
        Find the latest healthy anchor round:
        the latest round before current_round_id whose action is 'continue'.

        If round_1_not_rollback_target is True, round 1 is excluded.

        Args:
            current_round_id: Current round index.
            state_store: StateStore containing action history.

        Returns:
            The selected anchor round, or None if unavailable.
        """
        if state_store is None:
            return None

        action_history = state_store.get_action_history()
        candidate_rounds: list[int] = []

        for round_id, action in action_history.items():
            if round_id >= current_round_id:
                continue
            if action != "continue":
                continue
            if self.round_1_not_rollback_target and round_id == 1:
                continue
            candidate_rounds.append(round_id)

        if not candidate_rounds:
            return None

        return max(candidate_rounds)