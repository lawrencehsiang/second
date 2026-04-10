from __future__ import annotations

from src.components.state_store import StateStore
from src.schemas import RollbackDecision, RoundAction


class RollbackController:
    """
    Rule-based RollbackController.

    Responsibilities:
    1. Decide whether the current round should formally trigger rollback.
    2. Distinguish between immediate rollback and cumulative rollback.
    3. Optionally locate the latest healthy anchor round (latest continue round).
    """

    def __init__(
        self,
        max_rollbacks: int = 1,
        round_1_not_rollback_target: bool = True,
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
        Decide whether rollback should be triggered for the current round.

        Args:
            current_round_id: Current round index.
            current_round_action: Action mapped for the current round.
            previous_round_action: Action of the previous round, if any.
            has_used_rollback: Whether the system has already used rollback.
            state_store: Optional StateStore for locating rollback anchor round.

        Returns:
            RollbackDecision
        """
        # hard stop: no more rollback allowed
        if has_used_rollback:
            return RollbackDecision(
                trigger_rollback=False,
                rollback_to_round=None,
                reason="none: rollback budget already used.",
            )

        # round 1 should never trigger rollback in the normal protocol
        if current_round_id <= 1:
            return RollbackDecision(
                trigger_rollback=False,
                rollback_to_round=None,
                reason="none: round 1 does not trigger rollback.",
            )

        trigger_type = self._infer_trigger_type(
            current_round_action=current_round_action,
            previous_round_action=previous_round_action,
        )

        if trigger_type == "none":
            return RollbackDecision(
                trigger_rollback=False,
                rollback_to_round=None,
                reason="none: rollback trigger conditions are not met.",
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
                    f"{trigger_type}: rollback is triggered, "
                    "but no valid continue anchor round is available."
                ),
            )

        return RollbackDecision(
            trigger_rollback=True,
            rollback_to_round=rollback_to_round,
            reason=(
                f"{trigger_type}: rollback is triggered and will return to "
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
        derive previous_round_action from StateStore automatically.

        Args:
            current_round_id: Current round index.
            current_round_action: Action mapped for the current round.
            state_store: StateStore containing round action history.
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

    def _infer_trigger_type(
        self,
        current_round_action: RoundAction,
        previous_round_action: RoundAction | None,
    ) -> str:
        """
        Infer rollback trigger type according to the pilot rules.

        Rules:
        1. immediate:
           if current_round_action == "rollback"
        2. cumulative:
           if previous_round_action == "watch" and current_round_action == "watch"
           OR previous_round_action == "watch" and current_round_action == "rollback"
        3. none:
           otherwise

        Note:
        - The second cumulative case overlaps with immediate.
        - We prioritize "immediate" for current rollback action.
        """
        if current_round_action == "rollback":
            return "immediate"

        if previous_round_action == "watch" and current_round_action == "watch":
            return "cumulative"

        return "none"

    def _find_anchor_round(
        self,
        current_round_id: int,
        state_store: StateStore | None,
    ) -> int | None:
        """
        Find the latest healthy anchor round:
        the latest round before current_round_id whose action is 'continue'.

        If round_1_not_rollback_target is True, round 1 is excluded.
        """
        if state_store is None:
            return None

        action_history = state_store.get_action_history()
        candidate_rounds = []

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