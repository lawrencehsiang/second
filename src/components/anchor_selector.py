from __future__ import annotations

from src.schemas import AnchorSelectionResult, AnchorSelectorInput, StateRecord


class AnchorSelector:
    """
    Select the healthy anchor round after rollback is triggered.

    Main rule:
    - Search backward from trigger_round - 1
    - Find the latest round whose action is "continue"
    - Return both anchor_round and anchor_state
    """

    def select_anchor(
        self,
        selector_input: AnchorSelectorInput,
    ) -> AnchorSelectionResult:
        """
        Select anchor from structured input object.
        """
        trigger_round = selector_input.trigger_round
        action_history = selector_input.action_history
        state_record_pool = selector_input.state_record_pool

        anchor_round = self._find_anchor_round(
            trigger_round=trigger_round,
            action_history=action_history,
        )
        anchor_state = self._find_anchor_state(
            anchor_round=anchor_round,
            state_record_pool=state_record_pool,
        )

        return AnchorSelectionResult(
            anchor_round=anchor_round,
            anchor_state=anchor_state,
        )

    def select_anchor_from_parts(
        self,
        trigger_round: int,
        action_history: list[dict],
        state_record_pool: list[StateRecord],
    ) -> AnchorSelectionResult:
        """
        Convenience wrapper when caller does not want to construct AnchorSelectorInput manually.
        """
        selector_input = AnchorSelectorInput(
            trigger_round=trigger_round,
            action_history=action_history,
            state_record_pool=state_record_pool,
        )
        return self.select_anchor(selector_input)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _find_anchor_round(
        self,
        trigger_round: int,
        action_history: list[dict],
    ) -> int:
        """
        Find the latest round < trigger_round whose action is 'continue'.

        Expected action_history item format:
        {
            "round_id": 3,
            "action": "continue"
        }
        """
        if trigger_round <= 1:
            raise ValueError("trigger_round must be greater than 1 for anchor selection.")

        # normalize into dict: round_id -> action
        round_action_map: dict[int, str] = {}
        for item in action_history:
            if not isinstance(item, dict):
                continue

            round_id = item.get("round_id")
            action = item.get("action")

            if isinstance(round_id, int) and isinstance(action, str):
                round_action_map[round_id] = action

        for round_id in range(trigger_round - 1, 0, -1):
            if round_action_map.get(round_id) == "continue":
                return round_id

        raise ValueError(
            f"No valid continue anchor round found before trigger_round={trigger_round}."
        )

    def _find_anchor_state(
        self,
        anchor_round: int,
        state_record_pool: list[StateRecord],
    ) -> StateRecord:
        """
        Find the StateRecord corresponding to anchor_round.
        """
        for state in state_record_pool:
            if state.round_id == anchor_round:
                return state

        raise ValueError(
            f"Anchor round {anchor_round} found in action history, "
            "but corresponding StateRecord is missing."
        )