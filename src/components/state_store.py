from __future__ import annotations
from dataclasses import dataclass, field

from src.schemas import RoundAction, StateRecord


@dataclass
class StateStore:
    """
    In-memory store for:
    1. final effective state records
    2. round action history
    3. cached history units used by each round
    4. execution events (do NOT delete on rollback cleanup)
    """
    state_record_pool: list[StateRecord] = field(default_factory=list)
    round_action_history: dict[int, RoundAction] = field(default_factory=dict)
    history_unit_history: dict[int, list] = field(default_factory=dict)
    execution_events: list[dict] = field(default_factory=list)

    # ------------------------------------------------------------------
    # StateRecord operations
    # ------------------------------------------------------------------
    def add_state_record(self, state_record: StateRecord) -> None:
        existing_index = self._find_state_index(state_record.round_id)
        if existing_index is None:
            self.state_record_pool.append(state_record)
            self.state_record_pool.sort(key=lambda x: x.round_id)
        else:
            self.state_record_pool[existing_index] = state_record

    def get_state_record(self, round_id: int) -> StateRecord | None:
        for state in self.state_record_pool:
            if state.round_id == round_id:
                return state
        return None

    def get_latest_state_record(self) -> StateRecord | None:
        if not self.state_record_pool:
            return None
        return self.state_record_pool[-1]

    def get_previous_state_record(self) -> StateRecord | None:
        if len(self.state_record_pool) < 2:
            return None
        return self.state_record_pool[-2]

    def list_state_records(self) -> list[StateRecord]:
        return list(self.state_record_pool)

    # ------------------------------------------------------------------
    # RoundAction operations
    # ------------------------------------------------------------------
    def set_round_action(self, round_id: int, action: RoundAction) -> None:
        self.round_action_history[round_id] = action

    def get_round_action(self, round_id: int) -> RoundAction | None:
        return self.round_action_history.get(round_id)
    
    def get_action_history(self) -> dict[int, RoundAction]:
        return dict(self.round_action_history)

    # ------------------------------------------------------------------
    # History unit cache
    # ------------------------------------------------------------------
    def set_history_units(self, round_id: int, history_units: list) -> None:
        self.history_unit_history[round_id] = history_units

    def get_history_units(self, round_id: int) -> list | None:
        return self.history_unit_history.get(round_id)
    

    # ------------------------------------------------------------------
    # Execution events
    # ------------------------------------------------------------------
    def add_event(self, event: dict) -> None:
        self.execution_events.append(event)

    def list_events(self) -> list[dict]:
        return list(self.execution_events)

    # ------------------------------------------------------------------
    # Rollback cleanup
    # ------------------------------------------------------------------
    def remove_rounds_after(self, round_id: int) -> None:
        """
        Keep the anchor round itself.
        Remove failed suffix from final effective state/action/history cache.

        IMPORTANT:
        execution_events are NOT removed, because they represent the real
        execution history.
        """
        self.state_record_pool = [
            state for state in self.state_record_pool
            if state.round_id <= round_id
        ]
        self.round_action_history = {
            rid: action
            for rid, action in self.round_action_history.items()
            if rid <= round_id
        }
        self.history_unit_history = {
            rid: units
            for rid, units in self.history_unit_history.items()
            if rid <= round_id
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _find_state_index(self, round_id: int) -> int | None:
        for idx, state in enumerate(self.state_record_pool):
            if state.round_id == round_id:
                return idx
        return None