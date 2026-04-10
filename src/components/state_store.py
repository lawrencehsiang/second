from __future__ import annotations

from dataclasses import dataclass, field

from src.schemas import RoundAction, StateRecord


@dataclass
class StateStore:
    """
    In-memory store for debate states and round actions.

    Responsibilities:
    1. Store StateRecord for each round.
    2. Store mapped round actions (continue/watch/rollback).
    3. Provide convenient retrieval helpers for downstream modules.

    Notes:
    - This is an in-memory version only.
    - Persistence to disk can be added later if needed.

    用于存储辩论状态与回合操作的内存存储模块。
    职责：
    为每个回合存储状态记录。
    存储映射后的回合操作（继续 / 查看 / 回滚）。
    为下游模块提供便捷的检索辅助方法。
    说明：
    本实现仅为内存版本。
    若有需要，后续可添加持久化到磁盘的功能。
    """

    state_record_pool: list[StateRecord] = field(default_factory=list)
    round_action_history: dict[int, RoundAction] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # StateRecord operations
    # ------------------------------------------------------------------

    def add_state_record(self, state_record: StateRecord) -> None:
        """
        Add or replace the StateRecord for a given round_id.
        """
        existing_index = self._find_state_index(state_record.round_id)
        if existing_index is None:
            self.state_record_pool.append(state_record)
            self.state_record_pool.sort(key=lambda x: x.round_id)
        else:
            self.state_record_pool[existing_index] = state_record

    def get_state_record(self, round_id: int) -> StateRecord | None:
        """
        Get the StateRecord of a specific round.
        """
        for state in self.state_record_pool:
            if state.round_id == round_id:
                return state
        return None

    def get_latest_state_record(self) -> StateRecord | None:
        """
        Get the most recent StateRecord.
        """
        if not self.state_record_pool:
            return None
        return self.state_record_pool[-1]

    def get_previous_state_record(self) -> StateRecord | None:
        """
        Get the second most recent StateRecord.
        Useful when comparing (t-1) and t.
        """
        if len(self.state_record_pool) < 2:
            return None
        return self.state_record_pool[-2]

    def list_state_records(self) -> list[StateRecord]:
        """
        Return all StateRecords in ascending round order.
        """
        return list(self.state_record_pool)

    def has_round(self, round_id: int) -> bool:
        """
        Whether a StateRecord exists for the given round.
        """
        return self.get_state_record(round_id) is not None

    # ------------------------------------------------------------------
    # RoundAction operations
    # ------------------------------------------------------------------

    def set_round_action(self, round_id: int, action: RoundAction) -> None:
        """
        Record the mapped action for a round.
        """
        self.round_action_history[round_id] = action

    def get_round_action(self, round_id: int) -> RoundAction | None:
        """
        Get the mapped action of a specific round.
        """
        return self.round_action_history.get(round_id)

    def get_action_history(self) -> dict[int, RoundAction]:
        """
        Return a shallow copy of round action history.
        """
        return dict(self.round_action_history)

    # ------------------------------------------------------------------
    # Rollback helpers
    # ------------------------------------------------------------------

    def get_latest_continue_round(self) -> int | None:
        """
        Return the latest round id whose action is 'continue'.
        """
        continue_rounds = [
            round_id
            for round_id, action in self.round_action_history.items()
            if action == "continue"
        ]
        if not continue_rounds:
            return None
        return max(continue_rounds)

    def get_round_ids(self) -> list[int]:
        """
        Return all available round ids in ascending order.
        """
        return [state.round_id for state in self.state_record_pool]

    def clear(self) -> None:
        """
        Reset the store completely.
        """
        self.state_record_pool.clear()
        self.round_action_history.clear()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _find_state_index(self, round_id: int) -> int | None:
        """
        Find the index of the StateRecord with the given round_id.
        """
        for idx, state in enumerate(self.state_record_pool):
            if state.round_id == round_id:
                return idx
        return None