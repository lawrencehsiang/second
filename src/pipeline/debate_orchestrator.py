from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Optional

from src.pipeline.normal_round_executor import NormalRoundExecutor
from src.components.state_store import StateStore
from src.components.rollback_controller import RollbackController
from src.schemas import RoundResult, RollbackDecision, StateRecord


class DebateOrchestratorConfig:
    def __init__(self, question: str, agent_ids: list[str], max_round: int = 6):
        self.question = question
        self.agent_ids = agent_ids
        self.max_round = max_round


class DebateOrchestrator:
    """
    Orchestrates the debate in normal mode, controlling each round and managing
    rollback decisions.

    Responsibilities:
    - Execute each round in normal mode
    - Decide whether to rollback based on evaluator scores
    - Manage state transitions between rounds
    - Store each round's results in the StateStore
    """

    def __init__(
        self,
        config: DebateOrchestratorConfig,
        state_store: StateStore,
        normal_round_executor: NormalRoundExecutor,
    ) -> None:
        self.config = config
        self.state_store = state_store
        self.normal_round_executor = normal_round_executor

    def run_debate(self) -> Optional[dict]:
        """
        Runs the debate by executing multiple rounds until the final state is reached.

        If any round's action decision is rollback, the debate will be restarted with the appropriate round.
        If a rollback is triggered, returns a dictionary with information for repair mode.
        Otherwise, returns None.
        """
        round_id = 1
        used_rollback_count = 0

        while round_id <= self.config.max_round:
            print(f"Executing Round {round_id}...")

            # Get the previous state record if exists
            previous_state_record = self.state_store.get_state_record(round_id - 1)

            if previous_state_record:
                round_result = self.normal_round_executor.execute_round(
                    round_id=round_id,
                    used_rollback_count=used_rollback_count,
                )
            else:
                round_result = self.normal_round_executor.execute_round(
                    round_id=round_id,
                )

            self.state_store.add_state_record(round_result.state_record)

            if round_result.rollback_decision.trigger_rollback:
                print(f"Rollback decision made: {round_result.rollback_decision.reason}")
                rollback_decision = round_result.rollback_decision
                used_rollback_count += 1
                # Return the context needed for repair mode
                return {
                    "trigger_round": round_id,
                    "anchor_round": rollback_decision.rollback_to_round,
                    "anchor_state": self.state_store.get_state_record(rollback_decision.rollback_to_round),
                    "failed_suffix_state_records": self._get_failed_suffix(round_id),
                }

            # Proceed to the next round
            round_id += 1

        print("Debate completed.")
        return None

    def _get_failed_suffix(self, round_id: int) -> list[StateRecord]:
        """
        Retrieve all failed suffix state records up to the given round_id.
        This is used to collect all state records that occurred after the anchor round.
        """
        failed_suffix = []
        for state_record in self.state_store.get_all_state_records():
            if state_record.round_id >= round_id:
                failed_suffix.append(state_record)
        return failed_suffix