from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Optional

from src.pipeline.repair_round_executor import RepairRoundExecutor
from src.components.state_store import StateStore
from src.schemas import RepairRoundResult, RollbackDecision, StateRecord


class RepairOrchestratorConfig:
    def __init__(self, question: str, agent_ids: list[str], max_round: int = 6):
        self.question = question
        self.agent_ids = agent_ids
        self.max_round = max_round


class RepairOrchestrator:
    """
    Orchestrates the repair mode after rollback, controlling each repair round and managing
    rollback decisions for repair rounds.

    Responsibilities:
    - Execute each repair round
    - Decide whether to rollback based on evaluator scores
    - Manage state transitions between repair rounds
    - Store each repair round's results in the StateStore
    """

    def __init__(
        self,
        config: RepairOrchestratorConfig,
        state_store: StateStore,
        repair_round_executor: RepairRoundExecutor,
    ) -> None:
        self.config = config
        self.state_store = state_store
        self.repair_round_executor = repair_round_executor

    def run_repair(self, rollback_context: dict) -> None:
        """
        Runs the repair mode by executing multiple rounds after rollback.

        If any round's action decision is rollback, the repair mode will restart with the appropriate round.
        If a rollback is triggered, returns a dictionary with information for repair mode.
        Otherwise, returns None.
        """
        round_id = rollback_context["trigger_round"]
        anchor_state = rollback_context["anchor_state"]
        failed_suffix_state_records = rollback_context["failed_suffix_state_records"]

        print(f"Repair Mode initiated after rollback at round {round_id}...")

        used_rollback_count = 0

        while round_id <= self.config.max_round:
            print(f"Executing Repair Round {round_id}...")

            # Get the previous state record if exists
            previous_state_record = self.state_store.get_state_record(round_id - 1)

            # Execute repair round
            round_result = self.repair_round_executor.execute_repair_round(
                round_id=round_id,
                anchor_state=anchor_state,
                failed_suffix_state_records=failed_suffix_state_records,
                previous_repair_state_record=previous_state_record,
                repair_brief=rollback_context.get("repair_brief"),
            )

            self.state_store.add_state_record(round_result.state_record)

            if round_result.repair_action == "finalize":
                print(f"Repair mode finalized in round {round_id}.")
                break

            if round_result.rollback_decision.trigger_rollback:
                print(f"Rollback decision made: {round_result.rollback_decision.reason}")
                rollback_decision = round_result.rollback_decision
                used_rollback_count += 1
                round_id = rollback_decision.rollback_to_round
                continue  # Restart with the rollback round

            # Proceed to the next round
            round_id += 1

        print("Repair mode completed.")