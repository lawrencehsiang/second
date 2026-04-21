from __future__ import annotations

from dataclasses import dataclass

from src.components.state_store import StateStore
from src.pipeline.repair_round_executor import RepairRoundExecutor
from src.schemas import RepairBrief, StateRecord


@dataclass
class RepairOrchestratorConfig:
    question: str
    agent_ids: list[str]
    max_round: int = 6


class RepairOrchestrator:
    """
    Orchestrates repair mode after rollback.

    Responsibilities:
    - Execute each repair round
    - Stop when repair action becomes finalize
    - Manage previous_repair_state_record explicitly
    - Avoid duplicate state insertion (executor already writes state)
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
        Runs repair mode after rollback.

        Expected rollback_context keys:
        - trigger_round
        - anchor_round
        - anchor_state
        - failed_suffix_state_records
        - repair_brief
        """
        round_id = rollback_context["anchor_round"] + 1
        anchor_round = rollback_context["anchor_round"]
        anchor_state: StateRecord = rollback_context["anchor_state"]
        failed_suffix_state_records: list[StateRecord] = rollback_context[
            "failed_suffix_state_records"
        ]
        repair_brief: RepairBrief | None = rollback_context.get("repair_brief")

        print(f"Repair Mode initiated after rollback at round {round_id}...")

        self.state_store.add_event(
            {
                "type": "repair_started",
                "start_round": round_id,
                "anchor_round": anchor_round,
            }
        )

        previous_repair_state_record: StateRecord | None = None
        is_first_repair_round = True

        while round_id <= self.config.max_round:
            print(f"Executing Repair Round {round_id}...")

            round_result = self.repair_round_executor.execute_repair_round(
                round_id=round_id,
                anchor_state=anchor_state,
                failed_suffix_state_records=failed_suffix_state_records,
                previous_repair_state_record=previous_repair_state_record,
                repair_brief=repair_brief if is_first_repair_round else None,
            )

            # 不再重复写 state，executor 内部已经 add_state_record(...)
            previous_repair_state_record = round_result.state_record
            is_first_repair_round = False

            if round_result.repair_action == "finalize":
                print(f"Repair mode finalized in round {round_id}.")
                self.state_store.add_event(
                    {
                        "type": "repair_finalized",
                        "final_round": round_id,
                    }
                )
                break

            round_id += 1

        print("Repair mode completed.")