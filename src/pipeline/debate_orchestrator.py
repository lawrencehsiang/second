from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src.pipeline.normal_round_executor import NormalRoundExecutor
from src.components.state_store import StateStore
from src.schemas import StateRecord


@dataclass
class DebateOrchestratorConfig:
    question: str
    agent_ids: list[str]
    max_round: int = 6


class DebateOrchestrator:
    """
    Orchestrates the debate in normal mode.

    End conditions:
    1. rollback is triggered
    2. early-stop is triggered
    3. max_round is reached
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

    def run_debate(self) -> dict:
        """
        Returns:
        {
            "rollback_context": dict | None,
            "early_stopped": bool,
        }
        """
        round_id = 1
        used_rollback_count = 0

        while round_id <= self.config.max_round:
            print(f"Executing Round {round_id}...")

            previous_state_record = self.state_store.get_state_record(round_id - 1)
            # print(f"Previous state record for round {round_id - 1}: {previous_state_record}")

            if previous_state_record is not None:
                round_result = self.normal_round_executor.execute_round(
                    round_id=round_id,
                    used_rollback_count=used_rollback_count,
                )
            else:
                round_result = self.normal_round_executor.execute_round(
                    round_id=round_id,
                )

            # 兼容当前工程：即使 executor 里已经写入过，这里仍保留一次
            self.state_store.add_state_record(round_result.state_record)

            # 1. rollback 优先
            if round_result.rollback_decision.trigger_rollback:
                # print(f"Rollback decision made: {round_result.rollback_decision.reason}")
                rollback_decision = round_result.rollback_decision
                used_rollback_count += 1

                self.state_store.add_event({
                    "type": "rollback_triggered",
                    "trigger_round": round_id,
                    "anchor_round": rollback_decision.rollback_to_round,
                })

                rollback_context = {
                    "trigger_round": round_id,
                    "anchor_round": rollback_decision.rollback_to_round,
                    "anchor_state": self.state_store.get_state_record(
                        rollback_decision.rollback_to_round
                    ),
                    "failed_suffix_state_records": self._get_failed_suffix(
                                                        rollback_decision.rollback_to_round,
                                                        round_id,
                                                    ),
                }

                return {
                    "rollback_context": rollback_context,
                    "early_stopped": False,
                }

            # 2. early stop（只在 normal mode 做）
            if self._should_early_stop(current_round_id=round_id):
                print(
                    f"Early stop triggered at round {round_id}: "
                    f"debate has converged with no new information."
                )
                return {
                    "rollback_context": None,
                    "early_stopped": True,
                }

            round_id += 1

        print("Debate completed.")
        return {
            "rollback_context": None,
            "early_stopped": False,
        }

    def _get_failed_suffix(self, anchor_round: int, trigger_round: int) -> list[StateRecord]:
        failed_suffix: list[StateRecord] = []
        for state_record in self.state_store.list_state_records():
            if anchor_round < state_record.round_id <= trigger_round:
                failed_suffix.append(state_record)
        return failed_suffix

    def _should_early_stop(self, current_round_id: int) -> bool:
        """
        Early-stop rule:
        Stop only if the last TWO rounds both satisfy:
        - all current_answers are identical
        - unresolved_conflicts is empty
        - newly_added_claims is empty
        """
        if current_round_id < 2:
            return False

        prev_state = self.state_store.get_state_record(current_round_id - 1)
        curr_state = self.state_store.get_state_record(current_round_id)

        if prev_state is None or curr_state is None:
            return False

        return self._is_converged_state(prev_state) and self._is_converged_state(curr_state)

    def _is_converged_state(self, state_record: StateRecord) -> bool:
        """
        A converged state means:
        - all answers are identical
        """
        if not state_record.current_answers:
            return False

        all_answers_same = len(set(state_record.current_answers)) == 1
        #no_unresolved_conflicts = len(state_record.unresolved_conflicts) == 0
        #no_new_claims = len(state_record.newly_added_claims) == 0

        #return all_answers_same and no_unresolved_conflicts and no_new_claims
        return all_answers_same