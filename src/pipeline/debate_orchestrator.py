from __future__ import annotations

from dataclasses import dataclass

from src.components.state_store import StateStore
from src.pipeline.normal_round_executor import NormalRoundExecutor
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
    2. early-stop is triggered by the normal ActionMapper
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

            round_result = self.normal_round_executor.execute_round(
                round_id=round_id,
                used_rollback_count=used_rollback_count,
            )

            # 1) rollback 优先
            if (
                round_result.rollback_decision is not None
                and round_result.rollback_decision.trigger_rollback
            ):
                previous_state = self.state_store.get_state_record(round_id - 1)
                current_state = round_result.state_record

                # 保留这个保护逻辑：如果已经形成稳定共识，就不要因为某些旧规则误触 repair
                if self._should_force_early_stop_on_stable_consensus(
                    previous_state=previous_state,
                    current_state=current_state,
                ):
                    self.state_store.add_event(
                        {
                            "type": "stable_consensus_early_stop",
                            "round_id": round_id,
                            "mode": "normal",
                        }
                    )
                    print(
                        f"Stable consensus detected at round {round_id}: "
                        f"no unresolved conflicts and answers are already aligned. "
                        f"Early stop."
                    )
                    return {
                        "rollback_context": None,
                        "early_stopped": True,
                    }

                rollback_decision = round_result.rollback_decision
                used_rollback_count += 1

                anchor_round = rollback_decision.rollback_to_round
                anchor_state = (
                    self.state_store.get_state_record(anchor_round)
                    if anchor_round is not None
                    else None
                )

                self.state_store.add_event(
                    {
                        "type": "rollback_triggered",
                        "trigger_round": round_id,
                        "anchor_round": anchor_round,
                    }
                )

                failed_suffix_state_records = (
                    self._get_failed_suffix(anchor_round, round_id)
                    if anchor_round is not None
                    else []
                )

                rollback_context = {
                    "trigger_round": round_id,
                    "anchor_round": anchor_round,
                    "anchor_state": anchor_state,
                    "failed_suffix_state_records": failed_suffix_state_records,
                }
                return {
                    "rollback_context": rollback_context,
                    "early_stopped": False,
                }

            # 2) early stop（现在由 mapper 决定）
            if (
                round_result.action_decision is not None
                and round_result.action_decision.action == "early_stop"
            ):
                print(
                    f"Early stop triggered at round {round_id}: "
                    f"action mapper returned early_stop."
                )
                self.state_store.add_event(
                    {
                        "type": "early_stop_triggered",
                        "round_id": round_id,
                        "mode": "normal",
                    }
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

    def _get_failed_suffix(
        self,
        anchor_round: int,
        trigger_round: int,
    ) -> list[StateRecord]:
        failed_suffix: list[StateRecord] = []
        for state_record in self.state_store.list_state_records():
            if anchor_round < state_record.round_id <= trigger_round:
                failed_suffix.append(state_record)
        return failed_suffix

    def _answers_are_equivalent(
        self,
        answers: list[str],
    ) -> bool:
        if not answers:
            return False

        normalized = []
        for ans in answers:
            try:
                import re

                text = str(ans).strip().lower()
                text = text.replace(",", "")
                text = text.replace("$", "")
                text = text.replace("dollars", "").replace("dollar", "").strip()
                nums = re.findall(r"-?\d+(?:\.\d+)?", text)
                if nums:
                    normalized.append(f"num:{int(round(float(nums[-1])))}")
                else:
                    normalized.append(f"text:{text}")
            except Exception:
                normalized.append(f"text:{str(ans).strip().lower()}")

        return len(set(normalized)) == 1

    def _should_force_early_stop_on_stable_consensus(
        self,
        previous_state: StateRecord | None,
        current_state: StateRecord | None,
    ) -> bool:
        if current_state is None:
            return False

        # 1. 当前没有冲突
        if current_state.unresolved_conflicts:
            return False

        # 2. 当前答案已经一致 / 数值等价
        if not self._answers_are_equivalent(current_state.current_answers):
            return False

        # 3. 第一轮之后才有意义
        if previous_state is None:
            return False

        # 4. 上一轮如果也已经一致，说明只是稳定延续，不该 rollback
        if self._answers_are_equivalent(previous_state.current_answers):
            return True

        return False