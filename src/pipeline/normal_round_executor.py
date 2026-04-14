from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

from src.components.action_mapper import ActionMapper
from src.components.evaluator import Evaluator
from src.components.history_manager import HistoryManager
from src.components.recorder import Recorder
from src.components.rollback_controller import RollbackController
from src.components.state_store import StateStore
from src.pipeline.postprocess import apply_keep_or_update
from src.schemas import (
    ActionDecision,
    AgentInputNormal,
    AgentInputRound1,
    AgentOutputNormal,
    AgentOutputRound1,
    RollbackDecision,
    RoundResult,
    StateRecord,
)


class AgentRunnerProtocol(Protocol):
    """
    Minimal interface expected by NormalRoundExecutor.

    The real implementation can call one shared model multiple times
    or call separate model-backed agents.
    """

    def run_round_1(
        self,
        agent_id: str,
        agent_input: AgentInputRound1,
    ) -> AgentOutputRound1:
        ...

    def run_normal_round(
        self,
        agent_id: str,
        agent_input: AgentInputNormal,
    ) -> AgentOutputNormal:
        ...


@dataclass
class NormalRoundExecutorConfig:
    question: str
    agent_ids: list[str]
    max_round: int = 6


class NormalRoundExecutor:
    """
    Execute one round in the normal (pre-repair) debate stage.

    Responsibilities:
    - Build agent inputs
    - Run agents
    - Postprocess keep_or_update
    - Build StateRecord
    - Evaluate adjacent states for t >= 2
    - Map action
    - Decide rollback
    - Persist state + action into StateStore
    - Return a RoundResult
    """

    def __init__(
        self,
        config: NormalRoundExecutorConfig,
        agent_runner: AgentRunnerProtocol,
        state_store: StateStore,
        history_manager: HistoryManager,
        recorder: Recorder,
        evaluator: Evaluator,
        action_mapper: ActionMapper,
        rollback_controller: RollbackController,
    ) -> None:
        self.config = config
        self.agent_runner = agent_runner
        self.state_store = state_store
        self.history_manager = history_manager
        self.recorder = recorder
        self.evaluator = evaluator
        self.action_mapper = action_mapper
        self.rollback_controller = rollback_controller

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def execute_round(
        self,
        round_id: int,
        used_rollback_count: int = 0,
    ) -> RoundResult:
        """
        Execute one normal-stage round.

        Round 1:
        - independent answers
        - recorder -> state_record_1
        - no evaluator
        - default action = continue

        Round t >= 2:
        - history manager selects history units
        - agents run in parallel conceptually (sequentially in code)
        - postprocess keep_or_update
        - recorder -> state_record_t
        - evaluator compares t-1 and t
        - action mapper maps scores to continue/watch/rollback
        - rollback controller decides whether rollback formally triggers
        """
        self._validate_round_id(round_id)

        if round_id == 1:
            return self._execute_round_1(round_id=round_id)

        return self._execute_normal_round(
            round_id=round_id,
            used_rollback_count=used_rollback_count,
        )

    # ------------------------------------------------------------------
    # Round 1
    # ------------------------------------------------------------------

    def _execute_round_1(
        self,
        round_id: int,
    ) -> RoundResult:
        agent_inputs: list[AgentInputRound1] = []
        agent_outputs: list[AgentOutputRound1] = []

        for agent_id in self.config.agent_ids:
            agent_input = AgentInputRound1(
                question=self.config.question,
            )
            agent_output = self.agent_runner.run_round_1(
                agent_id=agent_id,
                agent_input=agent_input,
            )

            agent_inputs.append(agent_input)
            agent_outputs.append(agent_output)

        state_record = self.recorder.build_state_record(
            round_id=round_id,
            agent_outputs=agent_outputs,
            previous_state_record=None,
        )
        self.state_store.set_history_units(round_id, [])

        action_decision = ActionDecision(
            action="continue",
            reason="Round 1 defaults to continue in the normal-stage protocol.",
        )

        rollback_decision = RollbackDecision(
            trigger_rollback=False,
            rollback_to_round=None,
            reason="Round 1 does not trigger rollback.",
        )

        self.state_store.add_state_record(state_record)
        self.state_store.set_round_action(round_id, action_decision.action)

        return RoundResult(
            round_id=round_id,
            agent_inputs=[x.model_dump() for x in agent_inputs],
            agent_outputs=[x.model_dump() for x in agent_outputs],
            state_record=state_record,
            evaluator_scores=None,
            action_decision=action_decision,
            rollback_decision=rollback_decision,
        )

    # ------------------------------------------------------------------
    # Round t >= 2
    # ------------------------------------------------------------------

    def _execute_normal_round(
        self,
        round_id: int,
        used_rollback_count: int,
    ) -> RoundResult:
        previous_state_record = self.state_store.get_state_record(round_id - 1)
        if previous_state_record is None:
            raise ValueError(
                f"Previous StateRecord for round {round_id - 1} is missing."
            )

        agent_inputs: list[AgentInputNormal] = []
        agent_outputs: list[AgentOutputNormal] = []

        history_units = self.history_manager.build_history_units(
            question=self.config.question,
            current_round_id=round_id,
            state_store=self.state_store,
        )
        self.state_store.set_history_units(round_id, history_units)

        previous_answer_map = self._build_previous_answer_map(previous_state_record)

        for idx, agent_id in enumerate(self.config.agent_ids):
            own_previous_answer = previous_answer_map.get(agent_id)
            if own_previous_answer is None:
                raise ValueError(
                    f"Previous answer for agent {agent_id} is missing at round {round_id}."
                )

            agent_input = AgentInputNormal(
                question=self.config.question,
                own_previous_answer=own_previous_answer,
                history_units=history_units,
            )
            agent_output = self.agent_runner.run_normal_round(
                agent_id=agent_id,
                agent_input=agent_input,
            )

            agent_inputs.append(agent_input)
            agent_outputs.append(agent_output)

        agent_outputs = apply_keep_or_update(
            agent_outputs=agent_outputs,
            previous_answer_map=previous_answer_map,
        )

        state_record = self.recorder.build_state_record(
            round_id=round_id,
            agent_outputs=agent_outputs,
            previous_state_record=previous_state_record,
        )

        evaluator_scores = self.evaluator.evaluate_state(
            question=self.config.question,
            previous_state_record=previous_state_record,
            current_state_record=state_record,
        )

        action_decision = self.action_mapper.map_action(evaluator_scores)

        rollback_decision = self.rollback_controller.decide_rollback_from_store(
            current_round_id=round_id,
            current_round_action=action_decision.action,
            state_store=self.state_store,
            used_rollback_count=used_rollback_count,
        )

        self.state_store.add_state_record(state_record)
        self.state_store.set_round_action(round_id, action_decision.action)

        return RoundResult(
            round_id=round_id,
            agent_inputs=[x.model_dump() for x in agent_inputs],
            agent_outputs=[x.model_dump() for x in agent_outputs],
            state_record=state_record,
            evaluator_scores=evaluator_scores,
            action_decision=action_decision,
            rollback_decision=rollback_decision,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _validate_round_id(
        self,
        round_id: int,
    ) -> None:
        if round_id < 1:
            raise ValueError("round_id must be >= 1.")
        if round_id > self.config.max_round:
            raise ValueError(
                f"round_id={round_id} exceeds configured max_round={self.config.max_round}."
            )

    def _build_previous_answer_map(
        self,
        previous_state_record: StateRecord,
    ) -> dict[str, str]:
        """
        Map agent_id -> previous answer.

        Assumption:
        - current_answers in StateRecord preserve the same order as config.agent_ids.
        This is consistent with the current Recorder implementation.
        """
        answers = previous_state_record.current_answers
        if len(answers) != len(self.config.agent_ids):
            raise ValueError(
                "The number of previous answers does not match the number of configured agents."
            )

        return {
            agent_id: answer
            for agent_id, answer in zip(self.config.agent_ids, answers)
        }