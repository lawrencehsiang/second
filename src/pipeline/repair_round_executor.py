from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from src.components.recorder import Recorder
from src.components.repair_action_mapper import RepairActionMapper
from src.components.repair_brief_generator import RepairBriefGenerator
from src.components.repair_evaluator import RepairEvaluator
from src.components.state_store import StateStore
from src.pipeline.postprocess import apply_keep_or_update
from src.schemas import (
    AgentOutputNormal,
    RepairAgentInput,
    RepairBrief,
    RepairRoundResult,
    StateRecord,
)
from src.components.history_manager import HistoryManager

class RepairAgentRunnerProtocol(Protocol):
    """
    Minimal interface expected by RepairRoundExecutor.

    Current design:
    - reuse AgentOutputNormal as the repair-round agent output format
    - because repair mode still needs:
      current_answer / response_to_conflicts / brief_reason
    """

    def run_repair_round(
        self,
        agent_id: str,
        repair_agent_input: RepairAgentInput,
    ) -> AgentOutputNormal:
        ...


@dataclass
class RepairRoundExecutorConfig:
    question: str
    agent_ids: list[str]
    max_round: int = 6


class RepairRoundExecutor:
    """
    Execute one round in repair mode.

    Responsibilities:
    1. Generate or reuse repair brief.
    2. Build repair-mode agent inputs.
    3. Run repair agents.
    4. Postprocess keep_or_update.
    5. Build StateRecord via Recorder.
    6. Evaluate repair progress / quality / completion readiness.
    7. Map repair action: continue | finalize.
    8. Persist state into StateStore.
    9. Return RepairRoundResult.

    Notes:
    - This executor does NOT decide rollback again.
    - It assumes repair mode has already started after rollback.
    """

    def __init__(
        self,
        config: RepairRoundExecutorConfig,
        repair_agent_runner: RepairAgentRunnerProtocol,
        state_store: StateStore,
        repair_brief_generator: RepairBriefGenerator,
        recorder: Recorder,
        repair_evaluator: RepairEvaluator,
        repair_action_mapper: RepairActionMapper,
        history_manager: HistoryManager,
    ) -> None:
        self.config = config
        self.repair_agent_runner = repair_agent_runner
        self.state_store = state_store
        self.repair_brief_generator = repair_brief_generator
        self.recorder = recorder
        self.repair_evaluator = repair_evaluator
        self.repair_action_mapper = repair_action_mapper
        self.history_manager = history_manager

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def execute_repair_round(
        self,
        round_id: int,
        anchor_state: StateRecord,
        failed_suffix_state_records: list[StateRecord],
        previous_repair_state_record: StateRecord | None = None,
        repair_brief: RepairBrief | None = None,
    ) -> RepairRoundResult:
        """
        Execute one repair-mode round.

        Args:
            round_id: Global round id for this repair round.
            anchor_state: The selected healthy anchor state.
            failed_suffix_state_records: Old failed suffix states after the anchor.
            previous_repair_state_record: Previous repair-round state if this is not the first repair round.
            repair_brief: Optional pre-generated repair brief. If None, generate it here.

        Returns:
            RepairRoundResult
        """
        self._validate_round_id(round_id)
        self.state_store.add_event({
            "type": "repair_round_executed",
            "round_id": round_id,
            "mode": "repair",
        })

        if repair_brief is None:
            repair_brief = self.repair_brief_generator.generate_brief_from_parts(
                question=self.config.question,
                anchor_state=anchor_state,
                failed_suffix_state_records=failed_suffix_state_records,
            )

        agent_inputs: list[RepairAgentInput] = []
        agent_outputs: list[AgentOutputNormal] = []

        history_units = self._build_repair_history_units(
            round_id=round_id,
            previous_repair_state_record=previous_repair_state_record,
            anchor_state=anchor_state,
        )

        previous_answer_map = self._build_previous_answer_map_for_repair(
            anchor_state=anchor_state,
            previous_repair_state_record=previous_repair_state_record,
        )

        for agent_id in self.config.agent_ids:
            agent_input = RepairAgentInput(
                question=self.config.question,
                history_units=history_units,
                repair_brief=repair_brief,
            )
            agent_output = self.repair_agent_runner.run_repair_round(
                agent_id=agent_id,
                repair_agent_input=agent_input,
            )

            agent_inputs.append(agent_input)
            agent_outputs.append(agent_output)

        agent_outputs = apply_keep_or_update(
            agent_outputs=agent_outputs,
            previous_answer_map=previous_answer_map,
        )

        recorder_previous_state = (
            previous_repair_state_record if previous_repair_state_record is not None else anchor_state
        )

        state_record = self.recorder.build_state_record(
            round_id=round_id,
            agent_outputs=agent_outputs,
            previous_state_record=recorder_previous_state,
        )

        repair_scores = self.repair_evaluator.evaluate_repair(
            question=self.config.question,
            anchor_state=anchor_state,
            repair_brief=repair_brief,
            current_state_record=state_record,
            previous_repair_state_record=previous_repair_state_record,
        )

        repair_action_decision = self.repair_action_mapper.map_action(
            repair_scores=repair_scores,
            current_round=round_id,
            max_round=self.config.max_round,
        )

        self.state_store.add_state_record(state_record)
        # repair mode action is not part of normal RoundAction enum,
        # so we do not write it into round_action_history.

        return RepairRoundResult(
            round_id=round_id,
            mode="repair",
            agent_inputs=[x.model_dump() for x in agent_inputs],
            agent_outputs=[x.model_dump() for x in agent_outputs],
            state_record=state_record,
            repair_scores=repair_scores,
            repair_action=repair_action_decision.repair_action,
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

    def _build_previous_answer_map_for_repair(
        self,
        anchor_state: StateRecord,
        previous_repair_state_record: StateRecord | None,
    ) -> dict[str, str]:
        """
        For keep_or_update in repair mode:
        - if previous repair state exists, compare against it
        - otherwise compare against anchor_state
        """
        source_state = previous_repair_state_record or anchor_state
        answers = source_state.current_answers

        if len(answers) != len(self.config.agent_ids):
            raise ValueError(
                "The number of previous answers does not match the number of configured agents."
            )

        return {
            agent_id: answer
            for agent_id, answer in zip(self.config.agent_ids, answers)
        }

    def _build_repair_history_units(
        self,
        round_id: int,
        previous_repair_state_record: StateRecord | None,
        anchor_state: StateRecord,
    ) -> list:
        """
        First repair round:
            reuse the real cached history units of anchor round.
        Later repair rounds:
            rebuild history from anchor + previous repair states only.
            Old failed suffix has already been removed from the main store.
        """
        if previous_repair_state_record is None:
            cached_anchor_history = self.state_store.get_history_units(anchor_state.round_id)
            if cached_anchor_history is None:
                raise ValueError(
                    f"Missing cached history units for anchor round {anchor_state.round_id}."
                )
            return cached_anchor_history

        history_units = self.history_manager.build_history_units(
            question=self.config.question,
            current_round_id=round_id,
            state_store=self.state_store,
        )
        self.state_store.set_history_units(round_id, history_units)
        return history_units