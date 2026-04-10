from __future__ import annotations

from src.components.state_store import StateStore
from src.pipeline.normal_round_executor import NormalRoundExecutor, NormalRoundExecutorConfig
from src.pipeline.repair_round_executor import RepairRoundExecutor, RepairRoundExecutorConfig
from src.pipeline.debate_orchestrator import DebateOrchestrator, DebateOrchestratorConfig
from src.pipeline.repair_orchestrator import RepairOrchestrator, RepairOrchestratorConfig
from src.components.action_mapper import ActionMapper
from src.components.evaluator import Evaluator
from src.components.rollback_controller import RollbackController
from src.components.repair_brief_generator import RepairBriefGenerator
from src.components.repair_action_mapper import RepairActionMapper
from src.components.repair_evaluator import RepairEvaluator
from src.components.repair_agent_runner import RepairAgentRunner
from src.components.recorder import Recorder
from src.schemas import StateRecord


def run_normal_mode():
    """
    Run the normal stage (before rollback).
    """
    # Mocking components (in a real system, these would be implemented with actual logic)
    state_store = StateStore()
    normal_round_executor = NormalRoundExecutor(
        config=NormalRoundExecutorConfig(
            question="How many gems are in the chest?",
            agent_ids=["A", "B", "C"],
            max_round=6,
        ),
        agent_runner=MockAgentRunner(),
        state_store=state_store,
        history_manager=MockHistoryManager(),
        recorder=MockRecorder(),
        evaluator=Evaluator(llm_client=MockLLMClient()),
        action_mapper=ActionMapper(),
        rollback_controller=RollbackController(),
    )

    debate_orchestrator = DebateOrchestrator(
        config=DebateOrchestratorConfig(
            question="How many gems are in the chest?", agent_ids=["A", "B", "C"], max_round=6
        ),
        state_store=state_store,
        normal_round_executor=normal_round_executor,
    )

    # Start the debate
    print("Starting the debate in normal mode...")
    rollback_context = debate_orchestrator.run_debate()

    if rollback_context:
        print("Rollback detected, switching to repair mode...")
        run_repair_mode(rollback_context)


def run_repair_mode(rollback_context: dict):
    """
    Run the repair stage (after rollback).
    """
    # Mocking components (same here for repair phase)
    state_store = StateStore()
    repair_brief_generator = RepairBriefGenerator(llm_client=MockLLMClient())
    repair_evaluator = RepairEvaluator(llm_client=MockLLMClient())
    repair_action_mapper = RepairActionMapper()
    repair_agent_runner = RepairAgentRunner(llm_client=MockLLMClient())

    repair_round_executor = RepairRoundExecutor(
        config=RepairRoundExecutorConfig(
            question="How many gems are in the chest?", agent_ids=["A", "B", "C"], max_round=6
        ),
        repair_agent_runner=repair_agent_runner,
        state_store=state_store,
        repair_brief_generator=repair_brief_generator,
        recorder=MockRecorder(),
        repair_evaluator=repair_evaluator,
        repair_action_mapper=repair_action_mapper,
    )

    repair_orchestrator = RepairOrchestrator(
        config=RepairOrchestratorConfig(
            question="How many gems are in the chest?", agent_ids=["A", "B", "C"], max_round=6
        ),
        state_store=state_store,
        repair_round_executor=repair_round_executor,
    )

    # Start the repair process after rollback
    print("Starting the repair mode...")
    repair_orchestrator.run_repair(rollback_context=rollback_context)


if __name__ == "__main__":
    print("Starting the debate system...")

    # Start the normal mode first
    run_normal_mode()