from __future__ import annotations

import os
import re
import traceback
import pandas as pd
from dotenv import load_dotenv

from src.components.action_mapper import ActionMapper
from src.components.agent_runner import AgentRunner
from src.components.evaluator import Evaluator
from src.components.history_manager import HistoryManager
from src.components.qianfan_client import QianfanClient
from src.components.recorder import Recorder
from src.components.repair_action_mapper import RepairActionMapper
from src.components.repair_agent_runner import RepairAgentRunner
from src.components.repair_brief_generator import RepairBriefGenerator
from src.components.repair_evaluator import RepairEvaluator
from src.components.rollback_controller import RollbackController
from src.components.state_store import StateStore
from src.pipeline.debate_orchestrator import (
    DebateOrchestrator,
    DebateOrchestratorConfig,
)
from src.pipeline.normal_round_executor import (
    NormalRoundExecutor,
    NormalRoundExecutorConfig,
)
from src.pipeline.repair_orchestrator import (
    RepairOrchestrator,
    RepairOrchestratorConfig,
)
from src.pipeline.repair_round_executor import (
    RepairRoundExecutor,
    RepairRoundExecutorConfig,
)
from src.utils.result_writer import ResultWriter
from src.utils.result_utils import (
    build_trace,
    get_final_answers,
    get_round_1_answers,
    get_rounds_used,
    get_stop_reason,
    is_correct,
    majority_vote,
)

AGENT_IDS = ["A", "B", "C"]
MAX_ROUND = 5
# 强制禁用代理，直连国内网络
os.environ["HTTP_PROXY"] = ""
os.environ["HTTPS_PROXY"] = ""
os.environ["NO_PROXY"] = "qianfan.baidubce.com,localhost,127.0.0.1"
load_dotenv()

def build_llm_client() -> QianfanClient:
    api_key = os.getenv("QIANFAN_API_KEY")
    if not api_key:
        raise ValueError("Missing QIANFAN_API_KEY. Please set it in your .env file.")

    return QianfanClient(
        api_key=api_key,
        model="qwen2.5-7b-instruct",
    )


def load_gsm8k_samples(
    parquet_path: str = r"datasets\gsm8k\main\train-00000-of-00001.parquet",
    limit: int = 1,
) -> list[tuple[str, str, str]]:
    df = pd.read_parquet(parquet_path)

    samples = []
    for i in range(min(limit, len(df))):
        row = df.iloc[i]

        question = str(row["question"]).strip()
        answer_text = str(row["answer"])

        result = re.findall(r"####\s*(-?\d+(?:\.\d+)?)", answer_text)
        gold_answer = result[0] if result else answer_text.strip()

        sample_id = f"gsm8k_{i+1:04d}"
        samples.append((sample_id, question, gold_answer))

    return samples


def run_normal_mode(
    llm_client: QianfanClient,
    question: str,
    gold_answer: str,
    sample_id: str,
):
    state_store = StateStore()

    agent_runner = AgentRunner(llm_client=llm_client)
    history_manager = HistoryManager()
    recorder = Recorder(llm_client=llm_client)
    evaluator = Evaluator(llm_client=llm_client)

    normal_round_executor = NormalRoundExecutor(
        config=NormalRoundExecutorConfig(
            question=question,
            agent_ids=AGENT_IDS,
            max_round=MAX_ROUND,
        ),
        agent_runner=agent_runner,
        state_store=state_store,
        history_manager=history_manager,
        recorder=recorder,
        evaluator=evaluator,
        action_mapper=ActionMapper(),
        rollback_controller=RollbackController(),
    )

    debate_orchestrator = DebateOrchestrator(
        config=DebateOrchestratorConfig(
            question=question,
            agent_ids=AGENT_IDS,
            max_round=MAX_ROUND,
        ),
        state_store=state_store,
        normal_round_executor=normal_round_executor,
    )

    print(f"\n===== Running {sample_id} =====")
    print("Question:", question)
    print("Gold answer:", gold_answer)
    print("Starting the debate in normal mode...")

    debate_result = debate_orchestrator.run_debate()

    rollback_context = debate_result["rollback_context"]
    early_stopped = debate_result["early_stopped"]

    if rollback_context:
        anchor_round = rollback_context.get("anchor_round")
        anchor_state = rollback_context.get("anchor_state")
        failed_suffix_state_records = rollback_context.get("failed_suffix_state_records", [])

        if anchor_round is not None and anchor_state is not None:
            print("Rollback detected, switching to repair mode...")

            # 先保留 failed suffix，用它生成 repair_brief
            run_repair_mode(
                llm_client=llm_client,
                question=question,
                rollback_context=rollback_context,
                state_store=state_store,
                history_manager=history_manager,
            )
        else:
            print("Rollback detected, but no valid anchor is available. Skip repair mode.")

    round_1_answers = get_round_1_answers(state_store)
    final_answers = get_final_answers(state_store)

    single_agent_baseline_answer = round_1_answers[0] if round_1_answers else ""
    majority_voting_baseline_answer = majority_vote(round_1_answers)
    scrd_final_answer = majority_vote(final_answers)

    result = {
        "sample_id": sample_id,
        "question": question,
        "gold_answer": gold_answer,
        "round_1_answers": round_1_answers,
        "single_agent_baseline_answer": single_agent_baseline_answer,
        "majority_voting_baseline_answer": majority_voting_baseline_answer,
        "scrd_final_answer": scrd_final_answer,
        "single_agent_correct": is_correct(single_agent_baseline_answer, gold_answer),
        "majority_voting_correct": is_correct(majority_voting_baseline_answer, gold_answer),
        "scrd_correct": is_correct(scrd_final_answer, gold_answer),
        "rounds_used": get_rounds_used(state_store),
        "stop_reason": get_stop_reason(rollback_context, early_stopped),
    }

    trace = build_trace(state_store)
    return result, trace


def run_repair_mode(
    llm_client: QianfanClient,
    question: str,
    rollback_context: dict,
    state_store: StateStore,
    history_manager: HistoryManager,
) -> None:
    recorder = Recorder(llm_client=llm_client)
    repair_brief_generator = RepairBriefGenerator(llm_client=llm_client)
    repair_evaluator = RepairEvaluator(llm_client=llm_client)
    repair_action_mapper = RepairActionMapper()
    repair_agent_runner = RepairAgentRunner(llm_client=llm_client)

    anchor_round = rollback_context["anchor_round"]
    anchor_state = rollback_context["anchor_state"]
    failed_suffix_state_records = rollback_context["failed_suffix_state_records"]

    # Step 1: 先生成 repair_brief
    repair_brief = repair_brief_generator.generate_brief_from_parts(
        question=question,
        anchor_state=anchor_state,
        failed_suffix_state_records=failed_suffix_state_records,
    )

    # Step 2: 再删掉 anchor 之后的旧失败后缀
    state_store.remove_rounds_after(anchor_round)

    # Step 3: 用同一个主 store 跑 repair
    repair_round_executor = RepairRoundExecutor(
        config=RepairRoundExecutorConfig(
            question=question,
            agent_ids=AGENT_IDS,
            max_round=MAX_ROUND,
        ),
        repair_agent_runner=repair_agent_runner,
        state_store=state_store,
        repair_brief_generator=repair_brief_generator,
        recorder=recorder,
        repair_evaluator=repair_evaluator,
        repair_action_mapper=repair_action_mapper,
        history_manager=history_manager,
    )

    repair_orchestrator = RepairOrchestrator(
        config=RepairOrchestratorConfig(
            question=question,
            agent_ids=AGENT_IDS,
            max_round=MAX_ROUND,
        ),
        state_store=state_store,
        repair_round_executor=repair_round_executor,
    )

    print("Starting the repair mode...")
    repair_orchestrator.run_repair(
        rollback_context={
            **rollback_context,
            "repair_brief": repair_brief,
        }
    )


if __name__ == "__main__":
    print("Starting the debate system...")

    llm_client = build_llm_client()
    writer = ResultWriter(output_dir="outputs")

    samples = load_gsm8k_samples(limit=2)
    completed_sample_ids = writer.load_completed_sample_ids()
    if completed_sample_ids:
        print(f"Resume mode: found {len(completed_sample_ids)} completed samples in outputs/results.jsonl")

    total = 0
    single_correct_count = 0
    majority_correct_count = 0
    scrd_correct_count = 0
    skipped_count = 0
    failed_count = 0

    for sample_id, question, gold_answer in samples:
        if sample_id in completed_sample_ids:
            skipped_count += 1
            print(f"Skipping completed sample: {sample_id}")
            continue

        try:
            result, trace = run_normal_mode(
                llm_client=llm_client,
                question=question,
                gold_answer=gold_answer,
                sample_id=sample_id,
            )

            writer.append_result(result)
            writer.write_trace(sample_id, trace)

            total += 1
            single_correct_count += int(result["single_agent_correct"])
            majority_correct_count += int(result["majority_voting_correct"])
            scrd_correct_count += int(result["scrd_correct"])

            print("Result saved:", result)
        except KeyboardInterrupt:
            print("\nInterrupted by user. Progress has been saved. You can rerun to resume.")
            break
        except Exception as exc:
            failed_count += 1
            writer.append_error(
                {
                    "sample_id": sample_id,
                    "question": question,
                    "gold_answer": gold_answer,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )
            print(f"Failed sample {sample_id}: {exc}. Logged to outputs/errors.jsonl. Continuing...")
            continue

    print("\n===== Final Summary =====")
    print(f"Processed successfully in this run: {total}")
    print(f"Skipped (already completed): {skipped_count}")
    print(f"Failed in this run: {failed_count}")
    if total > 0:
        print(f"Single-agent baseline accuracy: {single_correct_count}/{total} = {single_correct_count / total:.4f}")
        print(f"Majority-voting baseline accuracy: {majority_correct_count}/{total} = {majority_correct_count / total:.4f}")
        print(f"SCRD accuracy: {scrd_correct_count}/{total} = {scrd_correct_count / total:.4f}")
    else:
        print("No new successful samples were processed in this run.")
