from __future__ import annotations

import os
import re
import traceback
import pandas as pd
from dotenv import load_dotenv
from src.components.usage_logger import UsageLogger
from src.utils.result_utils import build_usage_summary
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
    build_trace_bundle,
    get_final_answers,
    get_round_1_answers,
    get_stop_reason,
    is_correct,
    majority_vote,
    get_actual_rounds_executed,
    get_effective_rounds_used
)

from src.components.state_store import StateStore
import inspect
import json
from pathlib import Path


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
    samples: list[tuple[str, str, str]] = []

    for i in range(min(limit, len(df))):
        row = df.iloc[i]
        question = str(row["question"]).strip()
        answer_text = str(row["answer"])
        result = re.findall(r"####\s*(-?\d+(?:\.\d+)?)", answer_text)
        gold_answer = result[0] if result else answer_text.strip()
        sample_id = f"gsm8k_{i+1:04d}"
        samples.append((sample_id, question, gold_answer))

    return samples


def load_strategyqa_samples(
    json_path: str = r"datasets\strategyqa\strategyqa_train_filtered.json",
    limit: int = 1,
) -> list[tuple[str, str, str]]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    samples: list[tuple[str, str, str]] = []

    for i, item in enumerate(data[:limit]):
        question = str(item["question"]).strip()
        raw_answer = item["answer"]

        if isinstance(raw_answer, bool):
            gold_answer = "true" if raw_answer else "false"
        else:
            answer_text = str(raw_answer).strip().lower()
            if answer_text in {"true", "false"}:
                gold_answer = answer_text
            else:
                raise ValueError(
                    f"Unsupported StrategyQA answer format at index {i}: {raw_answer}"
                )

        sample_id = f"strategyqa_{i+1:04d}"
        samples.append((sample_id, question, gold_answer))

    return samples

def load_samples(dataset_name: str, limit: int) -> list[tuple[str, str, str]]:
    if dataset_name == "gsm8k":
        return load_gsm8k_samples(limit=limit)
    if dataset_name == "strategyqa":
        return load_strategyqa_samples(limit=limit)
    raise ValueError(f"Unsupported dataset: {dataset_name}")

print("StateStore loaded from:", inspect.getfile(StateStore))
print("Has get_action_history:", hasattr(StateStore, "get_action_history"))



def run_normal_mode(
    llm_client: QianfanClient,
    question: str,
    gold_answer: str,
    sample_id: str,
    dataset_name: str,
):
    state_store = StateStore()
    usage_logger = UsageLogger()

    agent_runner = AgentRunner(
        llm_client=llm_client,
        usage_logger=usage_logger,
        sample_id=sample_id,
        dataset_name=dataset_name,
    )
    history_manager = HistoryManager()
    recorder = Recorder(
        llm_client=llm_client,
        usage_logger=usage_logger,
        sample_id=sample_id,
    )
    evaluator = Evaluator(
        llm_client=llm_client,
        usage_logger=usage_logger,
        sample_id=sample_id,
    )

    normal_round_executor = NormalRoundExecutor(
        config=NormalRoundExecutorConfig(
            question=question,
            agent_ids=AGENT_IDS,
            max_round=MAX_ROUND,
            sample_id=sample_id,
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
    print("Dataset:", dataset_name)
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
                usage_logger=usage_logger,
                sample_id=sample_id,
                dataset_name=dataset_name,
            )
        else:
            print("Rollback detected, but no valid anchor is available. Skip repair mode.")

    round_1_answers = get_round_1_answers(state_store)
    final_answers = get_final_answers(state_store)

    single_agent_baseline_answer = round_1_answers[0] if round_1_answers else ""
    majority_voting_baseline_answer = majority_vote(round_1_answers,dataset_name=dataset_name)
    scrd_final_answer = majority_vote(final_answers,dataset_name=dataset_name)
    usage_summary = build_usage_summary(usage_logger)
    result = {
        "sample_id": sample_id,
        "dataset_name": dataset_name,
        "question": question,
        "gold_answer": gold_answer,
        "round_1_answers": round_1_answers,
        "single_agent_baseline_answer": single_agent_baseline_answer,
        "majority_voting_baseline_answer": majority_voting_baseline_answer,
        "scrd_final_answer": scrd_final_answer,
        "single_agent_correct": is_correct(single_agent_baseline_answer, gold_answer, dataset_name=dataset_name),
        "majority_voting_correct": is_correct(majority_voting_baseline_answer, gold_answer, dataset_name=dataset_name),
        "scrd_correct": is_correct(scrd_final_answer, gold_answer, dataset_name=dataset_name),
        "effective_rounds_used": get_effective_rounds_used(state_store),
        "actual_rounds_executed": get_actual_rounds_executed(state_store),
        "stop_reason": get_stop_reason(rollback_context, early_stopped),

        # token summary
        "single_agent_total_tokens": usage_summary["single_agent_total_tokens"],
        "majority_vote_total_tokens": usage_summary["majority_vote_total_tokens"],
        "scrd_total_tokens": usage_summary["scrd_total_tokens"],
        "scrd_prompt_tokens": usage_summary["scrd_prompt_tokens"],
        "scrd_completion_tokens": usage_summary["scrd_completion_tokens"],

        # component totals
        "agent_total_tokens": usage_summary["agent_total_tokens"],
        "recorder_total_tokens": usage_summary["recorder_total_tokens"],
        "evaluator_total_tokens": usage_summary["evaluator_total_tokens"],
        "repair_brief_total_tokens": usage_summary["repair_brief_total_tokens"],
        "repair_evaluator_total_tokens": usage_summary["repair_evaluator_total_tokens"],
        "repair_agent_total_tokens": usage_summary["repair_agent_total_tokens"],
    }

    trace = build_trace_bundle(state_store, usage_logger)
    return result, trace


def run_repair_mode(
    llm_client: QianfanClient,
    question: str,
    rollback_context: dict,
    state_store: StateStore,
    history_manager: HistoryManager,
    usage_logger: UsageLogger,
    sample_id: str,
    dataset_name: str,
) -> None:
    
    repair_action_mapper = RepairActionMapper()
    recorder = Recorder(
        llm_client=llm_client,
        usage_logger=usage_logger,
        sample_id=sample_id,
    )
    repair_brief_generator = RepairBriefGenerator(
        llm_client=llm_client,
        usage_logger=usage_logger,
        sample_id=sample_id,
    )
    repair_evaluator = RepairEvaluator(
        llm_client=llm_client,
        usage_logger=usage_logger,
        sample_id=sample_id,
    )
    repair_agent_runner = RepairAgentRunner(
        llm_client=llm_client,
        usage_logger=usage_logger,
        sample_id=sample_id,
        dataset_name=dataset_name,
    )

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
            sample_id=sample_id,
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
    DATASET_NAME = "gsm8k"
    OUTPUT_DIR = f"outputs/{DATASET_NAME}"
    writer = ResultWriter(output_dir=OUTPUT_DIR)

    samples = load_samples(DATASET_NAME, limit=300)
    completed_sample_ids = writer.load_completed_sample_ids()
    if completed_sample_ids:
        print(f"Resume mode: found {len(completed_sample_ids)} completed samples in"
              f"{OUTPUT_DIR}/results.jsonl")

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
                dataset_name=DATASET_NAME,
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
                    "dataset_name": DATASET_NAME,
                    "question": question,
                    "gold_answer": gold_answer,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )
            print(f"Failed sample {sample_id}: {exc}. Logged to "
                  f"{OUTPUT_DIR}/errors.jsonl. Continuing...")
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
