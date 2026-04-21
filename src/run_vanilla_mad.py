# src/run_vanilla_mad.py  python -m src.run_vanilla_mad --dataset gsm8k --limit 2 --max-round 7
from __future__ import annotations

import argparse
import json
import os
import re
import traceback
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from src.components.agent_runner import AgentRunner
from src.components.qianfan_client import QianfanClient
from src.components.usage_logger import UsageLogger
from src.pipeline.vanilla_mad_runner import VanillaMADRunner, VanillaMADRunnerConfig
from src.utils.result_writer import ResultWriter

AGENT_IDS = ["A", "B", "C"]
DEFAULT_MAX_ROUND = 7

# 强制禁用代理，直连国内网络（沿用你当前 main.py 的写法）
os.environ["HTTP_PROXY"] = ""
os.environ["HTTPS_PROXY"] = ""
os.environ["NO_PROXY"] = "qianfan.baidubce.com,localhost,127.0.0.1"

load_dotenv()


def build_llm_client(model_name: str = "qwen2.5-7b-instruct") -> QianfanClient:
    api_key = os.getenv("QIANFAN_API_KEY")
    if not api_key:
        raise ValueError("Missing QIANFAN_API_KEY. Please set it in your .env file.")
    return QianfanClient(
        api_key=api_key,
        model=model_name,
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


def build_output_dir(dataset_name: str, max_round: int) -> str:
    return f"outputs/{dataset_name}_vanilla_mad{max_round}"


def run_vanilla_mode(
    *,
    llm_client: QianfanClient,
    question: str,
    gold_answer: str,
    sample_id: str,
    dataset_name: str,
    max_round: int = DEFAULT_MAX_ROUND,
) -> tuple[dict, dict | list]:
    """
    单条样本的 vanilla MAD 入口。
    真正的 7 轮执行逻辑放在 VanillaMADRunner 里。
    """
    usage_logger = UsageLogger()
    agent_runner = AgentRunner(
        llm_client=llm_client,
        usage_logger=usage_logger,
        sample_id=sample_id,
        dataset_name=dataset_name,
    )

    runner = VanillaMADRunner(
        config=VanillaMADRunnerConfig(
            question=question,
            gold_answer=gold_answer,
            dataset_name=dataset_name,
            sample_id=sample_id,
            agent_ids=AGENT_IDS,
            max_round=max_round,
        ),
        agent_runner=agent_runner,
        usage_logger=usage_logger,
    )

    return runner.run()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Vanilla MAD baseline.")

    parser.add_argument(
        "--dataset",
        type=str,
        choices=["gsm8k", "strategyqa"],
        required=True,
        help="Dataset name.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=2,
        help="Number of samples to run.",
    )
    parser.add_argument(
        "--max-round",
        type=int,
        default=DEFAULT_MAX_ROUND,
        help="Number of vanilla MAD rounds. Recommended: 7.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="qwen2.5-7b-instruct",
        help="Qianfan model name.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional custom output directory.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dataset_name: str = args.dataset
    limit: int = args.limit
    max_round: int = args.max_round
    model_name: str = args.model

    output_dir = args.output_dir or build_output_dir(dataset_name, max_round)

    print("Starting Vanilla MAD baseline...")
    print("Dataset:", dataset_name)
    print("Limit:", limit)
    print("Max round:", max_round)
    print("Model:", model_name)
    print("Output dir:", output_dir)

    llm_client = build_llm_client(model_name=model_name)
    writer = ResultWriter(output_dir=output_dir)

    samples = load_samples(dataset_name, limit=limit)
    completed_sample_ids = writer.load_completed_sample_ids()

    if completed_sample_ids:
        print(
            f"Resume mode: found {len(completed_sample_ids)} completed samples in "
            f"{output_dir}/results.jsonl"
        )

    total = 0
    round1_majority_correct_count = 0
    round3_majority_correct_count = 0
    round5_majority_correct_count = 0
    round7_majority_correct_count = 0
    skipped_count = 0
    failed_count = 0

    for sample_id, question, gold_answer in samples:
        if sample_id in completed_sample_ids:
            skipped_count += 1
            print(f"Skipping completed sample: {sample_id}")
            continue

        try:
            result, trace = run_vanilla_mode(
                llm_client=llm_client,
                question=question,
                gold_answer=gold_answer,
                sample_id=sample_id,
                dataset_name=dataset_name,
                max_round=max_round,
            )

            writer.append_result(result)
            writer.write_trace(sample_id, trace)

            total += 1
            round1_majority_correct_count += int(result.get("round1_majority_correct", False))
            round3_majority_correct_count += int(result.get("round3_majority_correct", False))
            round5_majority_correct_count += int(result.get("round5_majority_correct", False))
            round7_majority_correct_count += int(result.get("round7_majority_correct", False))

            print("Result saved:", result)

        except KeyboardInterrupt:
            print("\nInterrupted by user. Progress has been saved. You can rerun to resume.")
            break

        except Exception as exc:
            failed_count += 1
            writer.append_error(
                {
                    "sample_id": sample_id,
                    "dataset_name": dataset_name,
                    "question": question,
                    "gold_answer": gold_answer,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )
            print(
                f"Failed sample {sample_id}: {exc}. "
                f"Logged to {output_dir}/errors.jsonl. Continuing..."
            )
            continue

    print("\n===== Final Summary =====")
    print(f"Processed successfully in this run: {total}")
    print(f"Skipped (already completed): {skipped_count}")
    print(f"Failed in this run: {failed_count}")

    if total > 0:
        print(
            f"Round 1 majority accuracy: "
            f"{round1_majority_correct_count}/{total} = "
            f"{round1_majority_correct_count / total:.4f}"
        )
        print(
            f"Round 3 majority accuracy: "
            f"{round3_majority_correct_count}/{total} = "
            f"{round3_majority_correct_count / total:.4f}"
        )
        print(
            f"Round 5 majority accuracy: "
            f"{round5_majority_correct_count}/{total} = "
            f"{round5_majority_correct_count / total:.4f}"
        )
        print(
            f"Round 7 majority accuracy: "
            f"{round7_majority_correct_count}/{total} = "
            f"{round7_majority_correct_count / total:.4f}"
        )
    else:
        print("No new successful samples were processed in this run.")


if __name__ == "__main__":
    main()