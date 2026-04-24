# src/run_vanilla_mad.py
# Example:
# python -m src.run_vanilla_mad --datasets gsm8k svamp multiarith --limit 80 --max-round 7

from __future__ import annotations

import argparse
import json
import os
import re
import string
import traceback

import pandas as pd
from dotenv import load_dotenv

from src.components.agent_runner import AgentRunner
from src.components.qianfan_client import QianfanClient
from src.components.usage_logger import UsageLogger
from src.pipeline.vanilla_mad_runner import VanillaMADRunner, VanillaMADRunnerConfig
from src.utils.result_writer import ResultWriter

AGENT_IDS = ["A", "B", "C"]
DEFAULT_MAX_ROUND = 7
OPTION_LABELS = list(string.ascii_uppercase)

SUPPORTED_DATASETS = [
    "gsm8k",
    "strategyqa",
    "aime2025",
    "aime2026",
    "mmlu",
    "mmlu_pro",
    "svamp",
    "multiarith",
    "addsub",
    "asdiv",
    "math",
    "singleeq",
]

# 强制禁用代理，直连国内网络
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


# =========================
# Dataset loaders
# =========================
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


def load_qa_jsonl_samples(
    jsonl_path: str,
    dataset_name: str,
    limit: int = 1,
    numeric_answer: bool = False,
) -> list[tuple[str, str, str]]:
    samples: list[tuple[str, str, str]] = []

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= limit:
                break

            item = json.loads(line)
            question = str(item["question"]).strip()
            raw_answer = str(item["answer"]).strip()

            if numeric_answer:
                matches = re.findall(r"-?\d+(?:\.\d+)?", raw_answer.replace(",", ""))
                gold_answer = matches[-1] if matches else raw_answer
            else:
                gold_answer = raw_answer

            sample_id = f"{dataset_name}_{i+1:04d}"
            samples.append((sample_id, question, gold_answer))

    return samples


def load_multiple_choice_samples(
    jsonl_path: str,
    dataset_name: str,
    limit: int = 1,
) -> list[tuple[str, str, str]]:
    samples: list[tuple[str, str, str]] = []

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= limit:
                break

            item = json.loads(line)
            question = str(item["question"]).strip()
            choices = item["choices"]
            answer_idx = int(item["answer"])

            if not isinstance(choices, list) or len(choices) == 0:
                raise ValueError(f"Invalid choices at line {i+1} in {jsonl_path}")

            if len(choices) > len(OPTION_LABELS):
                raise ValueError(
                    f"Too many choices ({len(choices)}) at line {i+1}; "
                    f"max supported is {len(OPTION_LABELS)}"
                )

            if not (0 <= answer_idx < len(choices)):
                raise ValueError(
                    f"Invalid answer index {answer_idx} at line {i+1} "
                    f"for {len(choices)} choices"
                )

            choice_lines = [
                f"{OPTION_LABELS[j]}. {str(choice).strip()}"
                for j, choice in enumerate(choices)
            ]
            question_with_choices = f"{question}\n\nOptions:\n" + "\n".join(choice_lines)
            gold_answer = OPTION_LABELS[answer_idx]

            sample_id = f"{dataset_name}_{i+1:04d}"
            samples.append((sample_id, question_with_choices, gold_answer))

    return samples


def load_samples(dataset_name: str, limit: int) -> list[tuple[str, str, str]]:
    if dataset_name == "gsm8k":
        return load_gsm8k_samples(limit=limit)

    if dataset_name == "strategyqa":
        return load_strategyqa_samples(limit=limit)

    if dataset_name == "aime2025":
        return load_qa_jsonl_samples(
            jsonl_path=r"datasets\AIME2025\aime2025.jsonl",
            dataset_name="aime2025",
            limit=limit,
            numeric_answer=True,
        )

    if dataset_name == "aime2026":
        return load_qa_jsonl_samples(
            jsonl_path=r"datasets\AIME2026\aime2026.jsonl",
            dataset_name="aime2026",
            limit=limit,
            numeric_answer=True,
        )

    if dataset_name == "mmlu":
        return load_multiple_choice_samples(
            jsonl_path=r"datasets\mmlu\mmlu.jsonl",
            dataset_name="mmlu",
            limit=limit,
        )

    if dataset_name == "mmlu_pro":
        return load_multiple_choice_samples(
            jsonl_path=r"datasets\mmlupro\mmlu_pro.jsonl",
            dataset_name="mmlu_pro",
            limit=limit,
        )

    if dataset_name == "svamp":
        return load_qa_jsonl_samples(
            jsonl_path=r"datasets\SVAMP\svamp.jsonl",
            dataset_name="svamp",
            limit=limit,
            numeric_answer=True,
        )

    if dataset_name == "multiarith":
        return load_qa_jsonl_samples(
            jsonl_path=r"datasets\MultiArith\multiarith.jsonl",
            dataset_name="multiarith",
            limit=limit,
            numeric_answer=True,
        )

    if dataset_name == "addsub":
        return load_qa_jsonl_samples(
            jsonl_path=r"datasets\addsub\addsub.jsonl",
            dataset_name="addsub",
            limit=limit,
            numeric_answer=True,
        )

    if dataset_name == "asdiv":
        return load_qa_jsonl_samples(
            jsonl_path=r"datasets\asdiv\asdiv.jsonl",
            dataset_name="asdiv",
            limit=limit,
            numeric_answer=True,
        )

    if dataset_name == "math":
        return load_qa_jsonl_samples(
            jsonl_path=r"datasets\math\math.jsonl",
            dataset_name="math",
            limit=limit,
            numeric_answer=True,
        )

    if dataset_name == "singleeq":
        return load_qa_jsonl_samples(
            jsonl_path=r"datasets\singleeq\singleeq.jsonl",
            dataset_name="singleeq",
            limit=limit,
            numeric_answer=True,
        )

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
    真正的多轮执行逻辑放在 VanillaMADRunner 里。
    """
    usage_logger = UsageLogger()

    agent_runner = AgentRunner(
        llm_client=llm_client,
        usage_logger=usage_logger,
        sample_id=sample_id,
        dataset_name=dataset_name,
        # 不传 role_by_agent_id：保持 vanilla MAD baseline 为同构 agent
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


def run_single_dataset(
    *,
    llm_client: QianfanClient,
    dataset_name: str,
    limit: int,
    max_round: int,
    output_dir: str,
) -> dict:
    print("\n" + "=" * 60)
    print(f"Starting Vanilla MAD baseline for dataset: {dataset_name}")
    print("Limit:", limit)
    print("Max round:", max_round)
    print("Output dir:", output_dir)

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
    interrupted = False

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
            round1_majority_correct_count += int(
                result.get("round1_majority_correct", False)
            )
            round3_majority_correct_count += int(
                result.get("round3_majority_correct", False)
            )
            round5_majority_correct_count += int(
                result.get("round5_majority_correct", False)
            )
            round7_majority_correct_count += int(
                result.get("round7_majority_correct", False)
            )

            print("Result saved:", result)

        except KeyboardInterrupt:
            interrupted = True
            print(
                "\nInterrupted by user. Progress for current dataset has been saved. "
                "You can rerun to resume."
            )
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

    print(f"\n===== Summary for {dataset_name} =====")
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

    return {
        "dataset_name": dataset_name,
        "processed": total,
        "skipped": skipped_count,
        "failed": failed_count,
        "round1_majority_correct_count": round1_majority_correct_count,
        "round3_majority_correct_count": round3_majority_correct_count,
        "round5_majority_correct_count": round5_majority_correct_count,
        "round7_majority_correct_count": round7_majority_correct_count,
        "interrupted": interrupted,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Vanilla MAD baseline on 3 datasets.")
    parser.add_argument(
        "--datasets",
        type=str,
        nargs=3,
        required=True,
        metavar=("DATASET1", "DATASET2", "DATASET3"),
        help=f"Exactly 3 dataset names. Supported: {', '.join(SUPPORTED_DATASETS)}",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=2,
        help="Number of samples to run per dataset.",
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
        "--output-root",
        type=str,
        default="outputs",
        help="Root output directory. Each dataset will have its own subdirectory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    datasets: list[str] = args.datasets
    limit: int = args.limit
    max_round: int = args.max_round
    model_name: str = args.model
    output_root: str = args.output_root

    invalid = [d for d in datasets if d not in SUPPORTED_DATASETS]
    if invalid:
        raise ValueError(
            f"Unsupported dataset(s): {invalid}. Supported: {SUPPORTED_DATASETS}"
        )

    if len(set(datasets)) != 3:
        raise ValueError(f"Please provide 3 distinct datasets, got: {datasets}")

    print("Starting Vanilla MAD baseline...")
    print("Datasets:", datasets)
    print("Limit per dataset:", limit)
    print("Max round:", max_round)
    print("Model:", model_name)
    print("Output root:", output_root)

    llm_client = build_llm_client(model_name=model_name)

    all_summaries: list[dict] = []

    for dataset_name in datasets:
        output_dir = os.path.join(output_root, f"{dataset_name}_vanilla_mad{max_round}")

        summary = run_single_dataset(
            llm_client=llm_client,
            dataset_name=dataset_name,
            limit=limit,
            max_round=max_round,
            output_dir=output_dir,
        )
        all_summaries.append(summary)

        if summary["interrupted"]:
            print("\nRun interrupted. Stopping before remaining datasets.")
            break

    print("\n" + "=" * 60)
    print("===== Overall Summary =====")
    for s in all_summaries:
        print(
            f"{s['dataset_name']}: processed={s['processed']}, "
            f"skipped={s['skipped']}, failed={s['failed']}"
        )
        if s["processed"] > 0:
            print(
                f"  R1={s['round1_majority_correct_count']}/{s['processed']} "
                f"({s['round1_majority_correct_count'] / s['processed']:.4f})"
            )
            print(
                f"  R3={s['round3_majority_correct_count']}/{s['processed']} "
                f"({s['round3_majority_correct_count'] / s['processed']:.4f})"
            )
            print(
                f"  R5={s['round5_majority_correct_count']}/{s['processed']} "
                f"({s['round5_majority_correct_count'] / s['processed']:.4f})"
            )
            print(
                f"  R7={s['round7_majority_correct_count']}/{s['processed']} "
                f"({s['round7_majority_correct_count'] / s['processed']:.4f})"
            )


if __name__ == "__main__":
    main()