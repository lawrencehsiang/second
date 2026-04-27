from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import traceback
from pathlib import Path
from typing import Any


DEFAULT_DATASETS = ["math", "gsm8k", "multiarith"]
ALL_SEVEN_DATASETS = [
    "addsub",
    "asdiv",
    "gsm8k",
    "math",
    "multiarith",
    "singleeq",
    "svamp",
]


def ensure_repo_imports(repo_root: Path) -> None:
    repo_root = repo_root.resolve()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        raise FileNotFoundError(f"Missing JSONL: {path}")

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def sample_index(sample_id: str) -> int | None:
    m = re.search(r"_(\d+)$", str(sample_id))
    if not m:
        return None
    return int(m.group(1))


def get_sample_by_id_or_index(
    *,
    sample_id: str,
    sample_by_id: dict[str, tuple[str, str, str]],
    sample_by_index: dict[int, tuple[str, str, str]],
) -> tuple[str, str, str] | None:
    if sample_id in sample_by_id:
        return sample_by_id[sample_id]

    idx = sample_index(sample_id)
    if idx is not None and idx in sample_by_index:
        return sample_by_index[idx]

    return None


def build_previous_answer_map(previous_state_record: Any, agent_ids: list[str]) -> dict[str, str]:
    answers = previous_state_record.current_answers
    if len(answers) != len(agent_ids):
        raise ValueError(
            "The number of previous answers does not match the number of configured agents."
        )
    return {agent_id: answer for agent_id, answer in zip(agent_ids, answers)}


def get_latest_answers(state_store: Any) -> list[str]:
    latest = state_store.get_latest_state_record()
    if latest is None:
        return []
    return [
        str(a).strip()
        for a in (latest.current_answers or [])
        if a is not None and str(a).strip()
    ]


class FixedFiveRoundSCRDRunner:
    """
    w/o Evaluator & Action Mapper ablation.

    This runner keeps:
    - AgentRunner
    - Recorder
    - HistoryManager / structured history filtering
    - StateStore
    - last-effective-round majority finalizer

    This runner removes:
    - TransitionExtractor
    - Evaluator
    - ActionMapper
    - RollbackController
    - Early stop
    - Rollback / repair

    It always executes exactly max_round normal rounds.
    """

    def __init__(
        self,
        *,
        question: str,
        sample_id: str,
        dataset_name: str,
        agent_ids: list[str],
        max_round: int,
        agent_runner: Any,
        recorder: Any,
        history_manager: Any,
        state_store: Any,
    ) -> None:
        self.question = question
        self.sample_id = sample_id
        self.dataset_name = dataset_name
        self.agent_ids = agent_ids
        self.max_round = max_round
        self.agent_runner = agent_runner
        self.recorder = recorder
        self.history_manager = history_manager
        self.state_store = state_store

    def run(self) -> None:
        for round_id in range(1, self.max_round + 1):
            print(f"Executing fixed round {round_id}/{self.max_round}...")

            if round_id == 1:
                self._run_round_1(round_id)
            else:
                self._run_normal_round(round_id)

        self.state_store.add_event(
            {
                "type": "fixed_max_round_reached",
                "round_id": self.max_round,
                "mode": "normal",
                "ablation": "wo_evaluator_action_mapper_fixed_round",
            }
        )

    def _run_round_1(self, round_id: int) -> None:
        from src.schemas import AgentInputRound1

        agent_inputs = []
        agent_outputs = []

        self.state_store.add_event(
            {
                "type": "normal_round_executed",
                "round_id": round_id,
                "mode": "normal",
                "ablation": "wo_evaluator_action_mapper_fixed_round",
            }
        )

        for agent_id in self.agent_ids:
            agent_input = AgentInputRound1(question=self.question)
            agent_output = self.agent_runner.run_round_1(
                agent_id=agent_id,
                agent_input=agent_input,
                round_id=round_id,
                sample_id=self.sample_id,
            )
            agent_inputs.append(agent_input)
            agent_outputs.append(agent_output)

        state_record = self.recorder.build_state_record(
            round_id=round_id,
            agent_outputs=agent_outputs,
            previous_state_record=None,
            sample_id=self.sample_id,
            mode="normal",
        )

        self.state_store.set_history_units(round_id, [])
        self.state_store.add_state_record(state_record)
        self.state_store.set_round_action(round_id, "fixed_continue")

    def _run_normal_round(self, round_id: int) -> None:
        from src.pipeline.postprocess import apply_keep_or_update
        from src.schemas import AgentInputNormal

        previous_state_record = self.state_store.get_state_record(round_id - 1)
        if previous_state_record is None:
            raise ValueError(f"Missing previous StateRecord for round {round_id - 1}.")

        history_units = self.history_manager.build_history_units(
            question=self.question,
            current_round_id=round_id,
            state_store=self.state_store,
        )
        self.state_store.set_history_units(round_id, history_units)

        self.state_store.add_event(
            {
                "type": "normal_round_executed",
                "round_id": round_id,
                "mode": "normal",
                "ablation": "wo_evaluator_action_mapper_fixed_round",
            }
        )

        previous_answer_map = build_previous_answer_map(
            previous_state_record,
            self.agent_ids,
        )

        agent_inputs = []
        agent_outputs = []

        for agent_id in self.agent_ids:
            own_previous_answer = previous_answer_map.get(agent_id)
            if own_previous_answer is None:
                raise ValueError(f"Missing previous answer for agent {agent_id}.")

            agent_input = AgentInputNormal(
                question=self.question,
                own_previous_answer=own_previous_answer,
                history_units=history_units,
            )
            agent_output = self.agent_runner.run_normal_round(
                agent_id=agent_id,
                agent_input=agent_input,
                round_id=round_id,
                sample_id=self.sample_id,
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
            sample_id=self.sample_id,
            mode="normal",
        )

        self.state_store.add_state_record(state_record)
        self.state_store.set_round_action(round_id, "fixed_continue")


def run_one_sample(
    *,
    llm_client: Any,
    question: str,
    gold_answer: str,
    sample_id: str,
    dataset_name: str,
    max_round: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    from src.components.agent_runner import AgentRunner
    from src.components.history_manager import HistoryManager
    from src.components.recorder import Recorder
    from src.components.state_store import StateStore
    from src.components.usage_logger import UsageLogger
    from src.main import AGENT_IDS, AGENT_ROLES
    from src.utils.result_utils import (
        build_trace_bundle,
        build_usage_summary,
        get_actual_rounds_executed,
        get_effective_rounds_used,
        get_round_1_answers,
        is_correct,
        majority_vote,
    )

    state_store = StateStore()
    usage_logger = UsageLogger()

    agent_runner = AgentRunner(
        llm_client=llm_client,
        usage_logger=usage_logger,
        sample_id=sample_id,
        dataset_name=dataset_name,
        role_by_agent_id=AGENT_ROLES,
    )
    recorder = Recorder(
        llm_client=llm_client,
        usage_logger=usage_logger,
        sample_id=sample_id,
    )
    history_manager = HistoryManager()

    runner = FixedFiveRoundSCRDRunner(
        question=question,
        sample_id=sample_id,
        dataset_name=dataset_name,
        agent_ids=AGENT_IDS,
        max_round=max_round,
        agent_runner=agent_runner,
        recorder=recorder,
        history_manager=history_manager,
        state_store=state_store,
    )

    print(f"\n===== w/o evaluator/action mapper fixed-{max_round}: {sample_id} =====")
    print("Dataset:", dataset_name)
    print("Gold answer:", gold_answer)

    runner.run()

    round_1_answers = get_round_1_answers(state_store)
    single_agent_baseline_answer = round_1_answers[0] if round_1_answers else ""
    majority_voting_baseline_answer = majority_vote(
        round_1_answers,
        dataset_name=dataset_name,
    )

    final_answers = get_latest_answers(state_store)
    final_answer = majority_vote(final_answers, dataset_name=dataset_name)

    usage_summary = build_usage_summary(usage_logger)

    result = {
        "sample_id": sample_id,
        "dataset_name": dataset_name,
        "question": question,
        "gold_answer": gold_answer,
        "ablation_name": "wo_evaluator_action_mapper_fixed_round",
        "fixed_rounds": max_round,
        "evaluator_disabled": True,
        "action_mapper_disabled": True,
        "rollback_disabled": True,
        "repair_disabled": True,
        "early_stop_disabled": True,
        "finalizer": "last_effective_round_majority_vote",
        "agent_roles": AGENT_ROLES,
        "round_1_answers": round_1_answers,
        "single_agent_baseline_answer": single_agent_baseline_answer,
        "majority_voting_baseline_answer": majority_voting_baseline_answer,
        "last_round_answers": final_answers,
        "scrd_final_answer": final_answer,
        "wo_evaluator_action_mapper_final_answer": final_answer,
        "single_agent_correct": is_correct(
            single_agent_baseline_answer,
            gold_answer,
            dataset_name=dataset_name,
        ),
        "majority_voting_correct": is_correct(
            majority_voting_baseline_answer,
            gold_answer,
            dataset_name=dataset_name,
        ),
        "scrd_correct": is_correct(final_answer, gold_answer, dataset_name=dataset_name),
        "wo_evaluator_action_mapper_correct": is_correct(
            final_answer,
            gold_answer,
            dataset_name=dataset_name,
        ),
        "effective_rounds_used": get_effective_rounds_used(state_store),
        "actual_rounds_executed": get_actual_rounds_executed(state_store),
        "stop_reason": "fixed_max_round",
        "single_agent_total_tokens": usage_summary["single_agent_total_tokens"],
        "majority_vote_total_tokens": usage_summary["majority_vote_total_tokens"],
        "scrd_total_tokens": usage_summary["scrd_total_tokens"],
        "wo_evaluator_action_mapper_total_tokens": usage_summary["scrd_total_tokens"],
        "scrd_prompt_tokens": usage_summary["scrd_prompt_tokens"],
        "scrd_completion_tokens": usage_summary["scrd_completion_tokens"],
        "agent_total_tokens": usage_summary["agent_total_tokens"],
        "recorder_total_tokens": usage_summary["recorder_total_tokens"],
        "evaluator_total_tokens": usage_summary["evaluator_total_tokens"],
        "repair_brief_total_tokens": usage_summary["repair_brief_total_tokens"],
        "repair_evaluator_total_tokens": usage_summary["repair_evaluator_total_tokens"],
        "repair_agent_total_tokens": usage_summary["repair_agent_total_tokens"],
    }

    trace = build_trace_bundle(state_store, usage_logger)
    trace["ablation_name"] = "wo_evaluator_action_mapper_fixed_round"
    trace["fixed_rounds"] = max_round
    trace["evaluator_disabled"] = True
    trace["action_mapper_disabled"] = True
    trace["rollback_disabled"] = True
    trace["repair_disabled"] = True
    trace["early_stop_disabled"] = True
    trace["finalizer"] = "last_effective_round_majority_vote"

    return result, trace


def summarize_dataset(dataset: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)

    full_correct = sum(bool(r.get("full_scrd_correct")) for r in rows)
    ablation_correct = sum(bool(r.get("wo_evaluator_action_mapper_correct")) for r in rows)

    full_wins = sum(
        bool(r.get("full_scrd_correct"))
        and not bool(r.get("wo_evaluator_action_mapper_correct"))
        for r in rows
    )
    ablation_wins = sum(
        (not bool(r.get("full_scrd_correct")))
        and bool(r.get("wo_evaluator_action_mapper_correct"))
        for r in rows
    )
    both_correct = sum(
        bool(r.get("full_scrd_correct"))
        and bool(r.get("wo_evaluator_action_mapper_correct"))
        for r in rows
    )
    both_wrong = sum(
        (not bool(r.get("full_scrd_correct")))
        and (not bool(r.get("wo_evaluator_action_mapper_correct")))
        for r in rows
    )

    full_tokens = [
        float(r.get("full_scrd_total_tokens") or 0)
        for r in rows
        if r.get("full_scrd_total_tokens") is not None
    ]
    ablation_tokens = [
        float(r.get("wo_evaluator_action_mapper_total_tokens") or 0)
        for r in rows
        if r.get("wo_evaluator_action_mapper_total_tokens") is not None
    ]

    full_rounds = [
        float(r.get("full_actual_rounds_executed") or 0)
        for r in rows
        if r.get("full_actual_rounds_executed") is not None
    ]
    ablation_rounds = [
        float(r.get("actual_rounds_executed") or 0)
        for r in rows
        if r.get("actual_rounds_executed") is not None
    ]

    return {
        "dataset": dataset,
        "n": n,
        "full_scrd_correct": full_correct,
        "full_scrd_acc": full_correct / n if n else None,
        "wo_evaluator_action_mapper_correct": ablation_correct,
        "wo_evaluator_action_mapper_acc": ablation_correct / n if n else None,
        "delta_ablation_minus_full_pp": (
            100.0 * ((ablation_correct / n) - (full_correct / n)) if n else None
        ),
        "both_correct": both_correct,
        "both_wrong": both_wrong,
        "full_wins": full_wins,
        "ablation_wins": ablation_wins,
        "net_full_minus_ablation": full_wins - ablation_wins,
        "avg_full_scrd_tokens": (
            sum(full_tokens) / len(full_tokens) if full_tokens else None
        ),
        "avg_ablation_tokens": (
            sum(ablation_tokens) / len(ablation_tokens) if ablation_tokens else None
        ),
        "avg_token_delta_ablation_minus_full": (
            (sum(ablation_tokens) / len(ablation_tokens))
            - (sum(full_tokens) / len(full_tokens))
            if ablation_tokens and full_tokens
            else None
        ),
        "avg_full_actual_rounds": (
            sum(full_rounds) / len(full_rounds) if full_rounds else None
        ),
        "avg_ablation_rounds": (
            sum(ablation_rounds) / len(ablation_rounds) if ablation_rounds else None
        ),
    }


def add_overall(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    n = sum(int(s["n"]) for s in summaries)

    def sum_key(key: str) -> int:
        return sum(int(s.get(key) or 0) for s in summaries)

    def weighted_avg(key: str) -> float | None:
        if not n:
            return None
        values = [
            (float(s.get(key) or 0), int(s["n"]))
            for s in summaries
            if s.get(key) is not None
        ]
        denom = sum(w for _, w in values)
        if not denom:
            return None
        return sum(v * w for v, w in values) / denom

    full_correct = sum_key("full_scrd_correct")
    ablation_correct = sum_key("wo_evaluator_action_mapper_correct")

    return {
        "dataset": "OVERALL",
        "n": n,
        "full_scrd_correct": full_correct,
        "full_scrd_acc": full_correct / n if n else None,
        "wo_evaluator_action_mapper_correct": ablation_correct,
        "wo_evaluator_action_mapper_acc": ablation_correct / n if n else None,
        "delta_ablation_minus_full_pp": (
            100.0 * ((ablation_correct / n) - (full_correct / n)) if n else None
        ),
        "both_correct": sum_key("both_correct"),
        "both_wrong": sum_key("both_wrong"),
        "full_wins": sum_key("full_wins"),
        "ablation_wins": sum_key("ablation_wins"),
        "net_full_minus_ablation": sum_key("net_full_minus_ablation"),
        "avg_full_scrd_tokens": weighted_avg("avg_full_scrd_tokens"),
        "avg_ablation_tokens": weighted_avg("avg_ablation_tokens"),
        "avg_token_delta_ablation_minus_full": weighted_avg(
            "avg_token_delta_ablation_minus_full"
        ),
        "avg_full_actual_rounds": weighted_avg("avg_full_actual_rounds"),
        "avg_ablation_rounds": weighted_avg("avg_ablation_rounds"),
    }


def fmt_pct(x: Any) -> str:
    if x is None:
        return "NA"
    return f"{100.0 * float(x):.2f}%"


def write_report(path: Path, summaries: list[dict[str, Any]], max_round: int) -> None:
    lines: list[str] = []
    lines.append("# Ablation: w/o Evaluator & Action Mapper")
    lines.append("")
    lines.append(
        f"Definition: run exactly {max_round} normal SCRD rounds with evaluator, "
        "action mapper, early stop, rollback, and repair disabled. Structured "
        "history filtering and recorder are preserved. Final answer is selected "
        "by last-effective-round majority vote."
    )
    lines.append("")
    lines.append("| Dataset | N | Full Acc | Fixed-5 Acc | Δ Acc | Full wins | Ablation wins | Net Full-Ablation | Full tok | Fixed-5 tok |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

    for s in summaries:
        lines.append(
            "| {dataset} | {n} | {full_acc} | {abl_acc} | {delta:.2f} pp | {full_wins} | {abl_wins} | {net} | {full_tok:.1f} | {abl_tok:.1f} |".format(
                dataset=s["dataset"],
                n=s["n"],
                full_acc=fmt_pct(s["full_scrd_acc"]),
                abl_acc=fmt_pct(s["wo_evaluator_action_mapper_acc"]),
                delta=float(s["delta_ablation_minus_full_pp"] or 0.0),
                full_wins=s["full_wins"],
                abl_wins=s["ablation_wins"],
                net=s["net_full_minus_ablation"],
                full_tok=float(s.get("avg_full_scrd_tokens") or 0.0),
                abl_tok=float(s.get("avg_ablation_tokens") or 0.0),
            )
        )

    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- This ablation must rerun all samples because early-stop samples in Full SCRD now continue to fixed max rounds.")
    lines.append("- Token cost is expected to increase because evaluator/action control no longer stops easy cases early.")
    lines.append("- If accuracy drops or token cost increases substantially, it supports the value of dynamic control.")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Ablation runner: w/o Evaluator & Action Mapper. "
            "Runs fixed normal rounds without early stop, rollback, repair, "
            "transition evaluator, or action mapper."
        )
    )
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--all-seven", action="store_true")
    parser.add_argument("--limit", type=int, default=80)
    parser.add_argument("--max-round", type=int, default=5)
    parser.add_argument(
        "--full-root",
        type=Path,
        default=Path("outputs_with_last_round_majority_vote"),
        help="Reference Full SCRD outputs, preferably after last-round MV finalizer rewrite.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/ablation/wo_evaluator_action_mapper_fixed5"),
    )
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.all_seven:
        args.datasets = ALL_SEVEN_DATASETS

    ensure_repo_imports(args.repo_root)

    os.environ["HTTP_PROXY"] = ""
    os.environ["HTTPS_PROXY"] = ""
    os.environ["NO_PROXY"] = "dashscope.aliyuncs.com,localhost,127.0.0.1"

    from dotenv import load_dotenv
    from src.main import build_llm_client, load_samples

    load_dotenv()
    llm_client = build_llm_client()

    args.output_root.mkdir(parents=True, exist_ok=True)

    all_summaries: list[dict[str, Any]] = []
    all_output_rows: list[dict[str, Any]] = []

    for dataset in args.datasets:
        print(f"\n######## Dataset: {dataset} ########")

        full_path = args.full_root / dataset / "results.jsonl"
        full_rows = load_jsonl(full_path)
        full_by_id = {str(r["sample_id"]): r for r in full_rows}

        samples = load_samples(dataset, limit=args.limit)
        sample_by_id = {str(sid): (str(sid), q, a) for sid, q, a in samples}
        sample_by_index = {
            idx: (str(sid), q, a)
            for idx, (sid, q, a) in enumerate(samples, start=1)
        }

        out_dir = args.output_root / dataset
        result_path = out_dir / "results.jsonl"
        error_path = out_dir / "errors.jsonl"
        trace_dir = out_dir / "traces"

        if args.overwrite:
            if result_path.exists():
                result_path.unlink()
            if error_path.exists():
                error_path.unlink()

        existing_ids: set[str] = set()
        if result_path.exists() and not args.overwrite:
            existing_ids = {str(r["sample_id"]) for r in load_jsonl(result_path)}

        for full_row in full_rows:
            sample_id = str(full_row["sample_id"])
            if sample_id in existing_ids:
                continue

            sample_tuple = get_sample_by_id_or_index(
                sample_id=sample_id,
                sample_by_id=sample_by_id,
                sample_by_index=sample_by_index,
            )

            if sample_tuple is None:
                append_jsonl(
                    error_path,
                    {
                        "sample_id": sample_id,
                        "dataset": dataset,
                        "error": "Sample not found by sample_id or numeric index.",
                    },
                )
                continue

            _, question, gold_answer = sample_tuple

            try:
                result, trace = run_one_sample(
                    llm_client=llm_client,
                    question=question,
                    gold_answer=gold_answer,
                    sample_id=sample_id,
                    dataset_name=dataset,
                    max_round=args.max_round,
                )

                result["full_scrd_final_answer"] = full_row.get("scrd_final_answer")
                result["full_scrd_correct"] = full_row.get("scrd_correct")
                result["full_scrd_total_tokens"] = full_row.get("scrd_total_tokens")
                result["full_effective_rounds_used"] = full_row.get("effective_rounds_used")
                result["full_actual_rounds_executed"] = full_row.get("actual_rounds_executed")
                result["full_stop_reason"] = full_row.get("stop_reason")
                result["answer_changed_vs_full_scrd"] = (
                    str(result.get("wo_evaluator_action_mapper_final_answer"))
                    != str(full_row.get("scrd_final_answer"))
                )

                append_jsonl(result_path, result)
                write_json(trace_dir / f"{sample_id}_trace.json", trace)

                print(
                    "Saved {} | full_correct={} | fixed_correct={} | fixed_answer={}".format(
                        sample_id,
                        full_row.get("scrd_correct"),
                        result.get("wo_evaluator_action_mapper_correct"),
                        result.get("wo_evaluator_action_mapper_final_answer"),
                    )
                )

            except KeyboardInterrupt:
                print("\nInterrupted by user. Partial progress saved.")
                raise
            except Exception as exc:
                append_jsonl(
                    error_path,
                    {
                        "sample_id": sample_id,
                        "dataset": dataset,
                        "question": question,
                        "gold_answer": gold_answer,
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    },
                )
                print(f"Failed {sample_id}: {exc}. Continuing...")

        dataset_rows = load_jsonl(result_path)
        summary = summarize_dataset(dataset, dataset_rows)
        write_json(out_dir / "summary.json", summary)

        all_summaries.append(summary)
        all_output_rows.extend(dataset_rows)

        print(
            "SUMMARY {}: full={:.2f}% | fixed={:.2f}% | delta={:+.2f} pp | tokens fixed-full={:+.1f}".format(
                dataset,
                100.0 * float(summary["full_scrd_acc"] or 0.0),
                100.0 * float(summary["wo_evaluator_action_mapper_acc"] or 0.0),
                float(summary["delta_ablation_minus_full_pp"] or 0.0),
                float(summary["avg_token_delta_ablation_minus_full"] or 0.0),
            )
        )

    overall = add_overall(all_summaries)
    summaries_with_overall = all_summaries + [overall]

    write_csv(args.output_root / "summary.csv", summaries_with_overall)
    write_json(args.output_root / "summary.json", summaries_with_overall)
    write_jsonl(args.output_root / "sample_level_results.jsonl", all_output_rows)
    write_report(args.output_root / "report.md", summaries_with_overall, args.max_round)

    print("")
    print(f"Saved outputs to: {args.output_root}")
    print(
        "OVERALL full={:.2f}% | fixed={:.2f}% | delta={:+.2f} pp | tokens fixed-full={:+.1f}".format(
            100.0 * float(overall["full_scrd_acc"] or 0.0),
            100.0 * float(overall["wo_evaluator_action_mapper_acc"] or 0.0),
            float(overall["delta_ablation_minus_full_pp"] or 0.0),
            float(overall["avg_token_delta_ablation_minus_full"] or 0.0),
        )
    )


if __name__ == "__main__":
    main()
