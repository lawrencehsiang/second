from __future__ import annotations

from collections import Counter
import re

from src.components.state_store import StateStore


# ------------------------------------------------------------------
# Shared normalizers
# ------------------------------------------------------------------

def normalize_text(text: str) -> str:
    if text is None:
        return ""
    text = str(text).strip().lower()
    text = text.replace(",", "")
    text = text.replace("$", "")
    text = text.replace("dollars", "")
    text = text.replace("dollar", "")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_bool_answer(text: str) -> str:
    """
    Normalize model outputs for boolean QA.
    Accept a few common variants, but always map to "true" / "false".
    """
    text = normalize_text(text)

    if text in {"true", "t", "yes"}:
        return "true"
    if text in {"false", "f", "no"}:
        return "false"

    if text.startswith("true"):
        return "true"
    if text.startswith("false"):
        return "false"

    return text


def extract_last_number(text: str) -> float | None:
    """
    Extract the last numeric value from model output.
    Examples:
    - "The answer is 10" -> 10.0
    - "$10.00" -> 10.0
    - "50/60 * 12 = 10" -> 10.0
    """
    if text is None:
        return None

    text = normalize_text(text)
    numbers = re.findall(r"-?\d+(?:\.\d+)?", text)
    if not numbers:
        return None

    try:
        return float(numbers[-1])
    except ValueError:
        return None

def normalize_multiple_choice_answer(text: str) -> str:
    if text is None:
        return ""

    s = str(text).strip().upper()

    if re.fullmatch(r"[A-Z]", s):
        return s

    patterns = [
        r"\bOPTION\s*([A-Z])\b",
        r"\bANSWER\s*(?:IS|:)?\s*([A-Z])\b",
        r"\bI\s+CHOOSE\s+([A-Z])\b",
        r"^\(?([A-Z])\)?[\.:\s]*$",
        r"\b([A-Z])\b",
    ]

    for pattern in patterns:
        m = re.search(pattern, s)
        if m:
            return m.group(1)

    return s


# ------------------------------------------------------------------
# Dataset-specific correctness
# ------------------------------------------------------------------

def is_correct_gsm8k(pred: str, gold: str) -> bool:
    pred_num = extract_last_number(pred)
    gold_num = extract_last_number(gold)

    if pred_num is None or gold_num is None:
        return False

    return int(round(pred_num)) == int(round(gold_num))


def is_correct_strategyqa(pred: str, gold: str) -> bool:
    return normalize_bool_answer(pred) == normalize_bool_answer(gold)

def is_correct_aime(pred: str, gold: str) -> bool:
    return is_correct_gsm8k(pred, gold)

def is_correct_svamp(pred: str, gold: str) -> bool:
    return is_correct_gsm8k(pred, gold)

def is_correct_multiarith(pred: str, gold: str) -> bool:
    return is_correct_gsm8k(pred, gold)

def is_correct_multiple_choice(pred: str, gold: str) -> bool:
    pred_norm = normalize_multiple_choice_answer(pred)
    gold_norm = normalize_multiple_choice_answer(gold)
    return pred_norm == gold_norm


def is_correct(pred: str, gold: str, dataset_name: str) -> bool:
    if dataset_name == "gsm8k":
        return is_correct_gsm8k(pred, gold)
    if dataset_name == "strategyqa":
        return is_correct_strategyqa(pred, gold)
    if dataset_name in {"aime2025", "aime2026"}:
        return is_correct_aime(pred, gold)
    if dataset_name in {"mmlu", "mmlu_pro"}:
        return is_correct_multiple_choice(pred, gold)
    if dataset_name == "svamp":
        return is_correct_svamp(pred, gold)
    if dataset_name in {"multiarith","addsub","asdiv","math","singleeq"}:
        return is_correct_multiarith(pred,gold)
    raise ValueError(f"Unsupported dataset: {dataset_name}")


# ------------------------------------------------------------------
# Dataset-specific majority vote
# ------------------------------------------------------------------

def majority_vote_gsm8k(answers: list[str]) -> str:
    """
    Prefer numeric majority when possible.
    If an answer contains a recoverable numeric value, vote by that value.
    Otherwise fall back to normalized text voting.
    """
    if not answers:
        return ""

    vote_keys: list[str] = []
    key_to_original: dict[str, str] = {}

    for ans in answers:
        num = extract_last_number(ans)
        if num is not None:
            key = f"num:{int(round(num))}"
        else:
            key = f"text:{normalize_text(ans)}"

        vote_keys.append(key)
        if key not in key_to_original:
            key_to_original[key] = ans

    count = Counter(vote_keys)
    majority_key = count.most_common(1)[0][0]
    return key_to_original[majority_key]


def majority_vote_strategyqa(answers: list[str]) -> str:
    if not answers:
        return ""

    normalized = [normalize_bool_answer(ans) for ans in answers]
    count = Counter(normalized)
    return count.most_common(1)[0][0]

def majority_vote_aime(answers: list[str]) -> str:
    return majority_vote_gsm8k(answers)

def majority_vote_svamp(answers: list[str]) -> str:
    return majority_vote_gsm8k(answers)

def majority_vote_multiarith(answers: list[str]) -> str:
    return majority_vote_gsm8k(answers)

def majority_vote_multiple_choice(answers: list[str]) -> str:
    normalized = [
        normalize_multiple_choice_answer(a)
        for a in answers
        if normalize_multiple_choice_answer(a)
    ]

    if not normalized:
        return ""

    counter = Counter(normalized)
    return counter.most_common(1)[0][0]

def majority_vote(answers: list[str], dataset_name: str) -> str:
    if dataset_name == "gsm8k":
        return majority_vote_gsm8k(answers)
    if dataset_name == "strategyqa":
        return majority_vote_strategyqa(answers)
    if dataset_name in {"aime2025", "aime2026"}:
        return majority_vote_aime(answers)
    if dataset_name in {"mmlu", "mmlu_pro"}:
        return majority_vote_multiple_choice(answers)
    if dataset_name == "svamp":
        return majority_vote_svamp(answers)
    if dataset_name in {"multiarith","addsub","asdiv","math","singleeq"}:
        return majority_vote_multiarith(answers)
    raise ValueError(f"Unsupported dataset: {dataset_name}")


# ------------------------------------------------------------------
# Existing trace helpers
# ------------------------------------------------------------------

def get_round_1_answers(state_store: StateStore) -> list[str]:
    state = state_store.get_state_record(1)
    if state is None:
        return []
    return state.current_answers


def get_final_answers(state_store: StateStore) -> list[str]:
    state = state_store.get_latest_state_record()
    if state is None:
        return []
    return state.current_answers


def build_final_trace(state_store: StateStore) -> list[dict]:
    return [state.model_dump() for state in state_store.list_state_records()]


def build_execution_events(state_store: StateStore) -> list[dict]:
    return state_store.list_events()


def build_trace_bundle(state_store: StateStore, usage_logger) -> dict:
    usage_summary = build_usage_summary(usage_logger)
    return {
        "final_trace": build_final_trace(state_store),
        "execution_events": build_execution_events(state_store),
        "usage_records": usage_summary["usage_records"],
        "usage_summary": {
            k: v for k, v in usage_summary.items() if k != "usage_records"
        },
    }


def get_effective_rounds_used(state_store: StateStore) -> int:
    """
    Number of rounds remaining in the final effective state trajectory.
    """
    return len(state_store.list_state_records())


def get_actual_rounds_executed(state_store: StateStore) -> int:
    """
    True execution cost: count all normal + repair rounds that were actually run,
    regardless of whether some failed suffix states were later removed.
    """
    events = state_store.list_events()
    count = 0
    for event in events:
        if event.get("type") in {"normal_round_executed", "repair_round_executed"}:
            count += 1
    return count


def get_stop_reason(
    rollback_context: dict | None,
    early_stopped: bool,
) -> str:
    if early_stopped:
        return "early_stop"

    if rollback_context:
        return "rollback"

    return "max_round"


def build_usage_summary(usage_logger) -> dict:
    records = usage_logger.list_records()

    def _sum_total(component_prefix: str) -> int:
        return sum(
            r.get("total_tokens", 0)
            for r in records
            if r.get("component", "").startswith(component_prefix)
        )

    scrd_total_tokens = sum(r.get("total_tokens", 0) for r in records)
    scrd_prompt_tokens = sum(r.get("prompt_tokens", 0) for r in records)
    scrd_completion_tokens = sum(r.get("completion_tokens", 0) for r in records)

    agent_total_tokens = _sum_total("agent")
    recorder_total_tokens = _sum_total("recorder")
    evaluator_total_tokens = _sum_total("evaluator")
    repair_brief_total_tokens = _sum_total("repair_brief")
    repair_evaluator_total_tokens = _sum_total("repair_evaluator")
    repair_agent_total_tokens = _sum_total("repair_agent")

    # baseline 口径沿用你现在的设计
    single_agent_total_tokens = sum(
        r.get("total_tokens", 0)
        for r in records
        if r.get("component") == "agent_round_1" and r.get("agent_id") == "A"
    )

    majority_vote_total_tokens = sum(
        r.get("total_tokens", 0)
        for r in records
        if str(r.get("component", "")).startswith("agent_round_1")
    )

    return {
        "usage_records": records,
        "single_agent_total_tokens": single_agent_total_tokens,
        "majority_vote_total_tokens": majority_vote_total_tokens,
        "scrd_total_tokens": scrd_total_tokens,
        "scrd_prompt_tokens": scrd_prompt_tokens,
        "scrd_completion_tokens": scrd_completion_tokens,
        "agent_total_tokens": agent_total_tokens,
        "recorder_total_tokens": recorder_total_tokens,
        "evaluator_total_tokens": evaluator_total_tokens,
        "repair_brief_total_tokens": repair_brief_total_tokens,
        "repair_evaluator_total_tokens": repair_evaluator_total_tokens,
        "repair_agent_total_tokens": repair_agent_total_tokens,
    }





