from __future__ import annotations

from collections import Counter
import re

from src.components.state_store import StateStore


def normalize_answer(text: str) -> str:
    """
    Lightweight normalization for voting / fallback comparison.
    """
    if text is None:
        return ""

    text = str(text).strip().lower()
    text = text.replace(",", "")
    text = text.replace("$", "")
    text = text.replace("dollars", "")
    text = text.replace("dollar", "")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_last_number(text: str) -> float | None:
    """
    Extract the last numeric value from model output.

    Examples:
    - 'The answer is 10' -> 10.0
    - '$10.00' -> 10.0
    - '50/60 * 12 = 10' -> 10.0
    """
    if text is None:
        return None

    text = normalize_answer(text)
    numbers = re.findall(r"-?\d+(?:\.\d+)?", text)
    if not numbers:
        return None

    try:
        return float(numbers[-1])
    except ValueError:
        return None


def is_correct(pred: str, gold: str) -> bool:
    """
    GSM8K-oriented correctness check.

    Assumption:
    gold answers are integers.
    So we extract the final numeric answer from pred/gold and compare as integers.
    """
    pred_num = extract_last_number(pred)
    gold_num = extract_last_number(gold)

    if pred_num is None or gold_num is None:
        return False

    return int(round(pred_num)) == int(round(gold_num))


def majority_vote(answers: list[str]) -> str:
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
            # GSM8K answers are integers, so vote on rounded integer form
            key = f"num:{int(round(num))}"
        else:
            key = f"text:{normalize_answer(ans)}"

        vote_keys.append(key)
        if key not in key_to_original:
            key_to_original[key] = ans

    count = Counter(vote_keys)
    majority_key = count.most_common(1)[0][0]
    return key_to_original[majority_key]


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


def build_trace_bundle(state_store: StateStore) -> dict:
    return {
        "final_trace": build_final_trace(state_store),
        "execution_events": build_execution_events(state_store),
    }


def get_effective_rounds_used(state_store: StateStore) -> int:
    """
    Number of rounds remaining in the final effective state trajectory.
    """
    return len(state_store.list_state_records())


def get_actual_rounds_executed(state_store: StateStore) -> int:
    """
    True execution cost:
    count all normal + repair rounds that were actually run,
    regardless of whether some failed suffix states were later removed.
    """
    return sum(
        1
        for event in state_store.list_events()
        if event["type"] in {"normal_round_executed", "repair_round_executed"}
    )


def get_stop_reason(rollback_context: dict | None, early_stopped: bool) -> str:
    if rollback_context is not None:
        return "rollback"
    if early_stopped:
        return "early_stop"
    return "max_round"