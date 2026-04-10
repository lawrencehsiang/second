from __future__ import annotations

import re
from collections import Counter
from typing import Optional

from src.components.state_store import StateStore


def extract_last_number(text: str) -> str:
    """
    Extract the last integer/decimal number from text.
    For GSM8K, this is usually enough for a first pass.
    """
    if text is None:
        return ""
    matches = re.findall(r"-?\d+(?:\.\d+)?", str(text))
    return matches[-1] if matches else str(text).strip()


def normalize_answer(text: str) -> str:
    return extract_last_number(text)


def is_correct(pred: str, gold: str) -> bool:
    return normalize_answer(pred) == normalize_answer(gold)


def majority_vote(answers: list[str]) -> str:
    if not answers:
        return ""
    normalized_answers = [normalize_answer(a) for a in answers]
    counter = Counter(normalized_answers)
    return counter.most_common(1)[0][0]


def get_round_1_answers(state_store: StateStore) -> list[str]:
    round_1 = state_store.get_state_record(1)
    if round_1 is None:
        return []
    return round_1.current_answers


def get_final_answers(state_store: StateStore) -> list[str]:
    all_records = state_store.list_state_records()
    if not all_records:
        return []
    final_record = all_records[-1]
    return final_record.current_answers


def build_trace(state_store: StateStore) -> list[dict]:
    trace = []
    for state in state_store.list_state_records():
        trace.append(state.model_dump())
    return trace


def get_rounds_used(state_store: StateStore) -> int:
    return len(state_store.list_state_records())


def get_stop_reason(rollback_context: Optional[dict], early_stopped: bool) -> str:
    if rollback_context is not None:
        return "rollback"
    if early_stopped:
        return "early_stop"
    return "max_round"