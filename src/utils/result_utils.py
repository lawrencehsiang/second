from __future__ import annotations
from collections import Counter
import re

from src.components.state_store import StateStore


def normalize_answer(answer: str) -> str:
    if answer is None:
        return ""
    answer = str(answer).strip().lower()
    answer = re.sub(r"[\$,]", "", answer)
    answer = answer.replace("dollars", "").replace("dollar", "").strip()
    return answer


def is_correct(pred: str, gold: str) -> bool:
    return normalize_answer(pred) == normalize_answer(gold)


def majority_vote(answers: list[str]) -> str:
    if not answers:
        return ""
    normalized_to_original = {}
    normalized_answers = []
    for ans in answers:
        norm = normalize_answer(ans)
        normalized_answers.append(norm)
        if norm not in normalized_to_original:
            normalized_to_original[norm] = ans
    count = Counter(normalized_answers)
    majority_norm = count.most_common(1)[0][0]
    return normalized_to_original[majority_norm]


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
    return len(state_store.list_state_records())


def get_actual_rounds_executed(state_store: StateStore) -> int:
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