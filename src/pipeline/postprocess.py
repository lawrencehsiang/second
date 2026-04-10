from __future__ import annotations

from typing import Literal

from src.schemas import AgentOutputNormal


KeepOrUpdate = Literal["keep", "update"]

# 这里是一个专门的后处理模块，负责根据当前回合的AgentOutputNormal和前一回合的答案，推断出每个AgentOutputNormal的keep_or_update字段应该是"keep"还是"update"。
def normalize_answer_text(text: str) -> str:
    """
    Lightweight normalization for answer comparison.

    Current strategy:
    - strip leading/trailing spaces
    - collapse internal whitespace
    - lowercase

    Notes:
    - This is intentionally simple for the first version.
    - We do NOT attempt semantic equivalence yet.
    """
    return " ".join(text.strip().lower().split())


def infer_keep_or_update(
    previous_answer: str | None,
    current_answer: str,
) -> KeepOrUpdate:
    """
    Infer whether the agent keeps or updates its answer.

    Rules:
    - If previous_answer is missing, default to "update".
    - If normalized previous/current answers are identical, return "keep".
    - Otherwise return "update".
    """
    if previous_answer is None:
        return "update"

    prev_norm = normalize_answer_text(previous_answer)
    curr_norm = normalize_answer_text(current_answer)

    if prev_norm == curr_norm:
        return "keep"
    return "update"


def apply_keep_or_update(
    agent_outputs: list[AgentOutputNormal],
    previous_answer_map: dict[str, str],
) -> list[AgentOutputNormal]:
    """
    Fill keep_or_update for a batch of AgentOutputNormal.

    Args:
        agent_outputs: Current round outputs from agents.
        previous_answer_map: Mapping from agent_id -> previous round answer.

    Returns:
        The same list of AgentOutputNormal objects with keep_or_update filled.
    """
    updated_outputs: list[AgentOutputNormal] = []

    for output in agent_outputs:
        previous_answer = previous_answer_map.get(output.agent_id)
        decision = infer_keep_or_update(
            previous_answer=previous_answer,
            current_answer=output.current_answer,
        )
        output.keep_or_update = decision
        updated_outputs.append(output)

    return updated_outputs