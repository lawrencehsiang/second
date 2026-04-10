from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from .history import HistoryUnit


class AgentInputRound1(BaseModel):
    question: str = Field(..., description="The original question for round 1.")


class AgentInputNormal(BaseModel):
    question: str = Field(..., description="The original question.")
    own_previous_answer: str = Field(
        ..., description="The agent's own answer from the previous round."
    )
    history_units: list[HistoryUnit] = Field(
        default_factory=list,
        description="Structured history units selected by the HistoryManager.",
    )


class ConflictResponse(BaseModel):
    conflict: str = Field(..., description="The unresolved conflict being addressed.")
    response: str = Field(..., description="The agent's response to the conflict.")
    status: Literal["resolved", "partially_resolved", "still_open"] = Field(
        ...,
        description="Whether the conflict is resolved, partially resolved, or still open.",
    )


class AgentOutputRound1(BaseModel):
    agent_id: str = Field(..., description="Agent identifier, e.g. A/B/C.")
    current_answer: str = Field(..., description="The agent's current answer.")
    brief_reason: str = Field(..., description="A short reason for the answer.")


class AgentOutputNormal(BaseModel):
    agent_id: str = Field(..., description="Agent identifier, e.g. A/B/C.")
    current_answer: str = Field(..., description="The agent's current answer.")
    response_to_conflicts: list[ConflictResponse] = Field(
        default_factory=list,
        description="Responses to the unresolved conflicts provided in the input.",
    )
    brief_reason: str = Field(..., description="A short reason for the answer.")
    keep_or_update: Literal["keep", "update"] | None = Field(
        default=None,
        description=(
            "Filled by system postprocessing. "
            "'keep' means answer unchanged from previous round; "
            "'update' means answer changed."
        ),
    )