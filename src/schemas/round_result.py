from __future__ import annotations

from pydantic import BaseModel, Field

from .action import ActionDecision, RollbackDecision
from .evaluation import EvaluatorScores
from .state import StateRecord


class RoundResult(BaseModel):
    round_id: int = Field(..., description="Current round index.")

    agent_inputs: list[dict] = Field(
        default_factory=list,
        description="Serialized agent inputs for this round.",
    )
    agent_outputs: list[dict] = Field(
        default_factory=list,
        description="Serialized agent outputs for this round.",
    )

    state_record: StateRecord = Field(
        ...,
        description="Recorder output for this round.",
    )

    evaluator_scores: EvaluatorScores | None = Field(
        default=None,
        description="Scores from Evaluator. None for round 1.",
    )
    action_decision: ActionDecision | None = Field(
        default=None,
        description="Mapped action decision for this round.",
    )
    rollback_decision: RollbackDecision | None = Field(
        default=None,
        description="Rollback decision for this round.",
    )