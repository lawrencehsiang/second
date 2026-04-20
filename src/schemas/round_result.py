from __future__ import annotations

from pydantic import BaseModel, Field

from .action import ActionDecision, RollbackDecision
from .evaluation import TransitionEvaluation
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

    # Keep the old field name for compatibility with the current pipeline,
    # but the actual payload is already the new unified TransitionEvaluation.
    evaluator_scores: TransitionEvaluation | None = Field(
        default=None,
        description="Unified evaluator output. None for round 1.",
    )

    action_decision: ActionDecision | None = Field(
        default=None,
        description="Mapped action decision for this round.",
    )
    rollback_decision: RollbackDecision | None = Field(
        default=None,
        description="Rollback decision for this round.",
    )