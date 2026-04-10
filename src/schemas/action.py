from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

RoundAction = Literal["continue", "watch", "rollback"]


class ActionDecision(BaseModel):
    action: RoundAction = Field(..., description="Mapped round action.")
    reason: str | None = Field(
        default=None,
        description="Optional explanation for the mapped action.",
    )


class RollbackDecision(BaseModel):
    trigger_rollback: bool = Field(
        ...,
        description="Whether rollback should be triggered.",
    )
    rollback_to_round: int | None = Field(
        default=None,
        description="Target round to roll back to, if any.",
    )
    reason: str | None = Field(
        default=None,
        description="Optional explanation for the rollback decision.",
    )