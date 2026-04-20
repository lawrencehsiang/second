from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


NormalRoundAction = Literal["continue", "early_stop", "rollback"]
RepairRoundAction = Literal["continue", "finalize"]
# backward-compatible alias for old imports
RoundAction = NormalRoundAction

class ActionDecision(BaseModel):
    """
    Action decision for normal debate mode.
    """
    action: NormalRoundAction = Field(
        ...,
        description="Mapped action for the normal debate stage.",
    )
    reason: str | None = Field(
        default=None,
        description="Optional explanation for the mapped action.",
    )


class RollbackDecision(BaseModel):
    """
    Extra rollback execution info for normal mode.

    Note:
    - action='rollback' means the mapper recommends rollback.
    - RollbackDecision says whether rollback is actually triggered,
      and where to roll back to.
    """
    trigger_rollback: bool = Field(
        ...,
        description="Whether rollback should actually be triggered.",
    )
    rollback_to_round: int | None = Field(
        default=None,
        description="Target round to roll back to, if any.",
    )
    reason: str | None = Field(
        default=None,
        description="Optional explanation for the rollback decision.",
    )


class RepairActionDecision(BaseModel):
    """
    Action decision for repair mode.
    """
    action: RepairRoundAction = Field(
        ...,
        description="Mapped action for the repair stage.",
    )
    reason: str | None = Field(
        default=None,
        description="Optional explanation for the mapped repair action.",
    )