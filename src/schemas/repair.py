from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from .history import HistoryUnit
from .state import StateRecord


RepairAction = Literal["continue", "finalize"]


class RemainingConflict(BaseModel):
    conflict: str = Field(..., description="A key conflict that still remains unresolved.")
    why_still_open: str = Field(
        ...,
        description="Why this conflict is still unresolved after the failed suffix.",
    )


class RepairBrief(BaseModel):
    remaining_conflicts: list[RemainingConflict] = Field(
        default_factory=list,
        description="Key remaining conflicts carried into repair mode.",
    )
    failure_summary: str = Field(
        ...,
        description="Compact summary of why the failed suffix went wrong.",
    )


class AnchorSelectorInput(BaseModel):
    trigger_round: int = Field(..., description="The round where rollback is triggered.")
    action_history: list[dict] = Field(
        default_factory=list,
        description="Round action history, e.g. [{'round_id': 1, 'action': 'continue'}].",
    )
    state_record_pool: list[StateRecord] = Field(
        default_factory=list,
        description="All available state records before rollback.",
    )


class AnchorSelectionResult(BaseModel):
    anchor_round: int = Field(..., description="The selected healthy anchor round.")
    anchor_state: StateRecord = Field(..., description="StateRecord at the anchor round.")


class RepairBriefGeneratorInput(BaseModel):
    question: str = Field(..., description="Original question.")
    anchor_state: StateRecord = Field(..., description="Healthy anchor state.")
    failed_suffix_state_records: list[StateRecord] = Field(
        default_factory=list,
        description="State records from old (t+1 ... k), excluding the anchor round itself.",
    )


class RepairAgentInput(BaseModel):
    question: str = Field(..., description="Original question.")
    history_units: list[HistoryUnit] = Field(
        default_factory=list,
        description="Reused / newly prepared history units for repair mode.",
    )
    repair_brief: RepairBrief = Field(
        ...,
        description="Compact repair brief injected into repair-mode agent input.",
    )


class RepairScores(BaseModel):
    progress_score: int = Field(..., ge=1, le=5)
    information_quality_score: int = Field(..., ge=1, le=5)
    completion_readiness_score: int = Field(..., ge=1, le=5)
    rationale: str | None = Field(
        default=None,
        description="Optional explanation of the repair scores.",
    )


class RepairRoundResult(BaseModel):
    round_id: int = Field(..., description="New round id in repair mode.")
    mode: Literal["repair"] = Field(default="repair")
    agent_inputs: list[dict] = Field(
        default_factory=list,
        description="Serialized repair-mode agent inputs.",
    )
    agent_outputs: list[dict] = Field(
        default_factory=list,
        description="Serialized repair-mode agent outputs.",
    )
    state_record: StateRecord = Field(..., description="StateRecord produced in repair mode.")
    repair_scores: RepairScores = Field(..., description="Repair-mode evaluator scores.")
    repair_action: RepairAction = Field(..., description="continue or finalize")


class RollbackEvent(BaseModel):
    trigger_round: int = Field(..., description="Old round where rollback was triggered.")
    anchor_round: int = Field(..., description="Healthy anchor round selected for repair.")


class RepairSessionResult(BaseModel):
    rollback_event: RollbackEvent = Field(..., description="Rollback metadata.")
    anchor_state: StateRecord = Field(..., description="Selected healthy anchor state.")
    repair_brief: RepairBrief = Field(..., description="Compact repair brief.")
    repair_rounds: list[RepairRoundResult] = Field(
        default_factory=list,
        description="All newly executed repair rounds.",
    )