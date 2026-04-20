from __future__ import annotations

from pydantic import BaseModel, Field


class AnswerTransition(BaseModel):
    answers_prev: list[str] = Field(
        default_factory=list,
        description="Answers from previous round, ordered by fixed agent order.",
    )
    answers_curr: list[str] = Field(
        default_factory=list,
        description="Answers from current round, ordered by fixed agent order.",
    )


class ConflictTransition(BaseModel):
    persistent_conflicts: list[str] = Field(
        default_factory=list,
        description="Conflicts appearing in both previous and current round.",
    )
    resolved_conflicts: list[str] = Field(
        default_factory=list,
        description="Conflicts present in previous round but absent in current round.",
    )
    new_conflicts: list[str] = Field(
        default_factory=list,
        description="Conflicts newly appearing in current round.",
    )


class ClaimsByAnswer(BaseModel):
    answer: str = Field(
        ...,
        description="The answer string that the grouped claims are attached to.",
    )
    support_claims: list[str] = Field(
        default_factory=list,
        description="New support-like claims added in current round for this answer.",
    )
    rebuttal_claims: list[str] = Field(
        default_factory=list,
        description="New rebuttal claims added in current round for this answer.",
    )


class ClaimTransition(BaseModel):
    new_claims_by_answer: list[ClaimsByAnswer] = Field(
        default_factory=list,
        description="Current round newly added claims grouped by related answer.",
    )


class TransitionDigest(BaseModel):
    answer_transition: AnswerTransition = Field(
        ...,
        description="Structured answer changes from previous to current round.",
    )
    conflict_transition: ConflictTransition = Field(
        ...,
        description="Structured unresolved-conflict changes between adjacent rounds.",
    )
    claim_transition: ClaimTransition = Field(
        ...,
        description="Structured grouping of current round newly added claims.",
    )