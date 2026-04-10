from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Claim(BaseModel):
    text: str = Field(..., description="The content of the newly added claim.")
    claim_type: Literal["support", "rebuttal", "constraint", "explanation"] = Field(
        ...,
        description="The functional type of the claim.",
    )
    related_answer: str | None = Field(
        default=None,
        description=(
            "Optional answer that this claim mainly supports or targets."
        ),
    )


class UnresolvedConflict(BaseModel):
    conflict: str = Field(..., description="Description of the unresolved conflict.")
    why_still_open: str = Field(
        ...,
        description="Why this conflict has not been resolved yet.",
    )
    involved_answers: list[str] = Field(
        default_factory=list,
        description="Optional list of answers involved in this conflict.",
    )


class StateRecord(BaseModel):
    round_id: int = Field(..., description="Current round index.")
    current_answers: list[str] = Field(
        default_factory=list,
        description="Current answers from all agents in this round.",
    )
    newly_added_claims: list[Claim] = Field(
        default_factory=list,
        description="Claims newly introduced or newly surfaced in this round.",
    )
    unresolved_conflicts: list[UnresolvedConflict] = Field(
        default_factory=list,
        description="Conflicts that remain unresolved after this round.",
    )
    key_raw_snippets: list[str] = Field(
        default_factory=list,
        description="Key raw snippets preserved from this round.",
    )