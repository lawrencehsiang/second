from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

HistoryUnitType = Literal[
    "mainstream_support",
    "key_rebuttal",
    "minority_objection",
    "core_unresolved_conflict",
]


class HistoryUnit(BaseModel):
    type: HistoryUnitType = Field(..., description="The category of the history unit.")

    answer: str | None = Field(
        default=None,
        description="Used mainly for mainstream_support.",
    )
    target_answer: str | None = Field(
        default=None,
        description="Used mainly for key_rebuttal; the answer being rebutted.",
    )
    minority_answer: str | None = Field(
        default=None,
        description="Used mainly for minority_objection.",
    )

    claim: str | None = Field(
        default=None,
        description="A structured summary of the key point or claim.",
    )
    snippet: str | None = Field(
        default=None,
        description="A raw or near-raw supporting text snippet.",
    )

    conflict: str | None = Field(
        default=None,
        description="Used mainly for core_unresolved_conflict.",
    )
    why_unresolved: str | None = Field(
        default=None,
        description="Why the minority objection remains unresolved.",
    )
    why_still_open: str | None = Field(
        default=None,
        description="Why the core conflict is still open.",
    )

    source_round: int | None = Field(
        default=None,
        description="Optional round index where this history unit originated.",
    )