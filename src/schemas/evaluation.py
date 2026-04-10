from __future__ import annotations

from pydantic import BaseModel, Field


class EvaluatorScores(BaseModel):
    progress_score: int = Field(
        ...,
        ge=1,
        le=5,
        description="How much meaningful progress this round makes.",
    )
    information_quality_score: int = Field(
        ...,
        ge=1,
        le=5,
        description="How informative and useful the state's information is.",
    )
    future_utility_score: int = Field(
        ...,
        ge=1,
        le=5,
        description="How useful this state is for future continuation.",
    )
    rationale: str | None = Field(
        default=None,
        description="Optional explanation of the scores.",
    )