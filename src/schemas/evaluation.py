from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


TransitionJudgement = Literal["improved", "plateau", "degraded"]
ContinueValue = Literal["high", "medium", "low"]


class TransitionEvaluation(BaseModel):
    """
    Unified evaluator output for both normal evaluator and repair evaluator.

    transition_judgement:
        - improved: the current transition clearly improves the debate state
        - plateau: the current transition does not clearly improve the state,
                   but it does not obviously make it worse either
        - degraded: the current transition makes the debate state worse

    continue_value:
        - high: continuing from the current state is likely worthwhile
        - medium: continuing may still be useful
        - low: continuing is unlikely to help much
    """

    transition_judgement: TransitionJudgement = Field(
        ...,
        description="High-level judgement of how the current round changed the debate state.",
    )
    continue_value: ContinueValue = Field(
        ...,
        description="Estimated value of continuing from the current state.",
    )
    reason: str = Field(
        ...,
        description="Short natural-language explanation for the judgement.",
    )