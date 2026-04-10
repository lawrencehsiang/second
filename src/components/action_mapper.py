from __future__ import annotations

from src.schemas import ActionDecision, EvaluatorScores


class ActionMapper:
    """
    Rule-based mapper from EvaluatorScores to ActionDecision.

    Strategy:
    1. continue: state looks healthy enough to move forward
    2. rollback: state looks clearly poor / low-value
    3. watch: intermediate zone
    """

    def map_action(self, scores: EvaluatorScores) -> ActionDecision:
        progress = scores.progress_score
        info = scores.information_quality_score
        future = scores.future_utility_score

        values = [progress, info, future]
        avg_score = sum(values) / 3.0
        min_score = min(values)
        low_count = sum(1 for v in values if v <= 2)

        # continue
        if all(v >= 3 for v in values):
            return ActionDecision(
                action="continue",
                reason=(
                    "All three evaluator scores are >= 3, "
                    "so the state is considered healthy enough to continue."
                ),
            )

        if avg_score >= 3.0 and min_score >= 2:
            return ActionDecision(
                action="continue",
                reason=(
                    "Average score is >= 3.0 and no dimension is weak (<2), "
                    "so the state is healthy enough to continue."
                ),
            )

        # rollback
        if low_count >= 2:
            return ActionDecision(
                action="rollback",
                reason=(
                    "At least two evaluator dimensions are weak (<=2), "
                    "so the state is poor enough to roll back."
                ),
            )

        if avg_score < 2.0:
            return ActionDecision(
                action="rollback",
                reason=(
                    "Average evaluator score is low (<2.0), "
                    "so the state is low-value and should roll back."
                ),
            )

        # watch
        return ActionDecision(
            action="watch",
            reason=(
                "The evaluator scores are mixed: not strong enough for continue, "
                "but not poor enough for rollback."
            ),
        )