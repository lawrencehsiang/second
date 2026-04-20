from __future__ import annotations

from src.schemas.action import RepairActionDecision
from src.schemas.evaluation import TransitionEvaluation


class RepairActionMapper:
    """
    Rule-based mapper from TransitionEvaluation to repair-mode action.

    Repair-mode action space:
    - continue
    - finalize

    Mapping idea:
    1. degraded:
       - finalize

    2. improved:
       - continue if continue_value is high/medium and not at max round
       - otherwise finalize

    3. plateau:
       - continue only if continue_value is high and not at max round
       - otherwise finalize
    """

    def map_action(
        self,
        evaluation: TransitionEvaluation,
        *,
        round_id: int,
        max_round: int,
    ) -> RepairActionDecision:
        """
        Map repair evaluator output into a repair-mode action.

        Args:
            evaluation: TransitionEvaluation from repair evaluator.
            round_id: Current repair round number.
            max_round: Max allowed repair round.

        Returns:
            RepairActionDecision
        """
        judgement = evaluation.transition_judgement
        continue_value = evaluation.continue_value

        # Hard stop when max round is reached
        if round_id >= max_round:
            return RepairActionDecision(
                action="finalize",
                reason=(
                    f"Current repair round {round_id} has reached max_round={max_round}, "
                    "so repair should finalize."
                ),
            )

        # Degraded repair transition: stop repair
        if judgement == "degraded":
            return RepairActionDecision(
                action="finalize",
                reason=(
                    "The repair transition is judged as degraded, so repair should "
                    "finalize instead of continuing."
                ),
            )

        # Improved repair transition: continue if future value remains meaningful
        if judgement == "improved":
            if continue_value in {"high", "medium"}:
                return RepairActionDecision(
                    action="continue",
                    reason=(
                        f"The repair transition is improved and continue_value is "
                        f"{continue_value}, so another repair round is justified."
                    ),
                )
            return RepairActionDecision(
                action="finalize",
                reason=(
                    "The repair transition is improved, but continue_value is low, "
                    "so further repair is unlikely to help much."
                ),
            )

        # Plateau repair transition: only continue if future value is clearly high
        if judgement == "plateau":
            if continue_value == "high":
                return RepairActionDecision(
                    action="continue",
                    reason=(
                        "The repair transition is plateau and continue_value is high, "
                        "so one more repair round is still justified."
                    ),
                )
            return RepairActionDecision(
                action="finalize",
                reason=(
                    f"The repair transition is plateau and continue_value is "
                    f"{continue_value}, so repair should finalize."
                ),
            )

        # Defensive fallback
        return RepairActionDecision(
            action="finalize",
            reason=(
                "Repair evaluator output was unexpected, so the mapper falls back "
                "to finalize for safety."
            ),
        )