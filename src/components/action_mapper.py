from __future__ import annotations

from src.schemas.action import ActionDecision
from src.schemas.evaluation import TransitionEvaluation


class ActionMapper:
    """
    Rule-based mapper from TransitionEvaluation to normal-mode ActionDecision.

    Normal-mode action space:
    - continue
    - early_stop
    - rollback

    Mapping idea:
    1. degraded:
       - if rollback is still available -> rollback
       - otherwise -> early_stop

    2. improved:
       - if continue_value is high/medium and not at max round -> continue
       - otherwise -> early_stop

    3. plateau:
       - if continue_value is high and not at max round -> continue
       - otherwise -> early_stop
    """

    def map_action(
        self,
        evaluation: TransitionEvaluation,
        *,
        round_id: int,
        max_round: int,
        rollback_available: bool,
    ) -> ActionDecision:
        """
        Map evaluator output into a normal-mode action.

        Args:
            evaluation: TransitionEvaluation from evaluator.
            round_id: Current round number.
            max_round: Max allowed round in normal mode.
            rollback_available: Whether rollback can still be used.

        Returns:
            ActionDecision
        """
        judgement = evaluation.transition_judgement
        continue_value = evaluation.continue_value

        # Hard stop when max round is reached
        if round_id >= max_round:
            return ActionDecision(
                action="early_stop",
                reason=(
                    f"Current round {round_id} has reached max_round={max_round}, "
                    "so the debate should stop."
                ),
            )

        # Degraded transition: prefer rollback if possible
        if judgement == "degraded":
            if rollback_available:
                return ActionDecision(
                    action="rollback",
                    reason=(
                        "The current transition is judged as degraded, and rollback "
                        "is still available, so rollback is preferred."
                    ),
                )
            return ActionDecision(
                action="early_stop",
                reason=(
                    "The current transition is judged as degraded, but rollback is "
                    "not available, so the debate should stop early."
                ),
            )

        # Improved transition: continue if future value remains meaningful
        if judgement == "improved":
            if continue_value in {"high", "medium"}:
                return ActionDecision(
                    action="continue",
                    reason=(
                        f"The transition is improved and continue_value is "
                        f"{continue_value}, so it is worth continuing."
                    ),
                )
            return ActionDecision(
                action="early_stop",
                reason=(
                    "The transition is improved, but continue_value is low, so "
                    "further rounds are unlikely to help much."
                ),
            )

        # Plateau transition: only continue when future value is clearly high
        if judgement == "plateau":
            if continue_value == "high":
                return ActionDecision(
                    action="continue",
                    reason=(
                        "The transition is plateau rather than degraded, and "
                        "continue_value is high, so one more round is still justified."
                    ),
                )
            return ActionDecision(
                action="early_stop",
                reason=(
                    f"The transition is plateau and continue_value is "
                    f"{continue_value}, so the debate should stop instead of dragging on."
                ),
            )

        # Defensive fallback
        return ActionDecision(
            action="early_stop",
            reason=(
                "Evaluator output was unexpected, so the mapper falls back to "
                "early_stop for safety."
            ),
        )