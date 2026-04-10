from __future__ import annotations

from dataclasses import dataclass

from src.schemas import RepairAction, RepairScores


@dataclass
class RepairActionDecision:
    repair_action: RepairAction
    reason: str | None = None


class RepairActionMapper:
    """
    Rule-based mapper from RepairScores to repair action.

    Pilot repair-mode rule:
    - finalize if completion_readiness_score >= 4
    - finalize if current_round == max_round
    - otherwise continue
    """

    def map_action(
        self,
        repair_scores: RepairScores,
        current_round: int,
        max_round: int,
    ) -> RepairActionDecision:
        """
        Map repair-mode scores into a repair action.

        Args:
            repair_scores: Repair-mode evaluator scores.
            current_round: Current repair round id in the global run.
            max_round: Global maximum round.

        Returns:
            RepairActionDecision
        """
        if repair_scores.completion_readiness_score >= 4:
            print("completion_readiness_score >= 4，状态被评估为修复完成准备就绪，进入finalize阶段。")
            return RepairActionDecision(
                repair_action="finalize",
                reason=(
                    "completion_readiness_score >= 4, "
                    "so repair mode is ready to finalize."
                ),
            )

        if current_round >= max_round:
            print("current_round has reached max_round，状态被评估为达到最大轮数，进入finalize阶段。")
            return RepairActionDecision(
                repair_action="finalize",
                reason=(
                    "current_round has reached max_round, "
                    "so repair mode must finalize."
                ),
            )

        return RepairActionDecision(
            print("completion_readiness_score < 4且current_round未达到max_round，状态被评估为修复继续中。"),  
            repair_action="continue",
            reason=(
                "completion_readiness_score is below finalize threshold "
                "and max_round is not reached, so repair continues."
            ),
        )