from src.components.repair_action_mapper import RepairActionMapper
from src.schemas import RepairScores


mapper = RepairActionMapper()

cases = [
    (
        "case_1_finalize_by_readiness",
        RepairScores(
            progress_score=4,
            information_quality_score=3,
            completion_readiness_score=4,
            rationale="Looks ready to stop."
        ),
        5,
        6,
    ),
    (
        "case_2_finalize_by_max_round",
        RepairScores(
            progress_score=3,
            information_quality_score=3,
            completion_readiness_score=2,
            rationale="Not ideal, but no more rounds allowed."
        ),
        6,
        6,
    ),
    (
        "case_3_continue",
        RepairScores(
            progress_score=3,
            information_quality_score=4,
            completion_readiness_score=2,
            rationale="Still needs another repair round."
        ),
        4,
        6,
    ),
]

for name, scores, current_round, max_round in cases:
    decision = mapper.map_action(
        repair_scores=scores,
        current_round=current_round,
        max_round=max_round,
    )
    print(name, decision)