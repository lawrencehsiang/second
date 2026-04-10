from src.components.action_mapper import ActionMapper
from src.schemas import EvaluatorScores

mapper = ActionMapper()

cases = [
    EvaluatorScores(
        progress_score=5,
        information_quality_score=4,
        future_utility_score=4,
        rationale="Healthy state."
    ),
    EvaluatorScores(
        progress_score=2,
        information_quality_score=2,
        future_utility_score=3,
        rationale="Weak state."
    ),
    EvaluatorScores(
        progress_score=3,
        information_quality_score=3,
        future_utility_score=2,
        rationale="Mixed state."
    ),
]

for i, case in enumerate(cases, start=1):
    decision = mapper.map_action(case)
    print(f"Case {i}: {decision.model_dump()}")