from src.components.repair_evaluator import RepairEvaluator
from src.schemas import (
    Claim,
    RepairBrief,
    RemainingConflict,
    StateRecord,
    UnresolvedConflict,
)


class MockLLMClient:
    def generate(self, prompt: str) -> str:
        return """
{
  "progress_score": 4,
  "information_quality_score": 4,
  "completion_readiness_score": 3,
  "rationale": "The current repair round directly addresses the main remaining conflict and improves answer stability, but another repair round may still help before finalization."
}
""".strip()


anchor_state = StateRecord(
    round_id=3,
    current_answers=["595", "595", "225"],
    newly_added_claims=[
        Claim(
            text="Rubies should be 140.",
            claim_type="support",
            related_answer="595",
        )
    ],
    unresolved_conflicts=[
        UnresolvedConflict(
            conflict="Whether rubies should be computed as 175 - 35",
            why_still_open="One agent still questions the ruby derivation.",
            involved_answers=["595", "225"],
        )
    ],
    key_raw_snippets=["Rubies should be 140."],
)

repair_brief = RepairBrief(
    remaining_conflicts=[
        RemainingConflict(
            conflict="Whether rubies should be computed as 175 - 35",
            why_still_open="The previous suffix repeatedly revisited the ruby derivation without closing the reasoning gap.",
        )
    ],
    failure_summary="The previous suffix repeatedly defended the same line of reasoning and did not directly resolve the core disagreement.",
)

previous_repair_state = StateRecord(
    round_id=4,
    current_answers=["595", "225", "225"],
    newly_added_claims=[
        Claim(
            text="The ruby derivation is still disputed.",
            claim_type="rebuttal",
            related_answer="225",
        )
    ],
    unresolved_conflicts=[
        UnresolvedConflict(
            conflict="Whether rubies should be computed as 175 - 35",
            why_still_open="The derivation remains disputed.",
            involved_answers=["595", "225"],
        )
    ],
    key_raw_snippets=["The ruby derivation is still disputed."],
)

current_repair_state = StateRecord(
    round_id=5,
    current_answers=["595", "595", "225"],
    newly_added_claims=[
        Claim(
            text="The ruby count can be justified directly from the problem statement.",
            claim_type="support",
            related_answer="595",
        ),
        Claim(
            text="The earlier objection does not provide a stronger alternative derivation.",
            claim_type="rebuttal",
            related_answer="225",
        ),
    ],
    unresolved_conflicts=[
        UnresolvedConflict(
            conflict="Whether rubies should be computed as 175 - 35",
            why_still_open="One minority objection still remains.",
            involved_answers=["595", "225"],
        )
    ],
    key_raw_snippets=[
        "The ruby count can be justified directly from the problem statement.",
        "The earlier objection does not provide a stronger alternative derivation.",
    ],
)

evaluator = RepairEvaluator(llm_client=MockLLMClient())

scores = evaluator.evaluate_repair(
    question="How many gems are in the chest?",
    anchor_state=anchor_state,
    repair_brief=repair_brief,
    current_state_record=current_repair_state,
    previous_repair_state_record=previous_repair_state,
)

print(scores.model_dump())