from src.components.evaluator import Evaluator
from src.schemas import Claim, StateRecord, UnresolvedConflict


class MockLLMClient:
    def generate(self, prompt: str) -> str:
        return """
{
  "progress_score": 4,
  "information_quality_score": 4,
  "future_utility_score": 3,
  "rationale": "Compared with the previous round, the current round adds clearer supporting and rebuttal claims, although one conflict remains unresolved."
}
""".strip()


previous_state = StateRecord(
    round_id=1,
    current_answers=["595", "225", "595"],
    newly_added_claims=[
        Claim(
            text="Rubies should be 140 because they are 35 fewer than 175 diamonds.",
            claim_type="support",
            related_answer="595",
        )
    ],
    unresolved_conflicts=[
        UnresolvedConflict(
            conflict="Whether rubies should be computed as 175 - 35",
            why_still_open="One agent still uses an alternative derivation.",
            involved_answers=["595", "225"],
        )
    ],
    key_raw_snippets=[
        "Rubies should be 140 because they are 35 fewer than 175 diamonds."
    ],
)

current_state = StateRecord(
    round_id=2,
    current_answers=["595", "595", "225"],
    newly_added_claims=[
        Claim(
            text="Rubies are 35 fewer than 175 diamonds, so rubies should be 140.",
            claim_type="support",
            related_answer="595",
        ),
        Claim(
            text="The alternative derivation remains insufficiently justified.",
            claim_type="rebuttal",
            related_answer="225",
        ),
    ],
    unresolved_conflicts=[
        UnresolvedConflict(
            conflict="Whether the ruby derivation is fully justified",
            why_still_open="One agent still challenges the derivation details.",
            involved_answers=["595", "225"],
        )
    ],
    key_raw_snippets=[
        "Rubies are 35 fewer than diamonds, so rubies should be 140.",
        "The alternative derivation remains insufficiently justified.",
    ],
)

evaluator = Evaluator(llm_client=MockLLMClient())
scores = evaluator.evaluate_state(
    question="How many gems are in the chest?",
    previous_state_record=previous_state,
    current_state_record=current_state,
)

print(scores.model_dump())