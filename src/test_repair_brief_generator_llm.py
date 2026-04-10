from src.components.repair_brief_generator import RepairBriefGenerator
from src.schemas import Claim, StateRecord, UnresolvedConflict


class MockLLMClient:
    def generate(self, prompt: str) -> str:
        return """
{
  "remaining_conflicts": [
    {
      "conflict": "Whether rubies should be computed as 175 - 35",
      "why_still_open": "The failed suffix repeatedly returned to the ruby derivation but did not settle whether it was sufficient to close the reasoning gap."
    }
  ],
  "failure_summary": "The previous suffix repeatedly defended the same line of reasoning and did not directly resolve the core disagreement."
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

failed_suffix_state_records = [
    StateRecord(
        round_id=4,
        current_answers=["595", "225", "225"],
        newly_added_claims=[],
        unresolved_conflicts=[
            UnresolvedConflict(
                conflict="Whether rubies should be computed as 175 - 35",
                why_still_open="The ruby derivation is still disputed.",
                involved_answers=["595", "225"],
            )
        ],
        key_raw_snippets=[],
    ),
    StateRecord(
        round_id=5,
        current_answers=["225", "225", "595"],
        newly_added_claims=[],
        unresolved_conflicts=[
            UnresolvedConflict(
                conflict="Whether rubies should be computed as 175 - 35",
                why_still_open="The ruby derivation is still disputed.",
                involved_answers=["595", "225"],
            )
        ],
        key_raw_snippets=[],
    ),
]

generator = RepairBriefGenerator(
    llm_client=MockLLMClient(),
    max_remaining_conflicts=2,
)

brief = generator.generate_brief_from_parts(
    question="How many gems are in the chest?",
    anchor_state=anchor_state,
    failed_suffix_state_records=failed_suffix_state_records,
)

print(brief.model_dump())