from src.components.recorder import Recorder
from src.schemas import AgentOutputNormal, ConflictResponse


class MockLLMClient:
    def generate(self, prompt: str) -> str:
        return """
{
  "round_id": 2,
  "current_answers": ["595", "225"],
  "newly_added_claims": [
    {
      "text": "Rubies should be 140 because they are 35 fewer than 175 diamonds.",
      "claim_type": "support",
      "related_answer": "595"
    },
    {
      "text": "The alternative derivation is still not fully justified.",
      "claim_type": "rebuttal",
      "related_answer": "225"
    }
  ],
  "unresolved_conflicts": [
    {
      "conflict": "Whether rubies should be computed as 175 - 35",
      "why_still_open": "One agent still questions the derivation.",
      "involved_answers": ["595", "225"]
    }
  ],
  "key_raw_snippets": [
    "Rubies should be 140 because they are 35 fewer than 175 diamonds.",
    "The alternative derivation is still not fully justified."
  ]
}
""".strip()


outputs = [
    AgentOutputNormal(
        agent_id="A",
        current_answer="595",
        response_to_conflicts=[
            ConflictResponse(
                conflict="Whether rubies should be computed as 175 - 35",
                response="Rubies should be 140 because they are 35 fewer than 175 diamonds.",
                status="still_open",
            )
        ],
        brief_reason="Because diamonds are 175 and rubies are 35 fewer, rubies are 140.",
        keep_or_update="keep",
    ),
    AgentOutputNormal(
        agent_id="B",
        current_answer="225",
        response_to_conflicts=[
            ConflictResponse(
                conflict="Whether rubies should be computed as 175 - 35",
                response="The alternative derivation is still not fully justified.",
                status="partially_resolved",
            )
        ],
        brief_reason="I still think the derivation may be flawed.",
        keep_or_update="update",
    ),
]

recorder = Recorder(llm_client=MockLLMClient())
state = recorder.build_state_record(
    round_id=2,
    agent_outputs=outputs,
)

print(state.model_dump())