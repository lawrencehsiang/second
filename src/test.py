# from src.components.state_store import StateStore
# from src.schemas import Claim, StateRecord, UnresolvedConflict

# store = StateStore()

# state_1 = StateRecord(
#     round_id=1,
#     current_answers=["595", "225", "595"],
#     newly_added_claims=[
#         Claim(text="rubies = 140", claim_type="support", related_answer="595")
#     ],
#     unresolved_conflicts=[
#         UnresolvedConflict(
#             conflict="Whether rubies should be computed as 175 - 35",
#             why_still_open="One agent still uses a different derivation",
#             involved_answers=["595", "225"],
#         )
#     ],
#     key_raw_snippets=["There were 35 fewer rubies than diamonds."],
# )

# store.add_state_record(state_1)
# store.set_round_action(1, "continue")

# print(store.get_latest_state_record())
# print(store.get_latest_continue_round())
# print(store.get_action_history())

# from src.components.history_manager import HistoryManager
# from src.components.state_store import StateStore
# from src.schemas import Claim, StateRecord, UnresolvedConflict

# store = StateStore()

# state_1 = StateRecord(
#     round_id=1,
#     current_answers=["595", "225", "595"],
#     newly_added_claims=[
#         Claim(text="rubies = 140", claim_type="support", related_answer="595"),
#         Claim(text="the previous derivation is invalid", claim_type="rebuttal", related_answer="225"),
#     ],
#     unresolved_conflicts=[
#         UnresolvedConflict(
#             conflict="Whether rubies should be computed as 175 - 35",
#             why_still_open="One agent still uses an alternative derivation",
#             involved_answers=["595", "225"],
#         )
#     ],
#     key_raw_snippets=["There were 35 fewer rubies than diamonds."],
# )

# state_2 = StateRecord(
#     round_id=2,
#     current_answers=["595", "595", "225"],
#     newly_added_claims=[
#         Claim(text="the minority answer still questions the ruby count", claim_type="explanation", related_answer="225"),
#     ],
#     unresolved_conflicts=[
#         UnresolvedConflict(
#             conflict="Whether the ruby count directly determines the final total",
#             why_still_open="The minority answer remains unconvinced",
#             involved_answers=["595", "225"],
#         )
#     ],
#     key_raw_snippets=["A minority agent still disputes the final count."],
# )

# store.add_state_record(state_1)
# store.add_state_record(state_2)

# hm = HistoryManager(history_window_rounds=3, normal_mode_history_unit_count=4)
# units = hm.build_history_units(
#     question="How many gems were there in the chest?",
#     current_round_id=3,
#     state_store=store,
# )

# for u in units:
#     print(u.model_dump())


# from src.pipeline.postprocess import apply_keep_or_update
# from src.schemas import AgentOutputNormal, ConflictResponse

# outputs = [
#     AgentOutputNormal(
#         agent_id="A",
#         current_answer="595",
#         response_to_conflicts=[],
#         brief_reason="Same as before."
#     ),
#     AgentOutputNormal(
#         agent_id="B",
#         current_answer="225",
#         response_to_conflicts=[],
#         brief_reason="I changed my answer."
#     ),
# ]

# previous_answer_map = {
#     "A": "595",
#     "B": "595",
# }

# updated = apply_keep_or_update(outputs, previous_answer_map)

# for item in updated:
#     print(item.agent_id, item.current_answer, item.keep_or_update)


from src.components.recorder import Recorder
from src.schemas import AgentOutputNormal, ConflictResponse

outputs = [
    AgentOutputNormal(
        agent_id="A",
        current_answer="595",
        response_to_conflicts=[
            ConflictResponse(
                conflict="Whether rubies should be computed as 175 - 35",
                response="I disagree with the alternative derivation because the problem directly states there are 35 fewer rubies than diamonds.",
                status="still_open",
            )
        ],
        brief_reason="Because diamonds are 175 and rubies are 35 fewer, rubies should be 140.",
        keep_or_update="keep",
    ),
    AgentOutputNormal(
        agent_id="B",
        current_answer="225",
        response_to_conflicts=[
            ConflictResponse(
                conflict="Whether rubies should be computed as 175 - 35",
                response="The previous derivation is not fully justified, so the conflict is still open.",
                status="partially_resolved",
            )
        ],
        brief_reason="I think the previous derivation may be flawed.",
        keep_or_update="update",
    ),
]

recorder = Recorder(max_snippets=5)
state = recorder.build_state_record(
    round_id=2,
    agent_outputs=outputs,
)

print(state.model_dump())