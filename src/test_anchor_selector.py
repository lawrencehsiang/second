from src.components.anchor_selector import AnchorSelector
from src.schemas import StateRecord


state_record_pool = [
    StateRecord(
        round_id=1,
        current_answers=["A", "B", "A"],
        newly_added_claims=[],
        unresolved_conflicts=[],
        key_raw_snippets=[],
    ),
    StateRecord(
        round_id=2,
        current_answers=["A", "A", "B"],
        newly_added_claims=[],
        unresolved_conflicts=[],
        key_raw_snippets=[],
    ),
    StateRecord(
        round_id=3,
        current_answers=["A", "A", "A"],
        newly_added_claims=[],
        unresolved_conflicts=[],
        key_raw_snippets=[],
    ),
    StateRecord(
        round_id=4,
        current_answers=["C", "A", "A"],
        newly_added_claims=[],
        unresolved_conflicts=[],
        key_raw_snippets=[],
    ),
]

action_history = [
    {"round_id": 1, "action": "continue"},
    {"round_id": 2, "action": "watch"},
    {"round_id": 3, "action": "continue"},
    {"round_id": 4, "action": "rollback"},
]

selector = AnchorSelector()

result = selector.select_anchor_from_parts(
    trigger_round=4,
    action_history=action_history,
    state_record_pool=state_record_pool,
)

print(result.model_dump())