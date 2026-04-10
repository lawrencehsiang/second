from src.components.rollback_controller import RollbackController
from src.components.state_store import StateStore
from src.schemas import StateRecord

store = StateStore()

# mock some state records so that rounds exist
store.add_state_record(
    StateRecord(
        round_id=1,
        current_answers=["A", "B", "A"],
        newly_added_claims=[],
        unresolved_conflicts=[],
        key_raw_snippets=[],
    )
)
store.add_state_record(
    StateRecord(
        round_id=2,
        current_answers=["A", "A", "B"],
        newly_added_claims=[],
        unresolved_conflicts=[],
        key_raw_snippets=[],
    )
)
store.add_state_record(
    StateRecord(
        round_id=3,
        current_answers=["A", "A", "A"],
        newly_added_claims=[],
        unresolved_conflicts=[],
        key_raw_snippets=[],
    )
)

store.set_round_action(1, "continue")
store.set_round_action(2, "watch")
store.set_round_action(3, "continue")

controller = RollbackController(
    max_rollbacks=1,
    round_1_not_rollback_target=True,
)

# case 1: immediate rollback
decision_1 = controller.decide_rollback_from_store(
    current_round_id=4,
    current_round_action="rollback",
    state_store=store,
    used_rollback_count=0,
)
print("case 1:", decision_1.model_dump())

# case 2: cumulative rollback (watch -> watch)
store.set_round_action(4, "watch")
decision_2 = controller.decide_rollback(
    current_round_id=5,
    current_round_action="watch",
    previous_round_action="watch",
    has_used_rollback=False,
    state_store=store,
)
print("case 2:", decision_2.model_dump())

# case 3: no rollback
decision_3 = controller.decide_rollback(
    current_round_id=4,
    current_round_action="watch",
    previous_round_action="continue",
    has_used_rollback=False,
    state_store=store,
)
print("case 3:", decision_3.model_dump())