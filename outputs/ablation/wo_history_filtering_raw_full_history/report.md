# Ablation: w/o History Filtering / Raw Full History

Definition: preserve the internal State Recorder, evaluator, action mapper, rollback/repair, and last-round majority finalizer. Replace the normal filtered HistoryManager with an unfiltered raw full-history manager. Agents receive raw previous-round state history instead of selected top-k structured history units.

| Dataset | N | Full Acc | Raw-History Acc | Δ Acc | Full wins | Ablation wins | Net Full-Ablation | Full tok | Raw-History tok |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| math | 195 | 56.41% | 49.74% | -6.67 pp | 30 | 17 | 13 | 11385.5 | 13717.8 |
| gsm8k | 200 | 57.00% | 55.50% | -1.50 pp | 29 | 26 | 3 | 11896.2 | 12187.9 |
| multiarith | 200 | 89.00% | 84.50% | -4.50 pp | 22 | 13 | 9 | 8140.9 | 9820.8 |
| OVERALL | 595 | 67.56% | 63.36% | -4.20 pp | 81 | 56 | 25 | 10466.5 | 11893.7 |

## Notes

- This is not a full removal of the State Recorder. The recorder is retained internally because evaluator/action mapper and rollback require StateRecord objects.
- The ablation removes selective history filtering from the agent input.
- Default history scope is all previous rounds. Use `--history-scope last` for a cheaper last-round-only variant.