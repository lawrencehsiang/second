# Ablation: w/o History Filtering / Raw Full History

Definition: preserve the internal State Recorder, evaluator, action mapper, rollback/repair, and last-round majority finalizer. Replace the normal filtered HistoryManager with an unfiltered raw full-history manager. Agents receive raw previous-round state history instead of selected top-k structured history units.

| Dataset | N | Full Acc | Raw-History Acc | Δ Acc | Full wins | Ablation wins | Net Full-Ablation | Full tok | Raw-History tok |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| math | 79 | 63.29% | 48.10% | -15.19 pp | 17 | 5 | 12 | 10128.0 | 13309.9 |
| gsm8k | 80 | 72.50% | 57.50% | -15.00 pp | 16 | 4 | 12 | 9381.3 | 11691.1 |
| multiarith | 80 | 93.75% | 80.00% | -13.75 pp | 13 | 2 | 11 | 6411.1 | 10008.3 |
| OVERALL | 239 | 76.57% | 61.92% | -14.64 pp | 46 | 11 | 35 | 8633.9 | 11662.9 |

## Notes

- This is not a full removal of the State Recorder. The recorder is retained internally because evaluator/action mapper and rollback require StateRecord objects.
- The ablation removes selective history filtering from the agent input.
- Default history scope is all previous rounds. Use `--history-scope last` for a cheaper last-round-only variant.