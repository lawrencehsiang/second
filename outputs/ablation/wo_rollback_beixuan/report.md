# Ablation: SCRD w/o Rollback / Repair

Definition: keep the normal SCRD evaluator/action mapper active, but when rollback would be triggered, suppress rollback and repair, and directly finalize the current normal trajectory.

Efficiency note: early-stop samples are reused from Full SCRD because removing rollback cannot affect samples that never triggered rollback. Only rollback-triggered samples are rerun.

## Full dataset summary

| Dataset | N | Rerun | Full Acc | w/o Rollback Acc | Δ Acc | Full wins | Ablation wins | Net Full-Ablation | Avg Full Tok | Avg w/o Tok | Δ Tok |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| multiarith | 200 | 31 | 89.00% | 87.50% | -1.50 pp | 3 | 0 | 3 | 8140.9 | 6968.3 | -1172.6 |
| OVERALL | 200 | 31 | 89.00% | 87.50% | -1.50 pp | 3 | 0 | 3 | 8140.9 | 6968.3 | -1172.6 |

## Rollback-subset summary

| Dataset | Rollback N | Full Acc on rollback subset | w/o Rollback Acc | Full Tok | w/o Tok |
|---|---:|---:|---:|---:|---:|
| multiarith | 31 | 80.65% | 70.97% | 14387.9 | 6822.9 |
| OVERALL | 31 | 80.65% | 70.97% | 14387.9 | 6822.9 |

## Interpretation guide

- `Full wins`: Full SCRD correct, w/o rollback wrong.
- `Ablation wins`: Full SCRD wrong, w/o rollback correct.
- Positive `Net Full-Ablation` means rollback/repair helps on net.
- Rollback-subset metrics are the most important numbers for this ablation.