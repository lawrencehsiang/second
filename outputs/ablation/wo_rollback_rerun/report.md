# Ablation: SCRD w/o Rollback / Repair

Definition: keep the normal SCRD evaluator/action mapper active, but when rollback would be triggered, suppress rollback and repair, and directly finalize the current normal trajectory.

Efficiency note: early-stop samples are reused from Full SCRD because removing rollback cannot affect samples that never triggered rollback. Only rollback-triggered samples are rerun.

## Full dataset summary

| Dataset | N | Rerun | Full Acc | w/o Rollback Acc | Δ Acc | Full wins | Ablation wins | Net Full-Ablation | Avg Full Tok | Avg w/o Tok | Δ Tok |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| math | 79 | 21 | 59.49% | 56.96% | -2.53 pp | 5 | 3 | 2 | 10053.6 | 8389.0 | -1664.6 |
| gsm8k | 80 | 22 | 60.00% | 62.50% | +2.50 pp | 3 | 5 | -2 | 9381.3 | 7493.0 | -1888.4 |
| multiarith | 80 | 10 | 96.25% | 93.75% | -2.50 pp | 2 | 0 | 2 | 6411.1 | 5783.9 | -627.2 |
| OVERALL | 239 | 53 | 71.97% | 71.13% | -0.84 pp | 10 | 8 | 2 | 8609.3 | 7217.1 | -1392.3 |

## Rollback-subset summary

| Dataset | Rollback N | Full Acc on rollback subset | w/o Rollback Acc | Full Tok | w/o Tok |
|---|---:|---:|---:|---:|---:|
| math | 21 | 38.10% | 28.57% | 16890.1 | 10628.0 |
| gsm8k | 22 | 36.36% | 45.45% | 15525.9 | 8659.1 |
| multiarith | 10 | 90.00% | 70.00% | 11477.8 | 6460.3 |
| OVERALL | 53 | 47.17% | 43.40% | 15302.6 | 9024.4 |

## Interpretation guide

- `Full wins`: Full SCRD correct, w/o rollback wrong.
- `Ablation wins`: Full SCRD wrong, w/o rollback correct.
- Positive `Net Full-Ablation` means rollback/repair helps on net.
- Rollback-subset metrics are the most important numbers for this ablation.