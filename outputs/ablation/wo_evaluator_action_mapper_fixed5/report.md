# Ablation: w/o Evaluator & Action Mapper

Definition: run exactly 5 normal SCRD rounds with evaluator, action mapper, early stop, rollback, and repair disabled. Structured history filtering and recorder are preserved. Final answer is selected by last-effective-round majority vote.

| Dataset | N | Full Acc | Fixed-5 Acc | Δ Acc | Full wins | Ablation wins | Net Full-Ablation | Full tok | Fixed-5 tok |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| math | 80 | 63.75% | 46.25% | -17.50 pp | 20 | 6 | 14 | 10078.2 | 17993.9 |
| gsm8k | 80 | 72.50% | 52.50% | -20.00 pp | 24 | 8 | 16 | 9381.3 | 17773.7 |
| multiarith | 80 | 93.75% | 86.25% | -7.50 pp | 9 | 3 | 6 | 6411.1 | 16588.3 |
| OVERALL | 240 | 76.67% | 61.67% | -15.00 pp | 53 | 17 | 36 | 8623.5 | 17452.0 |

## Notes

- This ablation must rerun all samples because early-stop samples in Full SCRD now continue to fixed max rounds.
- Token cost is expected to increase because evaluator/action control no longer stops easy cases early.
- If accuracy drops or token cost increases substantially, it supports the value of dynamic control.