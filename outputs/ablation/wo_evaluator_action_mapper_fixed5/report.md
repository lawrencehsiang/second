# Ablation: w/o Evaluator & Action Mapper

Definition: run exactly 5 normal SCRD rounds with evaluator, action mapper, early stop, rollback, and repair disabled. Structured history filtering and recorder are preserved. Final answer is selected by last-effective-round majority vote.

| Dataset | N | Full Acc | Fixed-5 Acc | Δ Acc | Full wins | Ablation wins | Net Full-Ablation | Full tok | Fixed-5 tok |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| math | 198 | 55.56% | 48.48% | -7.07 pp | 34 | 20 | 14 | 11469.2 | 17768.4 |
| gsm8k | 200 | 57.00% | 47.00% | -10.00 pp | 46 | 26 | 20 | 11896.2 | 17828.4 |
| multiarith | 200 | 89.00% | 86.00% | -3.00 pp | 20 | 14 | 6 | 8140.9 | 16324.6 |
| OVERALL | 598 | 67.22% | 60.54% | -6.69 pp | 100 | 60 | 40 | 10498.9 | 17305.6 |

## Notes

- This ablation must rerun all samples because early-stop samples in Full SCRD now continue to fixed max rounds.
- Token cost is expected to increase because evaluator/action control no longer stops easy cases early.
- If accuracy drops or token cost increases substantially, it supports the value of dynamic control.