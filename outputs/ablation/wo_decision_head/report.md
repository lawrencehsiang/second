# Ablation: SCRD w/o Decision Head

Definition: reuse the same Full SCRD trajectories, but replace the trajectory-aware decision head with majority vote over the last effective state's `current_answers`.

This is a post-hoc ablation. It does not call the LLM again; token cost is identical to Full SCRD.

## Summary

| Dataset | N | Full SCRD Acc | w/o Decision Head Acc | Δ Acc | Full wins | Ablation wins | Net Full-Ablation | Answer changed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| math | 80 | 58.75% | 63.75% | 5.00 pp | 2 | 6 | -4 | 19 |
| gsm8k | 80 | 60.00% | 72.50% | 12.50 pp | 1 | 11 | -10 | 17 |
| multiarith | 80 | 96.25% | 93.75% | -2.50 pp | 2 | 0 | 2 | 4 |
| OVERALL | 240 | 71.67% | 76.67% | 5.00 pp | 5 | 17 | -12 | 40 |

## Interpretation guide

- `Full wins`: Full SCRD correct, w/o Decision Head wrong.
- `Ablation wins`: Full SCRD wrong, w/o Decision Head correct.
- `Net Full-Ablation`: positive means the decision head helps on net.
- `Answer changed`: final answer differs between Full SCRD and last-round majority.