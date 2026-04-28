# Ablation: w/o Rollback / Repair

Policy: rollback is disabled. The evaluator and action mapper are kept. Degraded transitions map to early_stop; otherwise the debate may continue until early_stop or max_round.

Only samples whose reference Full SCRD run stopped by rollback are rerun. All other samples are reused from the reference outputs.

| Dataset | N | Rerun | Reference Acc | w/o Rollback Acc | Δ Acc | Full wins | Ablation wins | Net Full-Ablation | Ref tokens | w/o tokens |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| math | 80 | 22 | 63.75% | 62.50% | -1.25 pp | 2 | 1 | 1 | 10078.2 | 7883.2 |
| gsm8k | 80 | 22 | 72.50% | 63.75% | -8.75 pp | 8 | 1 | 7 | 9381.3 | 7720.8 |
| multiarith | 80 | 10 | 93.75% | 93.75% | 0.00 pp | 1 | 1 | 0 | 6411.1 | 5834.3 |
| OVERALL | 240 | 54 | 76.67% | 73.33% | -3.33 pp | 11 | 3 | 8 | 8623.5 | 7146.1 |