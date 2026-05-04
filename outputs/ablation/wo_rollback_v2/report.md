# Ablation: w/o Rollback / Repair

Policy: rollback is disabled. The evaluator and action mapper are kept. Degraded transitions map to early_stop; otherwise the debate may continue until early_stop or max_round.

Only samples whose reference Full SCRD run stopped by rollback are rerun. All other samples are reused from the reference outputs.

| Dataset | N | Rerun | Reference Acc | w/o Rollback Acc | Δ Acc | Full wins | Ablation wins | Net Full-Ablation | Ref tokens | w/o tokens |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| math | 197 | 75 | 55.33% | 51.78% | -3.55 pp | 12 | 5 | 7 | 11441.8 | 8572.0 |
| gsm8k | 200 | 88 | 57.00% | 52.50% | -4.50 pp | 23 | 14 | 9 | 11896.2 | 8144.2 |
| multiarith | 200 | 31 | 89.00% | 89.50% | 0.50 pp | 5 | 6 | -1 | 8140.9 | 7036.4 |
| OVERALL | 597 | 194 | 67.17% | 64.66% | -2.51 pp | 40 | 25 | 15 | 10488.2 | 7914.2 |