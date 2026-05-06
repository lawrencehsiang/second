# Ablation: w/o Rollback / Repair

Policy: rollback is disabled. The evaluator and action mapper are kept. Degraded transitions map to early_stop; otherwise the debate may continue until early_stop or max_round.

Reference Full SCRD is evaluated using last-round majority voting `last_round_majority_correct`, not decision-head `scrd_correct`.

Only samples whose reference Full SCRD run stopped by rollback are rerun. All other samples are reused from the reference outputs.

| Dataset | N | Rerun | Reference Acc | w/o Rollback Acc | Δ Acc | Full wins | Ablation wins | Net Full-Ablation | Ref tokens | w/o tokens | Fallback |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| math | 198 | 76 | 55.56% | 52.53% | -3.03 pp | 12 | 6 | 6 | 11469.2 | 8283.1 | 0 |
| gsm8k | 200 | 88 | 57.00% | 53.00% | -4.00 pp | 16 | 8 | 8 | 11896.2 | 8091.8 | 0 |
| multiarith | 200 | 31 | 89.00% | 88.00% | -1.00 pp | 5 | 3 | 2 | 8140.9 | 7228.7 | 0 |
| OVERALL | 598 | 195 | 67.22% | 64.55% | -2.68 pp | 33 | 17 | 16 | 10498.9 | 7866.5 | 0 |