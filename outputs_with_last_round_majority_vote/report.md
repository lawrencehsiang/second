# Outputs Recomputed with Last-Round Majority Vote

This report rewrites the Full SCRD outputs by replacing the original trajectory decision head with majority vote over the last effective state's `current_answers`. No LLM calls are made.

| Dataset | N | Decision Head Acc | Last-Round MV Acc | Δ Acc | DH wins | MV wins | Net MV-DH | Changed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| addsub | 80 | 100.00% | 100.00% | 0.00 pp | 0 | 0 | 0 | 0 |
| asdiv | 80 | 92.50% | 92.50% | 0.00 pp | 0 | 0 | 0 | 3 |
| gsm8k | 80 | 60.00% | 72.50% | 12.50 pp | 1 | 11 | 10 | 17 |
| math | 80 | 58.75% | 63.75% | 5.00 pp | 2 | 6 | 4 | 19 |
| multiarith | 80 | 96.25% | 93.75% | -2.50 pp | 2 | 0 | -2 | 4 |
| singleeq | 80 | 96.25% | 97.50% | 1.25 pp | 1 | 2 | 1 | 3 |
| svamp | 80 | 82.50% | 86.25% | 3.75 pp | 1 | 4 | 3 | 9 |
| OVERALL | 560 | 83.75% | 86.61% | 2.86 pp | 7 | 23 | 16 | 55 |