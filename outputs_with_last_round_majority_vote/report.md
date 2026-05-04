# Outputs Recomputed with Last-Round Majority Vote

This report rewrites the Full SCRD outputs by replacing the original trajectory decision head with majority vote over the last effective state's `current_answers`. No LLM calls are made.

| Dataset | N | Decision Head Acc | Last-Round MV Acc | Δ Acc | DH wins | MV wins | Net MV-DH | Changed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| addsub | 200 | 98.00% | 98.00% | 0.00 pp | 0 | 0 | 0 | 6 |
| asdiv | 200 | 86.50% | 87.00% | 0.50 pp | 0 | 1 | 1 | 17 |
| gsm8k | 200 | 53.50% | 57.00% | 3.50 pp | 12 | 19 | 7 | 59 |
| math | 198 | 52.53% | 55.56% | 3.03 pp | 9 | 15 | 6 | 49 |
| multiarith | 200 | 89.00% | 89.00% | 0.00 pp | 7 | 7 | 0 | 16 |
| singleeq | 200 | 97.00% | 96.50% | -0.50 pp | 5 | 4 | -1 | 9 |
| svamp | 200 | 86.00% | 87.50% | 1.50 pp | 6 | 9 | 3 | 25 |
| OVERALL | 1398 | 80.40% | 81.55% | 1.14 pp | 39 | 55 | 16 | 181 |