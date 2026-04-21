# SCRD Experiment Analysis Report
- Samples analyzed: **200**
- Single Agent accuracy: **0.5650**
- Majority Vote accuracy: **0.5700**
- SCRD accuracy: **0.7000**
## Cost Overview
- Mean single-agent tokens: **315.72**
- Mean majority-vote tokens: **947.10**
- Mean SCRD tokens: **8198.33**
- Mean SCRD prompt tokens: **7181.16**
- Mean SCRD completion tokens: **1017.16**
- SCRD / Majority token ratio: **8.66x**
- SCRD / Single token ratio: **25.97x**
## Efficiency
- Single Agent accuracy per 1k tokens: **1.7896**
- Majority Vote accuracy per 1k tokens: **0.6018**
- SCRD accuracy per 1k tokens: **0.0854**
## Main Takeaways Template
- Check whether SCRD improves over majority vote often enough to justify its extra tokens.
- Check which stop reasons are associated with the highest cost.
- Check whether recorder/evaluator dominate the prompt cost.
- Check whether degraded_from_majority cases are especially expensive.
