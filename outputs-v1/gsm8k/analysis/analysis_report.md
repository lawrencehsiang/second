# SCRD Experiment Analysis Report
- Samples analyzed: **291**
- Single Agent accuracy: **0.4433**
- Majority Vote accuracy: **0.4639**
- SCRD accuracy: **0.6186**
## Cost Overview
- Mean single-agent tokens: **299.64**
- Mean majority-vote tokens: **904.44**
- Mean SCRD tokens: **10845.44**
- Mean SCRD prompt tokens: **9394.13**
- Mean SCRD completion tokens: **1451.31**
- SCRD / Majority token ratio: **11.99x**
- SCRD / Single token ratio: **36.19x**
## Efficiency
- Single Agent accuracy per 1k tokens: **1.4794**
- Majority Vote accuracy per 1k tokens: **0.5129**
- SCRD accuracy per 1k tokens: **0.0570**
## Main Takeaways Template
- Check whether SCRD improves over majority vote often enough to justify its extra tokens.
- Check which stop reasons are associated with the highest cost.
- Check whether recorder/evaluator dominate the prompt cost.
- Check whether degraded_from_majority cases are especially expensive.
