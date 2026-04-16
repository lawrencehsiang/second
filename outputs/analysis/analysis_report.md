# SCRD Experiment Analysis Report
- Samples analyzed: **96**
- Single Agent accuracy: **0.3125**
- Majority Vote accuracy: **0.3125**
- SCRD accuracy: **0.4688**
## Cost Overview
- Mean single-agent tokens: **267.92**
- Mean majority-vote tokens: **804.58**
- Mean SCRD tokens: **12434.95**
- Mean SCRD prompt tokens: **10741.93**
- Mean SCRD completion tokens: **1693.02**
- SCRD / Majority token ratio: **15.46x**
- SCRD / Single token ratio: **46.41x**
## Efficiency
- Single Agent accuracy per 1k tokens: **1.1664**
- Majority Vote accuracy per 1k tokens: **0.3884**
- SCRD accuracy per 1k tokens: **0.0377**
## Main Takeaways Template
- Check whether SCRD improves over majority vote often enough to justify its extra tokens.
- Check which stop reasons are associated with the highest cost.
- Check whether recorder/evaluator dominate the prompt cost.
- Check whether degraded_from_majority cases are especially expensive.
