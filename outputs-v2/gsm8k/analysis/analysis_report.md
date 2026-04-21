# SCRD Experiment Analysis Report

- Samples analyzed: **200**
- Single Agent accuracy: **0.5650**
- Round-1 Majority Vote accuracy: **0.5700**
- Round-3 Majority Vote accuracy (subset n=72): **0.6667**
- SCRD Final accuracy: **0.7000**

## Cost Overview

- Mean Single-Agent tokens: **315.72**
- Mean Round-1 Vote tokens: **947.10**
- Mean Round-3 Vote tokens (cumulative to round 3, subset): **10470.58**
- Mean SCRD Final tokens: **8198.33**

## Efficiency

- Single-Agent accuracy per 1k tokens: **1.7896**
- Round-1 Vote accuracy per 1k tokens: **0.6018**
- Round-3 Vote accuracy per 1k tokens (subset): **0.0637**
- SCRD Final accuracy per 1k tokens: **0.0854**

## Output Files

- `sample_overview.csv`: readable per-sample summary
- `method_comparison.csv`: direct method-level comparison
- `stop_reason_summary.csv`: grouped by stop reason
- `outcome_vs_round1_vote.csv`: how SCRD changes outcomes relative to round-1 vote
- `outcome_vs_round3_vote.csv`: same, but relative to round-3 vote
- `round3_vote_samples.csv`: only samples with third-round vote available
- `trace_debug_details.csv`: trace-heavy debug fields kept separate from the readable sample table
