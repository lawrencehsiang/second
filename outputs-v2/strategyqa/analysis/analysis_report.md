# SCRD Experiment Analysis Report

- Samples analyzed: **199**
- Single Agent accuracy: **0.6432**
- Round-1 Majority Vote accuracy: **0.6432**
- Round-3 Majority Vote accuracy (subset n=32): **0.6562**
- SCRD Final accuracy: **0.5829**

## Cost Overview

- Mean Single-Agent tokens: **256.41**
- Mean Round-1 Vote tokens: **770.94**
- Mean Round-3 Vote tokens (cumulative to round 3, subset): **9782.59**
- Mean SCRD Final tokens: **5903.86**

## Efficiency

- Single-Agent accuracy per 1k tokens: **2.5085**
- Round-1 Vote accuracy per 1k tokens: **0.8343**
- Round-3 Vote accuracy per 1k tokens (subset): **0.0671**
- SCRD Final accuracy per 1k tokens: **0.0987**

## Output Files

- `sample_overview.csv`: readable per-sample summary
- `method_comparison.csv`: direct method-level comparison
- `stop_reason_summary.csv`: grouped by stop reason
- `outcome_vs_round1_vote.csv`: how SCRD changes outcomes relative to round-1 vote
- `outcome_vs_round3_vote.csv`: same, but relative to round-3 vote
- `round3_vote_samples.csv`: only samples with third-round vote available
- `trace_debug_details.csv`: trace-heavy debug fields kept separate from the readable sample table
