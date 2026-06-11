# Meridian Assistant — Eval Results

_Generated: 2026-06-11T15:51:13+00:00_

## 1. Retrieval

| Metric | Value |
|---|---|
| cases_evaluated | 13 |
| hit_at_3 | 0.8462 |
| hit_at_5 | 0.8462 |
| recall_at_3 | 0.8462 |
| recall_at_5 | 0.8462 |
| mrr | 0.8485 |

## 2. Action correctness

| Metric | Value |
|---|---|
| cases | 30 |
| cases_evaluated | 28 |
| cases_errored | 2 |
| action_accuracy | 0.9333 |
| action_accuracy_on_evaluated | 1.0 |
| handoff_precision | 1.0 |
| handoff_recall | 1.0 |
| handoff_f1 | 1.0 |

### By category

| Category | n | action_accuracy |
|---|---|---|
| booking_new | 2 | 1.0 |
| commercial | 1 | 1.0 |
| complaint | 1 | 1.0 |
| edge_commercial | 1 | 1.0 |
| edge_electrical_outside_area | 1 | 1.0 |
| edge_gas_emergency | 1 | 1.0 |
| edge_missing_zip | 1 | 1.0 |
| edge_out_of_area_zip | 1 | 0.0 |
| edge_partial_zip | 1 | 1.0 |
| edge_repeated_missing_info | 1 | 1.0 |
| edge_sunday_booking | 1 | 1.0 |
| edge_umd_campus | 1 | 0.0 |
| edge_warranty_booking | 1 | 1.0 |
| emergency | 1 | 1.0 |
| faq_hours | 1 | 1.0 |
| fee_dispute | 1 | 1.0 |
| out_of_area | 1 | 1.0 |
| payment | 1 | 1.0 |
| preferred_tech | 1 | 1.0 |
| pricing | 1 | 1.0 |
| pricing_estimate | 1 | 1.0 |
| pricing_plan | 1 | 1.0 |
| pricing_surcharge | 1 | 1.0 |
| reschedule | 1 | 1.0 |
| reschedule_same_day | 1 | 1.0 |
| service_area | 1 | 1.0 |
| status_check | 2 | 1.0 |
| warranty | 1 | 1.0 |

### Observed action distribution

- `answer`: 20
- `handoff`: 8
- `error`: 2

## 3. LLM judge (answer quality)

| Metric | Value |
|---|---|
| cases_evaluated | 20 |
| grounded_mean | 1.0 |
| correct_mean | 0.975 |
| cited_mean | 0.85 |

## How to interpret

- **Hit@k** = share of cases where a relevant source was in the top-k retrieved chunks. Target ≥ 0.85.
- **MRR** = mean reciprocal rank of the first relevant source. Target ≥ 0.75.
- **action_accuracy_on_evaluated** = accuracy excluding rate-limited/errored cases. Target ≥ 0.85.
- **handoff_f1** is the safety-critical metric; low precision = over-handoff, low recall = missed handoffs. Target ≥ 0.90.
- **grounded / correct / cited** are judge scores 0-1. Targets ≥ 0.80.
- Run with `--no-gate` to collect results without failing the build.

## Known limitations

- LLM judge uses `openai/gpt-4o-mini` (stronger than agent model) to reduce self-grading bias.
- 30 cases is a smoke-test set. Statistically meaningful conclusions need 200+.
- Action eval hits the live LLM; flaky under network or model-rate-limit pressure.
- Judge retrieves fresh context at eval time, which may differ from what the agent actually saw.
- Handoff cases are skipped in judge scoring (covered by action eval `handoff_f1`).
