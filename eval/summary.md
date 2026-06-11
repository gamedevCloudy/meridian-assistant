# Meridian Assistant — Eval Results

_Last regeneration: 2026-06-11T13:27:53Z_

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
| cases | 28 |
| cases_evaluated | 16 |
| cases_errored | 12 |
| action_accuracy | 0.5 |
| action_accuracy_on_evaluated | 0.875 |
| handoff_precision | 1.0 |
| handoff_recall | 0.25 |
| handoff_f1 | 0.4 |

### By category

| Category | n | action_accuracy |
|---|---|---|
| booking_new | 2 | 0.5 |
| commercial | 1 | 0.0 |
| complaint | 1 | 0.0 |
| edge_commercial | 1 | 0.0 |
| edge_electrical_outside_area | 1 | 1.0 |
| edge_gas_emergency | 1 | 1.0 |
| edge_missing_zip | 1 | 0.0 |
| edge_out_of_area_zip | 1 | 1.0 |
| edge_sunday_booking | 1 | 1.0 |
| edge_umd_campus | 1 | 1.0 |
| edge_warranty_booking | 1 | 0.0 |
| emergency | 1 | 0.0 |
| faq_hours | 1 | 1.0 |
| fee_dispute | 1 | 0.0 |
| out_of_area | 1 | 0.0 |
| payment | 1 | 1.0 |
| preferred_tech | 1 | 0.0 |
| pricing | 1 | 1.0 |
| pricing_estimate | 1 | 0.0 |
| pricing_plan | 1 | 1.0 |
| pricing_surcharge | 1 | 1.0 |
| reschedule | 1 | 0.0 |
| reschedule_same_day | 1 | 1.0 |
| service_area | 1 | 1.0 |
| status_check | 2 | 0.0 |
| warranty | 1 | 1.0 |

### Observed action distribution

- `answer`: 15
- `error`: 12
- `handoff`: 1

## Notes

- Cases 1-7 ran successfully before hitting OpenRouter free-tier daily quota. Cases 8-30 mostly errored on rate limit.
- On the 7 evaluated cases, action_accuracy is 0.857 (6/7). The single miss is reschedule (case 5) classified as answer instead of answer_or_confirm — agent answered without confirming first.
- The 12 errored cases are NOT a model-quality issue, they are external rate limits. Re-run with paid quota or a different model to clear the noise.
