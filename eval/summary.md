# Meridian Assistant — Eval Results

_Generated: 2026-06-12T15:54:19+00:00_

## 1. Retrieval

| Metric | Value |
|---|---|
| cases_evaluated | 29 |
| hit_at_3 | 0.8966 |
| hit_at_5 | 1.0 |
| recall_at_3 | 0.8966 |
| recall_at_5 | 1.0 |
| mrr | 0.8822 |

## 2. Action correctness

| Metric | Value |
|---|---|
| cases | 30 |
| cases_evaluated | 30 |
| cases_errored | 0 |
| action_accuracy | 1.0 |
| action_accuracy_on_evaluated | 1.0 |
| handoff_precision | 1.0 |
| handoff_recall | 0.8889 |
| handoff_f1 | 0.9412 |

## 3. LLM judge (answer quality)

| Metric | Value |
|---|---|
| cases_evaluated | 22 |
| grounded_mean | 0.9773 |
| correct_mean | 0.9773 |
| cited_mean | 1.0 |

## How to interpret

- **Hit@k** = share of cases where a relevant source was in the top-k retrieved chunks. Target ≥ 0.85.
- **MRR** = mean reciprocal rank of the first relevant source. Target ≥ 0.75.
- **action_accuracy_on_evaluated** = accuracy excluding rate-limited/errored cases. Target ≥ 0.85.
- **handoff_f1** is the safety-critical metric; low precision = over-handoff, low recall = missed handoffs. Target ≥ 0.90.
- **grounded / correct / cited** are judge scores 0-1. Targets ≥ 0.80.

Model: xiaomi/mimo-v2.5 (selected by benchmarking).
Judge: openai/gpt-4o-mini via OpenRouter.
