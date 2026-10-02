# Formal PoC Results

## Dataset

- 52 AAPL articles
- 21 prediction trading sessions
- 9 up / 12 non-up sessions
- common article universe for Gemini and FinBERT
- target reconstructed from adjacent observed closes
- trading-session time decay
- no-news session = neutral P(up)=0.5

## Gemini vs FinBERT

| Model | Half-life | Accuracy | F1 | ROC-AUC | AP | Brier |
|---|---:|---:|---:|---:|---:|---:|
| Gemini | 1 | 0.5238 | 0.6154 | 0.5093 | 0.4666 | 0.2599 |
| FinBERT | 1 | 0.4762 | 0.5600 | 0.4444 | 0.4891 | 0.3108 |
| Gemini | 2 | 0.4286 | 0.5714 | 0.5370 | 0.5393 | 0.2570 |
| FinBERT | 2 | 0.4762 | 0.5926 | 0.5278 | 0.5381 | 0.2855 |
| Gemini | 3 | 0.4286 | 0.5714 | 0.5278 | 0.5363 | 0.2556 |
| FinBERT | 3 | 0.4286 | 0.5714 | 0.5093 | 0.5312 | 0.2768 |
| Gemini | 5 | 0.3810 | 0.5517 | 0.5463 | 0.5471 | 0.2541 |
| FinBERT | 5 | 0.4286 | 0.6000 | 0.5185 | 0.5285 | 0.2690 |
| Gemini | 10 | 0.4286 | 0.6000 | 0.5185 | 0.5508 | 0.2525 |
| FinBERT | 10 | 0.4286 | 0.6000 | 0.5185 | 0.5270 | 0.2612 |

## Baseline interpretation

- neutral / fixed non-up accuracy: 0.5714
- Gemini same-day accuracy: 0.5714
- Gemini latest-carry reached 0.6667, but it intentionally carries stale signal and is not treated as evidence that old news remains valid indefinitely
- time decay did not produce a consistent accuracy improvement
- equal-blend and learned fusion did not establish a reliable incremental-news advantage in this small sample

## Paired uncertainty

Exact McNemar tests across H=1,2,3,5,10 did not provide evidence of a Gemini-vs-FinBERT accuracy difference after Holm correction.

Gemini had lower Brier scores than FinBERT across the five half-lives, but this should be treated as exploratory calibration evidence rather than a general superiority claim.

## Conclusion

The formal result is a **negative / inconclusive predictive finding** but a positive methodological result: the final pipeline is leakage-aware, PIT-aligned, paired, and reproducible.

The sample is too small for strong generalization claims. Larger historical coverage, more tickers, explicit independent temporal holdouts, and a task-specific next-day Gemini prompt are appropriate extensions.
