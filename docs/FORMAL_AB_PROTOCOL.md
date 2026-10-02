# FinBERT vs Gemini StockLLM — Formal A/B Protocol

## Why the old FinBERT notebook result is not formal

The earlier prototype created synthetic sentiment for missing dates using `future_return = Close.pct_change().shift(-1)`. Because that value contains future price movement, it leaks the answer into the input feature. Those numbers are prototype-only and are excluded from the formal comparison.

## Formal comparison

Both models consume the exact same article rows. FinBERT is run directly on the article-level Gemini output file, so `article_id`, timestamp, title, content, and PIT `information_date` are identical.

FinBERT's three probabilities are converted to an up-direction proxy as:

`P(up) = P(positive) + 0.5 * P(neutral)`

No future returns, target labels, fitted calibration parameters, or synthetic fills are used.

## Target

For market cutoff session D:

`Target(D) = 1[Close(D+1) > Close(D)]`

The old 10-session / 2% target is not used.

## PIT and holidays

The initial information-date rule shifts after-close news and weekends. During the A/B merge, any date that is not an observed market session is conservatively moved to the first market session on or after that date. This is done identically for Gemini and FinBERT.

## Time decay

Decay is measured in trading sessions. Articles on the same information date are first averaged into a daily probability P_t. A no-news session is neutral: P_t = 0.5.

Define `s_t = 2 P_t - 1`. For half-life H:

`w_k = 2^(-k/H)`

`S_D = sum_k w_k s_(D-k) / sum_k w_k`

`P_D = 0.5 + 0.5 S_D`

This daily-neutral formulation makes an isolated stale signal decay back toward 0.5.

Formal settings: Latest, H1, H2, H3, H5, H10.

## Evaluation population

The default comparison is restricted to cutoff dates within the observed common article span. This prevents incomplete news collection from causing old January news to be treated as current information months later.

Both models are hard-checked to have identical `(cutoff_date, target_date, target)` rows for every setting.

## Metrics

- Accuracy
- Precision
- Recall
- F1
- ROC-AUC
- Average Precision
- Brier score

The classification threshold is fixed ex ante at `P(up) >= 0.5`; it is never tuned on the test result.
