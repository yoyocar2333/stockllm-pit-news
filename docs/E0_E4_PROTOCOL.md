# StockLLM V2: E0–E4 preregistration draft

This is an **experimental plan**, NOT a claim of a successful new backtest.

| Arm | Sentiment text | Extra event features | News aggregation |
|---|---|---|---|
| E0 | Title | None | Daily equal mean |
| E1 | Title + body | None | Daily equal mean |
| E2 | Title | Actor/action/object, event type, ticker role | Daily equal mean |
| E3 | Title + body | Same event schema from full text | Daily equal mean |
| E4 | Title + body | Same as E3 | Prespecified relevance × importance × time-decay weighting |

Primary contrasts: E1–E0 (body), E2–E0 (title roles), E3–E1 (full-body roles), E4–E3 (weighting). E4 is **not** the learned HAN model from Hu et al.

## Evaluation controls

- Freeze identical article IDs, available title **and** body, evaluation sessions, and exact target definition across all arms.
- Keep original 52-article PoC unchanged. Use new historical data in a separate experiment version.
- Use PIT: publication time, time zone, exchange calendar, next-session cutoff, and no future price/text labels in features.
- Chronological train / validation / sealed test; all parameters tuned without test data.
- Fix downstream model, market OHLCV features, and data universe when changing one treatment.
- Treat FinBERT sentiment-derived P(up) as an **uncalibrated proxy**, not a proven forecast probability.
- Log tokenizer input truncation length for the full-body FinBERT condition.
- Annotate at least 100–200 items and assess actor/object/event-type extraction **independently** from stock accuracy.
- Compare Accuracy, F1, ROC-AUC, AP, Brier and the majority-class/market-only baselines. Report paired block-bootstrap intervals.
- For zero-news days use neutral value plus an explicit no-news indicator; don't carry stale news indefinitely.

## Prior literature

- Cortis et al. SemEval-2017 Task 5 (target-specific continuous financial sentiment).
- Ding et al. IJCAI 2015 (structured events and event embeddings).
- Hu et al. WSDM 2018 (news-level and temporal attention; self-paced learning).
