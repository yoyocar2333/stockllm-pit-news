# Actual local verification

2026-10-12, Windows, Python 3.12.14. Entire repository: 27 tests passed.
New tests cover date-only timing, exchange-session gaps, split-boundary label
purging, future-news/future-price invariance, finite memory decay, stable event
hashing and body extraction excluding press contacts/navigation.

450 real English articles, 795 actual daily OHLCV rows, 900 text feature rows,
1,809 FinBERT chunks. FinBERT revision is in model_registry.json. There was no
price input to text inference and no synthetic sentiment/future-return fill.
All observed market dates exactly matched XNYS sessions. Identical article IDs
and date/target populations were checked across arms.

The local resumable inference path was verified: the second run had zero uncached
chunks and produced an identical SHA256 for article_features.csv. The evaluation
was run once with frozen parameters. The new Colab notebook was structurally and
syntax checked locally; it was not executed in a hosted Colab runtime.

Independent human extraction annotations and historical source-text revisions
remain unverified. These checks do not establish a confirmatory forecast result.
