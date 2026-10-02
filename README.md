# StockLLM — Point-in-Time Financial News Evaluation

News / unstructured-data branch of my undergraduate research project on financial time-series prediction.

This repository studies whether financial news adds **incremental next-trading-day direction signal** beyond simple baselines, with strict controls for look-ahead leakage.

Companion structured market-model repository: https://github.com/yoyocar2333/stock1

## Research question

Can semantic news signals extracted by Gemini or FinBERT improve next-day stock-direction prediction when:

- both models see the **same article universe**;
- news is aligned by actual information availability;
- after-close / non-trading-day news is shifted conservatively;
- target labels are rebuilt from adjacent observed trading sessions;
- stale news is decayed through trading sessions;
- simple baselines are reported alongside complex models?

## Formal PoC dataset

The formal experiment used:

- 52 real AAPL news articles
- 2025-01-02 to 2025-02-03
- 21 prediction sessions
- 9 up / 12 non-up target days

The raw news corpus and Gemini cache are intentionally **not committed**. This avoids redistributing third-party article text and prevents API/cache artifacts from being mistaken for source code.

## Models

- Gemini structured semantic extraction
- ProsusAI/FinBERT
- fixed keyword baseline
- neutral / always-up / always-down references
- market-only and fusion experiments in the research report

## Core protocol

### Point-in-Time (PIT)

News is assigned to the earliest session on which it could have been known. After-close news is shifted forward. Holiday handling is checked again against observed market sessions during evaluation.

### Target

For market cutoff session D:

`Target(D) = 1[Close(D+1) > Close(D)]`

The older Phase-7A 10-session / 2% target is **not** used in this A/B test.

### Time decay

No-news days are neutral:

`P_t = 0.5`

With signal `s_t = 2P_t - 1` and half-life H:

`w_k = 2^(-k/H)`

Formal settings: Latest, H1, H2, H3, H5, H10.

## Key result

On this small PoC sample, there is **no reliable evidence that Gemini, FinBERT, or a more complex fusion model consistently beats simple baselines**.

For H=1:

| Model | Accuracy | F1 | ROC-AUC | AP | Brier |
|---|---:|---:|---:|---:|---:|
| Gemini | 52.38% | 0.6154 | 0.5093 | 0.4666 | 0.2599 |
| FinBERT | 47.62% | 0.5600 | 0.4444 | 0.4891 | 0.3108 |

A fixed non-up baseline reaches 57.14% accuracy. Paired exact McNemar tests across the five half-lives do not provide evidence of an accuracy difference after Holm correction.

This negative result is intentional to preserve: the research contribution is the **causally valid, reproducible evaluation pipeline**, not a claim that an LLM can already predict the market reliably.

See `RESULTS.md` for the concise result summary and `docs/FORMAL_AB_PROTOCOL.md` for the exact A/B design.

## Leakage fix

An early exploratory notebook generated missing sentiment features from `future_return`. That leaks future price movement into the model input and invalidates those prototype results.

The formal pipeline in this repository removes that approach completely:

- FinBERT reads text only;
- no synthetic sentiment fill;
- no future-return calibration;
- Gemini and FinBERT use identical article IDs;
- classification threshold is fixed ex ante;
- tests check target construction, decay behavior, and paired evaluation population.

## Public repository structure

```
data/
  align.py
  sample_news.csv
docs/
  FORMAL_AB_PROTOCOL.md
experiments/
  run_gemini_same_news.py
  run_finbert_same_news.py
  run_keyword_only.py
  compare_finbert_gemini_nextday_decay.py
llm/
  client.py
  schema.py
prompts/
  llm_only.txt
tests/
  test_finbert_gemini_ab.py
RESULTS.md
requirements.txt
```

## Reproduction outline

1. Prepare an article CSV with `article_id, ticker, timestamp, title, content`.
2. Set `GEMINI_API_KEY` and `GEMINI_MODEL`.
3. Run Gemini structured extraction.
4. Run FinBERT on the resulting exact article universe.
5. Provide a market CSV containing at least `Date` and `Close`.
6. Run the strict paired A/B evaluator.

Example:

```powershell
python -m experiments.run_gemini_same_news --input data/sample_news.csv --output outputs/gemini_articles.csv --ticker AAPL
python -m experiments.run_finbert_same_news --input outputs/gemini_articles.csv --output outputs/gemini_finbert_articles.csv --ticker AAPL
python -m experiments.compare_finbert_gemini_nextday_decay --articles outputs/gemini_finbert_articles.csv --market path/to/market.csv --output outputs/metrics.csv --ticker AAPL --half-lives 1 2 3 5 10
pytest -q
```

## Historical-model note

The recovered historical cache recorded a Gemini model key, but the original service metadata was not sufficient to independently prove the exact server-side model revision or training-data cutoff. New runs should therefore record `GEMINI_MODEL` explicitly and should not be expected to reproduce identical probabilities across model revisions.

## Scope

This is a research PoC, not trading advice.
