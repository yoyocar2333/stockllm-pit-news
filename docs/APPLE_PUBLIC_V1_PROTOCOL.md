# AAPL public-data E0–E4 pilot

This is a real-data **exploratory retrospective study**, not a confirmatory or
live-trading claim. Its settings are recorded in `experiments/apple_public_v1.json`.
No test-year performance is used to change this version's features, parameters,
threshold, text coverage, weights or source population.

## Data and timing

- Universe: all retrievable English articles with usable body text dated 2023–2025
  in the [Apple Newsroom archive](https://www.apple.com/newsroom/archive/).
- Articles include releases, updates and stories. This is a single corporate
  source, not a representative independent financial-news sample.
- Parse the displayed release date, h1 headline and unique `.pagebody-copy`
  prose blocks. Navigation, media captions and press-contact sections are excluded.
- Every article retains its canonical URL, text hash, HTTP-body hash and an
  explicit `historical_text_revision_verified=false` flag. Raw source text is
  cached locally and is not committed or included in public result archives.
- Archive pages are scanned in descending date order until the page is wholly
  earlier than the study's first year. Collection exclusions and scan hashes are
  logged. Both text conditions use the same complete-body article IDs.
- Price source: Yahoo public chart endpoint, AAPL daily raw Open/High/Low/Close/
  Volume, including a 2022 warm-up period and one extra 2026 session. Adjusted
  close is not used. Dividend and split events are retained as provenance.
- The available publication metadata is a date, not an exact timestamp.
  Treat the first usable close cutoff as the first **actual exchange session
  strictly after** that date. Even morning releases do not enter the same day.
  Validate that all price dates exactly match the XNYS exchange calendar;
  missing market dates cause a failure rather than skipping a target session.
- Forecast target is `1[raw Close(D+1)>raw Close(D)]`, using adjacent verified
  sessions. No strategy return, execution-price assumption or transaction cost
  result is reported.

## Splits and model

- Training: 2023; validation: 2024; test: 2025.
- Purge the last sample of each split when its target close is in a later year.
- Use 2023 only to fit StandardScaler and LogisticRegression. The validation
  period is reported as a diagnostic; there is no parameter search or refitting
  on it in this first version. The test is evaluated once after settings are fixed.
- LogisticRegression: C=0.1, max_iter=2000, random_state=20261012, threshold=0.5.
- All learned arms include the same nine market features available by close D:
  1-/5-session returns, deviations from 5-/20-session SMAs, 5-/20-session
  volatility, close/open return, high-low range/close and volume/20-session mean.
- Baselines: market-only logistic model, always-up, always-down, neutral 0.5,
  and probability equal to the training up-day fraction. No test-majority tuning.

## Text and event conditions

| Arm | Sentiment | Event features | News aggregation |
|---|---|---|---|
| E0 | Title | None | Same-day equal mean |
| E1 | Title + extracted body | None | Same-day equal mean |
| E2 | Title | Structured roles/types/triple hashes | Same-day equal mean |
| E3 | Title + extracted body | Same event schema | Same-day equal mean |
| E4 | Title + extracted body | Same event schema | Fixed relevance × importance × session decay |

FinBERT uses `ProsusAI/finbert` at exact revision
`0bc2ac10dda6929bcb1f77ccc43c7ec5c424f031`. Positive/negative/neutral probabilities
are input features, not asserted stock-direction probabilities.
Tokenize all extracted text, split into nonoverlapping 510-token chunks plus
CLS/SEP, and pool probabilities by original token count. No content tokens are
truncated. Independent chunks do not provide unrestricted cross-document context.
Record input token counts and chunks per article/source; archive reusable chunk
probabilities and the input/model hashes for resumable inference.

Events use the existing English spaCy syntactic baseline and fixed keyword
taxonomy. The full condition parses title and body separately. Event features:
six type fractions, five target-role fractions, event-presence and log-count,
plus deterministic signed SHA256 hashes of actor/action/object phrases with
8/16/16 dimensions. Hashing does not fit a future vocabulary. Counts are
normalized within each article; event mention duplicates are not merged.
There are 53 event dimensions. Corporate aliases, negation, modality and
coreference are not fully modeled.

All news arms include the current article indicator and log article count.
No-news sentiment is neutral [positive=0, negative=0, neutral=1]; event features
are zero. E0–E3 do not carry forward yesterday's article signals.

E4 relevance is `0.25+0.75*fraction(target_role in actor/object/both)`.
Importance is the mean of fixed type weights: earnings/regulation/litigation/
M&A=2, product_launch=1.5, other=1. Article weight is relevance × importance.
Use half-life 3 trading sessions, retain ages 0–19 sessions, and include a
neutral prior weight of 1. At cutoff D:

`feature(D) = neutral + sum(w_i*2^(-age_i/3)*(feature_i-neutral)) / (1+sum(w_i*2^(-age_i/3)))`

The neutral prior ensures old isolated signals decay. This weighted condition
jointly tests fixed relevance, importance and temporal memory. It is not a HAN.

## Evaluation and interpretation

Report identical session populations for all arms: Accuracy, F1, ROC-AUC,
Average Precision and Brier. Primary contrasts are E1–E0, E2–E0, E3–E1 and E4–E3.
Secondary contrasts compare E0–E4 with Market. Use paired 5-session moving-block
bootstrap, 2,000 draws, seed 20261012, for 95% percentile intervals of accuracy
and Brier differences. Also report exploratory blockwise sign-permutation
p-values and Holm adjustments within each prespecified comparison family.
These approximations depend on block/symmetry assumptions and are not a proof
of general forecasting ability.

This pilot has no independent human event-extraction gold set. Inference coverage
is audited, but extraction accuracy must not be inferred from the event counts
or the four synthetic example scores. A frozen 100–200 item real-news blind
annotation study is a separate prerequisite for a confirmatory extraction claim.

Historical web bodies may have changed since their displayed release dates.
Current pretrained models may also contain historical information. The date
alignment prevents obvious same-day and label leakage, but cannot independently
exclude historical text revision or pretrained-model contamination. Conclusions
must stay limited to this source, period and frozen retrospective specification.

## Run locally or in Colab

From this repository root (Python 3.12):

```bash
python -m pip install -r requirements-research.txt
python -m spacy download en_core_web_sm
python -m experiments.collect_apple_newsroom
python -m experiments.infer_apple_public --threads 8 --batch-size 8
python -m experiments.evaluate_apple_public
python -m experiments.report_apple_public
python -m pytest -q
```

Windows: use `.\.venv\Scripts\python.exe` instead of `python` after creating
`py -3.12 -m venv .venv`. Colab: run shell commands with `!` and use `%pip` for
installation. This pilot uses CPU; no Gemini or other paid model API is required.
Its full body corpus takes longer than the four-headline demonstration.

Outputs are in `outputs/apple_public_v1/`: article/market provenance, collection
audit, article features, model revision, chunk cache, inference audit, daily
features, fitted coefficients, per-session predictions, metrics, paired
contrasts and an evaluation audit. Raw texts/HTTP caches are in ignored
`data/raw/apple_public_v1/`. Cache a completed run; do not overwrite manually
reviewed labels or change this experiment version after seeing its test results.
