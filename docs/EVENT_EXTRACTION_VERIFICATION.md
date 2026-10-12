# Event extraction verification — 2026-10-12

This records actual local checks, not results of E0–E4 stock-prediction experiments.

- Original ZIP: `stockllm_v2_event_extraction.zip`, 19,674 bytes.
- ZIP SHA256: `E1DD000B29AA712609B46F44BD5D262A21469A0F5973772B403AD25B64175765`.
- Base commit: `fa0c6055d62920da1659274da468dec0e44e5afe`.
- Environment: Windows, Python 3.12.14, pandas 3.0.6, spaCy 3.8.16,
  en_core_web_sm 3.8.0, pytest 9.1.1; optional GLiNER2 2.1.0,
  torch 2.14.1, transformers 5.19.0.

## Actual checks

1. Original package: 3 tests passed; both spaCy title/full CLIs saved 4 article rows.
2. Original full-text parsing merged unpunctuated headlines into the body,
   producing actors such as `new car prototype ... Tesla`. The recovered version
   parses the two fields separately. The sample full-text run now retains `Tesla`
   as actor for both mentions. This does not resolve every possible parsing error.
3. Fixed rule priority and word forms: `Regulators investigate Apple` now maps to
   regulation; `launched a new investigation` maps to regulation rather than product launch.
4. Recovered code: spaCy title and full, GLiNER2 title and full all saved 4 article
   rows. Both GLiNER2 source modes were also run with HF_HUB_OFFLINE=1 after weights
   had been downloaded. The backend calls local inference, without cloud API credentials.
5. Standardized GLiNER2 action spans to contextual spaCy verb lemmas when aligned;
   e.g. `sues` becomes `sue`. Both title backends matched the four synthetic
   reference examples in `event_extraction/sample_gold_title.csv` on all five fields.
   This is only a functional demonstration, not independent human validation or
   evidence of general extraction accuracy.
6. Annotation export, no-overwrite protection, blind export, pending/incomplete
   labels, source mismatch, evidence mismatch, missed events and false positives
   have automated checks. Title annotation exports hide body text.
7. Existing four Gemini/FinBERT A/B evaluation tests passed with the new tests.
   No existing market/LLM extraction source file was changed.

## Reproduce

Install `requirements-event.txt`, download `en_core_web_sm`, and run:

```bash
python -m pytest tests/test_event_extractors.py tests/test_event_annotation.py tests/test_event_run.py -q
python -m event_extraction.run --backend spacy --source title --output outputs/events_title.csv
python -m event_extraction.annotate validate --predictions outputs/events_title.csv --annotations event_extraction/sample_gold_title.csv --output outputs/demo_metrics.json
```

For the whole repository test suite, also install scikit-learn (needed by the
existing A/B evaluator), then run `python -m pytest -q`.

## Limits

- The Colab notebook was structurally checked locally; it was not executed in a
  hosted Colab runtime. Local Windows commands and their equivalent model CLIs
  were executed. Linux/macOS installation was not separately tested.
- No real-news human reference set, downstream market features, sentiment training,
  PIT alignment for the new event data, E0–E4 training or backtest was performed.
- Event classes and company-role matching remain heuristics. Negation, modality,
  aliases, coreference, long-document coverage and repeated mentions need review.
- No promise of unlimited free Colab runtime or identical future model weights.
