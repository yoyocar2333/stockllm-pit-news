# StockLLM V2 — Local Event Extraction Prototype

This is an independent prototype intended for the `experiment/event-extraction-e0-e4` branch of StockLLM. **No real market-prediction experiment has been run in this package.** Sample headlines are synthetic.

## CPU baseline (no API)

```bash
pip install pandas pytest spacy
python -m spacy download en_core_web_sm
python -m event_extraction.run --backend spacy --source title
python -m event_extraction.run --backend spacy --source full --output outputs/events_full.csv
pytest -q
```

## Optional GLiNER2 comparison

```bash
pip install 'gliner2[local]'
python -m event_extraction.run --backend gliner2 --source title --output outputs/gliner2.csv
```

Model weights may require internet on first use. Check the installed GLiNER2 model interface; results and accuracy are not validated until real inference plus human annotation. Role labels are grammatical/semantic observations, NOT predicted financial impact.

See `docs/E0_E4_PROTOCOL.md` for frozen experiment comparisons before introducing historical news datasets.
