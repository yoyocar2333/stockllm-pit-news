from __future__ import annotations

"""Run FinBERT on the exact same article universe already scored by Gemini."""

import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from data.align import align_news

FINBERT_COLUMNS = [
    "FinBERT_Positive_Probability",
    "FinBERT_Negative_Probability",
    "FinBERT_Neutral_Probability",
    "FinBERT_Sentiment_Score",
    "FinBERT_Direction_Probability",
    "FinBERT_Confidence",
]


def _normalize_label(label: str) -> str:
    x = str(label).strip().lower()
    if "pos" in x:
        return "positive"
    if "neg" in x:
        return "negative"
    if "neu" in x:
        return "neutral"
    return x


def _label_indices(id2label: Dict[int, str]) -> Dict[str, int]:
    normalized = {_normalize_label(v): int(k) for k, v in id2label.items()}
    required = {"positive", "negative", "neutral"}
    missing = required - set(normalized)
    if missing:
        raise ValueError(
            "Could not identify FinBERT labels from "
            f"model.config.id2label={id2label}. Missing: {sorted(missing)}"
        )
    return normalized


def _build_text(df: pd.DataFrame) -> pd.Series:
    title = df["title"].fillna("").astype(str).str.strip()
    content = df["content"].fillna("").astype(str).str.strip()
    return (title + "\n\n" + content).str.strip()


def _validate_common_universe(df: pd.DataFrame) -> pd.DataFrame:
    required = {
        "article_id",
        "ticker",
        "timestamp",
        "information_date",
        "title",
        "content",
        "direction_probability",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input is missing required columns: {sorted(missing)}")

    if df["article_id"].isna().any() or df["article_id"].duplicated().any():
        raise ValueError("article_id must be present and unique")

    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="raise")
    out["information_date"] = pd.to_datetime(
        out["information_date"], errors="raise"
    ).dt.date

    aligned = align_news(
        out[["article_id", "ticker", "timestamp", "title", "content"]].copy()
    )
    recomputed = pd.to_datetime(aligned["information_date"]).dt.date
    mismatch = recomputed != out["information_date"]
    if mismatch.any():
        raise ValueError("Stored information_date does not match the PIT rule")

    out["direction_probability"] = pd.to_numeric(
        out["direction_probability"], errors="raise"
    ).clip(0.0, 1.0)
    return out


def _infer_finbert(
    texts: list[str],
    model_name: str,
    batch_size: int,
    max_length: int,
    device: str,
) -> pd.DataFrame:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    resolved_device = (
        "cuda" if device == "auto" and torch.cuda.is_available()
        else "cpu" if device == "auto"
        else device
    )

    print(f"Loading {model_name} on {resolved_device} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.to(resolved_device)
    model.eval()

    idx = _label_indices(model.config.id2label)
    rows = []

    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            enc = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(resolved_device) for k, v in enc.items()}
            probs = torch.softmax(model(**enc).logits, dim=-1).cpu().numpy()

            for pr in probs:
                p_pos = float(pr[idx["positive"]])
                p_neg = float(pr[idx["negative"]])
                p_neu = float(pr[idx["neutral"]])
                p_up = p_pos + 0.5 * p_neu
                rows.append(
                    {
                        "FinBERT_Positive_Probability": p_pos,
                        "FinBERT_Negative_Probability": p_neg,
                        "FinBERT_Neutral_Probability": p_neu,
                        "FinBERT_Sentiment_Score": p_pos - p_neg,
                        "FinBERT_Direction_Probability": float(
                            np.clip(p_up, 0.0, 1.0)
                        ),
                        "FinBERT_Confidence": max(p_pos, p_neg, p_neu),
                    }
                )

    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--ticker", default=None)
    ap.add_argument("--model", default="ProsusAI/finbert")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = ap.parse_args()

    source = _validate_common_universe(pd.read_csv(args.input, low_memory=False))
    if args.ticker:
        source = source[
            source["ticker"].astype(str).str.upper().eq(args.ticker.upper())
        ].copy()
    source = source.reset_index(drop=True)

    scored = _infer_finbert(
        _build_text(source).tolist(),
        args.model,
        args.batch_size,
        args.max_length,
        args.device,
    )
    result = pd.concat([source, scored], axis=1)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False, encoding="utf-8-sig")

    print(f"articles: {len(result):,}")
    print("synthetic/future-return fill: NONE")
    print("Gemini and FinBERT rows: IDENTICAL article universe")
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
