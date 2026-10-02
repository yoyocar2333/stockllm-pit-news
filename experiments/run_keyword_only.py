from __future__ import annotations

import argparse
from pathlib import Path
import re

import pandas as pd

from data.align import align_news

POSITIVE = {
    "beat", "beats", "strong", "growth", "surge", "profit", "profits",
    "revenue growth", "record", "upgrade", "bullish", "outperform", "buy",
    "approval", "gain", "rise", "rally", "optimistic", "positive", "success",
}
NEGATIVE = {
    "miss", "weak", "decline", "drop", "loss", "losses", "lawsuit",
    "investigation", "fine", "penalty", "downgrade", "bearish", "underperform",
    "sell", "recall", "fall", "negative", "risk", "warning", "cut", "cuts",
}


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).lower()).strip()


def count_terms(text: str, terms: set[str]) -> int:
    text = normalize_text(text)
    score = 0
    for term in terms:
        if " " in term:
            score += text.count(term)
        else:
            score += len(re.findall(rf"\b{re.escape(term)}\b", text))
    return score


def score_title(title: str):
    pos = count_terms(title, POSITIVE)
    neg = count_terms(title, NEGATIVE)
    total = pos + neg
    probability = 0.5 if total == 0 else (pos + 1.0) / (total + 2.0)
    confidence = 0.0 if total == 0 else min(1.0, total / 3.0)
    return pos, neg, probability, confidence


def main() -> None:
    ap = argparse.ArgumentParser(description="Transparent title-only keyword baseline.")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--ticker", default=None)
    args = ap.parse_args()

    df = pd.read_csv(args.input, low_memory=False)
    required = {"article_id", "ticker", "timestamp", "title"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if args.ticker:
        df = df[df["ticker"].astype(str).str.upper().eq(args.ticker.upper())].copy()

    df = align_news(df)
    rows = []
    for _, row in df.iterrows():
        pos, neg, p_up, confidence = score_title(row["title"])
        rows.append(
            {
                **row.to_dict(),
                "keyword_positive_count": pos,
                "keyword_negative_count": neg,
                "keyword_direction_probability": p_up,
                "keyword_confidence": confidence,
            }
        )

    out = pd.DataFrame(rows)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output, index=False, encoding="utf-8-sig")
    print(f"wrote {output} ({len(out):,} articles)")


if __name__ == "__main__":
    main()
