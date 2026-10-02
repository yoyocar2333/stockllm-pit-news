from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from data.align import align_news
from llm.client import GeminiQuotaError, LLMClient
from llm.schema import NewsSignal


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run Gemini structured extraction on a fixed article universe."
    )
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--ticker", default=None)
    ap.add_argument("--prompt", default="prompts/llm_only.txt")
    args = ap.parse_args()

    df = pd.read_csv(args.input, low_memory=False)
    required = {"article_id", "ticker", "timestamp", "title", "content"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if df["article_id"].duplicated().any():
        raise ValueError("article_id must be unique")

    if args.ticker:
        df = df[df["ticker"].astype(str).str.upper().eq(args.ticker.upper())].copy()

    df = align_news(df).reset_index(drop=True)
    prompt = Path(args.prompt).read_text(encoding="utf-8")
    client = LLMClient()

    rows = []
    for i, row in df.iterrows():
        article = {
            "article_id": row["article_id"],
            "ticker": row["ticker"],
            "timestamp": str(row["timestamp"]),
            "information_date": str(row["information_date"]),
            "title": row["title"],
            "content": row["content"],
            "source": row.get("source", ""),
            "url": row.get("url", ""),
        }
        try:
            signal = client.extract(prompt, article, NewsSignal).model_dump()
        except GeminiQuotaError:
            print(f"Gemini quota exhausted at {i + 1}/{len(df)}; saving completed rows.")
            break

        rows.append({**row.to_dict(), **signal})
        if (i + 1) % 25 == 0 or i == len(df) - 1:
            print(f"processed {i + 1}/{len(df)} articles")

    out = pd.DataFrame(rows)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output, index=False, encoding="utf-8-sig")
    print(f"wrote {output} ({len(out):,} articles)")


if __name__ == "__main__":
    main()
