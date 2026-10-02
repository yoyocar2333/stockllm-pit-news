from __future__ import annotations

"""Strict paired A/B comparison: Gemini vs FinBERT."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def build_next_day_market(market: pd.DataFrame, ticker: str | None = None) -> pd.DataFrame:
    required = {"Date", "Close"}
    missing = required - set(market.columns)
    if missing:
        raise ValueError(f"Market CSV missing columns: {sorted(missing)}")

    x = market.copy()
    if ticker and "ticker" in x.columns:
        x = x[x["ticker"].astype(str).str.upper().eq(ticker.upper())].copy()

    x["Date"] = pd.to_datetime(x["Date"], errors="raise").dt.normalize()
    x["Close"] = pd.to_numeric(x["Close"], errors="raise")
    x = x.sort_values("Date").drop_duplicates("Date", keep="last").reset_index(drop=True)

    x["target_date"] = x["Date"].shift(-1)
    next_close = x["Close"].shift(-1)
    x["Target_NextDay"] = np.where(
        next_close.notna(), (next_close > x["Close"]).astype(int), np.nan
    )
    return x.dropna(subset=["target_date", "Target_NextDay"]).copy()


def _map_to_market_session(
    info_dates: pd.Series, market_dates: pd.DatetimeIndex
) -> pd.Series:
    values = market_dates.values.astype("datetime64[ns]")
    d = pd.to_datetime(info_dates, errors="raise").dt.normalize()
    positions = np.searchsorted(
        values, d.values.astype("datetime64[ns]"), side="left"
    )
    mapped = [
        pd.NaT if pos >= len(market_dates) else market_dates[pos]
        for pos in positions
    ]
    return pd.Series(mapped, index=info_dates.index, dtype="datetime64[ns]")


def prepare_articles(
    articles: pd.DataFrame,
    market_dates: pd.DatetimeIndex,
    ticker: str | None = None,
) -> pd.DataFrame:
    required = {
        "article_id",
        "ticker",
        "information_date",
        "direction_probability",
        "FinBERT_Direction_Probability",
    }
    missing = required - set(articles.columns)
    if missing:
        raise ValueError(f"Article CSV missing columns: {sorted(missing)}")

    x = articles.copy()
    if ticker:
        x = x[x["ticker"].astype(str).str.upper().eq(ticker.upper())].copy()

    if x["article_id"].duplicated().any():
        raise ValueError("Duplicate article_id rows make the paired universe ambiguous")

    x["Gemini_Probability"] = pd.to_numeric(
        x["direction_probability"], errors="raise"
    ).clip(0, 1)
    x["FinBERT_Probability"] = pd.to_numeric(
        x["FinBERT_Direction_Probability"], errors="raise"
    ).clip(0, 1)
    x["information_date"] = pd.to_datetime(
        x["information_date"], errors="raise"
    ).dt.normalize()
    x["effective_information_date"] = _map_to_market_session(
        x["information_date"], market_dates
    )
    x = x.dropna(subset=["effective_information_date"]).copy()
    if x.empty:
        raise ValueError("No articles remain after validation")
    return x


def aggregate_same_day(articles: pd.DataFrame) -> pd.DataFrame:
    return (
        articles.groupby("effective_information_date", as_index=False)
        .agg(
            Gemini_Probability=("Gemini_Probability", "mean"),
            FinBERT_Probability=("FinBERT_Probability", "mean"),
            Article_Count=("article_id", "count"),
        )
        .sort_values("effective_information_date")
    )


def _decayed_probability(
    session_dates: pd.DatetimeIndex,
    daily_probability: pd.Series,
    half_life: float,
) -> np.ndarray:
    if half_life <= 0:
        raise ValueError("half_life must be > 0")

    p = pd.Series(0.5, index=session_dates, dtype=float)
    common = p.index.intersection(daily_probability.index)
    p.loc[common] = daily_probability.loc[common].astype(float)
    signal = 2.0 * p.to_numpy() - 1.0

    decay = 2.0 ** (-1.0 / float(half_life))
    numerator = 0.0
    denominator = 0.0
    out = np.zeros(len(signal), dtype=float)

    for i, s in enumerate(signal):
        numerator = s + decay * numerator
        denominator = 1.0 + decay * denominator
        out[i] = np.clip(
            0.5 + 0.5 * numerator / denominator, 0.0, 1.0
        )
    return out


def build_predictions(
    market_next: pd.DataFrame,
    daily: pd.DataFrame,
    half_lives: list[float],
    evaluation_start: str | None = None,
    evaluation_end: str | None = None,
) -> pd.DataFrame:
    all_dates = pd.DatetimeIndex(market_next["Date"].sort_values().unique())
    d = daily.copy()
    d["effective_information_date"] = pd.to_datetime(
        d["effective_information_date"]
    ).dt.normalize()
    d = d.set_index("effective_information_date").sort_index()

    start = (
        pd.Timestamp(evaluation_start).normalize()
        if evaluation_start
        else d.index.min()
    )
    end = (
        pd.Timestamp(evaluation_end).normalize()
        if evaluation_end
        else d.index.max()
    )
    eval_market = market_next[
        (market_next["Date"] >= start) & (market_next["Date"] <= end)
    ].copy()
    if eval_market.empty:
        raise ValueError("No market rows in evaluation window")

    rows = []
    for model, col in [
        ("Gemini", "Gemini_Probability"),
        ("FinBERT", "FinBERT_Probability"),
    ]:
        model_daily = d[col]
        latest = pd.Series(0.5, index=all_dates, dtype=float)
        common = latest.index.intersection(model_daily.index)
        latest.loc[common] = model_daily.loc[common].astype(float)

        settings = {"Latest": latest}
        for h in half_lives:
            settings[f"H{h:g}"] = pd.Series(
                _decayed_probability(all_dates, model_daily, h),
                index=all_dates,
            )

        for decay_name, probs in settings.items():
            half_life = (
                np.nan
                if decay_name == "Latest"
                else float(decay_name.removeprefix("H"))
            )
            for _, row in eval_market.iterrows():
                cutoff = pd.Timestamp(row["Date"])
                rows.append(
                    {
                        "model": model,
                        "decay": decay_name,
                        "half_life": half_life,
                        "cutoff_date": cutoff,
                        "target_date": pd.Timestamp(row["target_date"]),
                        "Target_NextDay": int(row["Target_NextDay"]),
                        "probability": float(probs.loc[cutoff]),
                    }
                )

    return pd.DataFrame(rows)


def classification_metrics(y_true, probability) -> dict:
    y = pd.Series(y_true).astype(int)
    p = pd.Series(probability).astype(float).clip(0.0, 1.0)
    pred = (p >= 0.5).astype(int)

    out = {
        "n": len(y),
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "brier": brier_score_loss(y, p),
    }
    out["roc_auc"] = (
        roc_auc_score(y, p)
        if y.nunique() > 1 and p.nunique() > 1
        else np.nan
    )
    out["pr_auc_ap"] = (
        average_precision_score(y, p) if y.nunique() > 1 else np.nan
    )
    return out


def summarize_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, decay, half_life), group in predictions.groupby(
        ["model", "decay", "half_life"], dropna=False
    ):
        metrics = classification_metrics(
            group["Target_NextDay"], group["probability"]
        )
        metrics.update(
            {"model": model, "decay": decay, "half_life": half_life}
        )
        rows.append(metrics)
    return pd.DataFrame(rows)


def validate_paired_population(predictions: pd.DataFrame) -> None:
    for decay, group in predictions.groupby("decay"):
        date_sets = {
            model: set(
                zip(
                    rows["cutoff_date"],
                    rows["target_date"],
                    rows["Target_NextDay"],
                )
            )
            for model, rows in group.groupby("model")
        }
        if set(date_sets) != {"Gemini", "FinBERT"}:
            raise ValueError(f"Unexpected models in {decay}")
        if date_sets["Gemini"] != date_sets["FinBERT"]:
            raise ValueError(f"Paired population mismatch for {decay}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--articles", required=True)
    ap.add_argument("--market", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--ticker", default=None)
    ap.add_argument(
        "--half-lives", nargs="+", type=float, default=[1, 2, 3, 5, 10]
    )
    ap.add_argument("--evaluation-start", default=None)
    ap.add_argument("--evaluation-end", default=None)
    args = ap.parse_args()

    articles_raw = pd.read_csv(args.articles, low_memory=False)
    market_raw = pd.read_csv(args.market, low_memory=False)

    market_next = build_next_day_market(market_raw, ticker=args.ticker)
    market_dates = pd.DatetimeIndex(
        pd.to_datetime(market_raw["Date"], errors="raise")
        .dt.normalize()
        .sort_values()
        .unique()
    )
    articles = prepare_articles(articles_raw, market_dates, ticker=args.ticker)
    daily = aggregate_same_day(articles)
    predictions = build_predictions(
        market_next,
        daily,
        args.half_lives,
        args.evaluation_start,
        args.evaluation_end,
    )
    validate_paired_population(predictions)
    summary = summarize_predictions(predictions)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output, index=False, encoding="utf-8-sig")
    predictions.to_csv(
        output.with_name(output.stem + "_predictions.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    daily.to_csv(
        output.with_name(output.stem + "_daily_common_news.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
