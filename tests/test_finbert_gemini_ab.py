import numpy as np
import pandas as pd

from experiments.compare_finbert_gemini_nextday_decay import (
    _decayed_probability,
    aggregate_same_day,
    build_next_day_market,
    build_predictions,
    validate_paired_population,
)


def test_next_day_target_is_next_observed_market_session():
    market = pd.DataFrame(
        {
            "Date": ["2025-01-03", "2025-01-06", "2025-01-07"],
            "Close": [100.0, 110.0, 105.0],
            "ticker": ["AAPL"] * 3,
        }
    )
    out = build_next_day_market(market, ticker="AAPL")
    assert out.iloc[0]["target_date"] == pd.Timestamp("2025-01-06")
    assert int(out.iloc[0]["Target_NextDay"]) == 1
    assert int(out.iloc[1]["Target_NextDay"]) == 0


def test_old_news_decays_toward_neutral_on_no_news_sessions():
    sessions = pd.DatetimeIndex(
        pd.to_datetime(
            ["2025-01-02", "2025-01-03", "2025-01-06", "2025-01-07", "2025-01-08"]
        )
    )
    daily_p = pd.Series([1.0], index=[pd.Timestamp("2025-01-02")])
    p = _decayed_probability(sessions, daily_p, half_life=1.0)
    assert p[0] == 1.0
    assert 0.5 < p[-1] < p[1] < p[0]


def test_same_daily_aggregation_rule_for_both_models():
    article = pd.DataFrame(
        {
            "effective_information_date": pd.to_datetime(
                ["2025-01-02", "2025-01-02"]
            ),
            "article_id": ["a", "b"],
            "Gemini_Probability": [0.8, 0.6],
            "FinBERT_Probability": [0.7, 0.5],
        }
    )
    daily = aggregate_same_day(article)
    assert np.isclose(daily.loc[0, "Gemini_Probability"], 0.7)
    assert np.isclose(daily.loc[0, "FinBERT_Probability"], 0.6)


def test_predictions_use_identical_paired_population():
    market = pd.DataFrame(
        {
            "Date": pd.to_datetime(
                ["2025-01-02", "2025-01-03", "2025-01-06", "2025-01-07"]
            ),
            "Close": [100.0, 101.0, 99.0, 102.0],
        }
    )
    market_next = build_next_day_market(market)
    daily = pd.DataFrame(
        {
            "effective_information_date": pd.to_datetime(
                ["2025-01-02", "2025-01-06"]
            ),
            "Gemini_Probability": [0.7, 0.4],
            "FinBERT_Probability": [0.6, 0.3],
            "Article_Count": [1, 1],
        }
    )
    pred = build_predictions(market_next, daily, half_lives=[1, 2])
    validate_paired_population(pred)
