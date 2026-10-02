import pandas as pd
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")


def information_date(ts, close_hour: int = 16):
    """Initial PIT rule: ET date, shift after close, then skip weekends.

    Formal evaluation additionally maps this date to the first observed market
    session on/after the date, which handles exchange holidays.
    """
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    t = t.tz_convert(ET)
    d = t.normalize()
    if t.hour >= close_hour:
        d += pd.Timedelta(days=1)
    while d.weekday() >= 5:
        d += pd.Timedelta(days=1)
    return d.date()


def align_news(df, close_hour: int = 16):
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    out["information_date"] = out["timestamp"].map(
        lambda x: information_date(x, close_hour)
    )
    return out
