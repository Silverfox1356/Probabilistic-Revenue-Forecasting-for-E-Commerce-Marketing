import numpy as np
import pandas as pd

QUANTILES = {"p10": 0.10, "p50": 0.50, "p90": 0.90}
N_SIMS = 2000

SOURCES = {
    "google_ads_campaign_stats.csv": ("Google Ads", "segments_date", "campaign_name",
                                      "campaign_advertising_channel_type", None, "metrics_conversions_value"),
    "bing_campaign_stats.csv": ("MS Ads", "TimePeriod", "CampaignName", "CampaignType", "Spend", "Revenue"),
    "meta_ads_campaign_stats.csv": ("Meta Ads", "date_start", "campaign_name", None, "spend", "conversion"),
}


def load_data(data_dir):
    frames = []
    for fname, (channel, date, name, ctype, spend, revenue) in SOURCES.items():
        df = pd.read_csv(f"{data_dir}/{fname}")
        out = pd.DataFrame({
            "date": pd.to_datetime(df[date]),
            "channel": channel,
            "campaign_type": df[ctype] if ctype else df[name].str.split("_").str[0],
            "campaign": df[name],
            "spend": df[spend] if spend else df["metrics_cost_micros"] / 1e6,
            "revenue": df[revenue],
        })
        frames.append(out)
    return pd.concat(frames, ignore_index=True)


def validate(df):
    issues = []
    for col in ("spend", "revenue"):
        n = int(df[col].isna().sum() + (df[col] < 0).sum())
        if n:
            issues.append(f"{n} null/negative values in '{col}'")
    mapping = df.groupby(["channel", "campaign"])["campaign_type"].nunique()
    bad = mapping[mapping > 1].index.tolist()
    if bad:
        issues.append(f"Campaigns mapped to multiple channels/types: {bad}")
    stale = df.groupby("campaign")["date"].max()
    n_stale = int((stale < df["date"].max() - pd.Timedelta(days=30)).sum())
    if n_stale:
        issues.append(f"{n_stale} campaigns inactive for >30 days (excluded from forecast)")
    return issues


def _active(df):
    cutoff = df["date"].max() - pd.Timedelta(days=30)
    recent = df.groupby("campaign")["date"].max()
    return df[df["campaign"].isin(recent[recent >= cutoff].index)]


def _elasticity(g):
    d = g.groupby("date")[["spend", "revenue"]].sum()
    d = d[(d["spend"] > 0) & (d["revenue"] > 0)]
    if len(d) < 20 or d["spend"].std() / d["spend"].mean() < 0.05:
        return 0.85
    return float(np.clip(np.polyfit(np.log(d["spend"]), np.log(d["revenue"]), 1)[0], 0.3, 1.0))


def forecast_entity(g, horizon_days, budget=None, rng=None):
    rng = rng or np.random.default_rng(42)
    daily = g.groupby("date")["revenue"].sum().asfreq("D", fill_value=0.0)
    recent = daily.tail(90)
    base = recent.mean()
    future = pd.date_range(daily.index.max() + pd.Timedelta(days=1), periods=horizon_days)
    factors = np.ones(horizon_days)
    trend = daily.rolling(28, center=True, min_periods=14).mean()
    ratio = (daily / trend).replace([np.inf, -np.inf], np.nan).dropna()
    if base > 0 and len(ratio) >= 28:
        dow = ratio.groupby(ratio.index.dayofweek).mean()
        month = ratio.groupby(ratio.index.month).mean()
        factors = dow.reindex(future.dayofweek, fill_value=1.0).values * \
                  np.clip(month.reindex(future.month, fill_value=1.0).values, 0.5, 2.0)
    expected = (base * factors).sum()
    resid = recent.values - base
    sims = np.maximum(expected + rng.choice(resid, (N_SIMS, horizon_days)).sum(axis=1), 0)
    spend = g.groupby("date")["spend"].sum().reindex(daily.index, fill_value=0.0).tail(90).mean() * horizon_days
    if budget is not None and spend > 0:
        sims = sims * (budget / spend) ** _elasticity(g)
        spend = budget
    rev = {k: float(np.quantile(sims, q)) for k, q in QUANTILES.items()}
    return {"planned_spend": round(spend, 2),
            "revenue": {k: round(v, 2) for k, v in rev.items()},
            "roas": {k: round(v / spend, 2) if spend else 0.0 for k, v in rev.items()}}


def run_forecast(df, horizon_days=30, channel_budgets=None):
    channel_budgets = channel_budgets or {}
    df = _active(df)
    rng = np.random.default_rng(42)
    out = {"horizon_days": horizon_days, "levels": {}}
    for level, keys in [("channel", ["channel"]),
                        ("campaign_type", ["channel", "campaign_type"]),
                        ("campaign", ["channel", "campaign_type", "campaign"])]:
        rows = []
        for name, g in df.groupby(keys):
            name = name if isinstance(name, tuple) else (name,)
            budget = channel_budgets.get(name[0]) if level == "channel" else None
            rows.append({**dict(zip(keys, name)), **forecast_entity(g, horizon_days, budget, rng)})
        out["levels"][level] = rows
    ch = out["levels"]["channel"]
    rev = {k: sum(r["revenue"][k] for r in ch) for k in QUANTILES}
    spend = sum(r["planned_spend"] for r in ch)
    out["aggregate"] = {
        "planned_spend": round(spend, 2),
        "revenue": {k: round(v, 2) for k, v in rev.items()},
        "blended_roas": {k: round(v / spend, 2) if spend else 0.0 for k, v in rev.items()},
        "revenue_contribution_pct": {r["channel"]: round(100 * r["revenue"]["p50"] / rev["p50"], 1)
                                     for r in ch if rev["p50"]}}
    return out


def backtest(df, horizon_days=30):
    cutoff = df["date"].max() - pd.Timedelta(days=horizon_days)
    train, test = df[df["date"] <= cutoff], df[df["date"] > cutoff]
    rows = []
    for ch, g in _active(train).groupby("channel"):
        pred = forecast_entity(g, horizon_days)["revenue"]
        actual = float(test.loc[test["channel"] == ch, "revenue"].sum())
        rows.append({"channel": ch, "actual": round(actual, 2), **pred,
                     "within_p10_p90": pred["p10"] <= actual <= pred["p90"]})
    total = {k: round(sum(r[k] for r in rows), 2) for k in ("actual", "p10", "p50", "p90")}
    rows.append({"channel": "TOTAL", **total,
                 "within_p10_p90": total["p10"] <= total["actual"] <= total["p90"]})
    return rows
