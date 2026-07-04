# Probabilistic Revenue Forecaster — AIgnition 2026

Forecasts ecommerce revenue and ROAS over 30/60/90-day horizons at channel, campaign-type, and campaign level, with per-channel budget simulation and AI-assisted causal summaries.

## Quick start

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py          # web UI
python app.py ./data --horizon 30 --no-llm   # CLI alternative
```

The `data` directory (CLI) or the sidebar uploaders (UI) take the three platform exports with their original filenames: `google_ads_campaign_stats.csv`, `bing_campaign_stats.csv`, `meta_ads_campaign_stats.csv`. Set `ANTHROPIC_API_KEY` to enable AI insights.

## Data preprocessing

Each platform export is normalized into `date, channel, campaign_type, campaign, spend, revenue`. Google spend is converted from micros and revenue taken from `metrics_conversions_value`; Bing columns map directly; Meta's `conversion` column is treated as conversion value (revenue) — validated by magnitude analysis (as a count it would imply a 306% conversion rate and $0.12 CPA, and 64% of values are non-integer) — and Meta campaign type is derived from the campaign-name prefix. Campaigns inactive for over 30 days are flagged and excluded from the forward view.

## Forecasting methodology

Per entity, the revenue level is the mean of the last 90 days of daily revenue. Day-of-week and month-of-year seasonality is estimated over the full history via ratio-to-28-day-centered-moving-average, which detrends the series so account growth is not mistaken for seasonality; month factors are clipped to [0.5, 2.0]. Uncertainty comes from bootstrap resampling of recent daily residuals (2,000 simulations) summed over the horizon, producing aggregate-period P10/P50/P90 ranges. Budget response applies a log-log spend elasticity fitted per channel, clipped to [0.3, 1.0] with a diminishing-returns default of 0.85 when spend variation is insufficient to fit. Blended ROAS is simulated total revenue over total planned spend.

## Model selection

A seasonal-naive level model with bootstrap uncertainty was chosen over ARIMA/Prophet/ML regressors because the account underwent a structural ~5x scale-up in late 2025, which breaks stationarity assumptions and makes long-history parametric fits misleading; the recent-window level plus detrended seasonal factors adapts to the new regime while still using the full history for seasonality. Bootstrap intervals make no distributional assumptions and are directly explainable to non-technical stakeholders, matching the brief's emphasis on realistic, explainable forecasting over theoretical modeling.

## Validation checks

Null/negative metrics, campaigns mapped to multiple types within a channel, and campaigns inactive >30 days.

## AI integration strategy

The LLM layer (Anthropic API) receives a monthly channel-level historical summary plus the forecast JSON and returns a three-part briefing: causal drivers, anomalies, and operational risks. The LLM interprets statistical output but never produces the numbers, keeping forecasts deterministic and reproducible (fixed random seed).

## Assumptions and limitations

Channel attribution is treated as the source of truth per the brief; Meta `conversion` equals revenue; the recent 90-day level persists over the horizon; budget response is a smooth power curve with no saturation cliffs or cross-channel effects; the final 2–3 days of data show a conversion-lag taper that slightly deflates estimates; seasonality for months seen only once in history is estimated from a single occurrence.
