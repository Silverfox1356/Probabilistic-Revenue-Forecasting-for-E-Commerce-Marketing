# Probabilistic Revenue Forecaster — AIgnition 2026

Forecasts ecommerce revenue and ROAS over 30/60/90-day horizons at channel, campaign-type, and campaign level, with per-channel budget simulation and AI-assisted causal summaries.

Python 3.13.

## Submission entry point

```bash
pip install -r requirements.txt
./run.sh ./data ./pickle/model.pkl ./output/predictions.csv
```

`run.sh` runs end to end: `generate_features.py` normalizes the CSVs in `data/`
into a feature table, then `predict.py` loads the committed `pickle/model.pkl`
and writes probabilistic forecasts to `output/predictions.csv`. It accepts the
three positional arguments above and falls back to those defaults with no
arguments. Rebuild the model artifact with `python train_model.py`.

Predictions cover the 30/60/90-day horizons at aggregate, channel,
campaign-type, and campaign level. Each row carries `planned_spend`,
`revenue_p10/p50/p90`, and `roas_p10/p50/p90`.

## Repository layout

```
run.sh                 entry point (feature generation + prediction)
generate_features.py   raw CSVs -> normalized feature table
predict.py             pickled model + features -> predictions.csv
train_model.py         builds pickle/model.pkl
forecaster.py          forecasting model
llm_insights.py        AI causal summary layer
app.py, streamlit_app.py   CLI and web UI for exploration
data/                  channel CSVs (dropped in at test time)
pickle/model.pkl       committed model artifact
```

## Interactive exploration

```bash
streamlit run streamlit_app.py               # web UI (upload the three CSVs)
python app.py ./data --horizon 30 --no-llm   # CLI with AI summary
```

The three platform exports keep their original filenames:
`google_ads_campaign_stats.csv`, `bing_campaign_stats.csv`,
`meta_ads_campaign_stats.csv`. Set `ANTHROPIC_API_KEY` (or `GEMINI_API_KEY` /
`OPENAI_API_KEY`) to enable AI insights.

## Data preprocessing

Each platform export is normalized into `date, channel, campaign_type, campaign, spend, revenue`. Google spend is converted from micros and revenue taken from `metrics_conversions_value`; Bing columns map directly; Meta's `conversion` column is treated as conversion value (revenue) — validated by magnitude analysis (as a count it would imply a 306% conversion rate and $0.12 CPA, and 64% of values are non-integer) — and Meta campaign type is derived from the campaign-name prefix. Campaigns inactive for over 30 days are flagged and excluded from the forward view.

## Forecasting methodology

The model is a multiplicative seasonal decomposition with a bootstrap simulation of the horizon total, run per entity (channel, campaign type, campaign).

1. **Seasonality.** Month-of-year factors are estimated as the mean ratio of daily revenue to its **365-day centered moving average**. The long window is essential: it captures the Nov/Dec holiday surge (December ≈ 3.6× a normal day). An earlier 28-day window failed here because a 28-day average tracks the surge itself and cancels the very seasonality it should measure. Factors are clipped to [0.3, 4.0].
2. **Level.** Daily revenue is deseasonalized (divided by its month factor) and the level is the mean of the last 90 deseasonalized days. Deseasonalizing first is what makes the level robust: a plain trailing mean is contaminated by wherever the holiday falls in the window, which is the main reason the naive version mis-forecast at 60–90 days.
3. **Uncertainty (2,000 simulations).** Two sources are combined and re-seasonalized onto the forecast window: *level uncertainty* (standard error of the mean, using a weekly-block effective sample size so autocorrelation is not ignored) and *day-to-day noise* (a weekly moving-block bootstrap of deseasonalized residuals). The horizon totals give aggregate-period P10/P50/P90 ranges.
4. **Budget response.** A log-log spend elasticity fitted per channel (clipped to [0.3, 1.0], diminishing-returns default 0.85 when spend varies too little to fit) scales revenue when a future budget is supplied. Blended ROAS is simulated total revenue over total planned spend.

**Backtest (rolling-origin, 8 cutoffs, aggregate revenue):** interval coverage 6/6/5 of 8 at 30/60/90 days with P50 error ≈ 39/26/30%, versus 2/0/2 coverage and ≈100/87/82% error for the naive-trailing-mean baseline. Run `streamlit_app.py` → "Backtest" to reproduce.

## Model selection

A decomposition + bootstrap model was chosen over ARIMA/Prophet/ML regressors because the account underwent a structural ~5x scale-up in late 2025, which breaks the stationarity assumptions those models rely on. A recent-window *level* adapts to the current regime while *seasonal factors* use the full history, and non-parametric bootstrap intervals make no distributional assumptions and are directly explainable to non-technical stakeholders — matching the brief's emphasis on realistic, explainable forecasting over theoretical modeling. Day-of-week seasonality was dropped after the data showed it was negligible (factors within 0.92–1.03).

## Validation checks

Null/negative metrics, campaigns mapped to multiple types within a channel, and campaigns inactive >30 days.

## AI integration strategy

The LLM layer (Anthropic API) receives a monthly channel-level historical summary plus the forecast JSON and returns a three-part briefing: causal drivers, anomalies, and operational risks. The LLM interprets statistical output but never produces the numbers, keeping forecasts deterministic and reproducible (fixed random seed).

## Assumptions and limitations

Channel attribution is treated as the source of truth per the brief; Meta `conversion` equals revenue; the recent deseasonalized level persists over the horizon (no local trend is extrapolated, so an unusually strong ramp is under-forecast and a decline over-forecast); budget response is a smooth power curve with no saturation cliffs or cross-channel effects; seasonality for months seen only once in history is estimated from a single occurrence; low-volume channels (MS Ads here) are inherently noisy and get wide, honest intervals rather than precise point forecasts.
