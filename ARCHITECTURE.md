# Architecture Overview

**Frontend stack.** Streamlit single-page app (`streamlit_app.py`) with matplotlib charts: CSV uploaders, validation panel, horizon selector, per-channel budget inputs, P10/P50/P90 metrics, channel revenue chart with uncertainty bars, contribution pie, campaign-type and campaign drill-down tables, on-demand AI summary, and JSON download. A CLI (`app.py`) exposes the same pipeline for scripted use.

**Backend stack.** Python with pandas/NumPy only. `forecaster.py` holds the entire modeling pipeline; `llm_insights.py` calls the Anthropic API over stdlib HTTP. No database or server components — the utility is stateless and file-driven.

**Forecasting pipeline.** `load_data` (three-source schema adapter) → `validate` (consistency checks) → `run_forecast`, which filters to active campaigns and, per entity at each aggregation level, estimates month-of-year seasonal factors (ratio to a 365-day trend), a deseasonalized recent-window level, and a bootstrap of the horizon total combining level and day-to-day uncertainty, then applies elasticity-based budget response and derives ROAS ranges → aggregate blended metrics → JSON. For the scoring pipeline, `generate_features.py` and `predict.py` wrap the same modeling code behind `run.sh` and a pickled config.

**LLM integration workflow.** After the statistical forecast completes, a prompt containing the monthly historical summary and forecast JSON is sent to `claude-sonnet-4-6`; the response (causal drivers, anomalies, operational risks) is attached to the output JSON and rendered in the UI. The key is read from `ANTHROPIC_API_KEY`; the layer degrades gracefully to a prepared prompt when unset.
