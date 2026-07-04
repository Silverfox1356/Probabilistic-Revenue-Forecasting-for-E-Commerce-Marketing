import json
import tempfile

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

import forecaster
import llm_insights

st.set_page_config(page_title="AIgnition Revenue Forecaster", layout="wide")
st.title("Probabilistic Revenue Forecaster")

FILES = list(forecaster.SOURCES)
uploads = {f: st.sidebar.file_uploader(f, type="csv") for f in FILES}

if not all(uploads.values()):
    st.info("Upload all three channel CSVs in the sidebar to begin.")
    st.stop()


@st.cache_data
def load(raw):
    with tempfile.TemporaryDirectory() as d:
        for name, data in raw.items():
            open(f"{d}/{name}", "wb").write(data)
        return forecaster.load_data(d)


df = load({name: f.getvalue() for name, f in uploads.items()})
issues = forecaster.validate(df)
with st.expander(f"Data validation — {len(issues) or 'no'} issue(s)", expanded=bool(issues)):
    for i in issues or ["All checks passed."]:
        st.write("•", i)

st.sidebar.divider()
horizon = st.sidebar.radio("Forecast horizon (days)", [30, 60, 90], horizontal=True)
st.sidebar.caption("Budget simulation (0 = keep current spend)")
budgets = {ch: st.sidebar.number_input(f"{ch} budget ($)", min_value=0, value=0, step=5000)
           for ch in sorted(df["channel"].unique())}
budgets = {k: float(v) for k, v in budgets.items() if v > 0}

result = forecaster.run_forecast(df, horizon, budgets)
agg = result["aggregate"]

c1, c2, c3 = st.columns(3)
c1.metric("Planned spend", f"${agg['planned_spend']:,.0f}")
c2.metric("Revenue P50", f"${agg['revenue']['p50']:,.0f}",
          f"P10 ${agg['revenue']['p10']:,.0f} — P90 ${agg['revenue']['p90']:,.0f}", delta_color="off")
c3.metric("Blended ROAS P50", agg["blended_roas"]["p50"],
          f"P10 {agg['blended_roas']['p10']} — P90 {agg['blended_roas']['p90']}", delta_color="off")

ch_df = pd.json_normalize(result["levels"]["channel"])
col_a, col_b = st.columns(2)
fig, ax = plt.subplots(figsize=(5, 3))
ax.bar(ch_df["channel"], ch_df["revenue.p50"], color="#4C78A8",
       yerr=[ch_df["revenue.p50"] - ch_df["revenue.p10"], ch_df["revenue.p90"] - ch_df["revenue.p50"]],
       capsize=5)
ax.set_ylabel("Revenue ($)")
ax.set_title(f"{horizon}-day channel revenue (P10–P90)")
col_a.pyplot(fig)

fig2, ax2 = plt.subplots(figsize=(5, 3))
contrib = agg["revenue_contribution_pct"]
ax2.pie(contrib.values(), labels=contrib.keys(), autopct="%1.1f%%")
ax2.set_title("Revenue contribution")
col_b.pyplot(fig2)

daily = df.groupby("date")["revenue"].sum().rolling(7).mean()
future = pd.date_range(df["date"].max(), periods=horizon + 1)
fig3, ax3 = plt.subplots(figsize=(10, 3))
ax3.plot(daily.index, daily.values, color="#4C78A8", label="7-day avg daily revenue")
ax3.fill_between(future, agg["revenue"]["p10"] / horizon, agg["revenue"]["p90"] / horizon,
                 color="#F58518", alpha=0.3, label="forecast P10-P90 (daily rate)")
ax3.plot(future, [agg["revenue"]["p50"] / horizon] * len(future), "--", color="#F58518")
ax3.legend()
ax3.set_title("Revenue history and forecast band")
st.pyplot(fig3)

with st.expander("Backtest — last 30 days held out"):
    st.dataframe(pd.DataFrame(forecaster.backtest(df)), use_container_width=True)

tab1, tab2 = st.tabs(["Campaign types", "Campaigns"])
tab1.dataframe(pd.json_normalize(result["levels"]["campaign_type"]), use_container_width=True)
tab2.dataframe(pd.json_normalize(result["levels"]["campaign"]), use_container_width=True)

st.subheader("AI causal summary")
if st.button("Generate insights"):
    with st.spinner("Analyzing..."):
        st.markdown(llm_insights.generate_insights(
            df, {"aggregate": agg, "channels": result["levels"]["channel"]}))

st.download_button("Download forecast JSON", json.dumps(result, indent=2),
                   "forecast_output.json", "application/json")
