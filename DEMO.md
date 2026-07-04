# Demo Workflow

1. **Launch.** `streamlit run streamlit_app.py`. The app opens with an upload prompt.
2. **Data ingestion.** Upload the three platform CSVs in the sidebar. The validation panel appears — point out the consistency checks and the flag for campaigns inactive >30 days, which are excluded from the forward view.
3. **Forecast generation.** Select the 30-day horizon. Walk through the headline metrics (planned spend, revenue P10–P90, blended ROAS range), the channel chart with uncertainty bars, and the contribution pie. Open the Campaigns tab to show campaign-level ranges.
4. **Budget simulation.** Set Google Ads budget to a higher figure (e.g. 100000) and leave others at 0. The dashboard recomputes live — highlight that blended ROAS compresses as spend scales, demonstrating the fitted diminishing-returns elasticity rather than naive linear scaling.
5. **AI-generated insights.** Click "Generate insights". The causal summary renders with drivers, anomalies (e.g. the late-2025 scale-up, end-of-data conversion-lag taper), and operational risks.
6. **Export.** Download the forecast JSON to show the full multi-level output that downstream tooling could consume.
