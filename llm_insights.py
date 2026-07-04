"""AI-assisted causal summary, anomaly interpretation, and risk identification."""
import json
import os
import urllib.request

API_URL = "https://api.anthropic.com/v1/messages"
MODEL = "claude-sonnet-4-6"

PROMPT = """You are a senior ecommerce marketing analyst. Given the historical summary and
probabilistic forecast below, write a concise business briefing with three sections:
1. Causal drivers — why revenue/ROAS is expected to move (seasonality, spend shifts, channel mix).
2. Anomalies — unusual patterns in the historical data worth investigating.
3. Operational risks — concrete risks to the forecast and mitigations.
Be specific, reference numbers, and avoid generic advice.

HISTORICAL SUMMARY:
{history}

FORECAST:
{forecast}"""


def historical_summary(df):
    monthly = df.groupby([df["date"].dt.to_period("M"), "channel"])[["spend", "revenue"]].sum()
    monthly["roas"] = (monthly["revenue"] / monthly["spend"]).round(2)
    return monthly.round(0).to_string()


def generate_insights(df, forecast):
    key = os.environ.get("ANTHROPIC_API_KEY")
    prompt = PROMPT.format(history=historical_summary(df), forecast=json.dumps(forecast, indent=1))
    if not key:
        return "[LLM insights skipped: set ANTHROPIC_API_KEY]\n\nPrompt prepared:\n" + prompt[:500]
    req = urllib.request.Request(
        API_URL,
        data=json.dumps({"model": MODEL, "max_tokens": 1000,
                         "messages": [{"role": "user", "content": prompt}]}).encode(),
        headers={"Content-Type": "application/json", "x-api-key": key,
                 "anthropic-version": "2023-06-01"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.load(resp)
    return "".join(b.get("text", "") for b in data["content"])
