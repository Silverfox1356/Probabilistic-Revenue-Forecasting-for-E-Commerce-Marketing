"""Step 2 of run.sh: load the pickled model config and the feature table, run
the probabilistic forecast, and write flat predictions to the output CSV."""
import argparse
import pickle

import pandas as pd

import forecaster


def _rows(result):
    """Flatten one horizon's forecast into prediction rows."""
    horizon = result["horizon_days"]
    rows = []
    for level, entries in result["levels"].items():
        for e in entries:
            rows.append({
                "horizon_days": horizon,
                "level": level,
                "channel": e.get("channel", ""),
                "campaign_type": e.get("campaign_type", ""),
                "campaign": e.get("campaign", ""),
                "planned_spend": e["planned_spend"],
                "revenue_p10": e["revenue"]["p10"],
                "revenue_p50": e["revenue"]["p50"],
                "revenue_p90": e["revenue"]["p90"],
                "roas_p10": e["roas"]["p10"],
                "roas_p50": e["roas"]["p50"],
                "roas_p90": e["roas"]["p90"],
            })
    agg = result["aggregate"]
    rows.append({
        "horizon_days": horizon,
        "level": "aggregate",
        "channel": "", "campaign_type": "", "campaign": "",
        "planned_spend": agg["planned_spend"],
        "revenue_p10": agg["revenue"]["p10"],
        "revenue_p50": agg["revenue"]["p50"],
        "revenue_p90": agg["revenue"]["p90"],
        "roas_p10": agg["blended_roas"]["p10"],
        "roas_p50": agg["blended_roas"]["p50"],
        "roas_p90": agg["blended_roas"]["p90"],
    })
    return rows


def main():
    p = argparse.ArgumentParser(description="Produce probabilistic revenue/ROAS forecasts")
    p.add_argument("--features", default="./output/features.csv", help="Feature table from generate_features.py")
    p.add_argument("--model", default="./pickle/model.pkl", help="Pickled model config")
    p.add_argument("--output", default="./output/predictions.csv", help="Where to write predictions")
    args = p.parse_args()

    with open(args.model, "rb") as f:
        model = pickle.load(f)

    df = pd.read_csv(args.features, parse_dates=["date"])

    rows = []
    for horizon in model["horizons"]:
        rows += _rows(forecaster.run_forecast(df, horizon))

    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(f"Wrote {len(rows)} prediction rows to {args.output}")


if __name__ == "__main__":
    main()
