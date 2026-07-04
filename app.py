import argparse
import json

import forecaster
import llm_insights


def main():
    p = argparse.ArgumentParser(description="Probabilistic ecommerce revenue/ROAS forecaster")
    p.add_argument("data_dir", help="Directory containing the three channel CSVs")
    p.add_argument("--horizon", type=int, default=30, choices=[30, 60, 90])
    p.add_argument("--budget", action="append", default=[], metavar="CHANNEL=AMOUNT")
    p.add_argument("--output", default="forecast_output.json")
    p.add_argument("--no-llm", action="store_true")
    args = p.parse_args()

    budgets = dict((k, float(v)) for k, v in (b.split("=") for b in args.budget))
    df = forecaster.load_data(args.data_dir)
    issues = forecaster.validate(df)
    print("Validation:", "OK" if not issues else "")
    for i in issues:
        print("  !", i)

    result = forecaster.run_forecast(df, args.horizon, budgets)
    agg = result["aggregate"]
    print(f"\n{args.horizon}-day forecast | planned spend ${agg['planned_spend']:,.0f}")
    print(f"  Revenue  P10 ${agg['revenue']['p10']:,.0f} | P50 ${agg['revenue']['p50']:,.0f} | P90 ${agg['revenue']['p90']:,.0f}")
    print(f"  ROAS     P10 {agg['blended_roas']['p10']} | P50 {agg['blended_roas']['p50']} | P90 {agg['blended_roas']['p90']}")
    print("  Contribution %:", agg["revenue_contribution_pct"])

    if not args.no_llm:
        result["ai_insights"] = llm_insights.generate_insights(
            df, {"aggregate": agg, "channels": result["levels"]["channel"]})
        print("\n--- AI Causal Summary ---\n" + result["ai_insights"])

    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nFull forecasts written to {args.output}")


if __name__ == "__main__":
    main()
