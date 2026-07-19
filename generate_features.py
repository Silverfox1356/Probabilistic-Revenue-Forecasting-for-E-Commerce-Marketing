"""Step 1 of run.sh: read the raw channel CSVs from the data folder and write
the normalized long-format feature table the forecaster consumes."""
import argparse

import forecaster


def main():
    p = argparse.ArgumentParser(description="Normalize channel CSVs into a single feature table")
    p.add_argument("--data-dir", default="./data", help="Folder containing the channel CSVs")
    p.add_argument("--out", default="./output/features.csv", help="Where to write the feature table")
    args = p.parse_args()

    df = forecaster.load_data(args.data_dir)
    issues = forecaster.validate(df)
    print("Validation:", "OK" if not issues else f"{len(issues)} issue(s)")
    for i in issues:
        print("  !", i)

    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df)} rows to {args.out}")


if __name__ == "__main__":
    main()
