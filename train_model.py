"""Build the committed model artifact (pickle/model.pkl).

The forecaster is a seasonal-naive level model with bootstrap uncertainty that
fits its parameters on whatever data it is given at run time, so the "trained"
artifact is the fixed configuration that drives every forecast: the planning
horizons to emit, the simulation quantiles, the number of bootstrap draws, and
the random seed. Storing plain Python primitives keeps the pickle robust across
library versions (the most common cause of unpickling failures)."""
import pickle

import forecaster

MODEL = {
    "horizons": [30, 60, 90],
    "quantiles": forecaster.QUANTILES,
    "n_sims": forecaster.N_SIMS,
    "seed": 42,
    "version": "1.0",
}


def main():
    with open("./pickle/model.pkl", "wb") as f:
        pickle.dump(MODEL, f)
    print("Wrote ./pickle/model.pkl:", MODEL)


if __name__ == "__main__":
    main()
