"""Clean the flights: remove impossible scheduled durations (rule in the README), keep everything else."""
import numpy as np
import pandas as pd

from src.config import CLEAN_PARQUET, FLIGHTS_PARQUET, TABLES_DIR
from src.data_loading import load_flights
from src.profiling import track

MAX_SPEED_MPH = 650
FIT_MONTHS = range(1, 7)  # the route medians use January-June only, like every other fitted step


def route_median_duration(df, months=FIT_MONTHS):
    """Median scheduled duration per route (origin, dest), from `months` only, so no information from the
    validation and test months is used. Routes that first appear later fall back to their own median."""
    keys = ["origin", "dest"]
    fit = df[df["month"].isin(list(months))].groupby(keys, observed=True)["crs_elapsed_time"].median()
    median = pd.Series(pd.MultiIndex.from_frame(df[keys]).map(fit), index=df.index)
    fallback = df.groupby(keys, observed=True)["crs_elapsed_time"].transform("median")
    return median.fillna(fallback)


def clean_flights(df):
    """Return (clean df, report). Cancelled and diverted flights are kept and labelled in `status`."""
    dur = df["crs_elapsed_time"]
    route_median = route_median_duration(df)
    rules = {
        "duration missing or <= 0": ~(dur > 0),
        f"speed > {MAX_SPEED_MPH} mph": (dur > 0) & (df["distance"] / (dur / 60) > MAX_SPEED_MPH),
        "duration < 0.5x or > 2x route median": (dur > 0) & ((dur < 0.5 * route_median) | (dur > 2 * route_median)),
    }
    remove = np.logical_or.reduce(list(rules.values()))

    report = pd.DataFrame({"flights_flagged": [int(m.sum()) for m in rules.values()]}, index=list(rules))
    report.loc["total removed (any rule)"] = int(remove.sum())
    report["share_pct"] = (100 * report["flights_flagged"] / len(df)).round(5)

    out = df[~remove].copy()
    for col in ["crs_dep_time", "crs_arr_time"]:
        out[col] = out[col].replace(2400, 0)
    out["status"] = pd.Categorical(
        np.select([out["cancelled"] == 1, out["diverted"] == 1], ["cancelled", "diverted"], "completed"),
        categories=["completed", "cancelled", "diverted"])
    return out.reset_index(drop=True), report


def make_clean_parquet():
    """Clean the full Parquet file and save the clean file and the removal report."""
    with track("clean_flights") as stats:
        df = load_flights(FLIGHTS_PARQUET)
        clean, report = clean_flights(df)
        clean.to_parquet(CLEAN_PARQUET, index=False)
        stats.update(rows_in=len(df), rows_out=len(clean))
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    report.to_csv(TABLES_DIR / "01c_cleaning_report.csv", float_format="%.5f")
    return report
