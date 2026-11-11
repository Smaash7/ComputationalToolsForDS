"""Construct pre-flight features and strictly historical delay rates."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.data_loading import load_flights

KEYS = ["origin", "dest", "op_unique_carrier"]
FLIGHT = ["month", "day_of_week", "sched_hour", "op_unique_carrier", "distance"]
RQ1 = ["route_cluster", "cluster_past_delay_rate"]
RQ2 = [
    "origin_community", "dest_community", "crosses_community",
    "origin_betweenness", "dest_betweenness",
]
BASE = ["route_past_delay_rate"]
FEATURES = {
    "flight": BASE + FLIGHT,
    "flight_rq1": BASE + FLIGHT + RQ1,
    "flight_rq2": BASE + FLIGHT + RQ2,
    "all": BASE + FLIGHT + RQ1 + RQ2,
}
CATEGORICAL = ["op_unique_carrier", "route_cluster", "origin_community", "dest_community"]
PROFILE_COLUMNS = [
    "mean_distance", "log_flights_per_week", "morning_share",
    "evening_night_share", "weekend_share",
]


def _route_profiles(frame, months):
    """Reproduce RQ1 Model B profiles from completed, valid scheduled flights."""
    data = frame.loc[frame["month"].isin(months)].copy()
    valid_time = (
        data["crs_dep_time"].notna()
        & data["crs_dep_time"].between(0, 2400)
        & ((data["crs_dep_time"] % 100) < 60)
    )
    data = data.loc[
        valid_time & (data["distance"] > 0) & (data["crs_elapsed_time"] > 0)
    ].copy()
    hour = (data["crs_dep_time"] // 100) % 24
    data["morning"] = hour.between(5, 11).astype("int8")
    data["evening_night"] = ((hour >= 18) | (hour < 5)).astype("int8")
    data["weekend"] = data["day_of_week"].isin([6, 7]).astype("int8")
    profiles = data.groupby(KEYS, observed=True).agg(
        n_flights=("distance", "size"), mean_distance=("distance", "mean"),
        morning_share=("morning", "mean"), evening_night_share=("evening_night", "mean"),
        weekend_share=("weekend", "mean"),
    ).reset_index()
    end = pd.Timestamp(2025, 1, 1) if max(months) == 12 else pd.Timestamp(2024, max(months) + 1, 1)
    weeks = (end - pd.Timestamp(2024, 1, 1)).days / 7
    profiles["log_flights_per_week"] = np.log1p(profiles["n_flights"] / weeks)
    return profiles


def assign_clusters(flights, routes):
    """Use RQ1 labels; assign other routes to fixed centroids using prior profiles."""
    if routes.duplicated(KEYS).any() or routes["route_cluster"].isna().any():
        raise ValueError("RQ1 route lookup must have unique keys and non-null cluster labels")
    june = _route_profiles(flights, list(range(1, 7)))
    weeks = (pd.Timestamp(2024, 7, 1) - pd.Timestamp(2024, 1, 1)).days / 7
    eligible = june.loc[june["n_flights"] / weeks >= 30 / 26]
    fitted = eligible.merge(routes, on=KEYS, how="inner", validate="one_to_one")
    if len(fitted) != len(eligible) or len(fitted) != len(routes):
        raise ValueError("RQ1 lookup does not match reconstructed January-June route profiles; rerun notebook 02")
    scaler = StandardScaler().fit(fitted[PROFILE_COLUMNS])
    transformed = scaler.transform(fitted[PROFILE_COLUMNS])
    labels = fitted["route_cluster"].to_numpy()
    cluster_ids = np.sort(np.unique(labels))
    centroids = np.vstack([transformed[labels == c].mean(axis=0) for c in cluster_ids])
    nearest = cluster_ids[((transformed[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2).argmin(axis=1)]
    if (nearest != labels).mean() > 0.001:
        raise ValueError("RQ1 lookup labels do not match reconstructed centroids; rerun notebook 02")

    data = flights.merge(routes, on=KEYS, how="left", validate="many_to_one")
    # Earlier-than-July rows use the frozen June label only as a training representation.
    # For unknown routes, each monthly snapshot uses schedules from previous months only.
    for month in range(2, 13):
        missing = (data["month"] == month) & data["route_cluster"].isna()
        if not missing.any():
            continue
        wanted = data.loc[missing, KEYS].drop_duplicates()
        previous = flights.loc[flights["month"] < month].merge(wanted, on=KEYS, how="inner")
        if previous.empty:
            continue
        profiles = _route_profiles(previous, list(range(1, month)))
        if profiles.empty:
            continue
        x = scaler.transform(profiles[PROFILE_COLUMNS])
        profiles["assigned_cluster"] = cluster_ids[((x[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2).argmin(axis=1)]
        mapping = profiles.set_index(KEYS)["assigned_cluster"]
        keys = pd.MultiIndex.from_frame(data.loc[missing, KEYS])
        data.loc[missing, "route_cluster"] = mapping.reindex(keys).to_numpy()
    data["route_cluster"] = data["route_cluster"].astype("Int64").astype("string").fillna("unknown")
    return data


def add_history(frame, keys, output, prior):
    """Attach expanding monthly rates from strictly preceding calendar months."""
    totals = frame.groupby(keys + ["month"], observed=True, dropna=False)["late_arrival"].agg(["sum", "count"]).reset_index()
    totals = totals.sort_values(keys + ["month"])
    grouped = totals.groupby(keys, observed=True, dropna=False)
    totals["previous_sum"] = grouped["sum"].cumsum() - totals["sum"]
    totals["previous_count"] = grouped["count"].cumsum() - totals["count"]
    totals[output] = (totals["previous_sum"] / totals["previous_count"].replace(0, np.nan)).fillna(prior)
    return frame.merge(totals[keys + ["month", output]], on=keys + ["month"], how="left", validate="many_to_one")


def prepare_data(flight_path, route_path, airport_path, prior):
    columns = [
        "month", "day_of_week", "op_unique_carrier", "origin", "dest", "crs_dep_time",
        "crs_elapsed_time", "distance", "status", "arr_delay",
    ]
    flights = load_flights(flight_path, columns=columns)
    flights = flights.loc[flights["status"] == "completed"].copy()
    if flights.empty:
        raise ValueError("No completed flights with an observed arrival delay")
    flights["sched_hour"] = ((flights["crs_dep_time"] // 100) % 24).astype("float32")
    routes = pd.read_csv(route_path)
    flights = assign_clusters(flights, routes)
    flights = flights.loc[flights["arr_delay"].notna()].copy()
    flights["late_arrival"] = (flights["arr_delay"] >= 15).astype("int8")
    airports = pd.read_csv(airport_path, usecols=["airport", "louvain", "betweenness"])
    if airports["airport"].duplicated().any():
        raise ValueError("Airport lookup contains duplicate airports")
    for side in ("origin", "dest"):
        lookup = airports.rename(columns={
            "airport": side, "louvain": f"{side}_community", "betweenness": f"{side}_betweenness",
        })
        flights = flights.merge(lookup, on=side, how="left", validate="many_to_one")
        flights[f"{side}_community"] = flights[f"{side}_community"].astype("Int64").astype("string").fillna("unknown")
    known = (flights["origin_community"] != "unknown") & (flights["dest_community"] != "unknown")
    flights["crosses_community"] = np.where(
        known, (flights["origin_community"] != flights["dest_community"]).astype(float), np.nan
    )
    flights = add_history(flights, KEYS, "route_past_delay_rate", prior)
    flights = add_history(flights, ["route_cluster"], "cluster_past_delay_rate", prior)
    return flights.drop(columns=["arr_delay", "status", "crs_dep_time", "crs_elapsed_time"])
