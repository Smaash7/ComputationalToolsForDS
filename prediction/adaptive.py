"""Exploratory rolling threshold calibration using the preceding month's labels."""

import json

import numpy as np
import pandas as pd

from prediction.experiment import fit_predict, folds, metrics, threshold_for_f2
from prediction.features import FEATURES


def adaptive_experiment(frame, settings, output):
    selection_path = output / "selection.json"
    if not selection_path.is_file():
        raise FileNotFoundError(f"Run the primary experiment before adaptive calibration: {selection_path}")
    with selection_path.open(encoding="utf-8") as file:
        selection = json.load(file)
    thresholds = np.linspace(settings["threshold_min"], settings["threshold_max"], settings["threshold_steps"])
    rows = []
    for month in settings["test_months"]:
        calibration_month = month - 1
        print(f"calibrating adaptive threshold on month {calibration_month}, evaluating month {month}", flush=True)
        prior_train, calibration, _ = folds(frame, calibration_month, settings, selection["train_window"])
        _, calibration_prob, _, _ = fit_predict(
            prior_train, calibration, FEATURES["all"], selection["model"], selection["parameters"], settings
        )
        threshold = threshold_for_f2(
            calibration["late_arrival"].to_numpy(), calibration_prob, thresholds,
            settings["min_validation_specificity"],
        )
        train, test, info = folds(frame, month, settings, selection["train_window"])
        _, probability, seconds, peak = fit_predict(
            train, test, FEATURES["all"], selection["model"], selection["parameters"], settings
        )
        rows.append({
            "month": month, "calibration_month": calibration_month, "threshold": threshold,
            "train_size": info["train_size"], "evaluation_size": info["evaluation_size"],
            "train_seconds": seconds, "process_peak_mb": peak,
            **metrics(test["late_arrival"], probability, threshold),
        })
        pd.DataFrame(rows).to_csv(output / "adaptive_metrics.csv", index=False)
    with (output / "adaptive_selection.json").open("w", encoding="utf-8") as file:
        json.dump({
            "method": "maximize F2 subject to minimum specificity on preceding month's out-of-sample predictions",
            "specificity_minimum": settings["min_validation_specificity"],
            "latest_test_month": rows[-1]["month"],
            "latest_threshold": rows[-1]["threshold"],
            "calibration_month": rows[-1]["calibration_month"],
            "caveat": "All 2024 test months were inspected during prior iterations; adaptive results are exploratory.",
        }, file, indent=2)
    return pd.DataFrame(rows)
