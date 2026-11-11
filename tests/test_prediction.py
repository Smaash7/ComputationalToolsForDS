import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from prediction.experiment import design, folds, make_model, metrics, threshold_for_f2
from prediction.features import FEATURES, add_history, assign_clusters
from prediction.main import load_config, ROOT
from prediction.adaptive import adaptive_experiment


class HistoricalFeaturesTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame({
            "route": ["A", "A", "A", "A", "B", "B"],
            "month": [1, 1, 3, 4, 1, 3],
            "late_arrival": [1, 0, 1, 0, 0, 1],
        })

    def test_rates_only_use_previous_months(self):
        result = add_history(self.frame, ["route"], "rate", 0.2)
        np.testing.assert_allclose(result["rate"], [0.2, 0.2, 0.5, 2 / 3, 0.2, 0.0])

    def test_changing_current_or_future_targets_cannot_change_earlier_features(self):
        original = add_history(self.frame, ["route"], "rate", 0.2)
        changed = self.frame.copy()
        changed.loc[changed["month"] >= 3, "late_arrival"] = 1 - changed.loc[changed["month"] >= 3, "late_arrival"]
        revised = add_history(changed, ["route"], "rate", 0.2)
        np.testing.assert_allclose(original.loc[original.month <= 3, "rate"], revised.loc[revised.month <= 3, "rate"])

    def test_unseen_route_uses_only_previous_schedule_profiles(self):
        rows = []
        for destination, distance in (("A", 100), ("B", 1000)):
            for index in range(40):
                rows.append({
                    "origin": "O", "dest": destination, "op_unique_carrier": "AA",
                    "month": index % 6 + 1, "day_of_week": 2, "crs_dep_time": 900,
                    "crs_elapsed_time": 60, "distance": distance,
                })
        for month in (7, 8):
            rows.append({
                "origin": "O", "dest": "C", "op_unique_carrier": "AA",
                "month": month, "day_of_week": 2, "crs_dep_time": 900,
                "crs_elapsed_time": 60, "distance": 100,
            })
        routes = pd.DataFrame({
            "origin": ["O", "O"], "dest": ["A", "B"], "op_unique_carrier": ["AA", "AA"],
            "route_cluster": [0, 1],
        })
        result = assign_clusters(pd.DataFrame(rows), routes)
        unseen = result.loc[result["dest"] == "C"].set_index("month")
        self.assertEqual(unseen.loc[7, "route_cluster"], "unknown")
        self.assertEqual(unseen.loc[8, "route_cluster"], "0")


class ExperimentTests(unittest.TestCase):
    def test_f2_threshold_and_confusion_counts(self):
        y = np.array([0, 0, 1, 1])
        probability = np.array([0.1, 0.4, 0.3, 0.8])
        threshold = threshold_for_f2(y, probability, np.array([0.2, 0.5]))
        self.assertEqual(threshold, 0.2)
        values = metrics(y, probability, threshold)
        self.assertEqual((values["tn"], values["fp"], values["fn"], values["tp"]), (1, 1, 0, 2))
        self.assertEqual(values["recall"], 1.0)
        constrained = threshold_for_f2(y, probability, np.array([0.2, 0.5]), min_specificity=1.0)
        self.assertEqual(constrained, 0.5)

    def test_folds_are_sampled_reproducibly_and_temporally(self):
        frame = pd.DataFrame({
            "month": [1] * 10 + [7] * 10,
            "late_arrival": [0, 1] * 10,
            "route_cluster": ["0"] * 20,
            "origin_community": ["0"] * 20,
            "dest_community": ["0"] * 20,
        })
        settings = {"train_size": 8, "evaluation_size": 8, "seed": 42}
        train, test, info = folds(frame, 7, settings)
        again, other, _ = folds(frame, 7, settings)
        self.assertTrue(train.index.equals(again.index))
        self.assertTrue(test.index.equals(other.index))
        self.assertTrue((train.month < 7).all())
        self.assertTrue((test.month == 7).all())
        self.assertEqual(info["evaluation_size"], 8)

    def test_recent_window_excludes_older_training_months(self):
        frame = pd.DataFrame({
            "month": np.repeat(np.arange(1, 9), 4),
            "late_arrival": [0, 1, 0, 1] * 8,
            "route_cluster": ["0"] * 32,
            "origin_community": ["0"] * 32,
            "dest_community": ["0"] * 32,
        })
        train, test, info = folds(frame, 8, {"train_size": 100, "evaluation_size": 100, "seed": 42}, 2)
        self.assertEqual(set(train.month), {6, 7})
        self.assertEqual(set(test.month), {8})
        self.assertEqual(info["train_window"], 2)

    def test_full_configuration_has_all_feature_groups(self):
        for config in ("flight", "flight_rq1", "flight_rq2"):
            self.assertTrue(set(FEATURES[config]).issubset(FEATURES["all"]))
        self.assertEqual(len(FEATURES["all"]), len(set(FEATURES["all"])))

    def test_logistic_handles_unknown_categories_and_missing_numeric_values(self):
        columns = ["op_unique_carrier", "distance"]
        training = pd.DataFrame({"op_unique_carrier": ["AA", "BB", "AA", "BB"], "distance": [1, 2, 3, 4]})
        model = make_model("logistic", {"C": 1.0}, columns, 42, 1)
        model.fit(design(training, columns), [0, 1, 0, 1])
        future = pd.DataFrame({"op_unique_carrier": ["unknown"], "distance": [np.nan]})
        self.assertTrue(np.isfinite(model.predict_proba(design(future, columns))).all())

    def test_default_configuration(self):
        settings, paths = load_config(ROOT / "prediction" / "settings.config")
        self.assertEqual(settings["train_size"], 200000)
        self.assertEqual(settings["evaluation_size"], 100000)
        self.assertEqual(settings["validation_months"], [7, 8])
        self.assertEqual(settings["test_months"], [9, 10, 11, 12])
        self.assertEqual(settings["train_windows"], [0, 2, 4])
        self.assertTrue(paths["flights"].is_absolute())

    def test_adaptive_threshold_uses_the_previous_month(self):
        frame = pd.DataFrame({
            "month": np.repeat(np.arange(1, 11), 4),
            "late_arrival": [0, 1, 0, 1] * 10,
            "route_cluster": ["0"] * 40,
            "origin_community": ["0"] * 40,
            "dest_community": ["0"] * 40,
        })
        settings = {
            "seed": 42, "train_size": 100, "evaluation_size": 4,
            "test_months": [9, 10], "threshold_min": 0.1, "threshold_max": 0.9,
            "threshold_steps": 9, "min_validation_specificity": 0.5,
        }

        def fake_fit(train, test, columns, kind, params, config):
            self.assertLess(train.month.max(), test.month.min())
            return None, np.where(test.late_arrival == 1, 0.8, 0.2), 0.1, 1.0

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "selection.json").write_text(json.dumps({"model": "xgboost", "parameters": {}, "train_window": 0}))
            with patch("prediction.adaptive.fit_predict", side_effect=fake_fit):
                result = adaptive_experiment(frame, settings, output)
            self.assertEqual(result["calibration_month"].tolist(), [8, 9])
            self.assertTrue((result["recall"] == 1).all())


if __name__ == "__main__":
    unittest.main()
