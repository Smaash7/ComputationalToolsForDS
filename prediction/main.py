"""Run the RQ3 experiments from the repository root or from the notebook."""

import argparse
import configparser
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prediction.experiment import experiment
from prediction.features import FEATURES, prepare_data
from prediction.resources import measure
from prediction.adaptive import adaptive_experiment


def load_config(path):
    parser = configparser.ConfigParser()
    if not parser.read(path):
        raise FileNotFoundError(f"Configuration not found: {path}")
    section = parser["experiment"]
    models = parser["models"]
    integers = lambda value: [int(x.strip()) for x in value.split(",")]
    settings = {
        "seed": section.getint("seed"),
        "train_size": section.getint("train_size"),
        "evaluation_size": section.getint("evaluation_size"),
        "importance_size": section.getint("importance_size"),
        "importance_repeats": section.getint("importance_repeats"),
        "prior_delay_rate": section.getfloat("prior_delay_rate"),
        "validation_months": integers(section["validation_months"]),
        "test_months": integers(section["test_months"]),
        "threshold_min": section.getfloat("threshold_min"),
        "threshold_max": section.getfloat("threshold_max"),
        "threshold_steps": section.getint("threshold_steps"),
        "min_validation_specificity": section.getfloat("min_validation_specificity"),
        "train_windows": integers(section["train_windows"]),
        "interactions": integers(models["ebm_interactions"]),
        "learning_rate": models.getfloat("ebm_learning_rate"),
        "logistic_c": [float(x.strip()) for x in models["logistic_c"].split(",")],
        "xgb_depth": integers(models["xgb_depth"]),
        "xgb_estimators": models.getint("xgb_estimators"),
        "workers": models.getint("workers"),
    }
    if not settings["validation_months"] or not settings["test_months"]:
        raise ValueError("Both validation and test months are required")
    if not settings["train_windows"] or any(window < 0 for window in settings["train_windows"]):
        raise ValueError("Training windows must be non-negative; zero means expanding")
    if min(settings["validation_months"]) < 7 or max(settings["validation_months"]) >= min(settings["test_months"]) or max(settings["test_months"]) > 12:
        raise ValueError("Validation must begin no earlier than July and precede all test months")
    if any(settings[key] < 1 for key in ("train_size", "evaluation_size", "importance_size", "importance_repeats", "workers", "threshold_steps")):
        raise ValueError("Sample sizes, repeats, workers and threshold steps must be positive")
    if not 0 <= settings["prior_delay_rate"] <= 1 or not 0 <= settings["threshold_min"] <= settings["threshold_max"] <= 1 or not 0 <= settings["min_validation_specificity"] <= 1:
        raise ValueError("Prior and thresholds must be probabilities")
    paths = {name: ROOT / value for name, value in parser["paths"].items()}
    return settings, paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "prediction" / "settings.config")
    parser.add_argument("--smoke", action="store_true", help="Use small samples and a single held-out month.")
    parser.add_argument("--adaptive-only", action="store_true", help="Recompute the supplementary adaptive-threshold experiment.")
    args = parser.parse_args()
    settings, paths = load_config(args.config)
    if args.smoke:
        settings.update(train_size=3000, evaluation_size=1000, importance_size=500,
                        importance_repeats=1, test_months=[9], interactions=[0], logistic_c=[1.0],
                        xgb_depth=[3], xgb_estimators=20, train_windows=[0, 2], workers=1)
        paths["output"] = paths["output"] / "smoke"
    for key in ("flights", "routes", "airports"):
        if not paths[key].is_file():
            raise FileNotFoundError(f"Missing {key} input: {paths[key]}; run the prerequisite notebooks")
    print("Preparing features from the cleaned Parquet and RQ1/RQ2 lookups", flush=True)
    with measure() as preparation:
        frame = prepare_data(paths["flights"], paths["routes"], paths["airports"], settings["prior_delay_rate"])
    print(f"Completed flights with target: {len(frame):,}; prevalence: {frame['late_arrival'].mean():.3f}", flush=True)
    print(f"Feature preparation: {preparation['seconds']:.1f} s, peak process RSS {preparation['peak_process_mb']:.1f} MiB", flush=True)
    output = paths["output"]
    output.mkdir(parents=True, exist_ok=True)
    with (output / "feature_manifest.json").open("w", encoding="utf-8") as file:
        json.dump(FEATURES, file, indent=2)
    with (output / "preparation_resources.json").open("w", encoding="utf-8") as file:
        json.dump(preparation, file, indent=2)
    if not args.adaptive_only:
        with measure() as run_resources:
            result = experiment(frame, settings, output)
        with (output / "experiment_resources.json").open("w", encoding="utf-8") as file:
            json.dump(run_resources, file, indent=2)
        print("\nPrimary held-out results:")
        print(result.to_string(index=False))
    with measure() as adaptive_resources:
        adaptive = adaptive_experiment(frame, settings, output)
    with (output / "adaptive_resources.json").open("w", encoding="utf-8") as file:
        json.dump(adaptive_resources, file, indent=2)
    print("\nSupplementary adaptive-threshold results:")
    print(adaptive.to_string(index=False))
    print(f"\nOutputs saved to {output}")


if __name__ == "__main__":
    main()
