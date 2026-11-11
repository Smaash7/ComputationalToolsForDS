"""Temporal model selection, held-out evaluation and feature interpretation."""

import json

import numpy as np
import pandas as pd
import joblib
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, brier_score_loss, confusion_matrix, fbeta_score,
    log_loss, precision_score, recall_score, roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from prediction.features import CATEGORICAL, FEATURES
from prediction.resources import measure


def sample(frame, size, seed):
    return frame.sample(n=min(size, len(frame)), random_state=seed)


def design(frame, columns):
    result = frame[columns].copy()
    for column in set(columns) & set(CATEGORICAL):
        result[column] = result[column].fillna("unknown").astype(str)
    for column in set(columns) - set(CATEGORICAL):
        result[column] = pd.to_numeric(result[column], errors="raise").astype(float)
    return result


def make_model(kind, params, columns, seed, workers):
    if kind == "ebm":
        return ExplainableBoostingClassifier(
            random_state=seed, n_jobs=workers, learning_rate=params["learning_rate"],
            interactions=params["interactions"],
        )
    if kind in ("logistic", "xgboost"):
        categorical = [c for c in columns if c in CATEGORICAL]
        numerical = [c for c in columns if c not in CATEGORICAL]
        preprocessing = ColumnTransformer([
            ("numeric", Pipeline([("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                                  ("scale", StandardScaler())]), numerical),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
        ])
        estimator = (
            LogisticRegression(C=params["C"], max_iter=500, random_state=seed)
            if kind == "logistic"
            else XGBClassifier(
                max_depth=params["max_depth"], n_estimators=params["n_estimators"],
                learning_rate=0.05, tree_method="hist", n_jobs=workers, random_state=seed,
            )
        )
        return Pipeline([
            ("preprocess", preprocessing),
            ("model", estimator),
        ])
    raise ValueError(f"Unknown model: {kind}")


def threshold_for_f2(y, probabilities, thresholds, min_specificity=0.0):
    y = np.asarray(y)
    if not (y == 0).any():
        raise ValueError("Threshold selection requires negative examples")
    eligible = []
    for threshold in thresholds:
        predicted = probabilities >= threshold
        specificity = np.mean(~predicted[y == 0])
        if specificity >= min_specificity:
            eligible.append((fbeta_score(y, predicted, beta=2, zero_division=0), float(threshold)))
    if not eligible:
        raise ValueError("No validation threshold meets the specificity constraint; expand the grid or change the constraint")
    return max(eligible, key=lambda item: item[0])[1]


def metrics(y, probability, threshold):
    predicted = probability >= threshold
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    return {
        "base_rate": float(np.mean(y)), "pr_auc": average_precision_score(y, probability),
        "roc_auc": roc_auc_score(y, probability), "log_loss": log_loss(y, probability, labels=[0, 1]),
        "brier": brier_score_loss(y, probability),
        "precision": precision_score(y, predicted, zero_division=0),
        "recall": recall_score(y, predicted, zero_division=0),
        "specificity": tn / (tn + fp) if tn + fp else np.nan,
        "f2": fbeta_score(y, predicted, beta=2, zero_division=0),
        "predicted_positive_rate": float(np.mean(predicted)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def fit_predict(train, test, columns, kind, params, settings):
    model = make_model(kind, params, columns, settings["seed"], settings["workers"])
    xtrain = design(train, columns)
    xtest = design(test, columns)
    with measure() as stats:
        model.fit(xtrain, train["late_arrival"])
    probability = model.predict_proba(xtest)[:, 1]
    return model, probability, stats["seconds"], stats["peak_process_mb"]


def folds(frame, month, settings, train_window=0):
    train_all = frame.loc[frame["month"] < month]
    if train_window:
        train_all = train_all.loc[train_all["month"] >= month - train_window]
    test_all = frame.loc[frame["month"] == month]
    if train_all.empty or test_all.empty:
        raise ValueError(f"Empty training or evaluation partition for month {month}")
    train = sample(train_all, settings["train_size"], settings["seed"] + month)
    test = sample(test_all, settings["evaluation_size"], settings["seed"] + 100 + month)
    if train["late_arrival"].nunique() != 2 or test["late_arrival"].nunique() != 2:
        raise ValueError(f"Both target classes must be present in fold {month}")
    info = {
        "month": month, "train_window": train_window, "available_train": len(train_all), "train_size": len(train),
        "available_evaluation": len(test_all), "evaluation_size": len(test),
        "full_month_base_rate": float(test_all["late_arrival"].mean()),
        "sample_base_rate": float(test["late_arrival"].mean()),
        "unknown_route_fraction": float((test["route_cluster"] == "unknown").mean()),
        "unknown_origin_fraction": float((test["origin_community"] == "unknown").mean()),
        "unknown_dest_fraction": float((test["dest_community"] == "unknown").mean()),
    }
    return train, test, info


def candidates(settings):
    for interactions in settings["interactions"]:
        yield "ebm", {"learning_rate": settings["learning_rate"], "interactions": interactions}
    for c in settings["logistic_c"]:
        yield "logistic", {"C": c}
    for depth in settings["xgb_depth"]:
        yield "xgboost", {"max_depth": depth, "n_estimators": settings["xgb_estimators"]}


def experiment(frame, settings, output):
    output.mkdir(parents=True, exist_ok=True)
    validation = {
        (m, window): folds(frame, m, settings, window)
        for m in settings["validation_months"] for window in settings["train_windows"]
    }
    search_rows, predictions = [], {}
    thresholds = np.linspace(settings["threshold_min"], settings["threshold_max"], settings["threshold_steps"])
    for kind, params in candidates(settings):
        for window in settings["train_windows"]:
            identifier = f"{kind}:{json.dumps({'params': params, 'train_window': window}, sort_keys=True)}"
            predictions[identifier] = []
            for month in settings["validation_months"]:
                train, test, _ = validation[(month, window)]
                print(f"tuning {identifier}, month {month}", flush=True)
                _, prob, seconds, peak = fit_predict(train, test, FEATURES["all"], kind, params, settings)
                predictions[identifier].append((test["late_arrival"].to_numpy(), prob))
                search_rows.append({
                    "candidate": identifier, "month": month, "pr_auc": average_precision_score(test["late_arrival"], prob),
                    "roc_auc": roc_auc_score(test["late_arrival"], prob), "train_seconds": seconds, "process_peak_mb": peak,
                })
    search = pd.DataFrame(search_rows)
    ranking = search.groupby("candidate", as_index=False).agg(
        mean_pr_auc=("pr_auc", "mean"), mean_roc_auc=("roc_auc", "mean")
    ).sort_values(["mean_pr_auc", "mean_roc_auc"], ascending=False)
    selected = ranking.iloc[0]["candidate"]
    kind, encoded = selected.split(":", 1)
    chosen = json.loads(encoded)
    params, window = chosen["params"], chosen["train_window"]
    selected_predictions = predictions[selected]
    y = np.concatenate([p[0] for p in selected_predictions])
    prob = np.concatenate([p[1] for p in selected_predictions])
    threshold = threshold_for_f2(y, prob, thresholds, settings["min_validation_specificity"])
    ebm_candidates = ranking.loc[ranking["candidate"].str.startswith("ebm:")]
    if ebm_candidates.empty:
        raise ValueError("At least one EBM candidate is required for the RQ3 comparison")
    ebm_identifier = ebm_candidates.iloc[0]["candidate"]
    ebm_choice = json.loads(ebm_identifier.split(":", 1)[1])
    ebm_params, ebm_window = ebm_choice["params"], ebm_choice["train_window"]
    ebm_validation = predictions[ebm_identifier]
    ebm_threshold = threshold_for_f2(
        np.concatenate([p[0] for p in ebm_validation]),
        np.concatenate([p[1] for p in ebm_validation]),
        thresholds, settings["min_validation_specificity"],
    )
    base_y = np.concatenate([validation[(m, window)][1]["late_arrival"].to_numpy() for m in settings["validation_months"]])
    base_prob = np.concatenate([validation[(m, window)][1]["route_past_delay_rate"].to_numpy() for m in settings["validation_months"]])
    baseline_threshold = threshold_for_f2(base_y, base_prob, thresholds, settings["min_validation_specificity"])
    search.to_csv(output / "validation_search.csv", index=False)
    ranking.to_csv(output / "validation_ranking.csv", index=False)
    with (output / "selection.json").open("w", encoding="utf-8") as file:
        json.dump({"model": kind, "parameters": params, "train_window": window, "threshold": threshold,
                   "baseline_threshold": baseline_threshold, "selection_metric": "mean validation PR-AUC",
                   "ebm_reference": {"parameters": ebm_params, "train_window": ebm_window, "threshold": ebm_threshold},
                   "threshold_metric": "pooled validation F2 subject to minimum validation specificity",
                   "settings": settings}, file, indent=2)

    rows, importance_rows, terms, fold_info = [], [], [], []
    for month in settings["test_months"]:
        train, test, info = folds(frame, month, settings, window)
        fold_info.append(info)
        baseline = test["route_past_delay_rate"].to_numpy()
        rows.append({"month": month, "configuration": "route_baseline", "threshold": baseline_threshold,
                     "train_seconds": 0.0, "process_peak_mb": 0.0,
                     **metrics(test["late_arrival"], baseline, baseline_threshold)})
        for config, columns in FEATURES.items():
            print(f"test {config}, month {month}, model {kind}", flush=True)
            model, prob, seconds, peak = fit_predict(train, test, columns, kind, params, settings)
            rows.append({"month": month, "configuration": config, "threshold": threshold,
                         "train_seconds": seconds, "process_peak_mb": peak,
                         **metrics(test["late_arrival"], prob, threshold)})
            if config != "all":
                continue
            if month == max(settings["test_months"]):
                joblib.dump(model, output / "final_model.joblib", compress=3)
            subset = sample(test, settings["importance_size"], settings["seed"] + 200 + month)
            explanation = permutation_importance(
                model, design(subset, columns), subset["late_arrival"], scoring="average_precision",
                n_repeats=settings["importance_repeats"], random_state=settings["seed"], n_jobs=1,
            )
            for name, mean, std in zip(columns, explanation.importances_mean, explanation.importances_std):
                importance_rows.append({"month": month, "feature": name, "mean_ap_decrease": mean, "std_ap_decrease": std})
            if kind == "ebm":
                for name, value in zip(model.term_names_, model.term_importances()):
                    terms.append({"month": month, "term": name, "importance": float(value)})
        if kind != "ebm":
            ebm_train, _, _ = folds(frame, month, settings, ebm_window)
            for config, columns in FEATURES.items():
                print(f"test EBM reference {config}, month {month}", flush=True)
                ebm, reference_prob, seconds, peak = fit_predict(
                    ebm_train, test, columns, "ebm", ebm_params, settings
                )
                rows.append({"month": month, "configuration": f"ebm_{config}", "threshold": ebm_threshold,
                             "train_seconds": seconds, "process_peak_mb": peak,
                             **metrics(test["late_arrival"], reference_prob, ebm_threshold)})
                if config == "all":
                    for name, value in zip(ebm.term_names_, ebm.term_importances()):
                        terms.append({"month": month, "term": name, "importance": float(value)})
        pd.DataFrame(rows).to_csv(output / "test_metrics.csv", index=False)
        pd.DataFrame(importance_rows).to_csv(output / "permutation_importance.csv", index=False)
        pd.DataFrame(fold_info).to_csv(output / "test_folds.csv", index=False)
        if terms:
            pd.DataFrame(terms).to_csv(output / "ebm_terms.csv", index=False)
    return pd.DataFrame(rows)
