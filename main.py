from __future__ import annotations

from pathlib import Path
from time import perf_counter
import csv

import pandas as pd
from sklearn.datasets import load_breast_cancer, load_digits, fetch_covtype
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from automl.classification import AutoMLClassifier


DATA_ROOT = Path(__file__).resolve().parent / "data"
RESULTS_FILE = DATA_ROOT / "main_results.csv"


def save_result(result: dict[str, object]) -> None:
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["dataset", "framework", "fit_time_sec", "test_accuracy"]
    file_exists = RESULTS_FILE.exists()
    with RESULTS_FILE.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(result)


def run_benchmark(
    dataset_name: str,
    X,
    y,
    *,
    test_size: float = 0.2,
    automl_trials: int = 8,
    autogluon_time_limit: int = 60,
) -> None:
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=42,
        stratify=y,
    )

    custom_automl = AutoMLClassifier(
        time_budget=1500,
        n_trials=automl_trials,
        cv=3,
        random_state=42,
        ensemble=True,
        ensemble_size=3,
        verbose=1,
    )
    started_at = perf_counter()
    custom_automl.fit(X_train, y_train)
    custom_duration = perf_counter() - started_at

    print(f"\n=== Dataset: {dataset_name} ===")
    print("=== Custom AutoML ===")
    print("Fit time (sec):", round(custom_duration, 3))
    print("Best score:", custom_automl.best_score_)
    print("Leaderboard:", custom_automl.leaderboard(top_n=5))
    custom_accuracy = custom_automl.score(X_test, y_test)
    print("Test accuracy:", custom_accuracy)

    save_result({
        "dataset": dataset_name,
        "framework": "custom_automl",
        "fit_time_sec": round(custom_duration, 3),
        "test_accuracy": float(custom_accuracy),
    })

    try:
        from autogluon.tabular import TabularPredictor
    except ImportError:
        print("AutoGluon is not installed. Skip comparison or install `autogluon.tabular`.")
        return

    feature_names = [f"feature_{idx}" for idx in range(X.shape[1])]
    train_df = pd.DataFrame(X_train, columns=feature_names)
    train_df["label"] = y_train
    test_df = pd.DataFrame(X_test, columns=feature_names)
    test_df["label"] = y_test

    predictor = TabularPredictor(
        label="label",
        problem_type="binary" if len(set(y)) == 2 else "multiclass",
        eval_metric="accuracy",
        verbosity=2,
    )
    started_at = perf_counter()
    predictor = predictor.fit(train_data=train_df, presets="medium_quality", time_limit=autogluon_time_limit)
    autogluon_duration = perf_counter() - started_at

    ag_predictions = predictor.predict(test_df.drop(columns=["label"]))
    ag_accuracy = accuracy_score(y_test, ag_predictions)

    print("=== AutoGluon ===")
    print("Fit time (sec):", round(autogluon_duration, 3))
    print("Test accuracy:", float(ag_accuracy))
    print("Leaderboard:")
    print(predictor.leaderboard(test_df, silent=True))

    save_result({
        "dataset": dataset_name,
        "framework": "autogluon",
        "fit_time_sec": round(autogluon_duration, 3),
        "test_accuracy": float(ag_accuracy),
    })


def main() -> None:
    datasets = [
        ("breast_cancer", *load_breast_cancer(return_X_y=True), 0.2, 20, 1500),
        ("digits", *load_digits(return_X_y=True), 0.2, 20, 1500),
        ("covtype", *fetch_covtype(return_X_y=True), 0.2, 20, 1500),
    ]

    for dataset_name, X, y, test_size, automl_trials, autogluon_time_limit in datasets:
        run_benchmark(
            dataset_name,
            X,
            y,
            test_size=test_size,
            automl_trials=automl_trials,
            autogluon_time_limit=autogluon_time_limit,
        )


if __name__ == "__main__":
    main()
