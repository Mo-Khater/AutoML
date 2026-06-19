from __future__ import annotations

import argparse
import csv
from pathlib import Path
from time import perf_counter

import pandas as pd
from sklearn.metrics import accuracy_score

from automl.classification import AutoMLClassifier


DATASET_NAME = "ai_student_impact"
DATASET_DIR = Path(__file__).resolve().parent / "data" / DATASET_NAME
RESULTS_FILE = DATASET_DIR / "automl_results.csv"   


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AutoML on the ai_student_impact dataset.")
    parser.add_argument("--time-budget", type=int, default=1500, help="Time budget in seconds.")
    parser.add_argument("--n-trials", type=int, default=100, help="Maximum number of trials.")
    parser.add_argument("--cv", type=int, default=3, help="Cross-validation folds.")
    parser.add_argument("--n-jobs", type=int, default=1, help="Parallel jobs for model training.")
    return parser.parse_args()


def load_dataset_split(dataset_dir: Path) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    X_train = pd.read_csv(dataset_dir / "X_train_processed.csv")
    y_train = pd.read_csv(dataset_dir / "y_train_processed.csv").iloc[:, 0]
    X_val = pd.read_csv(dataset_dir / "X_val_processed.csv")
    y_val = pd.read_csv(dataset_dir / "y_val_processed.csv").iloc[:, 0]
    return X_train, y_train, X_val, y_val


def save_result(result: dict[str, object]) -> None:
    fieldnames = ["dataset", "fit_time_sec", "val_accuracy", "time_budget", "n_trials", "cv"]
    file_exists = RESULTS_FILE.exists()
    with RESULTS_FILE.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(result)


def main() -> None:
    args = parse_args()
    X_train, y_train, X_val, y_val = load_dataset_split(DATASET_DIR)

    model = AutoMLClassifier(
        time_budget=args.time_budget,
        n_trials=args.n_trials,
        cv=args.cv,
        random_state=42,
        ensemble=True,
        ensemble_strategy="stacked",
        verbose=1,
        stacked_include_original_features_in_meta=False,
        stacked_include_base_predictions=False,
        n_jobs=args.n_jobs,
    )

    started_at = perf_counter()
    model.fit(X_train, y_train)
    fit_time_sec = perf_counter() - started_at

    predictions = model.predict(X_val)
    val_accuracy = accuracy_score(y_val, predictions)

    print("=== AutoML ===")
    print("dataset", DATASET_NAME)
    print("fit_time_sec", round(fit_time_sec, 3))
    print("best_score", model.best_score_)
    print("val_accuracy", float(val_accuracy))
    print("leaderboard_top3", model.leaderboard(top_n=3))

    save_result(
        {
            "dataset": DATASET_NAME,
            "fit_time_sec": round(fit_time_sec, 3),
            "val_accuracy": float(val_accuracy),
            "time_budget": args.time_budget,
            "n_trials": args.n_trials,
            "cv": args.cv,
        }
    )


if __name__ == "__main__":
    main()
