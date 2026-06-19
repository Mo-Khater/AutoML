from __future__ import annotations

from pathlib import Path
from time import perf_counter
import csv

import pandas as pd
from sklearn.metrics import accuracy_score

from automl.classification import AutoMLClassifier


DATA_ROOT = Path(r"d:\autosklearn\AutoML\data")
RESULTS_FILE = DATA_ROOT / "run_local_data_results.csv"


def load_dataset_split(dataset_dir: Path) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    X_train = pd.read_csv(dataset_dir / "X_train_processed.csv")
    y_train = pd.read_csv(dataset_dir / "y_train_processed.csv").iloc[:, 0]
    X_val = pd.read_csv(dataset_dir / "X_val_processed.csv")
    y_val = pd.read_csv(dataset_dir / "y_val_processed.csv").iloc[:, 0]
    return X_train, y_train, X_val, y_val


def choose_runtime_settings(_: int) -> dict[str, int]:
    return {"time_budget": 1500, "n_trials": 100}


def save_result(result: dict[str, object]) -> None:
    fieldnames = ["dataset", "framework", "fit_time_sec", "val_accuracy"]
    file_exists = RESULTS_FILE.exists()
    with RESULTS_FILE.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(result)


def run_custom_automl(
    dataset_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> dict[str, object]:
    settings = choose_runtime_settings(len(X_train))
    model = AutoMLClassifier(
        time_budget=settings["time_budget"],
        n_trials=settings["n_trials"],
        cv=3,
        random_state=42,
        ensemble=True,
        ensemble_strategy="stacked",
        verbose=1,
        stacked_include_original_features_in_meta=True,
        stacked_include_base_predictions=False,
        n_jobs=1,
    )

    started_at = perf_counter()
    model.fit(X_train, y_train)
    duration = perf_counter() - started_at
    preds = model.predict(X_val)
    val_accuracy = accuracy_score(y_val, preds)

    print("=== Custom AutoML ===")
    print("dataset", dataset_name)
    print("fit_time_sec", round(duration, 3))
    print("best_score", model.best_score_)
    print("val_accuracy", val_accuracy)
    print("leaderboard_top3", model.leaderboard(top_n=3))

    result = {
        "dataset": dataset_name,
        "framework": "custom_automl",
        "fit_time_sec": round(duration, 3),
        "val_accuracy": float(val_accuracy),
    }
    save_result(result)
    return result


def run_autogluon(
    dataset_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> dict[str, object]:
    try:
        from autogluon.tabular import TabularPredictor
    except ImportError:
        print("=== AutoGluon ===")
        print("dataset", dataset_name)
        print("status", "not_installed")
        return

    settings = choose_runtime_settings(len(X_train))
    train_df = X_train.copy()
    train_df["label"] = y_train.to_numpy()
    val_df = X_val.copy()
    val_df["label"] = y_val.to_numpy()

    predictor = TabularPredictor(
        label="label",
        problem_type="binary" if y_train.nunique() == 2 else "multiclass",
        eval_metric="accuracy",
        verbosity=2,
    
    )

    started_at = perf_counter()
    predictor = predictor.fit(
        train_data=train_df,
        feature_generator=None,
        presets="medium_quality",
        time_limit=settings["time_budget"],
    )
    duration = perf_counter() - started_at

    preds = predictor.predict(val_df.drop(columns=["label"]))
    accuracy = accuracy_score(y_val, preds)

    print("=== AutoGluon ===")
    print("dataset", dataset_name)
    print("fit_time_sec", round(duration, 3))
    print("val_accuracy", float(accuracy))
    print("leaderboard")
    print(predictor.leaderboard(val_df, silent=True))

    result = {
        "dataset": dataset_name,
        "framework": "autogluon",
        "fit_time_sec": round(duration, 3),
        "val_accuracy": float(accuracy),
    }
    save_result(result)
    return result


def main() -> None:
    # custom_datasets = ['heart','PurchaseStatus', 'student_report', 'titanic_preprocessing']
    custom_datasets = ['titanic_preprocessing']
    # custom_datasets = ['heart']
    # custom_datasets = ['datascienceproject']

    # Run both frameworks on the same processed splits for fair comparison.
    for dataset in custom_datasets:
        dataset_dir = DATA_ROOT / dataset
        if dataset_dir.exists():
            print()
            print(f"######## {dataset} ########")
            X_train, y_train, X_val, y_val = load_dataset_split(dataset_dir)
            run_custom_automl(dataset, X_train, y_train, X_val, y_val)
            # run_autogluon(dataset, X_train, y_train, X_val, y_val)


if __name__ == "__main__":
    main()
