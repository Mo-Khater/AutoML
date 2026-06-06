from __future__ import annotations

import argparse
import csv
from pathlib import Path
from time import perf_counter
from datetime import datetime

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


DATA_ROOT = Path(__file__).resolve().parent / "data"
DEFAULT_RESULTS_FILE = DATA_ROOT / "autogluon_processed_vs_unprocessed_results.csv"
DEFAULT_MODEL_ROOT = Path(__file__).resolve().parent / "autogluon_runs"


DATASET_CONFIGS = {
    "PurchaseStatus": {
        "processed_dir": DATA_ROOT / "PurchaseStatus",
        "raw_file": DATA_ROOT / "unprocessed" / "customer_purchase_data.csv",
        "target": "PurchaseStatus",
    },
    "student_report": {
        "processed_dir": DATA_ROOT / "student_report",
        "raw_file": DATA_ROOT / "unprocessed" / "student_placement_synthetic.csv",
        "target": "placement_status",
    },
    "titanic_preprocessing": {
        "processed_dir": DATA_ROOT / "titanic_preprocessing",
        "raw_file": DATA_ROOT / "unprocessed" / "Titanic-Dataset.csv",
        "target": "Survived",
    },
    "heart": {
        "processed_dir": DATA_ROOT / "heart",
        "raw_file": DATA_ROOT / "unprocessed" / "heart.csv",
        "target": "target",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare AutoGluon on processed splits with feature_generator disabled "
            "against AutoGluon on raw CSV data with feature generation enabled."
        )
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=sorted(DATASET_CONFIGS),
        default=sorted(DATASET_CONFIGS),
        help="Datasets to benchmark.",
    )
    parser.add_argument(
        "--time-limit",
        type=int,
        default=1500,
        help="AutoGluon fit time limit in seconds for each run.",
    )
    parser.add_argument(
        "--presets",
        default="medium_quality",
        help="AutoGluon presets passed to TabularPredictor.fit.",
    )
    parser.add_argument(
        "--results-file",
        type=Path,
        default=DEFAULT_RESULTS_FILE,
        help="CSV file to append results to.",
    )
    parser.add_argument(
        "--model-root",
        type=Path,
        default=DEFAULT_MODEL_ROOT,
        help="Directory where AutoGluon run artifacts will be stored.",
    )
    return parser.parse_args()


def load_processed_split(dataset_dir: Path) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    X_train = pd.read_csv(dataset_dir / "X_train_processed.csv")
    y_train = pd.read_csv(dataset_dir / "y_train_processed.csv").iloc[:, 0]
    X_val = pd.read_csv(dataset_dir / "X_val_processed.csv")
    y_val = pd.read_csv(dataset_dir / "y_val_processed.csv").iloc[:, 0]
    return X_train, y_train, X_val, y_val


def load_unprocessed_split(raw_file: Path, target: str) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    df = pd.read_csv(raw_file)
    X = df.drop(columns=[target])
    y = df[target]
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )
    return X_train, y_train, X_val, y_val


def infer_problem_type(y: pd.Series) -> str:
    return "binary" if y.nunique(dropna=False) == 2 else "multiclass"


def save_result(results_file: Path, result: dict[str, object]) -> None:
    results_file.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "variant",
        "source_data",
        "feature_generator",
        "fit_time_sec",
        "val_accuracy",
        "train_rows",
        "val_rows",
        "raw_file",
        "processed_dir",
    ]
    file_exists = results_file.exists()
    with results_file.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(result)


def run_autogluon_variant(
    *,
    dataset_name: str,
    variant: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    time_limit: int,
    presets: str,
    feature_generator,
    model_root: Path,
) -> dict[str, object]:
    from autogluon.tabular import TabularPredictor

    train_df = X_train.copy()
    train_df["label"] = y_train.to_numpy()
    val_df = X_val.copy()
    val_df["label"] = y_val.to_numpy()

    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = model_root / dataset_name / f"{variant}_{run_stamp}"
    model_path.parent.mkdir(parents=True, exist_ok=True)

    predictor = TabularPredictor(
        label="label",
        problem_type=infer_problem_type(y_train),
        eval_metric="accuracy",
        verbosity=2,
        path=str(model_path),
    )

    started_at = perf_counter()
    fit_kwargs = {
        "train_data": train_df,
        "presets": presets,
        "time_limit": time_limit,
    }
    if feature_generator is not ...:
        fit_kwargs["feature_generator"] = feature_generator

    predictor.fit(
        **fit_kwargs,
    )
    fit_time_sec = perf_counter() - started_at

    predictions = predictor.predict(val_df.drop(columns=["label"]))
    val_accuracy = accuracy_score(y_val, predictions)

    print(f"=== {dataset_name} | {variant} ===")
    print("fit_time_sec", round(fit_time_sec, 3))
    print("val_accuracy", float(val_accuracy))
    print("leaderboard")
    print(predictor.leaderboard(val_df, silent=True))

    return {
        "dataset": dataset_name,
        "variant": variant,
        "fit_time_sec": round(fit_time_sec, 3),
        "val_accuracy": float(val_accuracy),
        "train_rows": len(X_train),
        "val_rows": len(X_val),
    }


def main() -> None:
    args = parse_args()

    try:
        import autogluon.tabular  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "AutoGluon is not installed in the active environment. "
            "Install `autogluon.tabular` before running this script."
        ) from exc

    for dataset_name in args.datasets:
        config = DATASET_CONFIGS[dataset_name]
        print()
        print(f"######## {dataset_name} ########")

        processed_split = load_processed_split(config["processed_dir"])
        processed_result = run_autogluon_variant(
            dataset_name=dataset_name,
            variant="processed_feature_generator_false",
            X_train=processed_split[0],
            y_train=processed_split[1],
            X_val=processed_split[2],
            y_val=processed_split[3],
            time_limit=args.time_limit,
            presets=args.presets,
            feature_generator=None,
            model_root=args.model_root,
        )
        processed_result.update(
            {
                "source_data": "processed",
                "feature_generator": "False",
                "raw_file": str(config["raw_file"]),
                "processed_dir": str(config["processed_dir"]),
            }
        )
        save_result(args.results_file, processed_result)

        raw_split = load_unprocessed_split(config["raw_file"], config["target"])
        raw_result = run_autogluon_variant(
            dataset_name=dataset_name,
            variant="unprocessed_feature_generator_true",
            X_train=raw_split[0],
            y_train=raw_split[1],
            X_val=raw_split[2],
            y_val=raw_split[3],
            time_limit=args.time_limit,
            presets=args.presets,
            feature_generator=...,
            model_root=args.model_root,
        )
        raw_result.update(
            {
                "source_data": "unprocessed",
                "feature_generator": "True",
                "raw_file": str(config["raw_file"]),
                "processed_dir": str(config["processed_dir"]),
            }
        )
        save_result(args.results_file, raw_result)


if __name__ == "__main__":
    main()
