from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_RESULTS_CSV = Path(__file__).resolve().parent / "data" / "run_local_data_results.csv"
DEFAULT_OUTPUT_PNG = Path(__file__).resolve().parent / "data" / "run_local_data_comparison2.png"


DATASET_ALIASES = {
    "customer_purchase_data.csv": "PurchaseStatus",
    "heart.csv": "heart",
    "student_placement_synthetic.csv": "student_report",
    "Titanic-Dataset.csv": "titanic_preprocessing",
}


def normalize_dataset_name(name: str) -> str:
    return DATASET_ALIASES.get(name, name)


def load_latest_comparable_rows(results_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(results_csv)

    expected_cols = {"dataset", "framework", "fit_time_sec", "val_accuracy"}
    # Some historical files were appended without a header row.
    if not expected_cols.issubset(df.columns) and len(df.columns) == 4:
        df = pd.read_csv(
            results_csv,
            header=None,
            names=["dataset", "framework", "fit_time_sec", "val_accuracy"],
        )

    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in CSV: {sorted(missing)}")

    df = df.copy()
    df["dataset"] = df["dataset"].astype(str).map(normalize_dataset_name)
    df["_row_order"] = range(len(df))

    # Keep only the latest run for each (dataset, framework) pair.
    latest = (
        df.sort_values("_row_order")
        .groupby(["dataset", "framework"], as_index=False)
        .tail(1)
        .drop(columns=["_row_order"])
    )

    # Keep only datasets that have both frameworks for a fair comparison.
    counts = latest.groupby("dataset")["framework"].nunique()
    comparable_datasets = counts[counts >= 2].index
    latest = latest[latest["dataset"].isin(comparable_datasets)].copy()

    return latest.sort_values(["dataset", "framework"]).reset_index(drop=True)


def make_comparison_plot(df: pd.DataFrame, output_png: Path) -> None:
    accuracy_pivot = df.pivot(index="dataset", columns="framework", values="val_accuracy")
    time_pivot = df.pivot(index="dataset", columns="framework", values="fit_time_sec")

    datasets = list(accuracy_pivot.index)
    x = range(len(datasets))
    width = 0.35

    fig, axes = plt.subplots(2, 1, figsize=(12, 10), constrained_layout=True)

    left = [i - width / 2 for i in x]
    right = [i + width / 2 for i in x]

    custom_acc = accuracy_pivot.get("custom_automl", pd.Series(index=accuracy_pivot.index, dtype=float)).values
    ag_acc = accuracy_pivot.get("autogluon", pd.Series(index=accuracy_pivot.index, dtype=float)).values
    axes[0].bar(left, custom_acc, width=width, label="custom_automl")
    axes[0].bar(right, ag_acc, width=width, label="autogluon")
    axes[0].set_title("Validation Accuracy Comparison")
    axes[0].set_ylabel("val_accuracy")
    axes[0].set_xticks(list(x))
    axes[0].set_xticklabels(datasets, rotation=15, ha="right")
    axes[0].set_ylim(0, 1.05)
    axes[0].grid(axis="y", alpha=0.3)
    axes[0].legend()

    custom_time = time_pivot.get("custom_automl", pd.Series(index=time_pivot.index, dtype=float)).values
    ag_time = time_pivot.get("autogluon", pd.Series(index=time_pivot.index, dtype=float)).values
    axes[1].bar(left, custom_time, width=width, label="custom_automl")
    axes[1].bar(right, ag_time, width=width, label="autogluon")
    axes[1].set_title("Fit Time Comparison")
    axes[1].set_ylabel("fit_time_sec")
    axes[1].set_xticks(list(x))
    axes[1].set_xticklabels(datasets, rotation=15, ha="right")
    axes[1].grid(axis="y", alpha=0.3)
    axes[1].legend()

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize local AutoML comparison results")
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS_CSV, help="Path to results CSV")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PNG, help="Output plot PNG path")
    args = parser.parse_args()

    if not args.results.exists():
        raise FileNotFoundError(f"Results file not found: {args.results}")

    latest = load_latest_comparable_rows(args.results)
    if latest.empty:
        raise ValueError("No comparable rows found. Need both frameworks for at least one dataset.")

    make_comparison_plot(latest, args.output)

    summary = (
        latest.pivot(index="dataset", columns="framework", values=["val_accuracy", "fit_time_sec"])
        .sort_index()
    )
    print("Saved plot:", args.output)
    print("\nLatest comparable rows:")
    print(latest.to_string(index=False))
    print("\nPivot summary:")
    print(summary)


if __name__ == "__main__":
    main()
