from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def compute_basic_classification_metafeatures(X: Any, y: Any) -> dict[str, float]:
    X_df = _to_dataframe(X)
    X_arr = X_df.to_numpy()
    y_arr = np.asarray(y)

    if X_df.ndim != 2:
        raise ValueError("X must be 2-dimensional to compute metafeatures.")
    if y_arr.ndim != 1:
        y_arr = y_arr.reshape(-1)

    n_instances, n_features = X_df.shape
    classes, counts = np.unique(y_arr, return_counts=True)
    probs = counts / counts.sum()

    with np.errstate(divide="ignore", invalid="ignore"):
        class_entropy = float(-(probs * np.log2(probs + 1e-12)).sum())

    numeric_df, categorical_df = _split_feature_types(X_df)
    numeric_arr = numeric_df.to_numpy(dtype=float) if not numeric_df.empty else np.empty((n_instances, 0), dtype=float)

    if numeric_arr.shape[1] > 0:
        means = np.nanmean(numeric_arr, axis=0)
        stds = np.nanstd(numeric_arr, axis=0)
        centered = numeric_arr - means
        with np.errstate(divide="ignore", invalid="ignore"):
            skew = np.nanmean((centered / (stds + 1e-12)) ** 3, axis=0)
            kurt = np.nanmean((centered / (stds + 1e-12)) ** 4, axis=0) - 3.0
    else:
        skew = np.asarray([0.0])
        kurt = np.asarray([0.0])

    missing_mask = X_df.isna().to_numpy()
    n_missing_values = float(missing_mask.sum())
    n_instances_with_missing = float(np.any(missing_mask, axis=1).sum()) if n_instances else 0.0
    n_features_with_missing = float(np.any(missing_mask, axis=0).sum()) if n_features else 0.0

    dataset_ratio = float(n_features / max(n_instances, 1))
    inverse_dataset_ratio = float(n_instances / max(n_features, 1))
    n_categorical = float(categorical_df.shape[1])
    n_numeric = float(numeric_df.shape[1])
    symbol_counts = _categorical_symbol_counts(categorical_df)

    return {
        "ClassEntropy": class_entropy,
        "ClassProbabilityMax": float(probs.max()),
        "ClassProbabilityMean": float(probs.mean()),
        "ClassProbabilityMin": float(probs.min()),
        "ClassProbabilitySTD": float(probs.std()),
        "DatasetRatio": dataset_ratio,
        "InverseDatasetRatio": inverse_dataset_ratio,
        "KurtosisMax": float(np.nanmax(kurt)),
        "KurtosisMean": float(np.nanmean(kurt)),
        "KurtosisMin": float(np.nanmin(kurt)),
        "KurtosisSTD": float(np.nanstd(kurt)),
        "LogDatasetRatio": float(np.log(max(dataset_ratio, 1e-12))),
        "LogInverseDatasetRatio": float(np.log(max(inverse_dataset_ratio, 1e-12))),
        "LogNumberOfFeatures": float(np.log(max(n_features, 1))),
        "LogNumberOfInstances": float(np.log(max(n_instances, 1))),
        "NumberOfCategoricalFeatures": n_categorical,
        "NumberOfClasses": float(len(classes)),
        "NumberOfFeatures": float(n_features),
        "NumberOfFeaturesWithMissingValues": n_features_with_missing,
        "NumberOfInstances": float(n_instances),
        "NumberOfInstancesWithMissingValues": n_instances_with_missing,
        "NumberOfMissingValues": n_missing_values,
        "NumberOfNumericFeatures": n_numeric,
        "PercentageOfFeaturesWithMissingValues": float(n_features_with_missing / max(n_features, 1)),
        "PercentageOfInstancesWithMissingValues": float(n_instances_with_missing / max(n_instances, 1)),
        "PercentageOfMissingValues": float(n_missing_values / max(n_instances * n_features, 1)),
        "RatioNominalToNumerical": float(n_categorical / max(n_numeric, 1.0)),
        "RatioNumericalToNominal": float(n_numeric / max(n_categorical, 1.0)),
        "SkewnessMax": float(np.nanmax(skew)),
        "SkewnessMean": float(np.nanmean(skew)),
        "SkewnessMin": float(np.nanmin(skew)),
        "SkewnessSTD": float(np.nanstd(skew)),
        "SymbolsMax": float(symbol_counts.max(initial=0.0)),
        "SymbolsMean": float(symbol_counts.mean() if symbol_counts.size else 0.0),
        "SymbolsMin": float(symbol_counts.min(initial=0.0)),
        "SymbolsSTD": float(symbol_counts.std() if symbol_counts.size else 0.0),
        "SymbolsSum": float(symbol_counts.sum()),
    }


def compute_basic_regression_metafeatures(X: Any, y: Any) -> dict[str, float]:
    X_df = _to_dataframe(X)
    X_arr = X_df.to_numpy()
    y_arr = np.asarray(y, dtype=float).reshape(-1)

    if X_df.ndim != 2:
        raise ValueError("X must be 2-dimensional to compute metafeatures.")

    n_instances, n_features = X_df.shape
    numeric_df, categorical_df = _split_feature_types(X_df)
    numeric_arr = numeric_df.to_numpy(dtype=float) if not numeric_df.empty else np.empty((n_instances, 0), dtype=float)

    if numeric_arr.shape[1] > 0:
        means = np.nanmean(numeric_arr, axis=0)
        stds = np.nanstd(numeric_arr, axis=0)
        centered = numeric_arr - means
        with np.errstate(divide="ignore", invalid="ignore"):
            skew = np.nanmean((centered / (stds + 1e-12)) ** 3, axis=0)
            kurt = np.nanmean((centered / (stds + 1e-12)) ** 4, axis=0) - 3.0
    else:
        skew = np.asarray([0.0])
        kurt = np.asarray([0.0])

    target_mean = float(np.nanmean(y_arr)) if y_arr.size else 0.0
    target_std = float(np.nanstd(y_arr)) if y_arr.size else 0.0
    centered_y = y_arr - target_mean
    with np.errstate(divide="ignore", invalid="ignore"):
        target_skew = float(np.nanmean((centered_y / (target_std + 1e-12)) ** 3)) if y_arr.size else 0.0
        target_kurtosis = float(np.nanmean((centered_y / (target_std + 1e-12)) ** 4) - 3.0) if y_arr.size else 0.0

    missing_mask = X_df.isna().to_numpy()
    n_missing_values = float(missing_mask.sum())
    n_instances_with_missing = float(np.any(missing_mask, axis=1).sum()) if n_instances else 0.0
    n_features_with_missing = float(np.any(missing_mask, axis=0).sum()) if n_features else 0.0

    dataset_ratio = float(n_features / max(n_instances, 1))
    inverse_dataset_ratio = float(n_instances / max(n_features, 1))
    n_categorical = float(categorical_df.shape[1])
    n_numeric = float(numeric_df.shape[1])
    symbol_counts = _categorical_symbol_counts(categorical_df)

    return {
        "DatasetRatio": dataset_ratio,
        "InverseDatasetRatio": inverse_dataset_ratio,
        "KurtosisMax": float(np.nanmax(kurt)),
        "KurtosisMean": float(np.nanmean(kurt)),
        "KurtosisMin": float(np.nanmin(kurt)),
        "KurtosisSTD": float(np.nanstd(kurt)),
        "LogDatasetRatio": float(np.log(max(dataset_ratio, 1e-12))),
        "LogInverseDatasetRatio": float(np.log(max(inverse_dataset_ratio, 1e-12))),
        "LogNumberOfFeatures": float(np.log(max(n_features, 1))),
        "LogNumberOfInstances": float(np.log(max(n_instances, 1))),
        "NumberOfCategoricalFeatures": n_categorical,
        "NumberOfFeatures": float(n_features),
        "NumberOfFeaturesWithMissingValues": n_features_with_missing,
        "NumberOfInstances": float(n_instances),
        "NumberOfInstancesWithMissingValues": n_instances_with_missing,
        "NumberOfMissingValues": n_missing_values,
        "NumberOfNumericFeatures": n_numeric,
        "PercentageOfFeaturesWithMissingValues": float(n_features_with_missing / max(n_features, 1)),
        "PercentageOfInstancesWithMissingValues": float(n_instances_with_missing / max(n_instances, 1)),
        "PercentageOfMissingValues": float(n_missing_values / max(n_instances * n_features, 1)),
        "RatioNominalToNumerical": float(n_categorical / max(n_numeric, 1.0)),
        "RatioNumericalToNominal": float(n_numeric / max(n_categorical, 1.0)),
        "SkewnessMax": float(np.nanmax(skew)),
        "SkewnessMean": float(np.nanmean(skew)),
        "SkewnessMin": float(np.nanmin(skew)),
        "SkewnessSTD": float(np.nanstd(skew)),
        "SymbolsMax": float(symbol_counts.max(initial=0.0)),
        "SymbolsMean": float(symbol_counts.mean() if symbol_counts.size else 0.0),
        "SymbolsMin": float(symbol_counts.min(initial=0.0)),
        "SymbolsSTD": float(symbol_counts.std() if symbol_counts.size else 0.0),
        "SymbolsSum": float(symbol_counts.sum()),
        "TargetMean": target_mean,
        "TargetSTD": target_std,
        "TargetSkewness": target_skew,
        "TargetKurtosis": target_kurtosis,
        "LogTargetSTD": float(np.log(max(abs(target_std), 1e-12))),
    }


def _to_dataframe(X: Any) -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        return X.copy()

    X_arr = np.asarray(X)
    if X_arr.ndim != 2:
        raise ValueError("X must be 2-dimensional to compute metafeatures.")

    columns = [f"feature_{idx}" for idx in range(X_arr.shape[1])]
    return pd.DataFrame(X_arr, columns=columns)


def _split_feature_types(X_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    numeric_columns: list[str] = []
    categorical_columns: list[str] = []

    for column in X_df.columns:
        series = X_df[column]
        if pd.api.types.is_numeric_dtype(series):
            numeric_columns.append(column)
            continue

        coerced = pd.to_numeric(series, errors="coerce")
        non_missing_original = series.notna().sum()
        non_missing_coerced = coerced.notna().sum()
        if non_missing_original > 0 and non_missing_original == non_missing_coerced:
            numeric_columns.append(column)
        else:
            categorical_columns.append(column)

    numeric_df = X_df[numeric_columns].apply(pd.to_numeric, errors="coerce") if numeric_columns else pd.DataFrame(index=X_df.index)
    categorical_df = X_df[categorical_columns] if categorical_columns else pd.DataFrame(index=X_df.index)
    return numeric_df, categorical_df


def _categorical_symbol_counts(categorical_df: pd.DataFrame) -> np.ndarray:
    if categorical_df.empty:
        return np.asarray([], dtype=float)

    counts = []
    for column in categorical_df.columns:
        series = categorical_df[column]
        counts.append(float(series.dropna().nunique()))
    return np.asarray(counts, dtype=float)
