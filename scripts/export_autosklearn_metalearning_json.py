from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import arff


SUPPORTED_MODEL_MAPPERS = {
    "adaboost": "_map_adaboost",
    "extra_trees": "_map_extra_trees",
    "gaussian_nb": "_map_gaussian_nb",
    "gradient_boosting": "_map_gradient_boosting",
    "k_nearest_neighbors": "_map_knn",
    "lda": "_map_lda",
    "libsvm_svc": "_map_svc",
    "random_forest": "_map_random_forest",
}


def _decode(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def _none_if_missing(value: Any) -> Any:
    value = _decode(value)
    if value in {"", None, "None"}:
        return None
    return value


def _to_int(value: Any, default: int | None = None) -> int | None:
    value = _none_if_missing(value)
    if value is None:
        return default
    return int(float(value))


def _to_float(value: Any, default: float | None = None) -> float | None:
    value = _none_if_missing(value)
    if value is None:
        return default
    return float(value)


def _to_bool(value: Any, default: bool | None = None) -> bool | None:
    value = _none_if_missing(value)
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    value_str = str(value).strip().lower()
    if value_str in {"true", "1"}:
        return True
    if value_str in {"false", "0"}:
        return False
    return default


def _map_max_features(value: Any) -> str | None:
    value = _none_if_missing(value)
    if value is None:
        return None
    if isinstance(value, str) and value in {"sqrt", "log2"}:
        return value
    try:
        numeric = float(value)
    except Exception:
        return None

    if numeric <= 0.35:
        return "log2"
    if numeric <= 0.8:
        return "sqrt"
    return None


def _map_adaboost(row: dict[str, Any]) -> dict[str, Any] | None:
    return {
        "model_name": "adaboost",
        "params": {
            "n_estimators": _to_int(row.get("classifier:adaboost:n_estimators")),
            "learning_rate": _to_float(row.get("classifier:adaboost:learning_rate")),
        },
    }


def _map_gaussian_nb(row: dict[str, Any]) -> dict[str, Any] | None:
    return {
        "model_name": "gaussian_nb",
        "params": {
            "var_smoothing": 1e-9,
        },
    }


def _map_knn(row: dict[str, Any]) -> dict[str, Any] | None:
    return {
        "model_name": "knn",
        "params": {
            "n_neighbors": _to_int(row.get("classifier:k_nearest_neighbors:n_neighbors")),
            "p": _to_int(row.get("classifier:k_nearest_neighbors:p")),
            "weights": _none_if_missing(row.get("classifier:k_nearest_neighbors:weights")),
        },
    }


def _map_svc(row: dict[str, Any]) -> dict[str, Any] | None:
    return {
        "model_name": "svc",
        "params": {
            "C": _to_float(row.get("classifier:libsvm_svc:C")),
            "kernel": _none_if_missing(row.get("classifier:libsvm_svc:kernel")),
            "gamma": _to_float(row.get("classifier:libsvm_svc:gamma")),
            "degree": _to_int(row.get("classifier:libsvm_svc:degree"), default=3),
            "coef0": _to_float(row.get("classifier:libsvm_svc:coef0"), default=0.0),
        },
    }


def _map_random_forest(row: dict[str, Any]) -> dict[str, Any] | None:
    return {
        "model_name": "random_forest",
        "params": {
            "n_estimators": 512,
            "max_depth": _to_int(row.get("classifier:random_forest:max_depth"), default=64),
            "min_samples_split": _to_int(row.get("classifier:random_forest:min_samples_split"), default=2),
            "min_samples_leaf": _to_int(row.get("classifier:random_forest:min_samples_leaf"), default=1),
            "criterion": _none_if_missing(row.get("classifier:random_forest:criterion")) or "gini",
            "max_features": _map_max_features(row.get("classifier:random_forest:max_features")),
            "bootstrap": _to_bool(row.get("classifier:random_forest:bootstrap"), default=True),
        },
    }


def _map_extra_trees(row: dict[str, Any]) -> dict[str, Any] | None:
    return {
        "model_name": "extra_trees",
        "params": {
            "n_estimators": 512,
            "max_depth": _to_int(row.get("classifier:extra_trees:max_depth"), default=64),
            "min_samples_split": _to_int(row.get("classifier:extra_trees:min_samples_split"), default=2),
            "min_samples_leaf": _to_int(row.get("classifier:extra_trees:min_samples_leaf"), default=1),
            "criterion": _none_if_missing(row.get("classifier:extra_trees:criterion")) or "gini",
            "max_features": _map_max_features(row.get("classifier:extra_trees:max_features")),
        },
    }


def _map_gradient_boosting(row: dict[str, Any]) -> dict[str, Any] | None:
    learning_rate = _to_float(row.get("classifier:gradient_boosting:learning_rate"))
    max_depth = _to_int(row.get("classifier:gradient_boosting:max_depth"))
    min_samples_leaf = _to_int(row.get("classifier:gradient_boosting:min_samples_leaf"))
    l2_regularization = _to_float(row.get("classifier:gradient_boosting:l2_regularization"))

    if learning_rate is None or min_samples_leaf is None or l2_regularization is None:
        return None

    return {
        "model_name": "hist_gradient_boosting",
        "params": {
            "learning_rate": learning_rate,
            "max_iter": 200,
            "max_depth": max_depth if max_depth is not None else 12,
            "min_samples_leaf": min_samples_leaf,
            "l2_regularization": l2_regularization,
        },
    }


def _map_lda(row: dict[str, Any]) -> dict[str, Any] | None:
    shrinkage = _none_if_missing(row.get("classifier:lda:shrinkage"))
    if shrinkage not in {None, "auto"}:
        return None

    solver = "svd" if shrinkage is None else "lsqr"
    return {
        "model_name": "lda",
        "params": {
            "solver": solver,
            "shrinkage": shrinkage,
        },
    }


def map_supported_configuration(row: dict[str, Any]) -> dict[str, Any] | None:
    model_choice = _none_if_missing(row.get("classifier:__choice__"))
    if model_choice not in SUPPORTED_MODEL_MAPPERS:
        return None

    mapper = globals()[SUPPORTED_MODEL_MAPPERS[model_choice]]
    mapped = mapper(row)
    if mapped is None:
        return None

    params = mapped["params"]
    if any(value is None for value in params.values()):
        return None
    return mapped


def load_configurations(configurations_csv: Path) -> dict[str, dict[str, Any]]:
    configurations: dict[str, dict[str, Any]] = {}
    with configurations_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            config_id = row["idx"]
            configurations[config_id] = row
    return configurations


def load_algorithm_runs(algorithm_runs_arff: Path) -> dict[str, list[dict[str, Any]]]:
    with algorithm_runs_arff.open(encoding="utf-8") as handle:
        arff_dict = arff.load(handle)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in arff_dict["data"]:
        instance_id = _decode(row[0])
        runstatus = _decode(row[-1])
        if runstatus != "ok":
            continue
        grouped.setdefault(instance_id, []).append(
            {
                "algorithm_id": str(_decode(row[2])),
                "score": float(row[3]),
            }
        )
    return grouped


def load_metafeatures(feature_values_arff: Path) -> dict[str, dict[str, Any]]:
    with feature_values_arff.open(encoding="utf-8") as handle:
        arff_dict = arff.load(handle)
    attribute_names = [name for name, _ in arff_dict["attributes"][2:]]
    metafeatures: dict[str, dict[str, Any]] = {}
    for row in arff_dict["data"]:
        instance_id = _decode(row[0])
        values = {}
        for idx, attribute_name in enumerate(attribute_names, start=2):
            value = row[idx]
            decoded = _decode(value)
            if isinstance(decoded, float):
                values[attribute_name] = float(decoded)
            else:
                values[attribute_name] = decoded
        metafeatures[instance_id] = values
    return metafeatures


def convert_collection(source_dir: Path) -> dict[str, Any]:
    configurations = load_configurations(source_dir / "configurations.csv")
    algorithm_runs = load_algorithm_runs(source_dir / "algorithm_runs.arff")
    metafeatures = load_metafeatures(source_dir / "feature_values.arff")

    collection_name = source_dir.name
    parts = collection_name.split(".")
    metric = parts[0] if len(parts) > 0 else collection_name
    task_descriptor = parts[1] if len(parts) > 1 else ""
    data_descriptor = parts[2] if len(parts) > 2 else ""

    datasets: list[dict[str, Any]] = []
    for dataset_id, runs in sorted(algorithm_runs.items()):
        candidates: list[dict[str, Any]] = []
        for run in runs:
            config_row = configurations.get(run["algorithm_id"])
            if config_row is None:
                continue

            mapped = map_supported_configuration(config_row)
            if mapped is None:
                continue

            candidates.append(
                {
                    "source_algorithm_id": run["algorithm_id"],
                    "model_name": mapped["model_name"],
                    "params": mapped["params"],
                    "score": run["score"],
                }
            )

        if not candidates:
            continue

        candidates.sort(key=lambda row: row["score"], reverse=True)
        datasets.append(
            {
                "dataset_id": dataset_id,
                "metafeatures": metafeatures.get(dataset_id, {}),
                "candidates": candidates,
            }
        )

    return {
        "source_collection": collection_name,
        "metric": metric,
        "task_descriptor": task_descriptor,
        "data_descriptor": data_descriptor,
        "supported_models": sorted(
            {candidate["model_name"] for dataset in datasets for candidate in dataset["candidates"]}
        ),
        "datasets": datasets,
    }


def export_all(source_root: Path, output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    collections = sorted(path for path in source_root.iterdir() if path.is_dir())
    exported = 0

    for collection_dir in collections:
        converted = convert_collection(collection_dir)
        if not converted["datasets"]:
            continue

        output_path = output_root / f"{collection_dir.name}.json"
        output_path.write_text(json.dumps(converted, indent=2), encoding="utf-8")
        exported += 1
        print(f"Exported {collection_dir.name} -> {output_path}")

    print(f"Exported {exported} collections.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert auto-sklearn metalearning ASLib data to JSON usable by this AutoML library."
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("auto-sklearn/autosklearn/metalearning/files"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("AutoML/meta_learning/json"),
    )
    args = parser.parse_args()

    export_all(args.source_root, args.output_root)


if __name__ == "__main__":
    main()
