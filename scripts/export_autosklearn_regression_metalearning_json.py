from __future__ import annotations

import argparse
import ast
import csv
import json
import re
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

import arff

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
AUTO_SKLEARN_ROOT = REPO_ROOT.parent / "auto-sklearn"
if str(AUTO_SKLEARN_ROOT) not in sys.path:
    sys.path.insert(0, str(AUTO_SKLEARN_ROOT))
AUTO_SKLEARN_COMPONENT_ROOT = (
    AUTO_SKLEARN_ROOT / "autosklearn" / "pipeline" / "components" / "regression"
)


SUPPORTED_MODEL_MAPPERS = {
    "adaboost": "_map_adaboost",
    "ard_regression": "_map_ard_regression",
    "decision_tree": "_map_decision_tree",
    "extra_trees": "_map_extra_trees",
    "gaussian_process": "_map_gaussian_process",
    "gradient_boosting": "_map_gradient_boosting",
    "k_nearest_neighbors": "_map_knn",
    "liblinear_svr": "_map_liblinear_svr",
    "libsvm_svr": "_map_libsvm_svr",
    "mlp": "_map_mlp",
    "random_forest": "_map_random_forest",
    "sgd": "_map_sgd",
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


def _to_int_or_str(value: Any, default: int | str | None = None) -> int | str | None:
    value = _none_if_missing(value)
    if value is None:
        return default
    try:
        return int(float(value))
    except Exception:
        return str(value)


def _map_max_features(value: Any) -> float | None:
    value = _none_if_missing(value)
    if value is None:
        return None
    return float(value)


@lru_cache(maxsize=None)
def _askl_component_source(filename: str) -> str:
    return (AUTO_SKLEARN_COMPONENT_ROOT / filename).read_text(encoding="utf-8")


def _askl_get_max_iter(filename: str) -> int | None:
    source = _askl_component_source(filename)
    match = re.search(r"def get_max_iter\(\):\s+return\s+([0-9]+)", source)
    if not match:
        return None
    return int(match.group(1))


def _askl_constant_or_default(filename: str, hyperparameter_name: str) -> Any:
    source = _askl_component_source(filename)
    patterns = [
        rf'Constant\(\s*(?:name\s*=\s*)?["\']{re.escape(hyperparameter_name)}["\']\s*,\s*(?:value\s*=\s*)?([^)]+?)\s*\)',
        rf'UnParametrizedHyperparameter\(\s*(?:name\s*=\s*)?["\']{re.escape(hyperparameter_name)}["\']\s*,\s*(?:value\s*=\s*)?([^)]+?)\s*\)',
        rf'(?:UniformFloatHyperparameter|UniformIntegerHyperparameter|CategoricalHyperparameter)\([^)]*(?:name\s*=\s*)?["\']{re.escape(hyperparameter_name)}["\'][^)]*default_value\s*=\s*([^,\)]+)',
    ]
    for pattern in patterns:
        match = re.search(pattern, source)
        if match:
            try:
                return ast.literal_eval(match.group(1).strip())
            except Exception:
                return None
    return None


def _project_to_exportable_model(
    model_name: str,
    params: dict[str, Any],
) -> dict[str, Any] | None:
    cleaned = {key: value for key, value in params.items() if value is not None}
    if not cleaned:
        return None
    return {
        "model_name": model_name,
        "params": cleaned,
    }


def _map_adaboost(row: dict[str, Any]) -> dict[str, Any] | None:
    return _project_to_exportable_model(
        "adaboost",
        {
            "n_estimators": _to_int(row.get("regressor:adaboost:n_estimators")),
            "learning_rate": _to_float(row.get("regressor:adaboost:learning_rate")),
            "loss": _none_if_missing(row.get("regressor:adaboost:loss")),
            "max_depth": _to_int(row.get("regressor:adaboost:max_depth")),
        },
    )


def _map_ard_regression(row: dict[str, Any]) -> dict[str, Any] | None:
    return _project_to_exportable_model(
        "ard_regression",
        {
            "alpha_1": _to_float(row.get("regressor:ard_regression:alpha_1")),
            "alpha_2": _to_float(row.get("regressor:ard_regression:alpha_2")),
            "fit_intercept": _to_bool(row.get("regressor:ard_regression:fit_intercept")),
            "lambda_1": _to_float(row.get("regressor:ard_regression:lambda_1")),
            "lambda_2": _to_float(row.get("regressor:ard_regression:lambda_2")),
            "n_iter": _to_int(row.get("regressor:ard_regression:n_iter")),
            "threshold_lambda": _to_float(row.get("regressor:ard_regression:threshold_lambda")),
            "tol": _to_float(row.get("regressor:ard_regression:tol")),
        },
    )


def _map_decision_tree(row: dict[str, Any]) -> dict[str, Any] | None:
    max_depth_factor = _to_float(row.get("regressor:decision_tree:max_depth_factor"))
    max_depth = None
    if max_depth_factor is not None:
        max_depth = max(1, min(64, int(round(max_depth_factor * 32))))
    return _project_to_exportable_model(
        "decision_tree",
        {
            "criterion": _none_if_missing(row.get("regressor:decision_tree:criterion")) or "squared_error",
            "max_depth": max_depth,
            "max_features": _map_max_features(row.get("regressor:decision_tree:max_features")),
            "max_leaf_nodes": _to_int(row.get("regressor:decision_tree:max_leaf_nodes")),
            "min_impurity_decrease": _to_float(row.get("regressor:decision_tree:min_impurity_decrease"), default=0.0),
            "min_samples_leaf": _to_int(row.get("regressor:decision_tree:min_samples_leaf"), default=1),
            "min_samples_split": _to_int(row.get("regressor:decision_tree:min_samples_split"), default=2),
            "min_weight_fraction_leaf": _to_float(row.get("regressor:decision_tree:min_weight_fraction_leaf"), default=0.0),
        },
    )


def _map_extra_trees(row: dict[str, Any]) -> dict[str, Any] | None:
    return _project_to_exportable_model(
        "extra_trees",
        {
            "n_estimators": _askl_get_max_iter("extra_trees.py"),
            "bootstrap": _to_bool(row.get("regressor:extra_trees:bootstrap"), default=True),
            "criterion": _none_if_missing(row.get("regressor:extra_trees:criterion")),
            "max_depth": _to_int(row.get("regressor:extra_trees:max_depth")),
            "max_features": _map_max_features(row.get("regressor:extra_trees:max_features")),
            "max_leaf_nodes": _to_int(row.get("regressor:extra_trees:max_leaf_nodes")),
            "min_impurity_decrease": _to_float(row.get("regressor:extra_trees:min_impurity_decrease"), default=0.0),
            "min_samples_leaf": _to_int(row.get("regressor:extra_trees:min_samples_leaf"), default=1),
            "min_samples_split": _to_int(row.get("regressor:extra_trees:min_samples_split"), default=2),
            "min_weight_fraction_leaf": _to_float(row.get("regressor:extra_trees:min_weight_fraction_leaf"), default=0.0),
        },
    )


def _map_gaussian_process(row: dict[str, Any]) -> dict[str, Any] | None:
    return _project_to_exportable_model(
        "gaussian_process",
        {
            "alpha": _to_float(row.get("regressor:gaussian_process:alpha")),
            "thetaL": _to_float(row.get("regressor:gaussian_process:thetaL")),
            "thetaU": _to_float(row.get("regressor:gaussian_process:thetaU")),
        },
    )


def _map_gradient_boosting(row: dict[str, Any]) -> dict[str, Any] | None:
    return _project_to_exportable_model(
        "gradient_boosting",
        {
            "max_iter": _askl_get_max_iter("gradient_boosting.py"),
            "early_stop": _none_if_missing(row.get("regressor:gradient_boosting:early_stop")) or "off",
            "l2_regularization": _to_float(row.get("regressor:gradient_boosting:l2_regularization")),
            "learning_rate": _to_float(row.get("regressor:gradient_boosting:learning_rate")),
            "loss": _none_if_missing(row.get("regressor:gradient_boosting:loss")) or "least_squares",
            "max_bins": _to_int(
                row.get("regressor:gradient_boosting:max_bins"),
                default=_to_int(_askl_constant_or_default("gradient_boosting.py", "max_bins")),
            ),
            "max_depth": _to_int(row.get("regressor:gradient_boosting:max_depth")),
            "max_leaf_nodes": _to_int(row.get("regressor:gradient_boosting:max_leaf_nodes")),
            "min_samples_leaf": _to_int(row.get("regressor:gradient_boosting:min_samples_leaf")),
            "n_iter_no_change": _to_int(row.get("regressor:gradient_boosting:n_iter_no_change")),
            "scoring": _none_if_missing(row.get("regressor:gradient_boosting:scoring")) or "loss",
            "tol": _to_float(
                row.get("regressor:gradient_boosting:tol"),
                default=_to_float(_askl_constant_or_default("gradient_boosting.py", "tol")),
            ),
            "validation_fraction": _to_float(row.get("regressor:gradient_boosting:validation_fraction")),
        },
    )


def _map_knn(row: dict[str, Any]) -> dict[str, Any] | None:
    return _project_to_exportable_model(
        "k_nearest_neighbors",
        {
            "n_neighbors": _to_int(row.get("regressor:k_nearest_neighbors:n_neighbors")),
            "p": _to_int(row.get("regressor:k_nearest_neighbors:p")),
            "weights": _none_if_missing(row.get("regressor:k_nearest_neighbors:weights")),
        },
    )


def _map_liblinear_svr(row: dict[str, Any]) -> dict[str, Any] | None:
    return _project_to_exportable_model(
        "liblinear_svr",
        {
            "C": _to_float(row.get("regressor:liblinear_svr:C")),
            "dual": _to_bool(row.get("regressor:liblinear_svr:dual")),
            "epsilon": _to_float(row.get("regressor:liblinear_svr:epsilon")),
            "fit_intercept": _to_bool(row.get("regressor:liblinear_svr:fit_intercept")),
            "intercept_scaling": _to_float(row.get("regressor:liblinear_svr:intercept_scaling")),
            "loss": _none_if_missing(row.get("regressor:liblinear_svr:loss")),
            "tol": _to_float(row.get("regressor:liblinear_svr:tol")),
        },
    )


def _map_libsvm_svr(row: dict[str, Any]) -> dict[str, Any] | None:
    return _project_to_exportable_model(
        "libsvm_svr",
        {
            "C": _to_float(row.get("regressor:libsvm_svr:C")),
            "coef0": _to_float(
                row.get("regressor:libsvm_svr:coef0"),
                default=_to_float(_askl_constant_or_default("libsvm_svr.py", "coef0")),
            ),
            "degree": _to_int(
                row.get("regressor:libsvm_svr:degree"),
                default=_to_int(_askl_constant_or_default("libsvm_svr.py", "degree")),
            ),
            "epsilon": _to_float(row.get("regressor:libsvm_svr:epsilon")),
            "gamma": _to_float(row.get("regressor:libsvm_svr:gamma")),
            "kernel": _none_if_missing(row.get("regressor:libsvm_svr:kernel")),
            "max_iter": _to_int(
                row.get("regressor:libsvm_svr:max_iter"),
                default=_to_int(_askl_constant_or_default("libsvm_svr.py", "max_iter")),
            ),
            "shrinking": _to_bool(row.get("regressor:libsvm_svr:shrinking")),
            "tol": _to_float(row.get("regressor:libsvm_svr:tol")),
        },
    )


def _map_mlp(row: dict[str, Any]) -> dict[str, Any] | None:
    hidden_layer_depth = _to_int(row.get("regressor:mlp:hidden_layer_depth"), default=1)
    num_nodes_per_layer = _to_int(row.get("regressor:mlp:num_nodes_per_layer"))
    hidden_layer_sizes = None
    if num_nodes_per_layer is not None:
        hidden_layer_sizes = int(num_nodes_per_layer)
        if hidden_layer_depth is not None and hidden_layer_depth > 1:
            hidden_layer_sizes = min(512, hidden_layer_sizes * hidden_layer_depth)

    return _project_to_exportable_model(
        "mlp",
        {
            "activation": _none_if_missing(row.get("regressor:mlp:activation")),
            "alpha": _to_float(row.get("regressor:mlp:alpha")),
            "batch_size": _to_int_or_str(row.get("regressor:mlp:batch_size")),
            "beta_1": _to_float(row.get("regressor:mlp:beta_1")),
            "beta_2": _to_float(row.get("regressor:mlp:beta_2")),
            "early_stopping": _to_bool(row.get("regressor:mlp:early_stopping")),
            "epsilon": _to_float(row.get("regressor:mlp:epsilon")),
            "hidden_layer_sizes": hidden_layer_sizes,
            "learning_rate_init": _to_float(row.get("regressor:mlp:learning_rate_init")),
            "n_iter_no_change": _to_int(row.get("regressor:mlp:n_iter_no_change")),
            "shuffle": _to_bool(row.get("regressor:mlp:shuffle")),
            "solver": _none_if_missing(row.get("regressor:mlp:solver")),
            "tol": _to_float(row.get("regressor:mlp:tol")),
            "validation_fraction": _to_float(row.get("regressor:mlp:validation_fraction")),
        },
    )


def _map_random_forest(row: dict[str, Any]) -> dict[str, Any] | None:
    return _project_to_exportable_model(
        "random_forest",
        {
            "n_estimators": _askl_get_max_iter("random_forest.py"),
            "bootstrap": _to_bool(row.get("regressor:random_forest:bootstrap"), default=True),
            "criterion": _none_if_missing(row.get("regressor:random_forest:criterion")),
            "max_depth": _to_int(row.get("regressor:random_forest:max_depth")),
            "max_features": _map_max_features(row.get("regressor:random_forest:max_features")),
            "max_leaf_nodes": _to_int(row.get("regressor:random_forest:max_leaf_nodes")),
            "min_impurity_decrease": _to_float(row.get("regressor:random_forest:min_impurity_decrease"), default=0.0),
            "min_samples_leaf": _to_int(row.get("regressor:random_forest:min_samples_leaf"), default=1),
            "min_samples_split": _to_int(row.get("regressor:random_forest:min_samples_split"), default=2),
            "min_weight_fraction_leaf": _to_float(row.get("regressor:random_forest:min_weight_fraction_leaf"), default=0.0),
        },
    )


def _map_sgd(row: dict[str, Any]) -> dict[str, Any] | None:
    loss = _none_if_missing(row.get("regressor:sgd:loss"))
    if loss == "squared_loss":
        loss = "squared_error"

    learning_rate = _none_if_missing(row.get("regressor:sgd:learning_rate"))
    if learning_rate == "constant":
        learning_rate = "adaptive"

    return _project_to_exportable_model(
        "sgd",
        {
            "alpha": _to_float(row.get("regressor:sgd:alpha")),
            "average": _to_bool(row.get("regressor:sgd:average")),
            "epsilon": _to_float(row.get("regressor:sgd:epsilon")),
            "eta0": _to_float(row.get("regressor:sgd:eta0")),
            "fit_intercept": _to_bool(row.get("regressor:sgd:fit_intercept")),
            "l1_ratio": _to_float(row.get("regressor:sgd:l1_ratio")),
            "learning_rate": learning_rate,
            "loss": loss,
            "penalty": _none_if_missing(row.get("regressor:sgd:penalty")),
            "power_t": _to_float(row.get("regressor:sgd:power_t")),
            "tol": _to_float(row.get("regressor:sgd:tol")),
        },
    )


def map_supported_configuration(row: dict[str, Any]) -> dict[str, Any] | None:
    model_choice = _none_if_missing(row.get("regressor:__choice__"))
    if model_choice not in SUPPORTED_MODEL_MAPPERS:
        return None

    mapper = globals()[SUPPORTED_MODEL_MAPPERS[model_choice]]
    return mapper(row)


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
    parts = collection_name.split("_")
    metric = parts[0] if parts else collection_name
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

    mapped_models = {
        candidate["model_name"]
        for dataset in datasets
        for candidate in dataset["candidates"]
    }
    return {
        "source_collection": collection_name,
        "metric": metric,
        "task_descriptor": task_descriptor,
        "data_descriptor": data_descriptor,
        "supported_models": sorted(mapped_models),
        "datasets": datasets,
    }


def export_all(source_root: Path, output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    collections = sorted(
        path
        for path in source_root.iterdir()
        if path.is_dir() and "regression" in path.name
    )
    exported = 0

    for collection_dir in collections:
        converted = convert_collection(collection_dir)
        if not converted["datasets"]:
            continue

        output_path = output_root / f"{collection_dir.name}.json"
        output_path.write_text(json.dumps(converted, indent=2), encoding="utf-8")
        exported += 1
        print(f"Exported {collection_dir.name} -> {output_path}")

    print(f"Exported {exported} regression collections.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert auto-sklearn regression metalearning ASLib data to JSON."
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("auto-sklearn/autosklearn/metalearning/files"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("AutoML/meta_learning/json_regression"),
    )
    args = parser.parse_args()

    export_all(args.source_root, args.output_root)


if __name__ == "__main__":
    main()
