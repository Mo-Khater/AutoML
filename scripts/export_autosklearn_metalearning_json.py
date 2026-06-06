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
from sklearn.naive_bayes import GaussianNB as SklearnGaussianNB

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
AUTO_SKLEARN_ROOT = REPO_ROOT.parent / "auto-sklearn"
if str(AUTO_SKLEARN_ROOT) not in sys.path:
    sys.path.insert(0, str(AUTO_SKLEARN_ROOT))
AUTO_SKLEARN_COMPONENT_ROOT = (
    AUTO_SKLEARN_ROOT / "autosklearn" / "pipeline" / "components" / "classification"
)

from automl.components.classification import get_classification_components


SUPPORTED_MODEL_MAPPERS = {
    "adaboost": "_map_adaboost",
    "bernoulli_nb": "_map_bernoulli_nb",
    "decision_tree": "_map_decision_tree",
    "extra_trees": "_map_extra_trees",
    "gaussian_nb": "_map_gaussian_nb",
    "gradient_boosting": "_map_gradient_boosting",
    "k_nearest_neighbors": "_map_knn",
    "lda": "_map_lda",
    "liblinear_svc": "_map_liblinear_svc",
    "libsvm_svc": "_map_svc",
    "mlp": "_map_mlp",
    "multinomial_nb": "_map_multinomial_nb",
    "passive_aggressive": "_map_passive_aggressive",
    "qda": "_map_qda",
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


def _map_hist_gradient_boosting_early_stopping(value: Any) -> bool:
    value = _none_if_missing(value)
    if value is None:
        return False
    value_str = str(value).strip().lower()
    return value_str in {"train", "valid", "on", "true", "1"}


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


@lru_cache(maxsize=None)
def _current_model_defaults(model_name: str) -> dict[str, Any]:
    component = get_classification_components()[model_name]
    defaults: dict[str, Any] = {}
    prefix = f"{model_name}:"
    for hyperparameter in component.build_hyperparameters():
        hyper_name = getattr(hyperparameter, "name", "")
        if not hyper_name.startswith(prefix):
            continue
        param_name = hyper_name[len(prefix):]
        if hasattr(hyperparameter, "value"):
            defaults[param_name] = hyperparameter.value
        else:
            defaults[param_name] = getattr(hyperparameter, "default_value", None)
    return defaults


def _project_to_current_model(
    model_name: str,
    params: dict[str, Any],
) -> dict[str, Any] | None:
    defaults = dict(_current_model_defaults(model_name))
    if not defaults and model_name not in get_classification_components():
        return None
    for key, value in params.items():
        if value is not None:
            defaults[key] = value
    return {
        "model_name": model_name,
        "params": defaults,
    }


def _map_adaboost(row: dict[str, Any]) -> dict[str, Any] | None:
    return _project_to_current_model(
        "adaboost",
        {
            "n_estimators": _to_int(row.get("classifier:adaboost:n_estimators")),
            "learning_rate": _to_float(row.get("classifier:adaboost:learning_rate")),
        },
    )


def _map_bernoulli_nb(row: dict[str, Any]) -> dict[str, Any] | None:
    return _project_to_current_model(
        "bernoulli_nb",
        {
            "alpha": _to_float(row.get("classifier:bernoulli_nb:alpha")),
            "fit_prior": _to_bool(row.get("classifier:bernoulli_nb:fit_prior"), default=True),
        },
    )


def _map_decision_tree(row: dict[str, Any]) -> dict[str, Any] | None:
    max_depth_factor = _to_float(row.get("classifier:decision_tree:max_depth_factor"))
    max_depth = None
    if max_depth_factor is not None:
        max_depth = max(1, min(64, int(round(max_depth_factor * 32))))

    return _project_to_current_model(
        "decision_tree",
        {
            "criterion": _none_if_missing(row.get("classifier:decision_tree:criterion")) or "gini",
            "max_depth": max_depth,
            "min_samples_split": _to_int(row.get("classifier:decision_tree:min_samples_split"), default=2),
            "min_samples_leaf": _to_int(row.get("classifier:decision_tree:min_samples_leaf"), default=1),
            "max_features": _map_max_features(row.get("classifier:decision_tree:max_features")),
        },
    )


def _map_gaussian_nb(row: dict[str, Any]) -> dict[str, Any] | None:
    return _project_to_current_model(
        "gaussian_nb",
        {
            "var_smoothing": float(SklearnGaussianNB().get_params()["var_smoothing"]),
        },
    )


def _map_knn(row: dict[str, Any]) -> dict[str, Any] | None:
    return _project_to_current_model(
        "knn",
        {
            "n_neighbors": _to_int(row.get("classifier:k_nearest_neighbors:n_neighbors")),
            "p": _to_int(row.get("classifier:k_nearest_neighbors:p")),
            "weights": _none_if_missing(row.get("classifier:k_nearest_neighbors:weights")),
        },
    )


def _map_liblinear_svc(row: dict[str, Any]) -> dict[str, Any] | None:
    penalty = _none_if_missing(row.get("classifier:liblinear_svc:penalty")) or "l2"
    loss = _none_if_missing(row.get("classifier:liblinear_svc:loss")) or "squared_hinge"
    if penalty == "l1" and loss != "squared_hinge":
        loss = "squared_hinge"
    return _project_to_current_model(
        "liblinear_svc",
        {
            "C": _to_float(row.get("classifier:liblinear_svc:C")),
            "loss": loss,
        },
    )


def _map_svc(row: dict[str, Any]) -> dict[str, Any] | None:
    return _project_to_current_model(
        "svc",
        {
            "C": _to_float(row.get("classifier:libsvm_svc:C")),
            "kernel": _none_if_missing(row.get("classifier:libsvm_svc:kernel")),
            "gamma": _to_float(row.get("classifier:libsvm_svc:gamma")),
            "degree": _to_int(
                row.get("classifier:libsvm_svc:degree"),
                default=_to_int(_askl_constant_or_default("libsvm_svc.py", "degree")),
            ),
            "coef0": _to_float(
                row.get("classifier:libsvm_svc:coef0"),
                default=_to_float(_askl_constant_or_default("libsvm_svc.py", "coef0")),
            ),
        },
    )


def _map_random_forest(row: dict[str, Any]) -> dict[str, Any] | None:
    return _project_to_current_model(
        "random_forest",
        {
            "n_estimators": _askl_get_max_iter("random_forest.py"),
            "max_depth": _to_int(row.get("classifier:random_forest:max_depth"), default=64),
            "min_samples_split": _to_int(row.get("classifier:random_forest:min_samples_split"), default=2),
            "min_samples_leaf": _to_int(row.get("classifier:random_forest:min_samples_leaf"), default=1),
            "criterion": _none_if_missing(row.get("classifier:random_forest:criterion")) or "gini",
            "max_features": _map_max_features(row.get("classifier:random_forest:max_features")),
            "bootstrap": _to_bool(row.get("classifier:random_forest:bootstrap"), default=True),
            "max_leaf_nodes": _to_int(
                row.get("classifier:random_forest:max_leaf_nodes"),
                default=_to_int(_askl_constant_or_default("random_forest.py", "max_leaf_nodes")),
            ),
        },
    )


def _map_extra_trees(row: dict[str, Any]) -> dict[str, Any] | None:
    return _project_to_current_model(
        "extra_trees",
        {
            "n_estimators": _askl_get_max_iter("extra_trees.py"),
            "max_depth": _to_int(row.get("classifier:extra_trees:max_depth"), default=64),
            "min_samples_split": _to_int(row.get("classifier:extra_trees:min_samples_split"), default=2),
            "min_samples_leaf": _to_int(row.get("classifier:extra_trees:min_samples_leaf"), default=1),
            "criterion": _none_if_missing(row.get("classifier:extra_trees:criterion")) or "gini",
            "max_features": _map_max_features(row.get("classifier:extra_trees:max_features")),
            "max_leaf_nodes": _to_int(
                row.get("classifier:extra_trees:max_leaf_nodes"),
                default=_to_int(
                    _askl_constant_or_default("extra_trees.py", "max_leaf_nodes")
                ),
            ),
        },
    )


def _map_gradient_boosting(row: dict[str, Any]) -> dict[str, Any] | None:
    return _project_to_current_model(
        "gradient_boosting",
        {
            "n_estimators": _askl_get_max_iter("gradient_boosting.py"),
            "learning_rate": _to_float(row.get("classifier:gradient_boosting:learning_rate")),
            "max_depth": _to_int(row.get("classifier:gradient_boosting:max_depth")),
            "subsample": _to_float(
                row.get("classifier:gradient_boosting:subsample"),
                default=_to_float(_askl_constant_or_default("gradient_boosting.py", "subsample")),
            ),
            "max_leaf_nodes": _to_int(
                row.get("classifier:gradient_boosting:max_leaf_nodes"),
                default=_to_int(_askl_constant_or_default("gradient_boosting.py", "max_leaf_nodes")),
            ),
        },
    )


def _map_lda(row: dict[str, Any]) -> dict[str, Any] | None:
    shrinkage = _none_if_missing(row.get("classifier:lda:shrinkage"))
    if shrinkage not in {None, "auto"}:
        return None

    solver = "svd" if shrinkage is None else "lsqr"
    return _project_to_current_model(
        "lda",
        {
            "solver": solver,
            "shrinkage": shrinkage,
        },
    )


def _map_multinomial_nb(row: dict[str, Any]) -> dict[str, Any] | None:
    return _project_to_current_model(
        "multinomial_nb",
        {
            "alpha": _to_float(row.get("classifier:multinomial_nb:alpha")),
            "fit_prior": _to_bool(row.get("classifier:multinomial_nb:fit_prior"), default=True),
        },
    )


def _map_mlp(row: dict[str, Any]) -> dict[str, Any] | None:
    hidden_layer_depth = _to_int(row.get("classifier:mlp:hidden_layer_depth"), default=1)
    num_nodes_per_layer = _to_int(row.get("classifier:mlp:num_nodes_per_layer"))
    alpha = _to_float(row.get("classifier:mlp:alpha"))
    learning_rate_init = _to_float(row.get("classifier:mlp:learning_rate_init"))
    activation = _none_if_missing(row.get("classifier:mlp:activation")) or _askl_constant_or_default(
        "mlp.py", "activation"
    )
    solver = _none_if_missing(row.get("classifier:mlp:solver")) or _askl_constant_or_default(
        "mlp.py", "solver"
    )

    if num_nodes_per_layer is None:
        return None

    hidden_layer_sizes = int(num_nodes_per_layer)
    if hidden_layer_depth is not None and hidden_layer_depth > 1:
        hidden_layer_sizes = min(256, hidden_layer_sizes * hidden_layer_depth)

    return _project_to_current_model(
        "mlp",
        {
            "hidden_layer_sizes": hidden_layer_sizes,
            "alpha": alpha,
            "learning_rate_init": learning_rate_init,
            "activation": activation,
            "solver": solver,
        },
    )


def _map_passive_aggressive(row: dict[str, Any]) -> dict[str, Any] | None:
    return _project_to_current_model(
        "passive_aggressive",
        {
            "C": _to_float(row.get("classifier:passive_aggressive:C")),
            "loss": _none_if_missing(row.get("classifier:passive_aggressive:loss")) or "hinge",
            "average": _to_bool(row.get("classifier:passive_aggressive:average"), default=False),
        },
    )


def _map_qda(row: dict[str, Any]) -> dict[str, Any] | None:
    return _project_to_current_model(
        "qda",
        {
            "reg_param": _to_float(row.get("classifier:qda:reg_param"), default=0.0),
        },
    )


def _map_sgd(row: dict[str, Any]) -> dict[str, Any] | None:
    loss = _none_if_missing(row.get("classifier:sgd:loss")) or "log"
    if loss == "log":
        loss = "log_loss"
    elif loss in {"squared_hinge", "perceptron"}:
        loss = "hinge"

    learning_rate = _none_if_missing(row.get("classifier:sgd:learning_rate")) or "optimal"
    if learning_rate == "constant":
        learning_rate = "adaptive"

    return _project_to_current_model(
        "sgd",
        {
            "loss": loss,
            "penalty": _none_if_missing(row.get("classifier:sgd:penalty")) or "l2",
            "alpha": _to_float(row.get("classifier:sgd:alpha")),
            "learning_rate": learning_rate,
            "eta0": _to_float(
                row.get("classifier:sgd:eta0"),
                default=_to_float(_askl_constant_or_default("sgd.py", "eta0")),
            ),
            "average": _to_bool(row.get("classifier:sgd:average"), default=False),
        },
    )


def map_supported_configuration(row: dict[str, Any]) -> dict[str, Any] | None:
    model_choice = _none_if_missing(row.get("classifier:__choice__"))
    if model_choice not in SUPPORTED_MODEL_MAPPERS:
        return None

    mapper = globals()[SUPPORTED_MODEL_MAPPERS[model_choice]]
    mapped = mapper(row)
    if mapped is None:
        return None
    if any(value is None for value in mapped["params"].values()):
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

    exportable_models = set(get_classification_components())
    mapped_models = {candidate["model_name"] for dataset in datasets for candidate in dataset["candidates"]}
    return {
        "source_collection": collection_name,
        "metric": metric,
        "task_descriptor": task_descriptor,
        "data_descriptor": data_descriptor,
        "supported_models": sorted(exportable_models.intersection(mapped_models)),
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
