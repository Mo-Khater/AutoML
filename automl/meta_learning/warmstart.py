from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ..configspace_search_space import dict_to_configuration


def get_meta_learning_warmstarts(
    *,
    y: Any,
    metafeatures: dict[str, float],
    configspace: Any,
    scoring: str,
    allowed_models: list[str] | None = None,
    meta_learning_root: str | Path | None = None,
    top_datasets: int = 5,
    top_configs_per_dataset: int = 3,
    max_warmstarts: int = 10,
) -> list[Any]:
    collection_path = _resolve_collection_path(
        y=y,
        scoring=scoring,
        meta_learning_root=meta_learning_root,
    )
    if collection_path is None or not collection_path.exists():
        return []

    payload = json.loads(collection_path.read_text(encoding="utf-8"))
    datasets = payload.get("datasets", [])
    if not datasets:
        return []

    ranked_datasets = sorted(
        datasets,
        key=lambda dataset: _metafeature_distance(
            metafeatures,
            dataset.get("metafeatures", {}),
        ),
    )

    allowed = set(allowed_models) if allowed_models is not None else None
    selected_configs: list[Any] = []
    seen_signatures: set[tuple[Any, ...]] = set()

    for dataset in ranked_datasets[:top_datasets]:
        candidates = dataset.get("candidates", [])[:top_configs_per_dataset]
        for candidate in candidates:
            if allowed is not None and candidate["model_name"] not in allowed:
                continue

            signature = (
                candidate["model_name"],
                tuple(sorted(candidate["params"].items())),
            )
            if signature in seen_signatures:
                continue

            try:
                config = dict_to_configuration(
                    {
                        "model_name": candidate["model_name"],
                        "params": candidate["params"],
                    },
                    configspace,
                )
            except Exception:
                continue

            selected_configs.append(config)
            seen_signatures.add(signature)
            if len(selected_configs) >= max_warmstarts:
                return selected_configs

    return selected_configs


def _resolve_collection_path(
    *,
    y: Any,
    scoring: str,
    meta_learning_root: str | Path | None,
) -> Path | None:
    if meta_learning_root is not None:
        root = Path(meta_learning_root)
    else:
        root = Path(__file__).resolve().parents[2] / "meta_learning" / "json"
    n_classes = len(np.unique(np.asarray(y)))
    task_suffix = "binary" if n_classes <= 2 else "multiclass"

    metric_name = "accuracy"
    if scoring not in {None, "accuracy"}:
        metric_name = scoring

    candidate_names = [
        f"{metric_name}_{task_suffix}.classification_dense.json",
        f"{metric_name}.classification_dense.json",
    ]
    for name in candidate_names:
        path = root / name
        if path.exists():
            return path
    return None


def _metafeature_distance(
    current: dict[str, float],
    historical: dict[str, Any],
) -> float:
    common_keys = [
        key
        for key in current
        if key in historical and isinstance(historical[key], (int, float))
    ]
    if not common_keys:
        return float("inf")

    diffs = []
    for key in common_keys:
        current_value = float(current[key])
        historical_value = float(historical[key])
        scale = max(1.0, abs(current_value), abs(historical_value))
        diffs.append(((current_value - historical_value) / scale) ** 2)
    return float(np.sqrt(np.sum(diffs)))
