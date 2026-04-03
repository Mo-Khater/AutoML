from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score

from .base import BaseAutoML


class AutoMLClassifier(BaseAutoML):
    def __init__(
        self,
        *,
        time_budget: float | None = None,
        per_run_time_limit: float | None = None,
        n_trials: int = 50,
        cv: int = 5,
        scoring: str | None = None,
        random_state: int | None = None,
        ensemble: bool = True,
        ensemble_size: int = 10,
        n_jobs: int | None = None,
        verbose: int = 1,
        balance_classes: bool = False,
        allowed_models: list[str] | None = None,
        use_meta_learning: bool = True,
        meta_learning_root: str | None = None,
        meta_learning_top_datasets: int = 5,
        meta_learning_top_configs_per_dataset: int = 3,
        max_meta_learning_warmstarts: int = 10,
    ) -> None:
        super().__init__(
            time_budget=time_budget,
            per_run_time_limit=per_run_time_limit,
            n_trials=n_trials,
            cv=cv,
            scoring=scoring,
            random_state=random_state,
            ensemble=ensemble,
            ensemble_size=ensemble_size,
            n_jobs=n_jobs,
            verbose=verbose,
        )
        self.balance_classes = balance_classes
        self.allowed_models = allowed_models
        self.use_meta_learning = use_meta_learning
        self.meta_learning_root = meta_learning_root
        self.meta_learning_top_datasets = meta_learning_top_datasets
        self.meta_learning_top_configs_per_dataset = meta_learning_top_configs_per_dataset
        self.max_meta_learning_warmstarts = max_meta_learning_warmstarts

    def _task_name(self) -> str:
        return "classification"

    def _default_scoring(self) -> str | None:
        return "accuracy"

    def _build_engine(self):
        try:
            from .core.automl import AutoMLEngine
        except ImportError as exc:
            raise RuntimeError(
                "AutoMLClassifier requires `automl.core.automl.AutoMLEngine`, "
                "but that module has not been implemented yet."
            ) from exc

        return AutoMLEngine(
            task=self._task_name(),
            time_budget=self.time_budget,
            per_run_time_limit=self.per_run_time_limit,
            n_trials=self.n_trials,
            cv=self.cv,
            scoring=self._get_effective_scoring(),
            random_state=self.random_state,
            ensemble=self.ensemble,
            ensemble_size=self.ensemble_size,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
            balance_classes=self.balance_classes,
            allowed_models=self.allowed_models,
            use_meta_learning=self.use_meta_learning,
            meta_learning_root=self.meta_learning_root,
            meta_learning_top_datasets=self.meta_learning_top_datasets,
            meta_learning_top_configs_per_dataset=self.meta_learning_top_configs_per_dataset,
            max_meta_learning_warmstarts=self.max_meta_learning_warmstarts,
        )

    def fit(self, X: Any, y: Any):
        y_array = np.asarray(y)
        self.classes_ = np.unique(y_array)
        self.n_classes_ = int(self.classes_.shape[0])
        return super().fit(X, y)

    def score(self, X: Any, y: Any) -> float:
        predictions = self.predict(X)
        return float(accuracy_score(y, predictions))
