from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import r2_score

from .base import BaseAutoML


class AutoMLRegressor(BaseAutoML):
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
        ensemble_strategy: str = "stacked",
        stacked_base_size: int = 4,
        stacked_bagging_n_estimators: int = 5,
        stacked_meta_model_names: list[str] | None = None,
        stacked_include_base_predictions: bool = True,
        stacked_include_original_features_in_meta: bool = False,
        stacked_final_weight_optimizer: str = "greedy",
        n_jobs: int | None = None,
        search_n_parallel: int = 3,
        stack_n_jobs: int | None = None,
        inner_n_jobs: int | None = 1,
        verbose: int = 1,
        disable_evaluation_timeout: bool = False,
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
            ensemble_strategy=ensemble_strategy,
            stacked_base_size=stacked_base_size,
            stacked_bagging_n_estimators=stacked_bagging_n_estimators,
            stacked_meta_model_names=stacked_meta_model_names,
            stacked_include_base_predictions=stacked_include_base_predictions,
            stacked_include_original_features_in_meta=stacked_include_original_features_in_meta,
            stacked_final_weight_optimizer=stacked_final_weight_optimizer,
            n_jobs=n_jobs,
            search_n_parallel=search_n_parallel,
            stack_n_jobs=stack_n_jobs,
            inner_n_jobs=inner_n_jobs,
            verbose=verbose,
            disable_evaluation_timeout=disable_evaluation_timeout,
        )
        self.allowed_models = allowed_models
        self.use_meta_learning = use_meta_learning
        self.meta_learning_root = meta_learning_root
        self.meta_learning_top_datasets = meta_learning_top_datasets
        self.meta_learning_top_configs_per_dataset = meta_learning_top_configs_per_dataset
        self.max_meta_learning_warmstarts = max_meta_learning_warmstarts

    def _task_name(self) -> str:
        return "regression"

    def _default_scoring(self) -> str | None:
        return "r2"

    def _build_engine(self):
        try:
            from .core.automl import AutoMLEngine
        except ImportError as exc:
            raise RuntimeError(
                "AutoMLRegressor requires `automl.core.automl.AutoMLEngine`, "
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
            ensemble_strategy=self.ensemble_strategy,
            stacked_base_size=self.stacked_base_size,
            stacked_bagging_n_estimators=self.stacked_bagging_n_estimators,
            stacked_meta_model_names=self.stacked_meta_model_names,
            stacked_include_base_predictions=self.stacked_include_base_predictions,
            stacked_include_original_features_in_meta=self.stacked_include_original_features_in_meta,
            stacked_final_weight_optimizer=self.stacked_final_weight_optimizer,
            n_jobs=self.n_jobs,
            search_n_parallel=self.search_n_parallel,
            stack_n_jobs=self.stack_n_jobs,
            inner_n_jobs=self.inner_n_jobs,
            verbose=self.verbose,
            disable_evaluation_timeout=self.disable_evaluation_timeout,
            allowed_models=self.allowed_models,
            use_meta_learning=self.use_meta_learning,
            meta_learning_root=self.meta_learning_root,
            meta_learning_top_datasets=self.meta_learning_top_datasets,
            meta_learning_top_configs_per_dataset=self.meta_learning_top_configs_per_dataset,
            max_meta_learning_warmstarts=self.max_meta_learning_warmstarts,
        )

    def fit(self, X: Any, y: Any):
        self.n_outputs_ = 1 if np.asarray(y).ndim == 1 else int(np.asarray(y).shape[1])
        return super().fit(X, y)

    def score(self, X: Any, y: Any) -> float:
        predictions = self.predict(X)
        return float(r2_score(y, predictions))
