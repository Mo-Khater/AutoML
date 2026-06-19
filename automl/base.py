from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y, check_array


class BaseAutoML(BaseEstimator, ABC):
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
    ) -> None:
        
        self.time_budget = time_budget
        self.per_run_time_limit = per_run_time_limit
        self.n_trials = n_trials
        self.cv = cv
        self.scoring = scoring
        self.random_state = random_state
        self.ensemble = ensemble
        self.ensemble_size = ensemble_size
        self.ensemble_strategy = ensemble_strategy
        self.stacked_base_size = stacked_base_size
        self.stacked_bagging_n_estimators = stacked_bagging_n_estimators
        self.stacked_meta_model_names = stacked_meta_model_names
        self.stacked_include_base_predictions = stacked_include_base_predictions
        self.stacked_include_original_features_in_meta = stacked_include_original_features_in_meta
        self.stacked_final_weight_optimizer = stacked_final_weight_optimizer
        self.n_jobs = n_jobs
        self.search_n_parallel = search_n_parallel
        self.stack_n_jobs = stack_n_jobs
        self.inner_n_jobs = inner_n_jobs
        self.verbose = verbose
        self.disable_evaluation_timeout = disable_evaluation_timeout

        self._reset()

    @abstractmethod
    def _task_name(self) -> str:
        "return task name classification or regression"

    @abstractmethod
    def _default_scoring(self) -> str | None:
        "return default scoring metric name for the task"

    @abstractmethod
    def _build_engine(self):
        "build the search engine"

    def _reset(self) -> None:
        self.best_model_ = None
        self.best_score_ = None
        self.cv_results_ = None
        self.leaderboard_ = None
        self.ensemble_ = None
        self._final_model = None
        self._engine = None
        self._is_fitted = False

    def _get_effective_scoring(self) -> str | None:
        if self.scoring is not None:
            return self.scoring
        return self._default_scoring()

    def _validate_fit_inputs(self, X: Any, y: Any) -> tuple[Any, Any]:
        if isinstance(X, pd.DataFrame):
            y_valid = y.copy() if hasattr(y, "copy") else y
            if hasattr(y_valid, "to_numpy"):
                if len(X) != len(y_valid):
                    raise ValueError("X and y have inconsistent lengths.")
            else:
                if len(X) != len(y):
                    raise ValueError("X and y have inconsistent lengths.")
            return X.copy(), y_valid
        X_valid, y_valid = check_X_y(X, y, accept_sparse=True)
        return X_valid, y_valid

    def _validate_predict_input(self, X: Any) -> Any:
        if isinstance(X, pd.DataFrame):
            return X.copy()
        return check_array(X, accept_sparse=True)

    def _run_engine(self, X: Any, y: Any) -> Any:
        if self._engine is None:
            raise RuntimeError("Engine not built.")

        return self._engine.search(X, y)

    def _apply_fit_result(self, result: Any) -> None:
        self.best_model_ = self._read_result_value(result, "best_model")
        self.best_score_ = self._read_result_value(result, "best_score")
        self.cv_results_ = self._read_result_value(result, "cv_results")
        self.leaderboard_ = self._read_result_value(result, "leaderboard")
        self.ensemble_ = self._read_result_value(result, "ensemble")
        final_model = self._read_result_value(result, "final_model")

        if final_model is None:
            final_model = self.ensemble_ if self.ensemble_ is not None else self.best_model_

        self._final_model = final_model
        self._is_fitted = self._final_model is not None

        if not self._is_fitted:
            raise RuntimeError(
                "no valid fitted model found"
            )

    @staticmethod
    def _read_result_value(result: Any, name: str) -> Any:
        if result is None:
            return None

        if isinstance(result, Mapping):
            return result.get(name, None)

        if hasattr(result, name):
            return getattr(result, name)

        return None


    def _require_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError("call fit(X,y) before using this estimator.")

    def _get_final_model(self):
        self._require_fitted()
        return self._final_model

    def fit(self, X: Any, y: Any):
        self._reset()
        X_valid, y_valid = self._validate_fit_inputs(X, y)
        self._engine = self._build_engine()
        result = self._run_engine(X_valid, y_valid)
        self._apply_fit_result(result)
        return self

    def predict(self, X: Any):
        model = self._get_final_model()
        X_valid = self._validate_predict_input(X)
        return model.predict(X_valid)

    def predict_proba(self, X: Any):
        model = self._get_final_model()
        if not hasattr(model, "predict_proba"):
            raise AttributeError("The final model does not implement `predict_proba`.")
        X_valid = self._validate_predict_input(X)
        return model.predict_proba(X_valid)

    def leaderboard(self, top_n: int | None = None):
        self._require_fitted()

        if self.leaderboard_ is None or top_n is None:
            return self.leaderboard_

        return self.leaderboard_[:top_n]

    def get_cv_results(self):
        self._require_fitted()
        return self.cv_results_
