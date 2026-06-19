from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from joblib import Parallel, delayed
from scipy import sparse
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import BaggingRegressor, VotingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict


@dataclass
class RegressionForwardSelectionResult:
    selected_names: list[str]
    score: float


class StackedEnsembleRegressor(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        *,
        base_estimators: list[tuple[str, Any]],
        meta_estimators: list[tuple[str, Any]],
        cv: int = 5,
        random_state: int | None = None,
        n_jobs: int | None = None,
        inner_n_jobs: int | None = 1,
        bagging_n_estimators: int = 5,
        ensemble_iterations: int | None = None,
        include_base_predictions_in_final_ensemble: bool = True,
        include_original_features_in_meta: bool = True,
        final_weight_optimizer: str = "greedy",
        scoring: str = "r2",
        verbose: int = 0,
        min_base_models: int = 3,
        selection_tolerance: float = 1e-4,
    ) -> None:
        self.base_estimators = base_estimators
        self.meta_estimators = meta_estimators
        self.cv = cv
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.inner_n_jobs = inner_n_jobs
        self.bagging_n_estimators = bagging_n_estimators
        self.ensemble_iterations = ensemble_iterations
        self.include_base_predictions_in_final_ensemble = include_base_predictions_in_final_ensemble
        self.include_original_features_in_meta = include_original_features_in_meta
        self.final_weight_optimizer = final_weight_optimizer
        self.scoring = scoring
        self.verbose = verbose
        self.min_base_models = max(1, int(min_base_models))
        self.selection_tolerance = float(selection_tolerance)

    def fit(self, X: Any, y: Any):
        splitter = KFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)
        base_results = Parallel(n_jobs=self.n_jobs)(
            delayed(self._fit_base_model)(base_name, base_estimator, X, y, splitter)
            for base_name, base_estimator in self.base_estimators
        )
        if not base_results:
            raise RuntimeError("At least one base estimator is required to fit the stacked ensemble.")

        self.forward_selection_ = self._forward_select_base_models(X, np.asarray(y), base_results)
        selected_names = set(self.forward_selection_.selected_names)
        selected_results = [
            (base_name, fitted_model, oof_pred)
            for base_name, fitted_model, oof_pred in base_results
            if base_name in selected_names
        ]
        if len(selected_results) < self.min_base_models:
            raise RuntimeError("Forward selection did not produce the minimum number of base models.")

        self.base_models_ = [(base_name, fitted_model) for base_name, fitted_model, _ in selected_results]
        self.base_model_names_ = [base_name for base_name, _, _ in selected_results]
        base_prediction_blocks = [
            np.asarray(oof_pred, dtype=float).reshape(-1, 1)
            for _, _, oof_pred in selected_results
        ]
        stack_train_X = self._combine_meta_features(X, base_prediction_blocks)

        meta_name, meta_estimator = self._resolve_meta_estimator()
        self.meta_model_name_ = meta_name
        self.meta_model_ = clone(meta_estimator)
        self.meta_model_.fit(stack_train_X, y)

        self._log_ensemble_structure()
        return self

    def predict(self, X: Any):
        stack_features = self._build_stack_features(X)
        return np.asarray(self.meta_model_.predict(stack_features), dtype=float)

    def _fit_base_model(self, base_name: str, base_estimator: Any, X: Any, y: Any, splitter: KFold):
        bagged_model = self._make_bagged_model(base_estimator, n_jobs=self.inner_n_jobs)
        oof_pred = cross_val_predict(
            bagged_model,
            X,
            y,
            cv=splitter,
            method="predict",
            n_jobs=self.inner_n_jobs,
        )
        fitted_bagged_model = self._make_bagged_model(base_estimator, n_jobs=self.inner_n_jobs)
        fitted_bagged_model.fit(X, y)
        return base_name, fitted_bagged_model, np.asarray(oof_pred, dtype=float)

    def _forward_select_base_models(
        self,
        X: Any,
        y: np.ndarray,
        base_results: list[tuple[str, Any, np.ndarray]],
    ) -> RegressionForwardSelectionResult:
        remaining = list(base_results)
        best_single = max(remaining, key=lambda item: self._score_predictions(y, item[2]))
        selected = [best_single]
        remaining.remove(best_single)
        current_score = self._evaluate_meta_subset(X, y, selected)
        self._log_forward_selection_step(
            step=1,
            selected_names=[best_single[0]],
            score=current_score,
            improvement=None,
        )

        while remaining:
            best_candidate = None
            best_candidate_score = float("-inf")
            for candidate in remaining:
                trial_selected = selected + [candidate]
                trial_score = self._evaluate_meta_subset(X, y, trial_selected)
                if trial_score > best_candidate_score:
                    best_candidate_score = trial_score
                    best_candidate = candidate

            if best_candidate is None:
                break

            improvement = best_candidate_score - current_score
            if len(selected) >= self.min_base_models and improvement <= self.selection_tolerance:
                break

            selected.append(best_candidate)
            remaining.remove(best_candidate)
            current_score = best_candidate_score
            self._log_forward_selection_step(
                step=len(selected),
                selected_names=[name for name, _, _ in selected],
                score=current_score,
                improvement=improvement,
            )

        return RegressionForwardSelectionResult(
            selected_names=[name for name, _, _ in selected],
            score=current_score,
        )

    def _evaluate_meta_subset(
        self,
        X: Any,
        y: np.ndarray,
        selected_results: list[tuple[str, Any, np.ndarray]],
    ) -> float:
        blocks = [np.asarray(oof_pred, dtype=float).reshape(-1, 1) for _, _, oof_pred in selected_results]
        stack_train_X = self._combine_meta_features(X, blocks)
        splitter = KFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)
        _, meta_estimator = self._resolve_meta_estimator()
        meta_oof_pred = cross_val_predict(
            clone(meta_estimator),
            stack_train_X,
            y,
            cv=splitter,
            method="predict",
            n_jobs=self.inner_n_jobs,
        )
        return self._score_predictions(y, np.asarray(meta_oof_pred, dtype=float))

    def _resolve_meta_estimator(self) -> tuple[str, Any]:
        if self.meta_estimators:
            return self.meta_estimators[0][0], clone(self.meta_estimators[0][1])
        defaults = make_default_regression_meta_estimators(random_state=self.random_state)
        return defaults[0][0], clone(defaults[0][1])

    def _make_bagged_model(self, base_estimator: Any, n_jobs: int | None = None) -> BaggingRegressor:
        bagging_kwargs = {
            "n_estimators": self.bagging_n_estimators,
            "random_state": self.random_state,
            "n_jobs": n_jobs if n_jobs is not None else self.n_jobs,
        }
        try:
            return BaggingRegressor(estimator=clone(base_estimator), **bagging_kwargs)
        except TypeError:
            return BaggingRegressor(base_estimator=clone(base_estimator), **bagging_kwargs)

    def _build_stack_features(self, X: Any) -> Any:
        blocks = [np.asarray(model.predict(X), dtype=float).reshape(-1, 1) for _, model in self.base_models_]
        return self._combine_meta_features(X, blocks)

    def _log_ensemble_structure(self) -> None:
        if self.verbose <= 0:
            return
        print("[AutoML] Stacked ensemble summary")
        print(f"[AutoML] Selected base models: {', '.join(self.base_model_names_)}")
        print(f"[AutoML] Meta model: {self.meta_model_name_}")
        print(
            f"[AutoML] Forward selection score: {self.forward_selection_.score:.6f} "
            f"(uses original features={self.include_original_features_in_meta})"
        )

    def _log_forward_selection_step(
        self,
        *,
        step: int,
        selected_names: list[str],
        score: float,
        improvement: float | None,
    ) -> None:
        if self.verbose <= 0:
            return
        if improvement is None:
            print(
                f"[AutoML] Forward selection step {step}: "
                f"selected {selected_names[-1]} score={score:.6f}"
            )
            return
        print(
            f"[AutoML] Forward selection step {step}: "
            f"added {selected_names[-1]} score={score:.6f} improvement={improvement:.6f}"
        )

    def _score_predictions(self, y_true: np.ndarray, prediction: np.ndarray) -> float:
        if self.scoring in {"r2", None}:
            return float(r2_score(y_true, prediction))
        if self.scoring in {"neg_root_mean_squared_error", "root_mean_squared_error"}:
            return float(-np.sqrt(mean_squared_error(y_true, prediction)))
        if self.scoring in {"neg_mean_squared_error", "mean_squared_error"}:
            return float(-mean_squared_error(y_true, prediction))
        if self.scoring in {"neg_mean_absolute_error", "mean_absolute_error", "median_absolute_error"}:
            return float(-mean_absolute_error(y_true, prediction))
        return float(r2_score(y_true, prediction))

    def _combine_meta_features(self, X: Any, prediction_blocks: list[np.ndarray]) -> Any:
        prediction_matrix = np.hstack(prediction_blocks)
        if not self.include_original_features_in_meta:
            return prediction_matrix
        if sparse.issparse(X):
            return sparse.hstack([X, sparse.csr_matrix(prediction_matrix)], format="csr")
        return np.hstack([np.asarray(X), prediction_matrix])


def make_default_regression_meta_estimators(random_state: int | None = None) -> list[tuple[str, Any]]:
    return [
        (
            "meta_ridge_regressor",
            Ridge(alpha=1.0),
        ),
    ]


def make_voting_regressor(estimators: list[tuple[str, Any]]) -> VotingRegressor:
    return VotingRegressor(estimators=estimators)
