from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from joblib import Parallel, delayed
from scipy import sparse
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, cross_val_predict


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(shifted)
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)


class ProbabilityAdapterClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, base_estimator: Any):
        self.base_estimator = base_estimator

    def fit(self, X: Any, y: Any):
        self.estimator_ = clone(self.base_estimator)
        self.estimator_.fit(X, y)
        self.classes_ = getattr(self.estimator_, "classes_", np.unique(y))
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        if hasattr(self.estimator_, "predict_proba"):
            proba = self.estimator_.predict_proba(X)
            return np.asarray(proba, dtype=float)

        if hasattr(self.estimator_, "decision_function"):
            decision = np.asarray(self.estimator_.decision_function(X), dtype=float)
            if decision.ndim == 1:
                positive = 1.0 / (1.0 + np.exp(-decision))
                return np.column_stack([1.0 - positive, positive])
            return _softmax(decision)

        predictions = np.asarray(self.estimator_.predict(X))
        proba = np.zeros((predictions.shape[0], len(self.classes_)), dtype=float)
        for idx, class_label in enumerate(self.classes_):
            proba[:, idx] = (predictions == class_label).astype(float)
        return proba

    def predict(self, X: Any):
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]


@dataclass
class EnsembleSelectionResult:
    candidate_names: list[str]
    weights: np.ndarray
    score: float


class StackedEnsembleClassifier(BaseEstimator, ClassifierMixin):
    _WEIGHT_OPTIMIZATION_SAMPLE_SIZE = 300

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
        scoring: str = "accuracy",
        verbose: int = 0,
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

    def fit(self, X: Any, y: Any):
        self.classes_ = np.unique(y)
        splitter = StratifiedKFold(
            n_splits=self.cv,
            shuffle=True,
            random_state=self.random_state,
        )

        self.base_models_: list[tuple[str, Any]] = []
        base_oof_predictions: list[np.ndarray] = []
        base_feature_blocks: list[np.ndarray] = []
        self.base_model_names_: list[str] = []

        base_results = Parallel(n_jobs=self.n_jobs)(
            delayed(self._fit_base_model)(base_name, base_estimator, X, y, splitter)
            for base_name, base_estimator in self.base_estimators
        )
        for base_name, fitted_bagged_model, oof_proba in base_results:
            self.base_models_.append((base_name, fitted_bagged_model))
            self.base_model_names_.append(base_name)
            base_oof_predictions.append(np.asarray(oof_proba, dtype=float))
            base_feature_blocks.append(self._stack_features_from_proba(oof_proba))

        if not base_feature_blocks:
            raise RuntimeError("At least one base estimator is required to fit the stacked ensemble.")

        stack_train_X = self._combine_meta_features(X, base_feature_blocks)

        self.meta_models_: list[tuple[str, Any]] = []
        meta_oof_predictions: list[np.ndarray] = []
        self.meta_model_names_: list[str] = []

        meta_results = Parallel(n_jobs=self.n_jobs)(
            delayed(self._fit_meta_model)(meta_name, meta_estimator, stack_train_X, y, splitter)
            for meta_name, meta_estimator in self.meta_estimators
        )
        for meta_name, fitted_meta, meta_oof_proba in meta_results:
            self.meta_models_.append((meta_name, fitted_meta))
            self.meta_model_names_.append(meta_name)
            meta_oof_predictions.append(np.asarray(meta_oof_proba, dtype=float))

        ensemble_candidates: list[tuple[str, np.ndarray]] = list(
            zip(self.meta_model_names_, meta_oof_predictions)
        )
        if self.include_base_predictions_in_final_ensemble:
            ensemble_candidates.extend(zip(self.base_model_names_, base_oof_predictions))

        if not ensemble_candidates:
            raise RuntimeError("No ensemble candidates were produced for the final weighted ensemble.")

        self.ensemble_selection_ = self._fit_constrained_weighted_ensemble(
            y=np.asarray(y),
            candidate_predictions=ensemble_candidates,
        )
        self._log_ensemble_structure()
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        stack_features = self._build_stack_features(X)
        candidate_predictions = self._predict_ensemble_candidates(X, stack_features)
        final_proba = np.zeros_like(candidate_predictions[0][1], dtype=float)
        for candidate_name, proba in candidate_predictions:
            weight = self._weight_for_candidate(candidate_name)
            if weight > 0:
                final_proba += weight * proba
        return final_proba

    def predict(self, X: Any):
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    def _fit_base_model(
        self,
        base_name: str,
        base_estimator: Any,
        X: Any,
        y: Any,
        splitter: StratifiedKFold,
    ) -> tuple[str, Any, np.ndarray]:
        inner_n_jobs = self._inner_n_jobs_for_parallel_models()
        bagged_model = self._make_bagged_model(base_estimator, n_jobs=inner_n_jobs)
        oof_proba = cross_val_predict(
            bagged_model,
            X,
            y,
            cv=splitter,
            method="predict_proba",
            n_jobs=inner_n_jobs,
        )
        fitted_bagged_model = self._make_bagged_model(base_estimator, n_jobs=inner_n_jobs)
        fitted_bagged_model.fit(X, y)
        return base_name, fitted_bagged_model, np.asarray(oof_proba, dtype=float)

    def _fit_meta_model(
        self,
        meta_name: str,
        meta_estimator: Any,
        stack_train_X: Any,
        y: Any,
        splitter: StratifiedKFold,
    ) -> tuple[str, Any, np.ndarray]:
        inner_n_jobs = self._inner_n_jobs_for_parallel_models()
        adapted_meta = ProbabilityAdapterClassifier(meta_estimator)
        meta_oof_proba = cross_val_predict(
            adapted_meta,
            stack_train_X,
            y,
            cv=splitter,
            method="predict_proba",
            n_jobs=inner_n_jobs,
        )
        fitted_meta = ProbabilityAdapterClassifier(meta_estimator)
        fitted_meta.fit(stack_train_X, y)
        return meta_name, fitted_meta, np.asarray(meta_oof_proba, dtype=float)

    def _inner_n_jobs_for_parallel_models(self) -> int | None:
        return self.inner_n_jobs

    def _make_bagged_model(self, base_estimator: Any, n_jobs: int | None = None) -> BaggingClassifier:
        adapted_estimator = ProbabilityAdapterClassifier(base_estimator)
        bagging_kwargs = {
            "n_estimators": self.bagging_n_estimators,
            "random_state": self.random_state,
            "n_jobs": n_jobs if n_jobs is not None else self.n_jobs,
        }
        try:
            return BaggingClassifier(
                estimator=adapted_estimator,
                **bagging_kwargs,
            )
        except TypeError:
            return BaggingClassifier(
                base_estimator=adapted_estimator,
                **bagging_kwargs,
            )

    def _build_stack_features(self, X: Any) -> np.ndarray:
        blocks: list[np.ndarray] = []
        for _, model in self.base_models_:
            proba = model.predict_proba(X)
            blocks.append(self._stack_features_from_proba(proba))
        return self._combine_meta_features(X, blocks)

    def _predict_ensemble_candidates(
        self,
        X: Any,
        stack_features: np.ndarray,
    ) -> list[tuple[str, np.ndarray]]:
        predictions: list[tuple[str, np.ndarray]] = []

        for candidate_name, model in self.meta_models_:
            predictions.append((candidate_name, np.asarray(model.predict_proba(stack_features), dtype=float)))

        if self.include_base_predictions_in_final_ensemble:
            for candidate_name, model in self.base_models_:
                predictions.append((candidate_name, np.asarray(model.predict_proba(X), dtype=float)))

        return predictions

    def _fit_constrained_weighted_ensemble(
        self,
        *,
        y: np.ndarray,
        candidate_predictions: list[tuple[str, np.ndarray]],
    ) -> EnsembleSelectionResult:
        sampled_y, sampled_candidate_predictions = self._sample_weight_optimization_data(
            y,
            candidate_predictions,
        )
        if self.final_weight_optimizer == "greedy":
            return self._fit_greedy_weighted_ensemble(
                y=sampled_y,
                candidate_predictions=sampled_candidate_predictions,
            )
        if self.final_weight_optimizer != "slsqp":
            raise ValueError(
                f"Unsupported final_weight_optimizer `{self.final_weight_optimizer}`. "
                "Expected one of: greedy, slsqp."
            )

        candidate_names = [name for name, _ in candidate_predictions]
        prediction_tensor = np.stack(
            [np.asarray(prediction, dtype=float) for _, prediction in sampled_candidate_predictions],
            axis=0,
        )
        n_candidates = prediction_tensor.shape[0]

        def objective(weights: np.ndarray) -> float:
            blended = np.tensordot(weights, prediction_tensor, axes=(0, 0))
            blended = np.clip(blended, 1e-15, 1.0 - 1e-15)
            blended = blended / blended.sum(axis=1, keepdims=True)
            return float(log_loss(sampled_y, blended, labels=self.classes_))

        initial_weights = np.full(n_candidates, 1.0 / n_candidates, dtype=float)
        constraints = [{"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)}]
        bounds = [(0.0, 1.0)] * n_candidates
        result = minimize(
            objective,
            initial_weights,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": max(50, (self.ensemble_iterations or n_candidates * 25))},
        )

        if result.success and np.all(np.isfinite(result.x)):
            optimized_weights = np.clip(np.asarray(result.x, dtype=float), 0.0, 1.0)
            total = optimized_weights.sum()
            if total > 0:
                optimized_weights = optimized_weights / total
            else:
                optimized_weights = initial_weights
        else:
            optimized_weights = initial_weights

        blended = np.tensordot(optimized_weights, prediction_tensor, axes=(0, 0))
        blended = np.clip(blended, 1e-15, 1.0 - 1e-15)
        blended = blended / blended.sum(axis=1, keepdims=True)
        current_score = self._score_predictions(sampled_y, blended)
        return EnsembleSelectionResult(
            candidate_names=candidate_names,
            weights=optimized_weights,
            score=current_score,
        )

    def _fit_greedy_weighted_ensemble(
        self,
        *,
        y: np.ndarray,
        candidate_predictions: list[tuple[str, np.ndarray]],
    ) -> EnsembleSelectionResult:
        candidate_names = [name for name, _ in candidate_predictions]
        prediction_tensor = np.stack(
            [np.asarray(prediction, dtype=float) for _, prediction in candidate_predictions],
            axis=0,
        )
        n_candidates = prediction_tensor.shape[0]
        n_iterations = self.ensemble_iterations or max(20, n_candidates * 10)

        ensemble_sum = np.zeros_like(prediction_tensor[0], dtype=float)
        selection_counts = np.zeros(n_candidates, dtype=float)

        for iteration in range(n_iterations):
            best_idx = 0
            best_loss = float("inf")
            for idx in range(n_candidates):
                trial_blended = (ensemble_sum + prediction_tensor[idx]) / float(iteration + 1)
                trial_blended = np.clip(trial_blended, 1e-15, 1.0 - 1e-15)
                trial_blended = trial_blended / trial_blended.sum(axis=1, keepdims=True)
                trial_loss = float(log_loss(y, trial_blended, labels=self.classes_))
                if trial_loss < best_loss:
                    best_loss = trial_loss
                    best_idx = idx

            ensemble_sum += prediction_tensor[best_idx]
            selection_counts[best_idx] += 1.0

        weights = selection_counts / max(selection_counts.sum(), 1.0)
        blended = ensemble_sum / max(float(n_iterations), 1.0)
        blended = np.clip(blended, 1e-15, 1.0 - 1e-15)
        blended = blended / blended.sum(axis=1, keepdims=True)
        current_score = self._score_predictions(y, blended)
        return EnsembleSelectionResult(
            candidate_names=candidate_names,
            weights=weights,
            score=current_score,
        )

    def _sample_weight_optimization_data(
        self,
        y: np.ndarray,
        candidate_predictions: list[tuple[str, np.ndarray]],
    ) -> tuple[np.ndarray, list[tuple[str, np.ndarray]]]:
        if len(y) <= self._WEIGHT_OPTIMIZATION_SAMPLE_SIZE:
            return y, candidate_predictions

        splitter = StratifiedShuffleSplit(
            n_splits=1,
            train_size=self._WEIGHT_OPTIMIZATION_SAMPLE_SIZE,
            random_state=self.random_state,
        )
        indices, _ = next(splitter.split(np.zeros(len(y)), y))
        sampled_predictions = [
            (name, np.asarray(prediction, dtype=float)[indices])
            for name, prediction in candidate_predictions
        ]
        return np.asarray(y)[indices], sampled_predictions

    def _weight_for_candidate(self, candidate_name: str) -> float:
        for idx, name in enumerate(self.ensemble_selection_.candidate_names):
            if name == candidate_name:
                return float(self.ensemble_selection_.weights[idx])
        return 0.0

    def _log_ensemble_structure(self) -> None:
        if self.verbose <= 0:
            return

        print("[AutoML] Stacked ensemble summary")
        print(f"[AutoML] Layer 1 models: {', '.join(self.base_model_names_)}")
        print(
            f"[AutoML] Layer 2 models: {', '.join(self.meta_model_names_)} "
            f"(uses original features={self.include_original_features_in_meta})"
        )
        print(f"[AutoML] Final weight optimizer: {self.final_weight_optimizer}")

        weighted_parts: list[str] = []
        for name, weight in zip(
            self.ensemble_selection_.candidate_names,
            self.ensemble_selection_.weights,
        ):
            if weight > 0:
                weighted_parts.append(f"{name}={weight:.3f}")
        if not weighted_parts:
            weighted_parts.append("none")
        print(f"[AutoML] Final weighted ensemble: {', '.join(weighted_parts)}")

    def _score_predictions(self, y_true: np.ndarray, proba: np.ndarray) -> float:
        y_pred = self.classes_[np.argmax(proba, axis=1)]
        scoring = self.scoring

        if scoring == "accuracy":
            return float(accuracy_score(y_true, y_pred))
        if scoring == "balanced_accuracy":
            return float(balanced_accuracy_score(y_true, y_pred))
        if scoring == "f1":
            average = "binary" if len(self.classes_) == 2 else "macro"
            return float(f1_score(y_true, y_pred, average=average))
        if scoring == "f1_macro":
            return float(f1_score(y_true, y_pred, average="macro"))
        if scoring == "f1_micro":
            return float(f1_score(y_true, y_pred, average="micro"))
        if scoring == "f1_weighted":
            return float(f1_score(y_true, y_pred, average="weighted"))
        if scoring == "precision":
            average = "binary" if len(self.classes_) == 2 else "macro"
            return float(precision_score(y_true, y_pred, average=average, zero_division=0))
        if scoring == "precision_macro":
            return float(precision_score(y_true, y_pred, average="macro", zero_division=0))
        if scoring == "precision_micro":
            return float(precision_score(y_true, y_pred, average="micro", zero_division=0))
        if scoring == "precision_weighted":
            return float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
        if scoring == "recall":
            average = "binary" if len(self.classes_) == 2 else "macro"
            return float(recall_score(y_true, y_pred, average=average, zero_division=0))
        if scoring == "recall_macro":
            return float(recall_score(y_true, y_pred, average="macro", zero_division=0))
        if scoring == "recall_micro":
            return float(recall_score(y_true, y_pred, average="micro", zero_division=0))
        if scoring == "recall_weighted":
            return float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
        if scoring == "roc_auc" and proba.shape[1] == 2:
            return float(roc_auc_score(y_true, proba[:, 1]))
        if scoring == "average_precision" and proba.shape[1] == 2:
            return float(average_precision_score(y_true, proba[:, 1]))
        if scoring in {"neg_log_loss", "log_loss"}:
            return float(-log_loss(y_true, proba, labels=self.classes_))
        return float(accuracy_score(y_true, y_pred))

    @staticmethod
    def _stack_features_from_proba(proba: np.ndarray) -> np.ndarray:
        proba = np.asarray(proba, dtype=float)
        if proba.ndim != 2:
            raise ValueError("Expected probability predictions to be a 2D array.")
        if proba.shape[1] == 2:
            return proba[:, [1]]
        return proba

    def _combine_meta_features(
        self,
        X: Any,
        prediction_blocks: list[np.ndarray],
    ) -> Any:
        if not prediction_blocks:
            raise ValueError("At least one prediction block is required to build meta features.")

        prediction_matrix = np.hstack(prediction_blocks)
        if not self.include_original_features_in_meta:
            return prediction_matrix

        if sparse.issparse(X):
            return sparse.hstack([X, sparse.csr_matrix(prediction_matrix)], format="csr")

        original_matrix = np.asarray(X)
        return np.hstack([original_matrix, prediction_matrix])


def make_default_meta_estimators(random_state: int | None = None) -> list[tuple[str, Any]]:
    return [
        (
            "meta_random_forest",
            RandomForestClassifier(
                n_estimators=300,
                max_depth=6,
                min_samples_leaf=2,
                random_state=random_state,
            ),
        ),
        (
            "meta_logistic_regression",
            LogisticRegression(
                C=1.0,
                solver="lbfgs",
                penalty="l2",
                max_iter=1000,
                random_state=random_state,
            ),
        ),
    ]
