from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from joblib import Parallel, delayed
from scipy import sparse
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import BaggingClassifier
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
from sklearn.model_selection import StratifiedKFold, cross_val_predict


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
class ForwardSelectionResult:
    selected_names: list[str]
    score: float


class StackedEnsembleClassifier(BaseEstimator, ClassifierMixin):
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
        self.classes_ = np.unique(y)
        splitter = StratifiedKFold(
            n_splits=self.cv,
            shuffle=True,
            random_state=self.random_state,
        )

        base_results = Parallel(n_jobs=self.n_jobs)(
            delayed(self._fit_base_model)(base_name, base_estimator, X, y, splitter)
            for base_name, base_estimator in self.base_estimators
        )
        if not base_results:
            raise RuntimeError("At least one base estimator is required to fit the stacked ensemble.")

        self.forward_selection_ = self._forward_select_base_models(X, np.asarray(y), base_results)
        selected_names = set(self.forward_selection_.selected_names)
        selected_results = [
            (base_name, fitted_model, oof_proba)
            for base_name, fitted_model, oof_proba in base_results
            if base_name in selected_names
        ]
        if len(selected_results) < self.min_base_models:
            raise RuntimeError("Forward selection did not produce the minimum number of base models.")

        self.base_models_ = [(base_name, fitted_model) for base_name, fitted_model, _ in selected_results]
        self.base_model_names_ = [base_name for base_name, _, _ in selected_results]
        base_feature_blocks = [
            self._stack_features_from_proba(oof_proba)
            for _, _, oof_proba in selected_results
        ]
        stack_train_X = self._combine_meta_features(X, base_feature_blocks)

        meta_name, meta_estimator = self._resolve_meta_estimator()
        self.meta_model_name_ = meta_name
        self.meta_model_ = ProbabilityAdapterClassifier(meta_estimator)
        self.meta_model_.fit(stack_train_X, y)

        self._log_ensemble_structure()
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        stack_features = self._build_stack_features(X)
        return np.asarray(self.meta_model_.predict_proba(stack_features), dtype=float)

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

    def _forward_select_base_models(
        self,
        X: Any,
        y: np.ndarray,
        base_results: list[tuple[str, Any, np.ndarray]],
    ) -> ForwardSelectionResult:
        remaining = list(base_results)
        if not remaining:
            raise RuntimeError("No base model results available for forward selection.")

        selected: list[tuple[str, Any, np.ndarray]] = []
        best_single = max(remaining, key=lambda item: self._score_predictions(y, item[2]))
        selected.append(best_single)
        remaining.remove(best_single)
        current_score = self._evaluate_meta_subset(X, y, selected)
        self._log_forward_selection_step(
            step=1,
            selected_names=[best_single[0]],
            score=current_score,
            improvement=None,
        )

        while remaining:
            best_candidate: tuple[str, Any, np.ndarray] | None = None
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

        return ForwardSelectionResult(
            selected_names=[name for name, _, _ in selected],
            score=current_score,
        )

    def _evaluate_meta_subset(
        self,
        X: Any,
        y: np.ndarray,
        selected_results: list[tuple[str, Any, np.ndarray]],
    ) -> float:
        blocks = [
            self._stack_features_from_proba(oof_proba)
            for _, _, oof_proba in selected_results
        ]
        stack_train_X = self._combine_meta_features(X, blocks)
        splitter = StratifiedKFold(
            n_splits=self.cv,
            shuffle=True,
            random_state=self.random_state,
        )
        _, meta_estimator = self._resolve_meta_estimator()
        meta_oof_proba = cross_val_predict(
            ProbabilityAdapterClassifier(meta_estimator),
            stack_train_X,
            y,
            cv=splitter,
            method="predict_proba",
            n_jobs=self.inner_n_jobs,
        )
        return self._score_predictions(y, np.asarray(meta_oof_proba, dtype=float))

    def _resolve_meta_estimator(self) -> tuple[str, Any]:
        if self.meta_estimators:
            return self.meta_estimators[0][0], clone(self.meta_estimators[0][1])
        defaults = make_default_meta_estimators(random_state=self.random_state)
        return defaults[0][0], clone(defaults[0][1])

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
            return BaggingClassifier(estimator=adapted_estimator, **bagging_kwargs)
        except TypeError:
            return BaggingClassifier(base_estimator=adapted_estimator, **bagging_kwargs)

    def _build_stack_features(self, X: Any) -> np.ndarray:
        blocks: list[np.ndarray] = []
        for _, model in self.base_models_:
            proba = model.predict_proba(X)
            blocks.append(self._stack_features_from_proba(proba))
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
