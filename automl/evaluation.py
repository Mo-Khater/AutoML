from __future__ import annotations

import multiprocessing as mp
from dataclasses import dataclass
from time import perf_counter
from typing import Any

from ConfigSpace.configuration_space import Configuration
import numpy as np
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from .components import get_classification_components
from .configspace_search_space import configuration_to_dict


@dataclass
class EvaluationResult:
    model_name: str
    params: dict[str, Any]
    preprocessing: str
    score: float
    cost: float
    duration: float
    status: str
    error: str | None = None
    fidelity_stage: int = 0
    sample_fraction: float = 1.0
    cv_folds: int = 5
    model_budget: float = 1.0


@dataclass(frozen=True)
class FidelitySpec:
    stage: int = 0
    sample_fraction: float = 1.0
    cv_folds: int | None = None
    model_budget: float = 1.0


def _evaluate_config_worker(
    queue: Any,
    config: Configuration,
    X: Any,
    y: Any,
    settings: dict[str, Any],
    fidelity: FidelitySpec | None,
) -> None:
    evaluator = ClassificationEvaluator(
        cv=settings["cv"],
        scoring=settings["scoring"],
        random_state=settings["random_state"],
        n_jobs=settings["n_jobs"],
        balance_classes=settings["balance_classes"],
        per_run_time_limit=None,
    )
    try:
        result = evaluator._evaluate_inline(config, X, y, fidelity=fidelity)
        queue.put(result)
    except Exception as exc:
        config_dict = configuration_to_dict(config)
        queue.put(
            EvaluationResult(
                model_name=config_dict["model_name"],
                params=config_dict["params"],
                preprocessing="none",
                score=0.0,
                cost=evaluator._score_to_cost(0.0),
                duration=0.0,
                status="failed",
                error=str(exc),
            )
        )


class ClassificationEvaluator:
    def __init__(
        self,
        *,
        cv: int = 5,
        scoring: str = "accuracy",
        random_state: int | None = None,
        n_jobs: int | None = None,
        balance_classes: bool = False,
        per_run_time_limit: float | None = None,
    ) -> None:
        self.cv = cv
        self.scoring = scoring
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.balance_classes = balance_classes
        self.per_run_time_limit = per_run_time_limit

    def evaluate(
        self,
        config: Configuration,
        X: Any,
        y: Any,
        time_limit: float | None = None,
        fidelity: FidelitySpec | None = None,
    ) -> EvaluationResult:
        effective_time_limit = (
            self.per_run_time_limit if time_limit is None else time_limit
        )
        if effective_time_limit is None:
            return self._evaluate_inline(config, X, y, fidelity=fidelity)
        return self._evaluate_with_timeout(config, X, y, float(effective_time_limit), fidelity=fidelity)

    def _evaluate_inline(
        self,
        config: Configuration,
        X: Any,
        y: Any,
        fidelity: FidelitySpec | None = None,
    ) -> EvaluationResult:
        started_at = perf_counter()
        config_dict = configuration_to_dict(config)
        model_name = config_dict["model_name"]
        params = config_dict["params"]
        effective_fidelity = self._normalize_fidelity(fidelity)
        X_eval, y_eval = self._apply_sample_fraction(
            X,
            y,
            effective_fidelity.sample_fraction,
        )

        try:
            preprocessing_name, score = self._evaluate_preprocessing_variants(
                model_name,
                params,
                X_eval,
                y_eval,
                effective_fidelity=effective_fidelity,
            )
            status = "success"
            error = None
        except Exception as exc:
            preprocessing_name = "none"
            score = 0.0
            status = "failed"
            error = str(exc)

        duration = float(perf_counter() - started_at)
        cost = self._score_to_cost(score)
        return EvaluationResult(
            model_name=model_name,
            params=params,
            preprocessing=preprocessing_name,
            score=score,
            cost=cost,
            duration=duration,
            status=status,
            error=error,
            fidelity_stage=effective_fidelity.stage,
            sample_fraction=effective_fidelity.sample_fraction,
            cv_folds=effective_fidelity.cv_folds or self.cv,
            model_budget=effective_fidelity.model_budget,
        )

    def _evaluate_with_timeout(
        self,
        config: Configuration,
        X: Any,
        y: Any,
        time_limit: float,
        fidelity: FidelitySpec | None = None,
    ) -> EvaluationResult:
        if time_limit <= 0:
            config_dict = configuration_to_dict(config)
            effective_fidelity = self._normalize_fidelity(fidelity)
            return EvaluationResult(
                model_name=config_dict["model_name"],
                params=config_dict["params"],
                preprocessing="none",
                score=0.0,
                cost=self._score_to_cost(0.0),
                duration=0.0,
                status="timeout",
                error="per_run_time_limit exceeded before evaluation started",
                fidelity_stage=effective_fidelity.stage,
                sample_fraction=effective_fidelity.sample_fraction,
                cv_folds=effective_fidelity.cv_folds or self.cv,
                model_budget=effective_fidelity.model_budget,
            )

        started_at = perf_counter()
        ctx = mp.get_context("spawn")
        queue = ctx.Queue()
        settings = {
            "cv": self.cv,
            "scoring": self.scoring,
            "random_state": self.random_state,
            "n_jobs": self.n_jobs,
            "balance_classes": self.balance_classes,
        }
        process = ctx.Process(
            target=_evaluate_config_worker,
            args=(queue, config, X, y, settings, fidelity),
        )
        process.start()
        process.join(timeout=time_limit)

        if process.is_alive():
            process.terminate()
            process.join()
            config_dict = configuration_to_dict(config)
            effective_fidelity = self._normalize_fidelity(fidelity)
            return EvaluationResult(
                model_name=config_dict["model_name"],
                params=config_dict["params"],
                preprocessing="none",
                score=0.0,
                cost=self._score_to_cost(0.0),
                duration=float(perf_counter() - started_at),
                status="timeout",
                error=f"per_run_time_limit exceeded ({time_limit:.2f}s)",
                fidelity_stage=effective_fidelity.stage,
                sample_fraction=effective_fidelity.sample_fraction,
                cv_folds=effective_fidelity.cv_folds or self.cv,
                model_budget=effective_fidelity.model_budget,
            )

        if not queue.empty():
            result = queue.get()
            result.duration = float(perf_counter() - started_at)
            return result

        config_dict = configuration_to_dict(config)
        effective_fidelity = self._normalize_fidelity(fidelity)
        return EvaluationResult(
            model_name=config_dict["model_name"],
            params=config_dict["params"],
            preprocessing="none",
            score=0.0,
            cost=self._score_to_cost(0.0),
            duration=float(perf_counter() - started_at),
            status="failed",
            error="evaluation subprocess exited without returning a result",
            fidelity_stage=effective_fidelity.stage,
            sample_fraction=effective_fidelity.sample_fraction,
            cv_folds=effective_fidelity.cv_folds or self.cv,
            model_budget=effective_fidelity.model_budget,
        )

    def score_config(
        self,
        config: Configuration,
        X: Any,
        y: Any,
    ) -> float:
        return self.evaluate(config, X, y).cost

    def build_model(
        self,
        model_name: str,
        params: dict[str, Any],
        preprocessing: str = "none",
        fidelity: FidelitySpec | None = None,
    ):
        estimator = self._build_estimator(model_name, params, fidelity=fidelity)
        transformer = self._build_preprocessor(preprocessing)
        if transformer is None:
            return estimator
        return Pipeline(
            steps=[
                ("preprocessor", transformer),
                ("estimator", estimator),
            ]
        )

    def _build_estimator(
        self,
        model_name: str,
        params: dict[str, Any],
        fidelity: FidelitySpec | None = None,
    ):
        components = get_classification_components()
        if model_name not in components:
            raise ValueError(f"Unsupported classification model `{model_name}`.")
        component = components[model_name]
        estimator = component.build_estimator(
            params,
            self.random_state,
            self.n_jobs,
            self.balance_classes,
        )
        self._apply_model_budget(estimator, model_name, params, self._normalize_fidelity(fidelity))
        return estimator

    def _evaluate_preprocessing_variants(
        self,
        model_name: str,
        params: dict[str, Any],
        X: Any,
        y: Any,
        effective_fidelity: FidelitySpec,
    ) -> tuple[str, float]:
        n_splits = self._resolve_cv_folds(y, effective_fidelity.cv_folds or self.cv)
        splitter = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=self.random_state,
        )

        best_preprocessing = "none"
        best_score = float("-inf")

        for preprocessing in self._candidate_preprocessors(model_name, params):
            model = self.build_model(
                model_name,
                params,
                preprocessing=preprocessing,
                fidelity=effective_fidelity,
            )
            scores = cross_val_score(
                model,
                X,
                y,
                cv=splitter,
                scoring=self.scoring,
                n_jobs=self.n_jobs,
            )
            score = float(scores.mean())
            if score > best_score:
                best_score = score
                best_preprocessing = preprocessing

        return best_preprocessing, best_score

    @staticmethod
    def _candidate_preprocessors(
        model_name: str,
        params: dict[str, Any],
    ) -> list[str]:
        candidates = ["none"]

        if model_name in {
            "svc",
            "knn",
            "lda",
            "liblinear_svc",
            "passive_aggressive",
            "qda",
            "sgd",
            "ridge_classifier",
            "mlp",
            "gaussian_process",
        }:
            candidates.extend(["standard", "robust"])
        elif model_name == "logistic_regression":
            solver = params.get("solver")
            if solver in {"lbfgs", "saga", "liblinear"}:
                candidates.extend(["standard", "robust", "minmax"])

        # Preserve order while removing duplicates.
        return list(dict.fromkeys(candidates))

    @staticmethod
    def _build_preprocessor(name: str):
        if name == "none":
            return None
        if name == "standard":
            return StandardScaler()
        if name == "robust":
            return RobustScaler()
        if name == "minmax":
            return MinMaxScaler()
        raise ValueError(f"Unsupported preprocessing strategy `{name}`.")

    def _score_to_cost(self, score: float) -> float:
        if self.scoring == "accuracy":
            return float(1.0 - score)
        return float(-score)

    def _normalize_fidelity(self, fidelity: FidelitySpec | None) -> FidelitySpec:
        if fidelity is None:
            return FidelitySpec(stage=0, sample_fraction=1.0, cv_folds=self.cv, model_budget=1.0)
        return FidelitySpec(
            stage=int(fidelity.stage),
            sample_fraction=float(fidelity.sample_fraction),
            cv_folds=int(fidelity.cv_folds or self.cv),
            model_budget=float(fidelity.model_budget),
        )

    def _apply_sample_fraction(
        self,
        X: Any,
        y: Any,
        sample_fraction: float,
    ) -> tuple[Any, Any]:
        if sample_fraction >= 0.999:
            return X, y

        n_samples = len(y)
        target_size = max(2, int(round(n_samples * sample_fraction)))
        if target_size >= n_samples:
            return X, y

        splitter = StratifiedShuffleSplit(
            n_splits=1,
            train_size=target_size,
            random_state=self.random_state,
        )
        indices, _ = next(splitter.split(np.zeros(n_samples), y))
        return self._subset_rows(X, indices), self._subset_rows(y, indices)

    @staticmethod
    def _subset_rows(data: Any, indices: np.ndarray) -> Any:
        if hasattr(data, "iloc"):
            return data.iloc[indices]
        return data[indices]

    @staticmethod
    def _apply_model_budget(
        estimator: Any,
        model_name: str,
        params: dict[str, Any],
        fidelity: FidelitySpec,
    ) -> None:
        if fidelity.model_budget >= 0.999:
            return

        budget_fields: dict[str, tuple[str, int]] = {
            "random_forest": ("n_estimators", 16),
            "extra_trees": ("n_estimators", 16),
            "gradient_boosting": ("n_estimators", 16),
            "lightgbm": ("n_estimators", 16),
            "xgboost": ("n_estimators", 16),
            "catboost": ("iterations", 16),
            "hist_gradient_boosting": ("max_iter", 16),
            "adaboost": ("n_estimators", 8),
            "logistic_regression": ("max_iter", 50),
            "mlp": ("max_iter", 30),
            "gaussian_process": ("max_iter_predict", 10),
        }
        budget_field = budget_fields.get(model_name)
        if budget_field is None:
            return

        field_name, minimum = budget_field
        base_value = params.get(field_name)
        if base_value is None and hasattr(estimator, "get_params"):
            base_value = estimator.get_params().get(field_name)
        if base_value is None:
            return

        scaled_value = max(minimum, int(round(float(base_value) * fidelity.model_budget)))
        if hasattr(estimator, "set_params"):
            try:
                estimator.set_params(**{field_name: scaled_value})
            except ValueError:
                pass

    @staticmethod
    def _resolve_cv_folds(y: Any, requested_folds: int) -> int:
        values = np.asarray(y)
        _, counts = np.unique(values, return_counts=True)
        if counts.size == 0:
            return 2
        max_supported = int(np.min(counts))
        return max(2, min(int(requested_folds), max_supported))
