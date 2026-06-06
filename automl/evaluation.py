from __future__ import annotations

import multiprocessing as mp
from dataclasses import dataclass
from time import perf_counter
from typing import Any

from ConfigSpace.configuration_space import Configuration
from sklearn.model_selection import StratifiedKFold, cross_val_score
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


def _evaluate_config_worker(
    queue: Any,
    config: Configuration,
    X: Any,
    y: Any,
    settings: dict[str, Any],
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
        result = evaluator._evaluate_inline(config, X, y)
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
    ) -> EvaluationResult:
        effective_time_limit = (
            self.per_run_time_limit if time_limit is None else time_limit
        )
        if effective_time_limit is None:
            return self._evaluate_inline(config, X, y)
        return self._evaluate_with_timeout(config, X, y, float(effective_time_limit))

    def _evaluate_inline(
        self,
        config: Configuration,
        X: Any,
        y: Any,
    ) -> EvaluationResult:
        started_at = perf_counter()
        config_dict = configuration_to_dict(config)
        model_name = config_dict["model_name"]
        params = config_dict["params"]

        try:
            preprocessing_name, score = self._evaluate_preprocessing_variants(
                model_name,
                params,
                X,
                y,
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
        )

    def _evaluate_with_timeout(
        self,
        config: Configuration,
        X: Any,
        y: Any,
        time_limit: float,
    ) -> EvaluationResult:
        if time_limit <= 0:
            config_dict = configuration_to_dict(config)
            return EvaluationResult(
                model_name=config_dict["model_name"],
                params=config_dict["params"],
                preprocessing="none",
                score=0.0,
                cost=self._score_to_cost(0.0),
                duration=0.0,
                status="timeout",
                error="per_run_time_limit exceeded before evaluation started",
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
            args=(queue, config, X, y, settings),
        )
        process.start()
        process.join(timeout=time_limit)

        if process.is_alive():
            process.terminate()
            process.join()
            config_dict = configuration_to_dict(config)
            return EvaluationResult(
                model_name=config_dict["model_name"],
                params=config_dict["params"],
                preprocessing="none",
                score=0.0,
                cost=self._score_to_cost(0.0),
                duration=float(perf_counter() - started_at),
                status="timeout",
                error=f"per_run_time_limit exceeded ({time_limit:.2f}s)",
            )

        if not queue.empty():
            result = queue.get()
            result.duration = float(perf_counter() - started_at)
            return result

        config_dict = configuration_to_dict(config)
        return EvaluationResult(
            model_name=config_dict["model_name"],
            params=config_dict["params"],
            preprocessing="none",
            score=0.0,
            cost=self._score_to_cost(0.0),
            duration=float(perf_counter() - started_at),
            status="failed",
            error="evaluation subprocess exited without returning a result",
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
    ):
        estimator = self._build_estimator(model_name, params)
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
    ):
        components = get_classification_components()
        if model_name not in components:
            raise ValueError(f"Unsupported classification model `{model_name}`.")
        component = components[model_name]
        return component.build_estimator(
            params,
            self.random_state,
            self.n_jobs,
            self.balance_classes,
        )

    def _evaluate_preprocessing_variants(
        self,
        model_name: str,
        params: dict[str, Any],
        X: Any,
        y: Any,
    ) -> tuple[str, float]:
        splitter = StratifiedKFold(
            n_splits=self.cv,
            shuffle=True,
            random_state=self.random_state,
        )

        best_preprocessing = "none"
        best_score = float("-inf")

        for preprocessing in self._candidate_preprocessors(model_name, params):
            model = self.build_model(model_name, params, preprocessing=preprocessing)
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
