from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sklearn.base import clone
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

from ..configspace_search_space import (
    configuration_to_dict,
    get_classification_configspace,
)
from ..evaluation import ClassificationEvaluator
from ..meta_learning import (
    compute_basic_classification_metafeatures,
    get_meta_learning_warmstarts,
)
from ..optimization.smac_optimizer import SMACOptimizationResult, SMACOptimizer


@dataclass
class AutoMLResult:
    best_model: Any
    best_score: float
    cv_results: list[dict[str, Any]]
    leaderboard: list[dict[str, Any]]
    ensemble: Any
    final_model: Any
    incumbent_config: Any = None
    runhistory: Any = None


class AutoMLEngine:
    def __init__(
        self,
        *,
        task: str,
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
        self.task = task
        self.time_budget = time_budget
        self.per_run_time_limit = per_run_time_limit
        self.n_trials = n_trials
        self.cv = cv
        self.scoring = scoring or "accuracy"
        self.random_state = random_state
        self.ensemble = ensemble
        self.ensemble_size = ensemble_size
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.balance_classes = balance_classes
        self.allowed_models = allowed_models
        self.use_meta_learning = use_meta_learning
        self.meta_learning_root = meta_learning_root
        self.meta_learning_top_datasets = meta_learning_top_datasets
        self.meta_learning_top_configs_per_dataset = meta_learning_top_configs_per_dataset
        self.max_meta_learning_warmstarts = max_meta_learning_warmstarts

    def search(self, X: Any, y: Any) -> AutoMLResult:
        if self.task != "classification":
            raise NotImplementedError(
                "AutoMLEngine currently supports classification only."
            )

        configspace = self._build_configspace()
        warmstart_configs = self._get_meta_learning_warmstarts(configspace, X, y)
        evaluator = self._build_evaluator()
        optimizer = self._build_optimizer(configspace, evaluator, X, y, warmstart_configs)

        optimization_result = optimizer.optimize()
        incumbent = optimization_result.incumbent
        if incumbent is None:
            raise RuntimeError("Optimizer did not return an incumbent configuration.")

        incumbent_evaluation = optimization_result.incumbent_result
        cv_results = self._build_cv_results(optimization_result=optimization_result)
        leaderboard = self._build_leaderboard(cv_results)
        best_model = self._fit_final_model(
            incumbent,
            evaluator,
            X,
            y,
            preprocessing=incumbent_evaluation.preprocessing,
        )
        best_model_score = float(incumbent_evaluation.score)
        ensemble_model = self._build_ensemble(cv_results, evaluator, X, y)
        ensemble_score = None
        final_model = best_model
        final_score = best_model_score

        if ensemble_model is not None:
            ensemble_score = self._evaluate_estimator(ensemble_model, X, y)
            if ensemble_score >= best_model_score:
                final_model = ensemble_model
                final_score = ensemble_score

        return AutoMLResult(
            best_model=best_model,
            best_score=float(final_score),
            cv_results=cv_results,
            leaderboard=leaderboard,
            ensemble=ensemble_model,
            final_model=final_model,
            incumbent_config=incumbent,
            runhistory=optimization_result.runhistory,
        )

    def _build_configspace(self):
        return get_classification_configspace(allowed_models=self.allowed_models)

    def _build_evaluator(self) -> ClassificationEvaluator:
        return ClassificationEvaluator(
            cv=self.cv,
            scoring=self.scoring,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            balance_classes=self.balance_classes,
            per_run_time_limit=self._resolve_per_run_time_limit(),
        )

    def _build_optimizer(
        self,
        configspace: Any,
        evaluator: ClassificationEvaluator,
        X: Any,
        y: Any,
        initial_configurations: list[Any],
    ) -> SMACOptimizer:
        return SMACOptimizer(
            configspace=configspace,
            evaluator=evaluator,
            X=X,
            y=y,
            n_trials=self.n_trials,
            time_budget=self.time_budget,
            random_state=self.random_state,
            per_run_time_limit=self._resolve_per_run_time_limit(),
            initial_configurations=initial_configurations,
            verbose=self.verbose,
        )

    def _fit_final_model(
        self,
        incumbent: Any,
        evaluator: ClassificationEvaluator,
        X: Any,
        y: Any,
        preprocessing: str = "none",
    ) -> Any:
        config_dict = configuration_to_dict(incumbent)
        model = evaluator.build_model(
            config_dict["model_name"],
            config_dict["params"],
            preprocessing=preprocessing,
        )
        model.fit(X, y)
        return model

    def _build_cv_results(
        self,
        *,
        optimization_result: SMACOptimizationResult,
    ) -> list[dict[str, Any]]:
        return list(optimization_result.runhistory)

    def _build_ensemble(
        self,
        cv_results: list[dict[str, Any]],
        evaluator: ClassificationEvaluator,
        X: Any,
        y: Any,
    ) -> Any:
        if not self.ensemble:
            return None

        candidate_rows = [
            row
            for row in sorted(
                cv_results,
                key=lambda item: item.get("score", float("-inf")),
                reverse=True,
            )
            if row.get("status") == "success" and row.get("config") is not None
        ]

        if len(candidate_rows) < 2:
            return None

        estimators: list[tuple[str, Any]] = []
        seen_signatures: set[tuple[Any, ...]] = set()
        supports_soft_voting = True

        for row in candidate_rows:
            if len(estimators) >= self.ensemble_size:
                break

            config_dict = configuration_to_dict(row["config"])
            signature = (
                config_dict["model_name"],
                tuple(sorted(config_dict["params"].items())),
                row.get("preprocessing", "none"),
            )
            if signature in seen_signatures:
                continue

            estimator = evaluator.build_model(
                config_dict["model_name"],
                config_dict["params"],
                preprocessing=row.get("preprocessing", "none"),
            )
            if not hasattr(estimator, "predict_proba"):
                supports_soft_voting = False
            estimators.append(
                (f"{config_dict['model_name']}_{len(estimators)}", estimator)
            )
            seen_signatures.add(signature)

        if len(estimators) < 2:
            return None

        ensemble_model = VotingClassifier(
            estimators=estimators,
            voting="soft" if supports_soft_voting else "hard",
            n_jobs=self.n_jobs,
        )
        ensemble_model.fit(X, y)
        return ensemble_model

    def _evaluate_estimator(self, estimator: Any, X: Any, y: Any) -> float:
        splitter = StratifiedKFold(
            n_splits=self.cv,
            shuffle=True,
            random_state=self.random_state,
        )
        scores = cross_val_score(
            clone(estimator),
            X,
            y,
            cv=splitter,
            scoring=self.scoring,
            n_jobs=self.n_jobs,
        )
        return float(scores.mean())

    @staticmethod
    def _build_leaderboard(cv_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        ranked = sorted(
            cv_results,
            key=lambda row: row.get("score", float("-inf")),
            reverse=True,
        )

        leaderboard: list[dict[str, Any]] = []
        for rank, row in enumerate(ranked, start=1):
            leaderboard.append(
                {
                    "rank": rank,
                    "trial_id": row.get("trial_id", rank - 1),
                    "model_name": row["model_name"],
                    "preprocessing": row.get("preprocessing", "none"),
                    "cv_score": row["score"],
                    "error": row.get("error"),
                    "params": row["params"],
                }
            )

        return leaderboard

    def _get_meta_learning_warmstarts(self, configspace: Any, X: Any, y: Any) -> list[Any]:
        if not self.use_meta_learning or self.task != "classification":
            return []

        try:
            metafeatures = compute_basic_classification_metafeatures(X, y)
            warmstarts = get_meta_learning_warmstarts(
                y=y,
                metafeatures=metafeatures,
                configspace=configspace,
                scoring=self.scoring,
                allowed_models=self.allowed_models,
                meta_learning_root=self.meta_learning_root,
                top_datasets=self.meta_learning_top_datasets,
                top_configs_per_dataset=self.meta_learning_top_configs_per_dataset,
                max_warmstarts=self.max_meta_learning_warmstarts,
            )
        except Exception:
            return []

        if self.verbose and warmstarts:
            print(f"[AutoML] Meta-learning warmstarts loaded: {len(warmstarts)}")
        return warmstarts

    def _resolve_per_run_time_limit(self) -> float | None:
        if self.per_run_time_limit is not None:
            return float(self.per_run_time_limit)

        if self.time_budget is None:
            return 60.0

        effective_jobs = max(1, self.n_jobs or 1)
        per_run_time_limit = max(1.0, float(effective_jobs * self.time_budget) / 10.0)
        per_run_time_limit = min(per_run_time_limit, float(self.time_budget))

        if self.time_budget >= 2:
            per_run_time_limit = min(
                per_run_time_limit,
                max(1.0, float(self.time_budget) / 2.0),
            )

        return per_run_time_limit
