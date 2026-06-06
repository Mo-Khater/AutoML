from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, cross_val_predict, cross_val_score

from ..configspace_search_space import (
    configuration_to_dict,
    get_classification_configspace,
)
from ..evaluation import ClassificationEvaluator, FidelitySpec
from ..meta_learning import (
    compute_basic_classification_metafeatures,
    get_meta_learning_warmstarts,
)
from ..optimization.smac_optimizer import SMACOptimizationResult, SMACOptimizer
from ..stacked_ensemble import (
    ProbabilityAdapterClassifier,
    StackedEnsembleClassifier,
    make_default_meta_estimators,
)


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


@dataclass
class _BaseCandidate:
    name: str
    estimator: Any
    score: float
    oof_proba: np.ndarray


class AutoMLEngine:
    _STACK_SELECTION_SAMPLE_SIZE = 300
    _DEFAULT_FIDELITY_STAGES = (
        FidelitySpec(stage=0, sample_fraction=0.2, cv_folds=2, model_budget=0.25),
        FidelitySpec(stage=1, sample_fraction=0.5, cv_folds=3, model_budget=0.6),
        FidelitySpec(stage=2, sample_fraction=1.0, cv_folds=None, model_budget=1.0),
    )

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
        ensemble_strategy: str = "stacked",
        stacked_base_size: int = 4,
        stacked_bagging_n_estimators: int = 5,
        stacked_meta_model_names: list[str] | None = None,
        stacked_include_base_predictions: bool = True,
        stacked_include_original_features_in_meta: bool = True,
        stacked_final_weight_optimizer: str = "greedy",
        n_jobs: int | None = None,
        search_n_parallel: int = 3,
        stack_n_jobs: int | None = None,
        inner_n_jobs: int | None = 1,
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
        self.ensemble_strategy = ensemble_strategy
        self.stacked_base_size = stacked_base_size
        self.stacked_bagging_n_estimators = stacked_bagging_n_estimators
        self.stacked_meta_model_names = ["random_forest", "logistic_regression"]
        self.stacked_include_base_predictions = stacked_include_base_predictions
        self.stacked_include_original_features_in_meta = stacked_include_original_features_in_meta
        self.stacked_final_weight_optimizer = stacked_final_weight_optimizer
        self.n_jobs = n_jobs
        self.search_n_parallel = max(1, int(search_n_parallel))
        self.stack_n_jobs = n_jobs if stack_n_jobs is None else stack_n_jobs
        self.inner_n_jobs = inner_n_jobs
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

        effective_allowed_models = self._resolve_allowed_models(X)
        configspace = self._build_configspace(effective_allowed_models)
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
        ensemble_model = self._build_ensemble(cv_results, evaluator, X, y)
        best_model_score = float(incumbent_evaluation.score)
        ensemble_score = None
        final_model = best_model
        final_score = best_model_score

        if ensemble_model is not None:
            ensemble_score = self._evaluate_estimator(ensemble_model, X, y)
            leaderboard = self._append_ensemble_to_leaderboard(
                leaderboard=leaderboard,
                ensemble_score=ensemble_score,
            )
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

    def _build_configspace(self, allowed_models: list[str] | None):
        return get_classification_configspace(allowed_models=allowed_models)

    def _build_evaluator(self) -> ClassificationEvaluator:
        return ClassificationEvaluator(
            cv=self.cv,
            scoring=self.scoring,
            random_state=self.random_state,
            n_jobs=self.inner_n_jobs,
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
            n_parallel=self.search_n_parallel,
            verbose=self.verbose,
            fidelity_stages=list(self._DEFAULT_FIDELITY_STAGES),
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

        effective_ensemble_strategy = self._resolve_ensemble_strategy(X)

        if effective_ensemble_strategy == "stacked":
            return self._build_stacked_ensemble(cv_results, evaluator, X, y)
        if effective_ensemble_strategy != "voting":
            raise ValueError(f"Unsupported ensemble strategy `{effective_ensemble_strategy}`.")

        ranked_rows = [
            row
            for row in sorted(
                cv_results,
                key=lambda item: item.get("score", float("-inf")),
                reverse=True,
            )
            if row.get("status") == "success" and row.get("config") is not None
        ]

        if len(ranked_rows) < 2:
            return None

        candidate_estimators: list[tuple[tuple[Any, ...], str, Any]] = []
        seen_signatures: set[tuple[Any, ...]] = set()
        for row in ranked_rows:
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
            estimator_name = f"{config_dict['model_name']}_{len(candidate_estimators)}"
            candidate_estimators.append((signature, estimator_name, estimator))
            seen_signatures.add(signature)

            if len(candidate_estimators) >= max(self.ensemble_size * 3, self.ensemble_size):
                break

        if len(candidate_estimators) < 2:
            return None

        selected_estimators: list[tuple[str, Any]] = []
        remaining_candidates = list(candidate_estimators)
        best_ensemble_score = float("-inf")

        while remaining_candidates and len(selected_estimators) < self.ensemble_size:
            best_candidate_idx: int | None = None
            best_candidate_score = best_ensemble_score

            for idx, (_, candidate_name, candidate_estimator) in enumerate(remaining_candidates):
                trial_estimators = selected_estimators + [(candidate_name, candidate_estimator)]
                voting = self._get_voting_strategy([estimator for _, estimator in trial_estimators])
                trial_ensemble = VotingClassifier(
                    estimators=trial_estimators,
                    voting=voting,
                    n_jobs=self.inner_n_jobs,
                )
                trial_score = self._evaluate_estimator(trial_ensemble, X, y)
                if trial_score > best_candidate_score:
                    best_candidate_score = trial_score
                    best_candidate_idx = idx

            if best_candidate_idx is None:
                break

            _, candidate_name, candidate_estimator = remaining_candidates.pop(best_candidate_idx)
            selected_estimators.append((candidate_name, candidate_estimator))
            best_ensemble_score = best_candidate_score

        if len(selected_estimators) < 2:
            return None

        ensemble_model = VotingClassifier(
            estimators=selected_estimators,
            voting=self._get_voting_strategy([estimator for _, estimator in selected_estimators]),
            n_jobs=self.inner_n_jobs,
        )
        ensemble_model.fit(X, y)
        return ensemble_model

    def _resolve_ensemble_strategy(self, X: Any) -> str:
        n_rows = len(X) if hasattr(X, "__len__") else 0
        if self.ensemble_strategy == "stacked" and n_rows > 50_000:
            if self.verbose:
                print(
                    "[AutoML] Large dataset detected; switching ensemble strategy "
                    "from stacked to voting."
                )
            return "voting"
        return self.ensemble_strategy

    def _build_stacked_ensemble(
        self,
        cv_results: list[dict[str, Any]],
        evaluator: ClassificationEvaluator,
        X: Any,
        y: Any,
    ) -> Any:
        ranked_rows = [
            row
            for row in sorted(
                cv_results,
                key=lambda item: item.get("score", float("-inf")),
                reverse=True,
            )
            if row.get("status") == "success" and row.get("config") is not None
        ]
        if len(ranked_rows) < 2:
            return None

        base_estimators = self._select_diverse_base_estimators(ranked_rows, evaluator, X, y)
        if len(base_estimators) < 2:
            return None

        meta_estimators = self._select_meta_estimators(ranked_rows, evaluator)
        if len(meta_estimators) < 2:
            return None

        ensemble_model = StackedEnsembleClassifier(
            base_estimators=base_estimators,
            meta_estimators=meta_estimators,
            cv=self.cv,
            random_state=self.random_state,
            n_jobs=self.stack_n_jobs,
            inner_n_jobs=self.inner_n_jobs,
            bagging_n_estimators=self.stacked_bagging_n_estimators,
            include_base_predictions_in_final_ensemble=self.stacked_include_base_predictions,
            include_original_features_in_meta=self.stacked_include_original_features_in_meta,
            final_weight_optimizer=self.stacked_final_weight_optimizer,
            scoring=self.scoring,
            verbose=self.verbose,
        )
        ensemble_model.fit(X, y)
        return ensemble_model

    def _select_diverse_base_estimators(
        self,
        ranked_rows: list[dict[str, Any]],
        evaluator: ClassificationEvaluator,
        X: Any,
        y: Any,
    ) -> list[tuple[str, Any]]:
        candidate_rows: list[tuple[str, Any, float]] = []
        seen_signatures: set[tuple[Any, ...]] = set()
        candidate_pool_size = max(self.stacked_base_size * 3, self.stacked_base_size)

        for row in ranked_rows:
            config_dict = configuration_to_dict(row["config"])
            family_name = config_dict["model_name"]
            signature = (
                family_name,
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
            estimator_name = f"base_{family_name}_{len(candidate_rows)}"
            candidate_rows.append((estimator_name, estimator, float(row.get("score", float("-inf")))))
            seen_signatures.add(signature)

            if len(candidate_rows) >= candidate_pool_size:
                break

        if len(candidate_rows) <= self.stacked_base_size:
            return [(name, estimator) for name, estimator, _ in candidate_rows]

        candidate_predictions = self._get_candidate_oof_probabilities(candidate_rows, X, y)
        if len(candidate_predictions) <= self.stacked_base_size:
            return [(candidate.name, candidate.estimator) for candidate in candidate_predictions]

        return [
            (candidate.name, candidate.estimator)
            for candidate in self._greedy_select_diverse_candidates(candidate_predictions)
        ]

    def _get_candidate_oof_probabilities(
        self,
        candidate_rows: list[tuple[str, Any, float]],
        X: Any,
        y: Any,
    ) -> list[_BaseCandidate]:
        X_sample, y_sample = self._sample_for_stacked_selection(X, y)
        splitter = StratifiedKFold(
            n_splits=self.cv,
            shuffle=True,
            random_state=self.random_state,
        )
        candidates: list[_BaseCandidate] = []
        candidate_results = Parallel(n_jobs=self.stack_n_jobs, prefer="threads")(
            delayed(self._evaluate_base_candidate_oof)(
                candidate_name,
                estimator,
                score,
                X_sample,
                y_sample,
                splitter,
            )
            for candidate_name, estimator, score in candidate_rows
        )
        for candidate in candidate_results:
            if candidate is not None:
                candidates.append(candidate)

        return candidates

    def _evaluate_base_candidate_oof(
        self,
        candidate_name: str,
        estimator: Any,
        score: float,
        X: Any,
        y: Any,
        splitter: StratifiedKFold,
    ) -> _BaseCandidate | None:
        try:
            oof_proba = cross_val_predict(
                ProbabilityAdapterClassifier(estimator),
                X,
                y,
                cv=splitter,
                method="predict_proba",
                n_jobs=self.inner_n_jobs,
            )
        except Exception as exc:
            if self.verbose:
                print(
                    f"[AutoML] Skipping {candidate_name} during diverse base selection: {exc}"
                )
            return None

        return _BaseCandidate(
            name=candidate_name,
            estimator=estimator,
            score=score,
            oof_proba=np.asarray(oof_proba, dtype=float),
        )

    def _sample_for_stacked_selection(self, X: Any, y: Any) -> tuple[Any, Any]:
        if len(y) <= self._STACK_SELECTION_SAMPLE_SIZE:
            return X, y

        splitter = StratifiedShuffleSplit(
            n_splits=1,
            train_size=self._STACK_SELECTION_SAMPLE_SIZE,
            random_state=self.random_state,
        )
        indices, _ = next(splitter.split(np.zeros(len(y)), y))
        return self._subset_rows(X, indices), self._subset_rows(y, indices)

    @staticmethod
    def _subset_rows(data: Any, indices: np.ndarray) -> Any:
        if hasattr(data, "iloc"):
            return data.iloc[indices]
        return data[indices]

    def _greedy_select_diverse_candidates(
        self,
        candidates: list[_BaseCandidate],
    ) -> list[_BaseCandidate]:
        selected: list[_BaseCandidate] = []
        remaining = list(candidates)
        best_score = max(candidate.score for candidate in remaining)
        score_floor = best_score - max(abs(best_score) * 0.05, 1e-12)
        quality_candidates = [
            candidate for candidate in remaining if candidate.score >= score_floor
        ]
        if len(quality_candidates) >= self.stacked_base_size:
            remaining = quality_candidates

        score_values = np.asarray([candidate.score for candidate in remaining], dtype=float)
        score_min = float(np.min(score_values))
        score_range = float(np.max(score_values) - score_min)

        selected.append(max(remaining, key=lambda candidate: candidate.score))
        remaining.remove(selected[0])

        while remaining and len(selected) < self.stacked_base_size:
            best_candidate = max(
                remaining,
                key=lambda candidate: self._base_selection_score(
                    candidate,
                    selected,
                    score_min,
                    score_range,
                ),
            )
            selected.append(best_candidate)
            remaining.remove(best_candidate)

        if self.verbose:
            selected_names = ", ".join(candidate.name for candidate in selected)
            print(f"[AutoML] Diversity-selected stacked base models: {selected_names}")

        return selected

    def _base_selection_score(
        self,
        candidate: _BaseCandidate,
        selected: list[_BaseCandidate],
        score_min: float,
        score_range: float,
    ) -> float:
        if score_range <= 0:
            normalized_quality = 1.0
        else:
            normalized_quality = (candidate.score - score_min) / score_range

        average_diversity = float(
            np.mean(
                [
                    self._prediction_diversity(candidate.oof_proba, selected_candidate.oof_proba)
                    for selected_candidate in selected
                ]
            )
        )
        return 0.7 * normalized_quality + 0.3 * average_diversity

    @staticmethod
    def _prediction_diversity(left: np.ndarray, right: np.ndarray) -> float:
        if left.shape != right.shape:
            return 0.0
        return float(np.mean(np.abs(left - right)))

    def _select_meta_estimators(
        self,
        ranked_rows: list[dict[str, Any]],
        evaluator: ClassificationEvaluator,
    ) -> list[tuple[str, Any]]:
        _ = ranked_rows, evaluator
        return make_default_meta_estimators(random_state=self.random_state)

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
            n_jobs=self.inner_n_jobs,
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

    def _append_ensemble_to_leaderboard(
        self,
        *,
        leaderboard: list[dict[str, Any]],
        ensemble_score: float,
    ) -> list[dict[str, Any]]:
        updated = list(leaderboard)
        updated.append(
            {
                "rank": None,
                "trial_id": "ensemble",
                "model_name": f"{self.ensemble_strategy}_ensemble",
                "preprocessing": "stacked",
                "cv_score": float(ensemble_score),
                "error": None,
                "params": {
                    "stacked_base_size": self.stacked_base_size,
                    "stacked_bagging_n_estimators": self.stacked_bagging_n_estimators,
                    "stacked_meta_model_names": list(self.stacked_meta_model_names),
                    "include_base_predictions": self.stacked_include_base_predictions,
                    "include_original_features_in_meta": self.stacked_include_original_features_in_meta,
                    "final_weight_optimizer": self.stacked_final_weight_optimizer,
                },
            }
        )
        reranked = sorted(
            updated,
            key=lambda row: row.get("cv_score", float("-inf")),
            reverse=True,
        )
        for rank, row in enumerate(reranked, start=1):
            row["rank"] = rank
        return reranked

    def _resolve_allowed_models(self, X: Any) -> list[str] | None:
        base_models = (
            sorted(set(self.allowed_models))
            if self.allowed_models is not None
            else None
        )

        candidate_models = list(base_models) if base_models is not None else None
        disabled_models: list[str] = []
        n_samples = len(X) if hasattr(X, "__len__") else 0

        if n_samples >= 10_000:
            disabled_models.append("knn")
            disabled_models.append("gaussian_process")
        if n_samples >= 50_000:
            disabled_models.append("svc")
            disabled_models.append("mlp")

        if self._features_have_negative_values(X):
            disabled_models.append("multinomial_nb")

        if not disabled_models:
            return candidate_models

        disabled_set = set(disabled_models)
        if candidate_models is None:
            filtered_models = [
                model_name
                for model_name in get_classification_configspace().get_hyperparameter("model_name").choices
                if model_name not in disabled_set
            ]
        else:
            filtered_models = [
                model_name for model_name in candidate_models if model_name not in disabled_set
            ]

        if self.verbose:
            print(
                "[AutoML] Dataset-aware filtering disabled models:",
                sorted(disabled_set),
            )
        return filtered_models

    @staticmethod
    def _features_have_negative_values(X: Any) -> bool:
        try:
            if hasattr(X, "select_dtypes"):
                numeric_frame = X.select_dtypes(include=["number"])
                if numeric_frame.shape[1] == 0:
                    return False
                numeric_values = numeric_frame.to_numpy()
            else:
                numeric_values = np.asarray(X)
                if numeric_values.dtype.kind not in {"i", "u", "f", "b"}:
                    return False
            return bool(np.nanmin(numeric_values) < 0)
        except Exception:
            return False

    @staticmethod
    def _get_voting_strategy(estimators: list[Any]) -> str:
        if all(hasattr(estimator, "predict_proba") for estimator in estimators):
            return "soft"
        return "hard"

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
                allowed_models=list(configspace.get_hyperparameter("model_name").choices),
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

        per_run_time_limit = max(1.0, float(self.time_budget) / 10.0)
        per_run_time_limit = min(per_run_time_limit, float(self.time_budget))

        if self.time_budget >= 2:
            per_run_time_limit = min(
                per_run_time_limit,
                max(1.0, float(self.time_budget) / 2.0),
            )

        return per_run_time_limit
