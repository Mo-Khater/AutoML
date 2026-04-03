from __future__ import annotations

from dataclasses import dataclass
from statistics import NormalDist
from time import perf_counter
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestRegressor


@dataclass
class SMACOptimizationResult:
    incumbent: Any
    incumbent_cost: float
    incumbent_result: Any
    runhistory: list[dict[str, Any]]
    optimizer: Any


class SMACOptimizer:

    def __init__(
        self,
        *,
        configspace: Any,
        evaluator: Any,
        X: Any,
        y: Any,
        n_trials: int = 50,
        time_budget: float | None = None,
        random_state: int | None = None,
        output_directory: str | None = None,
        deterministic: bool = True,
        per_run_time_limit: float | None = None,
        initial_configurations: list[Any] | None = None,
        n_initial_points: int = 8,
        candidate_pool_size: int = 128,
        verbose: int = 1,
    ) -> None:
        self.configspace = configspace
        self.evaluator = evaluator
        self.X = X
        self.y = y
        self.n_trials = n_trials
        self.time_budget = time_budget
        self.random_state = random_state
        self.output_directory = output_directory
        self.deterministic = deterministic
        self.per_run_time_limit = per_run_time_limit
        self.initial_configurations = list(initial_configurations or [])
        self.n_initial_points = max(1, n_initial_points)
        self.candidate_pool_size = max(16, candidate_pool_size)
        self.verbose = verbose

        self._rng = np.random.default_rng(random_state)
        self._observations: list[dict[str, Any]] = []
        self._incumbent = None
        self._incumbent_cost = float("inf")
        self._incumbent_result = None

        if hasattr(self.configspace, "seed"):
            self.configspace.seed(random_state)

    def optimize(self) -> SMACOptimizationResult:
        started_at = perf_counter()

        for trial_id in range(self.n_trials):
            if self._time_budget_exceeded(started_at):
                break

            config = self._suggest()
            if self.verbose:
                model_name = self._safe_model_name(config)
                print(
                    f"[AutoML] Trial {trial_id + 1}/{self.n_trials} - evaluating {model_name}"
                )
            result = self.evaluator.evaluate(
                config,
                self.X,
                self.y,
                time_limit=self._current_per_run_time_limit(started_at),
            )
            row = {
                "trial_id": trial_id,
                "config": config,
                "model_name": result.model_name,
                "params": result.params,
                "preprocessing": result.preprocessing,
                "score": result.score,
                "cost": result.cost,
                "duration": result.duration,
                "status": result.status,
                "error": result.error,
            }
            self._observations.append(row)

            if self.verbose:
                print(
                    f"[AutoML] Trial {trial_id + 1}/{self.n_trials} - "
                    f"score={result.score:.6f} cost={result.cost:.6f} status={result.status}"
                )
                if result.error and self.verbose > 1:
                    print(f"[AutoML] Trial {trial_id + 1}/{self.n_trials} - error: {result.error}")

            if result.cost < self._incumbent_cost:
                self._incumbent = config
                self._incumbent_cost = float(result.cost)
                self._incumbent_result = result
                if self.verbose:
                    print(
                        f"[AutoML] New incumbent - model={result.model_name} "
                        f"preprocessing={result.preprocessing} score={result.score:.6f}"
                    )

        return SMACOptimizationResult(
            incumbent=self._incumbent,
            incumbent_cost=self._incumbent_cost,
            incumbent_result=self._incumbent_result,
            runhistory=list(self._observations),
            optimizer=self,
        )

    def get_target_function(self):
        def target_function(config: Any, seed: int | None = None) -> float:
            return self.evaluator.score_config(config, self.X, self.y)

        return target_function

    def _suggest(self):
        if self.initial_configurations:
            return self.initial_configurations.pop(0)

        if len(self._observations) < self.n_initial_points:
            return self._sample_random_configuration()

        X_obs = np.vstack([row["vector"] for row in self._vectorized_observations()])
        y_obs = np.asarray([row["cost"] for row in self._vectorized_observations()], dtype=float)

        if len(np.unique(y_obs)) <= 1:
            return self._sample_random_configuration()

        surrogate = RandomForestRegressor(
            n_estimators=200,
            random_state=self.random_state,
            min_samples_leaf=2,
            n_jobs=1,
        )
        surrogate.fit(X_obs, y_obs)

        candidate_configs = self._sample_candidate_pool()
        candidate_vectors = np.vstack([self._config_to_vector(config) for config in candidate_configs])
        mean, std = self._predict_with_uncertainty(surrogate, candidate_vectors)
        acquisition = self._expected_improvement(mean, std, best=self._incumbent_cost)
        best_idx = int(np.argmax(acquisition))
        return candidate_configs[best_idx]

    def _vectorized_observations(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for row in self._observations:
            vector = row.get("vector")
            if vector is None:
                vector = self._config_to_vector(row["config"])
                row["vector"] = vector
            rows.append(row)
        return rows

    def _sample_random_configuration(self):
        return self.configspace.sample_configuration()

    def _sample_candidate_pool(self) -> list[Any]:
        candidates: list[Any] = []
        seen: set[tuple[float, ...]] = set()
        max_attempts = self.candidate_pool_size * 4

        for _ in range(max_attempts):
            config = self._sample_random_configuration()
            vector_key = tuple(self._config_to_vector(config).tolist())
            if vector_key in seen:
                continue
            seen.add(vector_key)
            candidates.append(config)
            if len(candidates) >= self.candidate_pool_size:
                break

        if not candidates:
            candidates.append(self._sample_random_configuration())

        return candidates

    @staticmethod
    def _predict_with_uncertainty(
        surrogate: RandomForestRegressor,
        X: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        tree_predictions = np.vstack([tree.predict(X) for tree in surrogate.estimators_])
        mean = tree_predictions.mean(axis=0)
        std = tree_predictions.std(axis=0)
        std = np.maximum(std, 1e-9)
        return mean, std

    @staticmethod
    def _expected_improvement(
        mean: np.ndarray,
        std: np.ndarray,
        *,
        best: float,
        xi: float = 0.01,
    ) -> np.ndarray:
        improvement = best - mean - xi
        z = improvement / std
        normal = NormalDist()
        phi = np.asarray([normal.pdf(float(value)) for value in z], dtype=float)
        Phi = np.asarray([normal.cdf(float(value)) for value in z], dtype=float)
        ei = improvement * Phi + std * phi
        ei[std <= 1e-12] = 0.0
        return ei

    @staticmethod
    def _config_to_vector(config: Any) -> np.ndarray:
        if hasattr(config, "get_array"):
            vector = np.asarray(config.get_array(), dtype=float)
        else:
            raise TypeError("Configuration object must implement `get_array()`.")

        return np.nan_to_num(vector, nan=-1.0, posinf=1.0, neginf=-1.0)

    def _time_budget_exceeded(self, started_at: float) -> bool:
        if self.time_budget is None:
            return False
        return (perf_counter() - started_at) >= float(self.time_budget)

    def _current_per_run_time_limit(self, started_at: float) -> float | None:
        if self.per_run_time_limit is None:
            return None

        if self.time_budget is None:
            return float(self.per_run_time_limit)

        elapsed = perf_counter() - started_at
        remaining = max(1.0, float(self.time_budget) - elapsed)
        current = min(float(self.per_run_time_limit), remaining)

        if remaining >= 2.0 and remaining / max(current, 1.0) < 2.0:
            current = max(1.0, remaining / 2.0)

        return current

    @staticmethod
    def _safe_model_name(config: Any) -> str:
        try:
            config_dict = dict(config)
            return str(config_dict.get("model_name", "<unknown>"))
        except Exception:
            return "<unknown>"
