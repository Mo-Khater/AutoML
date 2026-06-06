from __future__ import annotations

from dataclasses import dataclass
from multiprocessing import Process, Queue
from statistics import NormalDist
from threading import Thread
from time import perf_counter, sleep
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from ..evaluation import EvaluationResult

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
        n_initial_points: int = 5,
        candidate_pool_size: int = 256,
        verbose: int = 1,
        n_parallel: int = 3,
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
        self.n_parallel = max(1, n_parallel)

        self._rng = np.random.default_rng(random_state)
        self._observations: list[dict[str, Any]] = []
        self._incumbent = None
        self._incumbent_cost = float("inf")
        self._incumbent_result = None
        self._seen_signatures: set[tuple[float, ...]] = set()

        if hasattr(self.configspace, "seed"):
            self.configspace.seed(random_state)

    def optimize(self) -> SMACOptimizationResult:
        started_at = perf_counter()

        trial_id = 0
        while trial_id < self.n_trials:
            if self._time_budget_exceeded(started_at):
                break

            # Determine how many trials to run in this batch
            remaining_trials = self.n_trials - trial_id
            batch_size = min(self.n_parallel, remaining_trials)

            # Get batch of configurations
            batch_configs = self._suggest_batch(batch_size)

            if self.verbose:
                print(f"[AutoML] Starting batch of {batch_size} trials ({trial_id + 1}/{self.n_trials})")

            # Evaluate in parallel with manual process management
            if batch_size > 1:
                batch_results = self._evaluate_batch_with_processes(batch_configs)
            else:
                batch_results = [self._evaluate_single_config(batch_configs[0])]

            # Process results
            for batch_idx, (config, result) in enumerate(zip(batch_configs, batch_results)):
                current_trial_id = trial_id + batch_idx

                if self.verbose:
                    model_name = self._safe_model_name(config)
                    print(
                        f"[AutoML] Trial {current_trial_id + 1}/{self.n_trials} - {model_name}: "
                        f"score={result.score:.6f} cost={result.cost:.6f} status={result.status}"
                    )
                    if result.error and self.verbose > 1:
                        print(f"[AutoML] Trial {current_trial_id + 1}/{self.n_trials} - error: {result.error}")

                row = {
                    "trial_id": current_trial_id,
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

                if result.cost < self._incumbent_cost:
                    self._incumbent = config
                    self._incumbent_cost = float(result.cost)
                    self._incumbent_result = result
                    if self.verbose:
                        print(
                            f"[AutoML] New incumbent - model={result.model_name} "
                            f"preprocessing={result.preprocessing} score={result.score:.6f}"
                        )

            trial_id += batch_size

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
        initial_config = self._pop_next_initial_configuration()
        if initial_config is not None:
            self._mark_seen(initial_config)
            return initial_config

        if len(self._observations) < self.n_initial_points:
            config = self._sample_random_unseen_configuration()
            self._mark_seen(config)
            return config

        X_obs = np.vstack([row["vector"] for row in self._vectorized_observations()])
        y_obs = np.asarray([row["cost"] for row in self._vectorized_observations()], dtype=float)

        if len(np.unique(y_obs)) <= 1:
            config = self._sample_random_unseen_configuration()
            self._mark_seen(config)
            return config

        surrogate = RandomForestRegressor(
            n_estimators=100,
            bootstrap=True,
            max_features=5/6,
            min_samples_split=3,
            min_samples_leaf=3,
            max_depth=20,
            n_jobs=1,
            random_state=self.random_state,
        )
        surrogate.fit(X_obs, y_obs)

        candidate_configs = self._sample_candidate_pool()
        candidate_vectors = np.vstack([self._config_to_vector(config) for config in candidate_configs])
        mean, std = self._predict_with_uncertainty(surrogate, candidate_vectors)
        acquisition = self._expected_improvement(mean, std, best=self._incumbent_cost)
        ranked_indices = np.argsort(acquisition)[::-1]

        for idx in ranked_indices[:10]:
            candidate = candidate_configs[int(idx)]
            if self._is_unseen(candidate):
                self._mark_seen(candidate)
                return candidate

        config = self._sample_random_unseen_configuration()
        self._mark_seen(config)
        return config

    def _suggest_batch(self, batch_size: int) -> list[Any]:
        """Suggest multiple unseen configurations based on acquisition function."""
        batch_configs = []

        # First, use any remaining initial configurations
        for _ in range(batch_size):
            initial_config = self._pop_next_initial_configuration()
            if initial_config is not None:
                self._mark_seen(initial_config)
                batch_configs.append(initial_config)
            else:
                break

        if len(batch_configs) >= batch_size:
            return batch_configs[:batch_size]

        # Then, use random configurations during initial exploration
        if len(self._observations) < self.n_initial_points:
            for _ in range(batch_size - len(batch_configs)):
                config = self._sample_random_unseen_configuration()
                self._mark_seen(config)
                batch_configs.append(config)
            return batch_configs

        # Finally, use acquisition function for guided exploration
        X_obs = np.vstack([row["vector"] for row in self._vectorized_observations()])
        y_obs = np.asarray([row["cost"] for row in self._vectorized_observations()], dtype=float)

        if len(np.unique(y_obs)) <= 1:
            # Not enough variance, sample randomly
            for _ in range(batch_size - len(batch_configs)):
                config = self._sample_random_unseen_configuration()
                self._mark_seen(config)
                batch_configs.append(config)
            return batch_configs

        # Fit surrogate model
        surrogate = RandomForestRegressor(
            n_estimators=100,
            bootstrap=True,
            max_features=5/6,
            min_samples_split=3,
            min_samples_leaf=3,
            max_depth=20,
            n_jobs=1,
            random_state=self.random_state,
        )
        surrogate.fit(X_obs, y_obs)

        # Sample candidate pool and rank by acquisition function
        candidate_configs = self._sample_candidate_pool()
        candidate_vectors = np.vstack([self._config_to_vector(config) for config in candidate_configs])
        mean, std = self._predict_with_uncertainty(surrogate, candidate_vectors)
        acquisition = self._expected_improvement(mean, std, best=self._incumbent_cost)
        ranked_indices = np.argsort(acquisition)[::-1]

        # Select top N unseen candidates
        for idx in ranked_indices:
            if len(batch_configs) >= batch_size:
                break
            candidate = candidate_configs[int(idx)]
            if self._is_unseen(candidate):
                self._mark_seen(candidate)
                batch_configs.append(candidate)

        # If we still need more configurations, sample randomly
        while len(batch_configs) < batch_size:
            config = self._sample_random_unseen_configuration()
            self._mark_seen(config)
            batch_configs.append(config)

        return batch_configs

    def _evaluate_batch_with_processes(self, configs: list[Any]) -> list[Any]:
        """Evaluate a batch of configurations using manual process management with per-process timeout."""
        processes = []
        queues = []
        results = [None] * len(configs)
        process_start_times = {}  # Track start time for each process

        # Start all processes
        batch_start_time = perf_counter()
        for i, config in enumerate(configs):
            queue = Queue()
            queues.append(queue)

            process = Process(
                target=self._evaluate_config_worker,
                args=(queue, config, i),
                daemon=True
            )
            processes.append(process)
            process_start_times[i] = perf_counter()  # Record individual start time
            process.start()

        # Wait for all processes with per-process timeout management
        completed = 0
        remaining_indices = list(range(len(configs)))

        while completed < len(configs) and remaining_indices:
            current_time = perf_counter()

            # Check for completed processes
            still_running = []
            for idx in remaining_indices:
                if not queues[idx].empty():
                    try:
                        result = queues[idx].get_nowait()
                        results[idx] = result
                        completed += 1
                        processes[idx].join()  # Clean up
                    except:
                        pass
                else:
                    still_running.append(idx)

            remaining_indices = still_running

            # Kill timed-out processes (per-process timeout)
            if self.per_run_time_limit is not None:
                for idx in remaining_indices:
                    elapsed = current_time - process_start_times[idx]  # Individual process elapsed time
                    if elapsed >= self.per_run_time_limit:
                        if processes[idx].is_alive():
                            if self.verbose > 1:
                                config_dict = dict(configs[idx])
                                print(f"[AutoML] Killing process for {config_dict.get('model_name', 'unknown')} (timeout)")
                            processes[idx].terminate()
                            processes[idx].join(timeout=1.0)  # Give it 1 second to terminate gracefully
                            if processes[idx].is_alive():
                                processes[idx].kill()  # Force kill if needed

                            # Create timeout result
                            config_dict = dict(configs[idx])
                            results[idx] = EvaluationResult(
                                model_name=config_dict.get("model_name", "unknown"),
                                params=config_dict,
                                preprocessing="none",
                                score=0.0,
                                cost=1.0,
                                duration=elapsed,
                                status="timeout",
                                error=f"per_run_time_limit exceeded ({self.per_run_time_limit:.2f}s)"
                            )
                            completed += 1

            # Small sleep to avoid busy waiting
            if remaining_indices:
                sleep(0.01)

        # Clean up any remaining processes
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=0.1)
                if process.is_alive():
                    process.kill()

        return results

    def _evaluate_single_config(self, config: Any) -> Any:
        """Evaluate a single configuration with optional timeout."""
        if self.per_run_time_limit is None:
            return self.evaluator._evaluate_inline(config, self.X, self.y)
        return self._evaluate_batch_with_processes([config])[0]

    def _evaluate_config_worker(self, queue: Queue, config: Any, index: int) -> None:
        """Worker function for process-based evaluation."""
        try:
            result = self._evaluate_single_inline(config)
            queue.put(result)
        except Exception as e:
            config_dict = dict(config)
            queue.put(EvaluationResult(
                model_name=config_dict.get("model_name", "unknown"),
                params=config_dict,
                preprocessing="none",
                score=0.0,
                cost=1.0,
                duration=0.0,
                status="error",
                error=str(e)
            ))

    def _evaluate_single_inline(self, config: Any) -> Any:
        """Evaluate a single configuration inline (no timeout, used by worker processes)."""
        return self.evaluator._evaluate_inline(config, self.X, self.y)

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

    def _sample_random_unseen_configuration(self):
        max_attempts = max(self.candidate_pool_size * 8, 128)
        for _ in range(max_attempts):
            config = self._sample_random_configuration()
            if self._is_unseen(config):
                return config
        return self._sample_random_configuration()

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

    def _config_signature(self, config: Any) -> tuple[float, ...]:
        return tuple(self._config_to_vector(config).tolist())

    def _is_unseen(self, config: Any) -> bool:
        return self._config_signature(config) not in self._seen_signatures

    def _mark_seen(self, config: Any) -> None:
        self._seen_signatures.add(self._config_signature(config))

    def _pop_next_initial_configuration(self):
        while self.initial_configurations:
            config = self.initial_configurations.pop(0)
            if self._is_unseen(config):
                return config
        return None

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
