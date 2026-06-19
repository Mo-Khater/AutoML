from __future__ import annotations

from dataclasses import dataclass, field
from math import ceil
from multiprocessing import Process, Queue
from statistics import NormalDist
from time import perf_counter, sleep
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from ..evaluation import EvaluationResult, FidelitySpec

@dataclass
class SMACOptimizationResult:
    incumbent: Any
    incumbent_cost: float
    incumbent_result: Any
    runhistory: list[dict[str, Any]]
    optimizer: Any


@dataclass
class _ScheduledTrial:
    config: Any
    fidelity: FidelitySpec
    source: str
    parent_trial_id: int | None = None
    priority: float = float("inf")


@dataclass
class _CandidateState:
    config: Any
    signature: tuple[float, ...]
    completed_stage: int = -1
    highest_scheduled_stage: int = -1
    successful_stages: dict[int, float] = field(default_factory=dict)
    promoted_to_stage: set[int] = field(default_factory=set)
    latest_trial_id: int | None = None
    latest_result: EvaluationResult | None = None


@dataclass
class _StageStats:
    scheduled: int = 0
    completed: int = 0
    successes: int = 0
    failures: int = 0


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
        fidelity_stages: list[FidelitySpec] | None = None,
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
        self.fidelity_stages = list(
            fidelity_stages
            or [FidelitySpec(stage=0, sample_fraction=1.0, cv_folds=None, model_budget=1.0)]
        )

        self._rng = np.random.default_rng(random_state)
        self._observations: list[dict[str, Any]] = []
        self._incumbent = None
        self._incumbent_cost = float("inf")
        self._incumbent_result = None
        self._seen_signatures: set[tuple[float, ...]] = set()
        self._promotion_queue: list[_ScheduledTrial] = []
        self._candidate_states: dict[tuple[float, ...], _CandidateState] = {}
        self._stage_stats: dict[int, _StageStats] = {
            index: _StageStats() for index in range(len(self.fidelity_stages))
        }
        self._promotion_quantile = 0.35
        self._min_stage_observations_for_promotion = max(3, self.n_parallel + 1)

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
            scheduled_trials = self._suggest_batch(batch_size)

            if self.verbose:
                print(f"[AutoML] Starting batch of {batch_size} trials ({trial_id + 1}/{self.n_trials})")

            # Evaluate in parallel with manual process management
            if batch_size > 1:
                batch_results = self._evaluate_batch_with_processes(scheduled_trials)
            else:
                batch_results = [self._evaluate_single_config(scheduled_trials[0])]

            # Process results
            for batch_idx, (scheduled, result) in enumerate(zip(scheduled_trials, batch_results)):
                current_trial_id = trial_id + batch_idx
                config = scheduled.config

                if self.verbose:
                    model_name = self._safe_model_name(config)
                    print(
                        f"[AutoML] Trial {current_trial_id + 1}/{self.n_trials} - {model_name}: "
                        f"score={result.score:.6f} cost={result.cost:.6f} status={result.status} "
                        f"stage={result.fidelity_stage}"
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
                    "fidelity_stage": result.fidelity_stage,
                    "sample_fraction": result.sample_fraction,
                    "cv_folds": result.cv_folds,
                    "model_budget": result.model_budget,
                    "source": scheduled.source,
                    "parent_trial_id": scheduled.parent_trial_id,
                }
                self._observations.append(row)
                self._record_observation(config, result, current_trial_id)

                if self._is_full_fidelity(result) and result.cost < self._incumbent_cost:
                    self._incumbent = config
                    self._incumbent_cost = float(result.cost)
                    self._incumbent_result = result
                    if self.verbose:
                        print(
                            f"[AutoML] New incumbent - model={result.model_name} "
                            f"preprocessing={result.preprocessing} score={result.score:.6f}"
                        )

            trial_id += batch_size

        if self._incumbent is None and self._observations:
            finite_rows = [row for row in self._observations if np.isfinite(row["cost"])]
            if finite_rows:
                best_row = min(finite_rows, key=lambda item: item["cost"])
                self._incumbent = best_row["config"]
                self._incumbent_cost = float(best_row["cost"])
                self._incumbent_result = EvaluationResult(
                    model_name=best_row["model_name"],
                    params=best_row["params"],
                    preprocessing=best_row["preprocessing"],
                    score=best_row["score"],
                    cost=best_row["cost"],
                    duration=best_row["duration"],
                    status=best_row["status"],
                    error=best_row["error"],
                    fidelity_stage=best_row.get("fidelity_stage", 0),
                    sample_fraction=best_row.get("sample_fraction", 1.0),
                    cv_folds=best_row.get("cv_folds", self.fidelity_stages[-1].cv_folds or 1),
                    model_budget=best_row.get("model_budget", 1.0),
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
        initial_config = self._pop_next_initial_configuration()
        if initial_config is not None:
            self._mark_seen(initial_config)
            return initial_config

        if len(self._observations) < self.n_initial_points:
            config = self._sample_random_unseen_configuration()
            self._mark_seen(config)
            return config

        finite_observations = self._finite_vectorized_observations()
        if len(finite_observations) < self.n_initial_points:
            config = self._sample_random_unseen_configuration()
            self._mark_seen(config)
            return config

        X_obs = np.vstack([row["augmented_vector"] for row in finite_observations])
        y_obs = np.asarray([row["cost"] for row in finite_observations], dtype=float)

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
        candidate_vectors = np.vstack(
            [self._augment_vector(self._config_to_vector(config), self.fidelity_stages[0]) for config in candidate_configs]
        )
        mean, std = self._predict_with_uncertainty(surrogate, candidate_vectors)
        acquisition = self._expected_improvement(mean, std, best=self._best_observed_cost())
        ranked_indices = np.argsort(acquisition)[::-1]

        for idx in ranked_indices[:10]:
            candidate = candidate_configs[int(idx)]
            if self._is_unseen(candidate):
                self._mark_seen(candidate)
                return candidate

        config = self._sample_random_unseen_configuration()
        self._mark_seen(config)
        return config

    def _suggest_batch(self, batch_size: int) -> list[_ScheduledTrial]:
        """Suggest multiple trials with either promotion or cheap exploration."""
        scheduled_trials: list[_ScheduledTrial] = []

        promotion_slots = self._promotion_slots(batch_size)
        scheduled_trials.extend(self._take_promotions(promotion_slots))

        # First, use any remaining initial configurations
        for _ in range(batch_size - len(scheduled_trials)):
            initial_config = self._pop_next_initial_configuration()
            if initial_config is not None:
                scheduled_trials.append(
                    _ScheduledTrial(
                        config=initial_config,
                        fidelity=self.fidelity_stages[0],
                        source="warmstart",
                        priority=self._best_observed_cost(),
                    )
                )
                self._register_scheduled_trial(scheduled_trials[-1])
            else:
                break

        if len(scheduled_trials) >= batch_size:
            return scheduled_trials[:batch_size]

        # Then, use random configurations during initial exploration
        if len(self._observations) < self.n_initial_points:
            for _ in range(batch_size - len(scheduled_trials)):
                config = self._sample_random_unseen_configuration()
                scheduled_trials.append(
                    _ScheduledTrial(
                        config=config,
                        fidelity=self.fidelity_stages[0],
                        source="random",
                        priority=self._best_observed_cost(),
                    )
                )
                self._register_scheduled_trial(scheduled_trials[-1])
            return scheduled_trials

        # Finally, use acquisition function for guided exploration
        finite_observations = self._finite_vectorized_observations()
        if len(finite_observations) < self.n_initial_points:
            for _ in range(batch_size - len(scheduled_trials)):
                config = self._sample_random_unseen_configuration()
                scheduled_trials.append(
                    _ScheduledTrial(
                        config=config,
                        fidelity=self.fidelity_stages[0],
                        source="random",
                        priority=self._best_observed_cost(),
                    )
                )
                self._register_scheduled_trial(scheduled_trials[-1])
            return scheduled_trials

        X_obs = np.vstack([row["augmented_vector"] for row in finite_observations])
        y_obs = np.asarray([row["cost"] for row in finite_observations], dtype=float)

        if len(np.unique(y_obs)) <= 1:
            # Not enough variance, sample randomly
            for _ in range(batch_size - len(scheduled_trials)):
                config = self._sample_random_unseen_configuration()
                scheduled_trials.append(
                    _ScheduledTrial(
                        config=config,
                        fidelity=self.fidelity_stages[0],
                        source="random",
                        priority=self._best_observed_cost(),
                    )
                )
                self._register_scheduled_trial(scheduled_trials[-1])
            return scheduled_trials

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
        candidate_vectors = np.vstack(
            [self._augment_vector(self._config_to_vector(config), self.fidelity_stages[0]) for config in candidate_configs]
        )
        mean, std = self._predict_with_uncertainty(surrogate, candidate_vectors)
        acquisition = self._expected_improvement(mean, std, best=self._best_observed_cost())
        ranked_indices = np.argsort(acquisition)[::-1]

        # Select top N unseen candidates
        for idx in ranked_indices:
            if len(scheduled_trials) >= batch_size:
                break
            candidate = candidate_configs[int(idx)]
            if self._is_unseen(candidate):
                scheduled_trials.append(
                    _ScheduledTrial(
                        config=candidate,
                        fidelity=self.fidelity_stages[0],
                        source="ei",
                        priority=float(mean[int(idx)]),
                    )
                )
                self._register_scheduled_trial(scheduled_trials[-1])

        # If we still need more configurations, sample randomly
        while len(scheduled_trials) < batch_size:
            config = self._sample_random_unseen_configuration()
            scheduled_trials.append(
                _ScheduledTrial(
                    config=config,
                    fidelity=self.fidelity_stages[0],
                    source="random",
                    priority=self._best_observed_cost(),
                )
            )
            self._register_scheduled_trial(scheduled_trials[-1])

        return scheduled_trials

    def _evaluate_batch_with_processes(self, scheduled_trials: list[_ScheduledTrial]) -> list[Any]:
        """Evaluate a batch of configurations using manual process management with per-process timeout."""
        processes = []
        queues = []
        results = [None] * len(scheduled_trials)
        process_start_times = {}  # Track start time for each process

        # Start all processes
        for i, scheduled in enumerate(scheduled_trials):
            queue = Queue()
            queues.append(queue)

            process = Process(
                target=self._evaluate_config_worker,
                args=(queue, scheduled.config, scheduled.fidelity, i),
                daemon=True
            )
            processes.append(process)
            process_start_times[i] = perf_counter()  # Record individual start time
            process.start()

        # Wait for all processes with per-process timeout management
        completed = 0
        remaining_indices = list(range(len(scheduled_trials)))

        while completed < len(scheduled_trials) and remaining_indices:
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
                                config_dict = dict(scheduled_trials[idx].config)
                                print(f"[AutoML] Killing process for {config_dict.get('model_name', 'unknown')} (timeout)")
                            processes[idx].terminate()
                            processes[idx].join(timeout=1.0)  # Give it 1 second to terminate gracefully
                            if processes[idx].is_alive():
                                processes[idx].kill()  # Force kill if needed

                            # Create timeout result
                            config_dict = dict(scheduled_trials[idx].config)
                            results[idx] = EvaluationResult(
                                model_name=config_dict.get("model_name", "unknown"),
                                params=config_dict,
                                preprocessing="none",
                                score=0.0,
                                cost=1.0,
                                duration=elapsed,
                                status="timeout",
                                error=f"per_run_time_limit exceeded ({self.per_run_time_limit:.2f}s)",
                                fidelity_stage=scheduled_trials[idx].fidelity.stage,
                                sample_fraction=scheduled_trials[idx].fidelity.sample_fraction,
                                cv_folds=scheduled_trials[idx].fidelity.cv_folds or 0,
                                model_budget=scheduled_trials[idx].fidelity.model_budget,
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

    def _evaluate_single_config(self, scheduled_trial: _ScheduledTrial) -> Any:
        """Evaluate a single configuration with optional timeout."""
        if self.per_run_time_limit is None:
            return self.evaluator._evaluate_inline(
                scheduled_trial.config,
                self.X,
                self.y,
                fidelity=scheduled_trial.fidelity,
            )
        return self._evaluate_batch_with_processes([scheduled_trial])[0]

    def _evaluate_config_worker(self, queue: Queue, config: Any, fidelity: FidelitySpec, index: int) -> None:
        """Worker function for process-based evaluation."""
        try:
            result = self._evaluate_single_inline(config, fidelity)
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
                error=str(e),
                fidelity_stage=fidelity.stage,
                sample_fraction=fidelity.sample_fraction,
                cv_folds=fidelity.cv_folds or 0,
                model_budget=fidelity.model_budget,
            ))

    def _evaluate_single_inline(self, config: Any, fidelity: FidelitySpec) -> Any:
        """Evaluate a single configuration inline (no timeout, used by worker processes)."""
        return self.evaluator._evaluate_inline(config, self.X, self.y, fidelity=fidelity)

    def _vectorized_observations(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for row in self._observations:
            vector = row.get("vector")
            if vector is None:
                vector = self._config_to_vector(row["config"])
                row["vector"] = vector
            augmented_vector = row.get("augmented_vector")
            if augmented_vector is None:
                fidelity = FidelitySpec(
                    stage=int(row.get("fidelity_stage", 0)),
                    sample_fraction=float(row.get("sample_fraction", 1.0)),
                    cv_folds=int(row.get("cv_folds", self.fidelity_stages[-1].cv_folds or 1)),
                    model_budget=float(row.get("model_budget", 1.0)),
                )
                augmented_vector = self._augment_vector(vector, fidelity)
                row["augmented_vector"] = augmented_vector
            rows.append(row)
        return rows

    def _finite_vectorized_observations(self) -> list[dict[str, Any]]:
        rows = []
        for row in self._vectorized_observations():
            if not np.isfinite(row.get("cost", np.nan)):
                continue
            augmented_vector = np.asarray(row["augmented_vector"], dtype=float)
            if not np.all(np.isfinite(augmented_vector)):
                continue
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

    @staticmethod
    def _augment_vector(config_vector: np.ndarray, fidelity: FidelitySpec) -> np.ndarray:
        cv_component = 0.0 if fidelity.cv_folds is None else float(fidelity.cv_folds)
        fidelity_vector = np.asarray(
            [
                float(fidelity.stage),
                float(fidelity.sample_fraction),
                cv_component,
                float(fidelity.model_budget),
            ],
            dtype=float,
        )
        return np.concatenate([config_vector, fidelity_vector], dtype=float)

    def _is_unseen(self, config: Any) -> bool:
        return self._config_signature(config) not in self._seen_signatures

    def _mark_seen(self, config: Any) -> None:
        self._seen_signatures.add(self._config_signature(config))

    def _record_observation(
        self,
        config: Any,
        result: EvaluationResult,
        parent_trial_id: int,
    ) -> None:
        signature = self._config_signature(config)
        state = self._candidate_states.get(signature)
        if state is None:
            state = _CandidateState(config=config, signature=signature)
            self._candidate_states[signature] = state

        current_stage = int(result.fidelity_stage)
        stage_stats = self._stage_stats[current_stage]
        stage_stats.completed += 1
        state.latest_trial_id = parent_trial_id
        state.latest_result = result

        if result.status == "success":
            stage_stats.successes += 1
            state.completed_stage = max(state.completed_stage, current_stage)
            state.successful_stages[current_stage] = float(result.cost)
        else:
            stage_stats.failures += 1

        next_stage = current_stage + 1
        if next_stage >= len(self.fidelity_stages):
            return
        if result.status != "success":
            return
        if next_stage in state.promoted_to_stage:
            return
        if not self._should_promote(state, current_stage):
            return

        promoted_trial = _ScheduledTrial(
            config=config,
            fidelity=self.fidelity_stages[next_stage],
            source="promotion",
            parent_trial_id=parent_trial_id,
            priority=float(result.cost),
        )
        self._promotion_queue.append(promoted_trial)
        self._promotion_queue.sort(key=lambda trial: (trial.fidelity.stage, trial.priority))
        state.promoted_to_stage.add(next_stage)
        state.highest_scheduled_stage = max(state.highest_scheduled_stage, next_stage)

    def _should_promote(self, candidate_state: _CandidateState, stage: int) -> bool:
        if stage not in candidate_state.successful_stages:
            return False
        eligible_states = [
            state for state in self._candidate_states.values()
            if stage in state.successful_stages
        ]
        if not eligible_states:
            return True
        if len(eligible_states) < self._min_stage_observations_for_promotion:
            return True

        ranked_states = sorted(
            eligible_states,
            key=lambda state: (state.successful_stages[stage], state.latest_trial_id or -1),
        )
        promotion_count = max(1, ceil(len(ranked_states) * self._promotion_quantile))
        promoted_signatures = {state.signature for state in ranked_states[:promotion_count]}
        return candidate_state.signature in promoted_signatures

    def _is_full_fidelity(self, result: EvaluationResult) -> bool:
        return int(result.fidelity_stage) >= (len(self.fidelity_stages) - 1)

    def _register_scheduled_trial(self, scheduled_trial: _ScheduledTrial) -> None:
        signature = self._config_signature(scheduled_trial.config)
        if scheduled_trial.fidelity.stage == 0:
            self._mark_seen(scheduled_trial.config)

        state = self._candidate_states.get(signature)
        if state is None:
            state = _CandidateState(config=scheduled_trial.config, signature=signature)
            self._candidate_states[signature] = state
        state.highest_scheduled_stage = max(state.highest_scheduled_stage, scheduled_trial.fidelity.stage)
        self._stage_stats[scheduled_trial.fidelity.stage].scheduled += 1

    def _take_promotions(self, limit: int) -> list[_ScheduledTrial]:
        if limit <= 0 or not self._promotion_queue:
            return []
        taken = self._promotion_queue[:limit]
        self._promotion_queue = self._promotion_queue[limit:]
        for trial in taken:
            self._register_scheduled_trial(trial)
        return taken

    def _promotion_slots(self, batch_size: int) -> int:
        if not self._promotion_queue:
            return 0
        if len(self._observations) < self.n_initial_points:
            return 0
        queued = len(self._promotion_queue)
        if self._incumbent_result is None:
            return min(queued, max(1, batch_size // 2))
        return min(queued, max(1, (2 * batch_size) // 3))

    def _best_observed_cost(self) -> float:
        if np.isfinite(self._incumbent_cost):
            return float(self._incumbent_cost)
        finite_costs = [float(row["cost"]) for row in self._observations if np.isfinite(row["cost"])]
        if finite_costs:
            return float(min(finite_costs))
        return 1.0

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
