# from __future__ import annotations

# from dataclasses import dataclass
# from typing import Any

# import math
# import random


# @dataclass(frozen=True)
# class Parameter:
#     name: str
#     kind: str


# @dataclass(frozen=True)
# class CategoricalParameter(Parameter):
#     choices: tuple[Any, ...]

#     def __init__(self, name: str, choices: list[Any] | tuple[Any, ...]) -> None:
#         object.__setattr__(self, "name", name)
#         object.__setattr__(self, "kind", "categorical")
#         object.__setattr__(self, "choices", tuple(choices))


# @dataclass(frozen=True)
# class IntegerParameter(Parameter):
#     low: int
#     high: int
#     log: bool = False

#     def __init__(self, name: str, low: int, high: int, log: bool = False) -> None:
#         object.__setattr__(self, "name", name)
#         object.__setattr__(self, "kind", "integer")
#         object.__setattr__(self, "low", low)
#         object.__setattr__(self, "high", high)
#         object.__setattr__(self, "log", log)


# @dataclass(frozen=True)
# class FloatParameter(Parameter):
#     low: float
#     high: float
#     log: bool = False

#     def __init__(self, name: str, low: float, high: float, log: bool = False) -> None:
#         object.__setattr__(self, "name", name)
#         object.__setattr__(self, "kind", "float")
#         object.__setattr__(self, "low", float(low))
#         object.__setattr__(self, "high", float(high))
#         object.__setattr__(self, "log", log)


# def logistic_regression_space() -> list[Parameter]:
#     return [
#         FloatParameter("C", 1e-4, 1e2, log=True),
#         CategoricalParameter("solver", ("lbfgs", "liblinear")),
#         IntegerParameter("max_iter", 100, 1000),
#     ]


# def random_forest_space() -> list[Parameter]:
#     return [
#         IntegerParameter("n_estimators", 100, 600),
#         IntegerParameter("max_depth", 2, 32),
#         IntegerParameter("min_samples_split", 2, 20),
#         IntegerParameter("min_samples_leaf", 1, 20),
#         CategoricalParameter("criterion", ("gini", "entropy")),
#     ]


# def svc_space() -> list[Parameter]:
#     return [
#         FloatParameter("C", 1e-4, 1e2, log=True),
#         FloatParameter("gamma", 1e-5, 1.0, log=True),
#     ]


# def knn_space() -> list[Parameter]:
#     return [
#         IntegerParameter("n_neighbors", 1, 50),
#         CategoricalParameter("weights", ("uniform", "distance")),
#         IntegerParameter("p", 1, 2),
#     ]


# def get_classification_search_space(
#     allowed_models: list[str] | None = None,
# ) -> dict[str, Any]:
#     model_spaces = {
#         "logistic_regression": logistic_regression_space(),
#         "random_forest": random_forest_space(),
#         "svc": svc_space(),
#         "knn": knn_space(),
#     }

#     if allowed_models is not None:
#         allowed = set(allowed_models)
#         unknown = sorted(allowed - set(model_spaces))
#         if unknown:
#             raise ValueError(f"Unknown classification models requested: {unknown}")
#         model_spaces = {name: specs for name, specs in model_spaces.items() if name in allowed}

#     if not model_spaces:
#         raise ValueError("Classification search space must contain at least one model.")

#     return {
#         "task": "classification",
#         "model_name": CategoricalParameter("model_name", tuple(model_spaces.keys())),
#         "models": model_spaces,
#     }


# def get_model_parameter_space(model_name: str, search_space: dict[str, Any]) -> list[Parameter]:
#     models = search_space["models"]
#     if model_name not in models:
#         raise ValueError(f"Unknown model_name `{model_name}`.")
#     return models[model_name]


# def _sample_parameter(spec: Parameter, rng: random.Random) -> Any:
#     if isinstance(spec, CategoricalParameter):
#         return rng.choice(spec.choices)

#     if isinstance(spec, IntegerParameter):
#         if spec.log:
#             sampled = math.exp(rng.uniform(math.log(spec.low), math.log(spec.high)))
#             return int(round(min(max(sampled, spec.low), spec.high)))
#         return rng.randint(spec.low, spec.high)

#     if isinstance(spec, FloatParameter):
#         if spec.log:
#             return float(math.exp(rng.uniform(math.log(spec.low), math.log(spec.high))))
#         return float(rng.uniform(spec.low, spec.high))

#     raise TypeError(f"Unsupported parameter spec: {type(spec)!r}")


# def sample_classification_config(
#     search_space: dict[str, Any],
#     rng: random.Random | None = None,
# ) -> dict[str, Any]:
#     rng = rng or random.Random()
#     model_name_spec: CategoricalParameter = search_space["model_name"]
#     model_name = rng.choice(model_name_spec.choices)
#     params = {
#         spec.name: _sample_parameter(spec, rng)
#         for spec in get_model_parameter_space(model_name, search_space)
#     }
#     return {"model_name": model_name, "params": params}


# def encode_classification_config(
#     config: dict[str, Any],
#     search_space: dict[str, Any],
# ) -> list[float]:

#     model_name_spec: CategoricalParameter = search_space["model_name"]
#     model_name = config["model_name"]
#     params = config["params"]
#     encoded: list[float] = []

#     encoded.append(float(model_name_spec.choices.index(model_name)))

#     for model in model_name_spec.choices:
#         specs = get_model_parameter_space(model, search_space)
#         if model != model_name:
#             encoded.extend(_default_encoded_value(spec) for spec in specs)
#             continue

#         for spec in specs:
#             encoded.append(_encode_parameter_value(spec, params[spec.name]))

#     return encoded


# def _encode_parameter_value(spec: Parameter, value: Any) -> float:
#     if isinstance(spec, CategoricalParameter):
#         return float(spec.choices.index(value))

#     if isinstance(spec, IntegerParameter):
#         if spec.log:
#             return float(math.log(value))
#         return float(value)

#     if isinstance(spec, FloatParameter):
#         numeric_value = float(value)
#         if spec.log:
#             return float(math.log(numeric_value))
#         return numeric_value

#     raise TypeError(f"Unsupported parameter spec: {type(spec)!r}")


# def _default_encoded_value(spec: Parameter) -> float:
#     if isinstance(spec, CategoricalParameter):
#         return -1.0
#     if isinstance(spec, IntegerParameter):
#         return -1.0
#     if isinstance(spec, FloatParameter):
#         return -1.0
#     raise TypeError(f"Unsupported parameter spec: {type(spec)!r}")
