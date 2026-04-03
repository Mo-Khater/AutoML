from __future__ import annotations

from ConfigSpace.configuration_space import Configuration, ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter

from .components import get_classification_components


def get_classification_configspace(
    allowed_models: list[str] | None = None,
) -> ConfigurationSpace:
    cs = ConfigurationSpace()
    components = get_classification_components(allowed_models=allowed_models)
    selected_models = sorted(components)

    model_name = CategoricalHyperparameter("model_name", choices=selected_models)
    cs.add(model_name)

    for component_name in selected_models:
        component = components[component_name]
        hyperparameters = component.build_hyperparameters()
        cs.add(hyperparameters)
        params = {param.name: param for param in hyperparameters}
        conditions = component.build_conditions(
            {"model_name": model_name, "params": params}
        )
        if conditions:
            cs.add(conditions)

    return cs


def sample_classification_configuration(
    configspace: ConfigurationSpace,
) -> Configuration:
    return configspace.sample_configuration()


def configuration_to_dict(config: Configuration) -> dict:
    raw = dict(config)
    model_name = raw["model_name"]
    prefix = f"{model_name}:"

    params = {
        key[len(prefix) :]: value
        for key, value in raw.items()
        if key.startswith(prefix)
    }
    return {
        "model_name": model_name,
        "params": params,
    }


def dict_to_configuration(
    config_dict: dict,
    configspace: ConfigurationSpace,
) -> Configuration:
    model_name = config_dict["model_name"]
    params = config_dict.get("params", {})

    flat_config = {"model_name": model_name}
    for key, value in params.items():
        flat_config[f"{model_name}:{key}"] = value

    return Configuration(configspace, values=flat_config)
