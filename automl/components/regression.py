from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from ConfigSpace.conditions import AndConjunction, EqualsCondition, InCondition
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
)
from sklearn.ensemble import AdaBoostRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.linear_model import ARDRegression, SGDRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import LinearSVR, SVR
from sklearn.tree import DecisionTreeRegressor


@dataclass(frozen=True)
class RegressionComponent:
    name: str
    build_hyperparameters: Callable[[], list[Any]]
    build_conditions: Callable[[dict[str, Any]], list[Any]]
    build_estimator: Callable[[dict[str, Any], int | None, int | None], Any]


def _all_equal_conditions(model_name_hyperparameter: Any, component_name: str, params: dict[str, Any]) -> list[Any]:
    return [
        EqualsCondition(param, model_name_hyperparameter, component_name)
        for param in params.values()
    ]


def _adaboost_hyperparameters() -> list[Any]:
    return [
        UniformIntegerHyperparameter("adaboost:n_estimators", lower=25, upper=500),
        UniformFloatHyperparameter("adaboost:learning_rate", lower=1e-3, upper=2.0, log=True),
        CategoricalHyperparameter("adaboost:loss", choices=["linear", "square", "exponential"]),
        UniformIntegerHyperparameter("adaboost:max_depth", lower=1, upper=8),
    ]


def _adaboost_conditions(ctx: dict[str, Any]) -> list[Any]:
    return _all_equal_conditions(ctx["model_name"], "adaboost", ctx["params"])


def _build_adaboost(params: dict[str, Any], random_state: int | None, n_jobs: int | None):
    base_estimator = DecisionTreeRegressor(
        max_depth=int(params["max_depth"]),
        random_state=random_state,
    )
    try:
        return AdaBoostRegressor(
            estimator=base_estimator,
            n_estimators=int(params["n_estimators"]),
            learning_rate=float(params["learning_rate"]),
            loss=params["loss"],
            random_state=random_state,
        )
    except TypeError:
        return AdaBoostRegressor(
            base_estimator=base_estimator,
            n_estimators=int(params["n_estimators"]),
            learning_rate=float(params["learning_rate"]),
            loss=params["loss"],
            random_state=random_state,
        )


def _ard_regression_hyperparameters() -> list[Any]:
    return [
        UniformFloatHyperparameter("ard_regression:alpha_1", lower=1e-8, upper=1e-2, log=True),
        UniformFloatHyperparameter("ard_regression:alpha_2", lower=1e-8, upper=1e-2, log=True),
        CategoricalHyperparameter("ard_regression:fit_intercept", choices=[False, True]),
        UniformFloatHyperparameter("ard_regression:lambda_1", lower=1e-8, upper=1e-2, log=True),
        UniformFloatHyperparameter("ard_regression:lambda_2", lower=1e-8, upper=1e-2, log=True),
        UniformIntegerHyperparameter("ard_regression:n_iter", lower=100, upper=1000),
        UniformFloatHyperparameter("ard_regression:threshold_lambda", lower=1e3, upper=1e6, log=True),
        UniformFloatHyperparameter("ard_regression:tol", lower=1e-6, upper=1e-2, log=True),
    ]


def _ard_regression_conditions(ctx: dict[str, Any]) -> list[Any]:
    return _all_equal_conditions(ctx["model_name"], "ard_regression", ctx["params"])


def _build_ard_regression(params: dict[str, Any], random_state: int | None, n_jobs: int | None):
    estimator_params = {
        "alpha_1": float(params["alpha_1"]),
        "alpha_2": float(params["alpha_2"]),
        "fit_intercept": bool(params["fit_intercept"]),
        "lambda_1": float(params["lambda_1"]),
        "lambda_2": float(params["lambda_2"]),
        "threshold_lambda": float(params["threshold_lambda"]),
        "tol": float(params["tol"]),
    }
    try:
        estimator_params["max_iter"] = int(params["n_iter"])
        return ARDRegression(**estimator_params)
    except TypeError:
        estimator_params["n_iter"] = int(params["n_iter"])
        return ARDRegression(**estimator_params)


def _decision_tree_hyperparameters() -> list[Any]:
    return [
        CategoricalHyperparameter("decision_tree:criterion", choices=["squared_error", "friedman_mse", "absolute_error"]),
        UniformIntegerHyperparameter("decision_tree:max_depth", lower=1, upper=64),
        UniformFloatHyperparameter("decision_tree:max_features", lower=0.1, upper=1.0),
        UniformIntegerHyperparameter("decision_tree:max_leaf_nodes", lower=2, upper=512),
        UniformFloatHyperparameter("decision_tree:min_impurity_decrease", lower=0.0, upper=0.1),
        UniformIntegerHyperparameter("decision_tree:min_samples_leaf", lower=1, upper=20),
        UniformIntegerHyperparameter("decision_tree:min_samples_split", lower=2, upper=20),
        UniformFloatHyperparameter("decision_tree:min_weight_fraction_leaf", lower=0.0, upper=0.4),
    ]


def _decision_tree_conditions(ctx: dict[str, Any]) -> list[Any]:
    return _all_equal_conditions(ctx["model_name"], "decision_tree", ctx["params"])


def _build_decision_tree(params: dict[str, Any], random_state: int | None, n_jobs: int | None):
    return DecisionTreeRegressor(
        criterion=params["criterion"],
        max_depth=int(params["max_depth"]),
        max_features=float(params["max_features"]),
        max_leaf_nodes=int(params["max_leaf_nodes"]),
        min_impurity_decrease=float(params["min_impurity_decrease"]),
        min_samples_leaf=int(params["min_samples_leaf"]),
        min_samples_split=int(params["min_samples_split"]),
        min_weight_fraction_leaf=float(params["min_weight_fraction_leaf"]),
        random_state=random_state,
    )


def _extra_trees_hyperparameters() -> list[Any]:
    return [
        UniformIntegerHyperparameter("extra_trees:n_estimators", lower=100, upper=1000),
        CategoricalHyperparameter("extra_trees:bootstrap", choices=[True, False]),
        CategoricalHyperparameter("extra_trees:criterion", choices=["squared_error", "friedman_mse", "absolute_error"]),
        UniformIntegerHyperparameter("extra_trees:max_depth", lower=2, upper=64),
        UniformFloatHyperparameter("extra_trees:max_features", lower=0.1, upper=1.0),
        UniformIntegerHyperparameter("extra_trees:max_leaf_nodes", lower=2, upper=512),
        UniformFloatHyperparameter("extra_trees:min_impurity_decrease", lower=0.0, upper=0.1),
        UniformIntegerHyperparameter("extra_trees:min_samples_leaf", lower=1, upper=20),
        UniformIntegerHyperparameter("extra_trees:min_samples_split", lower=2, upper=20),
        UniformFloatHyperparameter("extra_trees:min_weight_fraction_leaf", lower=0.0, upper=0.4),
    ]


def _extra_trees_conditions(ctx: dict[str, Any]) -> list[Any]:
    return _all_equal_conditions(ctx["model_name"], "extra_trees", ctx["params"])


def _build_extra_trees(params: dict[str, Any], random_state: int | None, n_jobs: int | None):
    return ExtraTreesRegressor(
        n_estimators=int(params["n_estimators"]),
        bootstrap=bool(params["bootstrap"]),
        criterion=params["criterion"],
        max_depth=int(params["max_depth"]),
        max_features=float(params["max_features"]),
        max_leaf_nodes=int(params["max_leaf_nodes"]),
        min_impurity_decrease=float(params["min_impurity_decrease"]),
        min_samples_leaf=int(params["min_samples_leaf"]),
        min_samples_split=int(params["min_samples_split"]),
        min_weight_fraction_leaf=float(params["min_weight_fraction_leaf"]),
        random_state=random_state,
        n_jobs=n_jobs,
    )


def _gaussian_process_hyperparameters() -> list[Any]:
    return [
        UniformFloatHyperparameter("gaussian_process:alpha", lower=1e-12, upper=1e-1, log=True),
        UniformFloatHyperparameter("gaussian_process:thetaL", lower=1e-2, upper=1e2, log=True),
        UniformFloatHyperparameter("gaussian_process:thetaU", lower=1e-1, upper=1e3, log=True),
    ]


def _gaussian_process_conditions(ctx: dict[str, Any]) -> list[Any]:
    return _all_equal_conditions(ctx["model_name"], "gaussian_process", ctx["params"])


def _build_gaussian_process(params: dict[str, Any], random_state: int | None, n_jobs: int | None):
    theta_l = float(params["thetaL"])
    theta_u = float(params["thetaU"])
    length_scale = max(theta_l, min(theta_u, (theta_l + theta_u) / 2.0))
    return GaussianProcessRegressor(
        alpha=float(params["alpha"]),
        kernel=1.0 * RBF(length_scale=length_scale),
        random_state=random_state,
        normalize_y=True,
    )


def _gradient_boosting_hyperparameters() -> list[Any]:
    return [
        CategoricalHyperparameter("gradient_boosting:early_stop", choices=["off", "valid", "train"]),
        UniformFloatHyperparameter("gradient_boosting:l2_regularization", lower=1e-10, upper=1.0, log=True),
        UniformFloatHyperparameter("gradient_boosting:learning_rate", lower=1e-3, upper=0.5, log=True),
        CategoricalHyperparameter("gradient_boosting:loss", choices=["squared_error", "absolute_error"]),
        UniformIntegerHyperparameter("gradient_boosting:max_bins", lower=32, upper=255),
        UniformIntegerHyperparameter("gradient_boosting:max_depth", lower=2, upper=32),
        UniformIntegerHyperparameter("gradient_boosting:max_iter", lower=50, upper=500),
        UniformIntegerHyperparameter("gradient_boosting:max_leaf_nodes", lower=3, upper=255),
        UniformIntegerHyperparameter("gradient_boosting:min_samples_leaf", lower=1, upper=200),
        UniformIntegerHyperparameter("gradient_boosting:n_iter_no_change", lower=1, upper=20),
        CategoricalHyperparameter("gradient_boosting:scoring", choices=["loss"]),
        UniformFloatHyperparameter("gradient_boosting:tol", lower=1e-8, upper=1e-2, log=True),
        UniformFloatHyperparameter("gradient_boosting:validation_fraction", lower=0.01, upper=0.4),
    ]


def _gradient_boosting_conditions(ctx: dict[str, Any]) -> list[Any]:
    params = ctx["params"]
    conditions = _all_equal_conditions(ctx["model_name"], "gradient_boosting", params)
    hidden = {"gradient_boosting:n_iter_no_change", "gradient_boosting:validation_fraction"}
    conditions = [cond for cond in conditions if cond.child.name not in hidden]
    conditions.extend(
        [
            AndConjunction(
                EqualsCondition(params["gradient_boosting:n_iter_no_change"], ctx["model_name"], "gradient_boosting"),
                InCondition(params["gradient_boosting:n_iter_no_change"], params["gradient_boosting:early_stop"], ["valid", "train"]),
            ),
            AndConjunction(
                EqualsCondition(params["gradient_boosting:validation_fraction"], ctx["model_name"], "gradient_boosting"),
                EqualsCondition(params["gradient_boosting:validation_fraction"], params["gradient_boosting:early_stop"], "valid"),
            ),
        ]
    )
    return conditions


def _build_gradient_boosting(params: dict[str, Any], random_state: int | None, n_jobs: int | None):
    early_stop = params["early_stop"]
    n_iter_no_change = None
    validation_fraction = None
    if early_stop in {"valid", "train"}:
        n_iter_no_change = int(params["n_iter_no_change"])
    if early_stop == "valid":
        validation_fraction = float(params["validation_fraction"])
    return HistGradientBoostingRegressor(
        loss=params["loss"],
        learning_rate=float(params["learning_rate"]),
        max_iter=int(params["max_iter"]),
        min_samples_leaf=int(params["min_samples_leaf"]),
        max_depth=int(params["max_depth"]),
        max_leaf_nodes=int(params["max_leaf_nodes"]),
        max_bins=int(params["max_bins"]),
        l2_regularization=float(params["l2_regularization"]),
        early_stopping=(early_stop != "off"),
        tol=float(params["tol"]),
        scoring=params["scoring"],
        n_iter_no_change=n_iter_no_change,
        validation_fraction=validation_fraction,
        random_state=random_state,
    )


def _knn_hyperparameters() -> list[Any]:
    return [
        UniformIntegerHyperparameter("k_nearest_neighbors:n_neighbors", lower=1, upper=50),
        CategoricalHyperparameter("k_nearest_neighbors:p", choices=[1, 2]),
        CategoricalHyperparameter("k_nearest_neighbors:weights", choices=["uniform", "distance"]),
    ]


def _knn_conditions(ctx: dict[str, Any]) -> list[Any]:
    return _all_equal_conditions(ctx["model_name"], "k_nearest_neighbors", ctx["params"])


def _build_knn(params: dict[str, Any], random_state: int | None, n_jobs: int | None):
    return KNeighborsRegressor(
        n_neighbors=int(params["n_neighbors"]),
        p=int(params["p"]),
        weights=params["weights"],
        n_jobs=n_jobs,
    )


def _liblinear_svr_hyperparameters() -> list[Any]:
    return [
        UniformFloatHyperparameter("liblinear_svr:C", lower=1e-4, upper=1e4, log=True),
        CategoricalHyperparameter("liblinear_svr:dual", choices=[False, True]),
        UniformFloatHyperparameter("liblinear_svr:epsilon", lower=1e-4, upper=1.0, log=True),
        CategoricalHyperparameter("liblinear_svr:fit_intercept", choices=[False, True]),
        UniformFloatHyperparameter("liblinear_svr:intercept_scaling", lower=0.1, upper=10.0, log=True),
        CategoricalHyperparameter("liblinear_svr:loss", choices=["epsilon_insensitive", "squared_epsilon_insensitive"]),
        UniformFloatHyperparameter("liblinear_svr:tol", lower=1e-5, upper=1e-1, log=True),
    ]


def _liblinear_svr_conditions(ctx: dict[str, Any]) -> list[Any]:
    return _all_equal_conditions(ctx["model_name"], "liblinear_svr", ctx["params"])


def _build_liblinear_svr(params: dict[str, Any], random_state: int | None, n_jobs: int | None):
    return LinearSVR(
        C=float(params["C"]),
        dual=bool(params["dual"]),
        epsilon=float(params["epsilon"]),
        fit_intercept=bool(params["fit_intercept"]),
        intercept_scaling=float(params["intercept_scaling"]),
        loss=params["loss"],
        random_state=random_state,
        tol=float(params["tol"]),
    )


def _libsvm_svr_hyperparameters() -> list[Any]:
    return [
        UniformFloatHyperparameter("libsvm_svr:C", lower=1e-4, upper=1e4, log=True),
        UniformFloatHyperparameter("libsvm_svr:coef0", lower=-1.0, upper=1.0),
        UniformIntegerHyperparameter("libsvm_svr:degree", lower=2, upper=5),
        UniformFloatHyperparameter("libsvm_svr:epsilon", lower=1e-4, upper=1.0, log=True),
        UniformFloatHyperparameter("libsvm_svr:gamma", lower=1e-5, upper=10.0, log=True),
        CategoricalHyperparameter("libsvm_svr:kernel", choices=["linear", "poly", "rbf", "sigmoid"]),
        CategoricalHyperparameter("libsvm_svr:shrinking", choices=[False, True]),
        UniformFloatHyperparameter("libsvm_svr:tol", lower=1e-5, upper=1e-1, log=True),
    ]


def _libsvm_svr_conditions(ctx: dict[str, Any]) -> list[Any]:
    params = ctx["params"]
    conditions = _all_equal_conditions(ctx["model_name"], "libsvm_svr", params)
    conditions = [
        cond for cond in conditions
        if cond.child.name not in {"libsvm_svr:degree", "libsvm_svr:gamma", "libsvm_svr:coef0"}
    ]
    conditions.extend(
        [
            AndConjunction(
                EqualsCondition(params["libsvm_svr:degree"], ctx["model_name"], "libsvm_svr"),
                EqualsCondition(params["libsvm_svr:degree"], params["libsvm_svr:kernel"], "poly"),
            ),
            AndConjunction(
                EqualsCondition(params["libsvm_svr:gamma"], ctx["model_name"], "libsvm_svr"),
                InCondition(params["libsvm_svr:gamma"], params["libsvm_svr:kernel"], ["poly", "rbf"]),
            ),
            AndConjunction(
                EqualsCondition(params["libsvm_svr:coef0"], ctx["model_name"], "libsvm_svr"),
                InCondition(params["libsvm_svr:coef0"], params["libsvm_svr:kernel"], ["poly", "sigmoid"]),
            ),
        ]
    )
    return conditions


def _build_libsvm_svr(params: dict[str, Any], random_state: int | None, n_jobs: int | None):
    return SVR(
        C=float(params["C"]),
        coef0=float(params.get("coef0", 0.0)),
        degree=int(params.get("degree", 3)),
        epsilon=float(params["epsilon"]),
        gamma=float(params.get("gamma", 0.1)),
        kernel=params["kernel"],
        shrinking=bool(params["shrinking"]),
        tol=float(params["tol"]),
    )


def _mlp_hyperparameters() -> list[Any]:
    return [
        CategoricalHyperparameter("mlp:activation", choices=["identity", "logistic", "tanh", "relu"]),
        UniformFloatHyperparameter("mlp:alpha", lower=1e-6, upper=1e-1, log=True),
        CategoricalHyperparameter("mlp:batch_size", choices=["auto", 32, 64, 128, 256]),
        UniformFloatHyperparameter("mlp:beta_1", lower=0.8, upper=0.9999),
        UniformFloatHyperparameter("mlp:beta_2", lower=0.9, upper=0.999999),
        CategoricalHyperparameter("mlp:early_stopping", choices=[False, True]),
        UniformFloatHyperparameter("mlp:epsilon", lower=1e-9, upper=1e-6, log=True),
        UniformIntegerHyperparameter("mlp:hidden_layer_sizes", lower=32, upper=512),
        UniformFloatHyperparameter("mlp:learning_rate_init", lower=1e-4, upper=1e-1, log=True),
        UniformIntegerHyperparameter("mlp:max_iter", lower=100, upper=800),
        UniformIntegerHyperparameter("mlp:n_iter_no_change", lower=5, upper=30),
        CategoricalHyperparameter("mlp:shuffle", choices=[False, True]),
        CategoricalHyperparameter("mlp:solver", choices=["adam", "sgd"]),
        UniformFloatHyperparameter("mlp:tol", lower=1e-6, upper=1e-2, log=True),
        UniformFloatHyperparameter("mlp:validation_fraction", lower=0.05, upper=0.4),
    ]


def _mlp_conditions(ctx: dict[str, Any]) -> list[Any]:
    return _all_equal_conditions(ctx["model_name"], "mlp", ctx["params"])


def _build_mlp(params: dict[str, Any], random_state: int | None, n_jobs: int | None):
    return MLPRegressor(
        activation=params["activation"],
        alpha=float(params["alpha"]),
        batch_size=params["batch_size"],
        beta_1=float(params["beta_1"]),
        beta_2=float(params["beta_2"]),
        early_stopping=bool(params["early_stopping"]),
        epsilon=float(params["epsilon"]),
        hidden_layer_sizes=(int(params["hidden_layer_sizes"]),),
        learning_rate_init=float(params["learning_rate_init"]),
        max_iter=int(params["max_iter"]),
        n_iter_no_change=int(params["n_iter_no_change"]),
        shuffle=bool(params["shuffle"]),
        solver=params["solver"],
        tol=float(params["tol"]),
        validation_fraction=float(params["validation_fraction"]),
        random_state=random_state,
    )


def _random_forest_hyperparameters() -> list[Any]:
    return [
        UniformIntegerHyperparameter("random_forest:n_estimators", lower=100, upper=1000),
        CategoricalHyperparameter("random_forest:bootstrap", choices=[True, False]),
        CategoricalHyperparameter("random_forest:criterion", choices=["squared_error", "friedman_mse", "absolute_error"]),
        UniformIntegerHyperparameter("random_forest:max_depth", lower=2, upper=64),
        UniformFloatHyperparameter("random_forest:max_features", lower=0.1, upper=1.0),
        UniformIntegerHyperparameter("random_forest:max_leaf_nodes", lower=2, upper=512),
        UniformFloatHyperparameter("random_forest:min_impurity_decrease", lower=0.0, upper=0.1),
        UniformIntegerHyperparameter("random_forest:min_samples_leaf", lower=1, upper=20),
        UniformIntegerHyperparameter("random_forest:min_samples_split", lower=2, upper=20),
        UniformFloatHyperparameter("random_forest:min_weight_fraction_leaf", lower=0.0, upper=0.4),
    ]


def _random_forest_conditions(ctx: dict[str, Any]) -> list[Any]:
    return _all_equal_conditions(ctx["model_name"], "random_forest", ctx["params"])


def _build_random_forest(params: dict[str, Any], random_state: int | None, n_jobs: int | None):
    return RandomForestRegressor(
        n_estimators=int(params["n_estimators"]),
        bootstrap=bool(params["bootstrap"]),
        criterion=params["criterion"],
        max_depth=int(params["max_depth"]),
        max_features=float(params["max_features"]),
        max_leaf_nodes=int(params["max_leaf_nodes"]),
        min_impurity_decrease=float(params["min_impurity_decrease"]),
        min_samples_leaf=int(params["min_samples_leaf"]),
        min_samples_split=int(params["min_samples_split"]),
        min_weight_fraction_leaf=float(params["min_weight_fraction_leaf"]),
        random_state=random_state,
        n_jobs=n_jobs,
    )


def _sgd_hyperparameters() -> list[Any]:
    return [
        UniformFloatHyperparameter("sgd:alpha", lower=1e-6, upper=1e-1, log=True),
        CategoricalHyperparameter("sgd:average", choices=[False, True]),
        UniformFloatHyperparameter("sgd:epsilon", lower=1e-4, upper=1.0, log=True),
        UniformFloatHyperparameter("sgd:eta0", lower=1e-4, upper=1e-1, log=True),
        CategoricalHyperparameter("sgd:fit_intercept", choices=[False, True]),
        UniformFloatHyperparameter("sgd:l1_ratio", lower=0.0, upper=1.0),
        CategoricalHyperparameter("sgd:learning_rate", choices=["constant", "optimal", "invscaling", "adaptive"]),
        CategoricalHyperparameter("sgd:loss", choices=["squared_error", "huber", "epsilon_insensitive", "squared_epsilon_insensitive"]),
        CategoricalHyperparameter("sgd:penalty", choices=["l2", "l1", "elasticnet"]),
        UniformFloatHyperparameter("sgd:power_t", lower=0.1, upper=0.9),
        UniformFloatHyperparameter("sgd:tol", lower=1e-5, upper=1e-1, log=True),
    ]


def _sgd_conditions(ctx: dict[str, Any]) -> list[Any]:
    return _all_equal_conditions(ctx["model_name"], "sgd", ctx["params"])


def _build_sgd(params: dict[str, Any], random_state: int | None, n_jobs: int | None):
    return SGDRegressor(
        alpha=float(params["alpha"]),
        average=bool(params["average"]),
        epsilon=float(params["epsilon"]),
        eta0=float(params["eta0"]),
        fit_intercept=bool(params["fit_intercept"]),
        l1_ratio=float(params["l1_ratio"]),
        learning_rate=params["learning_rate"],
        loss=params["loss"],
        penalty=params["penalty"],
        power_t=float(params["power_t"]),
        random_state=random_state,
        tol=float(params["tol"]),
    )


REGRESSION_COMPONENTS: dict[str, RegressionComponent] = {
    "adaboost": RegressionComponent("adaboost", _adaboost_hyperparameters, _adaboost_conditions, _build_adaboost),
    "ard_regression": RegressionComponent("ard_regression", _ard_regression_hyperparameters, _ard_regression_conditions, _build_ard_regression),
    "decision_tree": RegressionComponent("decision_tree", _decision_tree_hyperparameters, _decision_tree_conditions, _build_decision_tree),
    "extra_trees": RegressionComponent("extra_trees", _extra_trees_hyperparameters, _extra_trees_conditions, _build_extra_trees),
    "gaussian_process": RegressionComponent("gaussian_process", _gaussian_process_hyperparameters, _gaussian_process_conditions, _build_gaussian_process),
    "gradient_boosting": RegressionComponent("gradient_boosting", _gradient_boosting_hyperparameters, _gradient_boosting_conditions, _build_gradient_boosting),
    "k_nearest_neighbors": RegressionComponent("k_nearest_neighbors", _knn_hyperparameters, _knn_conditions, _build_knn),
    "liblinear_svr": RegressionComponent("liblinear_svr", _liblinear_svr_hyperparameters, _liblinear_svr_conditions, _build_liblinear_svr),
    "libsvm_svr": RegressionComponent("libsvm_svr", _libsvm_svr_hyperparameters, _libsvm_svr_conditions, _build_libsvm_svr),
    "mlp": RegressionComponent("mlp", _mlp_hyperparameters, _mlp_conditions, _build_mlp),
    "random_forest": RegressionComponent("random_forest", _random_forest_hyperparameters, _random_forest_conditions, _build_random_forest),
    "sgd": RegressionComponent("sgd", _sgd_hyperparameters, _sgd_conditions, _build_sgd),
}


def get_regression_components(allowed_models: list[str] | None = None) -> dict[str, RegressionComponent]:
    if allowed_models is None:
        return dict(REGRESSION_COMPONENTS)

    selected_models = sorted(set(allowed_models))
    unknown = sorted(set(selected_models) - set(REGRESSION_COMPONENTS))
    if unknown:
        raise ValueError(f"Unknown regression models requested: {unknown}")
    if not selected_models:
        raise ValueError("At least one regression model must be selected.")
    return {name: REGRESSION_COMPONENTS[name] for name in selected_models}
