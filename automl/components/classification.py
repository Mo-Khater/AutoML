from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from ConfigSpace.conditions import AndConjunction, EqualsCondition, InCondition
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
)
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier, SGDClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


@dataclass(frozen=True)
class ClassificationComponent:
    name: str
    build_hyperparameters: Callable[[], list[Any]]
    build_conditions: Callable[[dict[str, Any]], list[Any]]
    build_estimator: Callable[[dict[str, Any], int | None, int | None, bool], Any]


def _all_equal_conditions(model_name_hyperparameter: Any, component_name: str, params: dict[str, Any]) -> list[Any]:
    return [
        EqualsCondition(param, model_name_hyperparameter, component_name)
        for param in params.values()
    ]


def _set_optional_class_weight(
    estimator_params: dict[str, Any],
    params: dict[str, Any],
    *,
    balance_classes: bool,
    key: str = "class_weight",
) -> None:
    class_weight = params.get(key, "none")
    if balance_classes:
        estimator_params[key] = "balanced"
    elif class_weight != "none":
        estimator_params[key] = class_weight


def _logistic_regression_hyperparameters() -> list[Any]:
    return [
        UniformFloatHyperparameter("logistic_regression:C", lower=1e-4, upper=1e4, log=True),
        CategoricalHyperparameter("logistic_regression:solver", choices=["lbfgs", "liblinear", "saga"]),
        CategoricalHyperparameter("logistic_regression:penalty", choices=["l2", "l1"]),
        UniformIntegerHyperparameter("logistic_regression:max_iter", lower=100, upper=2000),
        CategoricalHyperparameter("logistic_regression:class_weight", choices=["none", "balanced"]),
    ]


def _logistic_regression_conditions(ctx: dict[str, Any]) -> list[Any]:
    params = ctx["params"]
    conditions = _all_equal_conditions(ctx["model_name"], "logistic_regression", params)
    conditions = [
        cond
        for cond in conditions
        if cond.child.name != "logistic_regression:penalty"
    ]
    conditions.append(
        AndConjunction(
            EqualsCondition(
                params["logistic_regression:penalty"],
                ctx["model_name"],
                "logistic_regression",
            ),
            InCondition(
                params["logistic_regression:penalty"],
                params["logistic_regression:solver"],
                ["liblinear", "saga"],
            ),
        )
    )
    return conditions


def _build_logistic_regression(params: dict[str, Any], random_state: int | None, n_jobs: int | None, balance_classes: bool):
    estimator_params = {
        "C": float(params["C"]),
        "solver": params["solver"],
        "penalty": params.get("penalty", "l2"),
        "max_iter": int(params["max_iter"]),
        "random_state": random_state,
    }
    _set_optional_class_weight(estimator_params, params, balance_classes=balance_classes)
    if estimator_params["solver"] == "lbfgs":
        estimator_params["penalty"] = "l2"
    return LogisticRegression(**estimator_params)


def _random_forest_hyperparameters() -> list[Any]:
    return [
        UniformIntegerHyperparameter("random_forest:n_estimators", lower=100, upper=1000),
        UniformIntegerHyperparameter("random_forest:max_depth", lower=2, upper=64),
        UniformIntegerHyperparameter("random_forest:min_samples_split", lower=2, upper=20),
        UniformIntegerHyperparameter("random_forest:min_samples_leaf", lower=1, upper=20),
        CategoricalHyperparameter("random_forest:criterion", choices=["gini", "entropy"]),
        CategoricalHyperparameter("random_forest:max_features", choices=["sqrt", "log2", None]),
        CategoricalHyperparameter("random_forest:bootstrap", choices=[True, False]),
        UniformIntegerHyperparameter("random_forest:max_leaf_nodes", lower=2, upper=512),
        CategoricalHyperparameter("random_forest:class_weight", choices=["none", "balanced"]),
    ]


def _random_forest_conditions(ctx: dict[str, Any]) -> list[Any]:
    return _all_equal_conditions(ctx["model_name"], "random_forest", ctx["params"])


def _build_random_forest(params: dict[str, Any], random_state: int | None, n_jobs: int | None, balance_classes: bool):
    estimator_params = {
        "n_estimators": int(params["n_estimators"]),
        "max_depth": int(params["max_depth"]),
        "min_samples_split": int(params["min_samples_split"]),
        "min_samples_leaf": int(params["min_samples_leaf"]),
        "criterion": params["criterion"],
        "max_features": params.get("max_features"),
        "bootstrap": params.get("bootstrap", True),
        "max_leaf_nodes": int(params["max_leaf_nodes"]),
        "random_state": random_state,
        "n_jobs": n_jobs,
    }
    _set_optional_class_weight(estimator_params, params, balance_classes=balance_classes)
    return RandomForestClassifier(**estimator_params)


def _extra_trees_hyperparameters() -> list[Any]:
    return [
        UniformIntegerHyperparameter("extra_trees:n_estimators", lower=100, upper=1000),
        UniformIntegerHyperparameter("extra_trees:max_depth", lower=2, upper=64),
        UniformIntegerHyperparameter("extra_trees:min_samples_split", lower=2, upper=20),
        UniformIntegerHyperparameter("extra_trees:min_samples_leaf", lower=1, upper=20),
        CategoricalHyperparameter("extra_trees:criterion", choices=["gini", "entropy"]),
        CategoricalHyperparameter("extra_trees:max_features", choices=["sqrt", "log2", None]),
        UniformIntegerHyperparameter("extra_trees:max_leaf_nodes", lower=2, upper=512),
        CategoricalHyperparameter("extra_trees:class_weight", choices=["none", "balanced"]),
    ]


def _extra_trees_conditions(ctx: dict[str, Any]) -> list[Any]:
    return _all_equal_conditions(ctx["model_name"], "extra_trees", ctx["params"])


def _build_extra_trees(params: dict[str, Any], random_state: int | None, n_jobs: int | None, balance_classes: bool):
    estimator_params = {
        "n_estimators": int(params["n_estimators"]),
        "max_depth": int(params["max_depth"]),
        "min_samples_split": int(params["min_samples_split"]),
        "min_samples_leaf": int(params["min_samples_leaf"]),
        "criterion": params["criterion"],
        "max_features": params.get("max_features"),
        "max_leaf_nodes": int(params["max_leaf_nodes"]),
        "random_state": random_state,
        "n_jobs": n_jobs,
    }
    _set_optional_class_weight(estimator_params, params, balance_classes=balance_classes)
    return ExtraTreesClassifier(**estimator_params)


def _svc_hyperparameters() -> list[Any]:
    return [
        UniformFloatHyperparameter("svc:C", lower=1e-4, upper=1e4, log=True),
        CategoricalHyperparameter("svc:kernel", choices=["rbf", "poly", "sigmoid"]),
        UniformFloatHyperparameter("svc:gamma", lower=1e-5, upper=10.0, log=True),
        UniformIntegerHyperparameter("svc:degree", lower=2, upper=5),
        UniformFloatHyperparameter("svc:coef0", lower=-1.0, upper=1.0),
        CategoricalHyperparameter("svc:class_weight", choices=["none", "balanced"]),
    ]


def _svc_conditions(ctx: dict[str, Any]) -> list[Any]:
    params = ctx["params"]
    conditions = _all_equal_conditions(ctx["model_name"], "svc", params)
    conditions = [
        cond
        for cond in conditions
        if cond.child.name not in {"svc:degree", "svc:coef0"}
    ]
    conditions.extend(
        [
            AndConjunction(
                EqualsCondition(params["svc:degree"], ctx["model_name"], "svc"),
                EqualsCondition(params["svc:degree"], params["svc:kernel"], "poly"),
            ),
            AndConjunction(
                EqualsCondition(params["svc:coef0"], ctx["model_name"], "svc"),
                InCondition(params["svc:coef0"], params["svc:kernel"], ["poly", "sigmoid"]),
            ),
        ]
    )
    return conditions


def _build_svc(params: dict[str, Any], random_state: int | None, n_jobs: int | None, balance_classes: bool):
    estimator_params = {
        "C": float(params["C"]),
        "kernel": params.get("kernel", "rbf"),
        "gamma": float(params["gamma"]),
        "degree": int(params.get("degree", 3)),
        "coef0": float(params.get("coef0", 0.0)),
        "probability": True,
        "random_state": random_state,
    }
    _set_optional_class_weight(estimator_params, params, balance_classes=balance_classes)
    return SVC(**estimator_params)


def _knn_hyperparameters() -> list[Any]:
    return [
        UniformIntegerHyperparameter("knn:n_neighbors", lower=1, upper=50),
        CategoricalHyperparameter("knn:weights", choices=["uniform", "distance"]),
        CategoricalHyperparameter("knn:p", choices=[1, 2]),
        CategoricalHyperparameter("knn:algorithm", choices=["auto", "ball_tree", "kd_tree", "brute"]),
        UniformIntegerHyperparameter("knn:leaf_size", lower=10, upper=100),
    ]


def _knn_conditions(ctx: dict[str, Any]) -> list[Any]:
    return _all_equal_conditions(ctx["model_name"], "knn", ctx["params"])


def _build_knn(params: dict[str, Any], random_state: int | None, n_jobs: int | None, balance_classes: bool):
    return KNeighborsClassifier(
        n_neighbors=int(params["n_neighbors"]),
        weights=params["weights"],
        p=int(params["p"]),
        algorithm=params.get("algorithm", "auto"),
        leaf_size=int(params.get("leaf_size", 30)),
    )


def _gradient_boosting_hyperparameters() -> list[Any]:
    return [
        UniformIntegerHyperparameter("gradient_boosting:n_estimators", lower=50, upper=500),
        UniformFloatHyperparameter("gradient_boosting:learning_rate", lower=1e-3, upper=0.5, log=True),
        UniformIntegerHyperparameter("gradient_boosting:max_depth", lower=1, upper=10),
        UniformFloatHyperparameter("gradient_boosting:subsample", lower=0.5, upper=1.0),
        UniformIntegerHyperparameter("gradient_boosting:max_leaf_nodes", lower=2, upper=128),
    ]


def _gradient_boosting_conditions(ctx: dict[str, Any]) -> list[Any]:
    return _all_equal_conditions(ctx["model_name"], "gradient_boosting", ctx["params"])


def _build_gradient_boosting(params: dict[str, Any], random_state: int | None, n_jobs: int | None, balance_classes: bool):
    return GradientBoostingClassifier(
        n_estimators=int(params["n_estimators"]),
        learning_rate=float(params["learning_rate"]),
        max_depth=int(params["max_depth"]),
        subsample=float(params["subsample"]),
        max_leaf_nodes=int(params["max_leaf_nodes"]),
        random_state=random_state,
    )


def _lightgbm_hyperparameters() -> list[Any]:
    return [
        UniformIntegerHyperparameter("lightgbm:n_estimators", lower=100, upper=1200),
        UniformFloatHyperparameter("lightgbm:learning_rate", lower=1e-3, upper=0.3, log=True),
        UniformIntegerHyperparameter("lightgbm:num_leaves", lower=15, upper=255),
        UniformIntegerHyperparameter("lightgbm:max_depth", lower=-1, upper=16),
        UniformIntegerHyperparameter("lightgbm:min_child_samples", lower=5, upper=100),
        UniformFloatHyperparameter("lightgbm:subsample", lower=0.5, upper=1.0),
        UniformFloatHyperparameter("lightgbm:colsample_bytree", lower=0.5, upper=1.0),
        UniformFloatHyperparameter("lightgbm:reg_alpha", lower=1e-8, upper=10.0, log=True),
        UniformFloatHyperparameter("lightgbm:reg_lambda", lower=1e-8, upper=10.0, log=True),
        CategoricalHyperparameter("lightgbm:class_weight", choices=["none", "balanced"]),
    ]


def _lightgbm_conditions(ctx: dict[str, Any]) -> list[Any]:
    return _all_equal_conditions(ctx["model_name"], "lightgbm", ctx["params"])


def _build_lightgbm(params: dict[str, Any], random_state: int | None, n_jobs: int | None, balance_classes: bool):
    estimator_params = {
        "n_estimators": int(params["n_estimators"]),
        "learning_rate": float(params["learning_rate"]),
        "num_leaves": int(params["num_leaves"]),
        "max_depth": int(params["max_depth"]),
        "min_child_samples": int(params["min_child_samples"]),
        "subsample": float(params["subsample"]),
        "colsample_bytree": float(params["colsample_bytree"]),
        "reg_alpha": float(params["reg_alpha"]),
        "reg_lambda": float(params["reg_lambda"]),
        "random_state": random_state,
        "n_jobs": n_jobs,
        "verbosity": -1,
    }
    _set_optional_class_weight(estimator_params, params, balance_classes=balance_classes)
    return LGBMClassifier(**estimator_params)


def _xgboost_hyperparameters() -> list[Any]:
    return [
        UniformIntegerHyperparameter("xgboost:n_estimators", lower=100, upper=1200),
        UniformFloatHyperparameter("xgboost:learning_rate", lower=1e-3, upper=0.3, log=True),
        UniformIntegerHyperparameter("xgboost:max_depth", lower=2, upper=12),
        UniformFloatHyperparameter("xgboost:min_child_weight", lower=1e-2, upper=32.0, log=True),
        UniformFloatHyperparameter("xgboost:subsample", lower=0.5, upper=1.0),
        UniformFloatHyperparameter("xgboost:colsample_bytree", lower=0.5, upper=1.0),
        UniformFloatHyperparameter("xgboost:reg_alpha", lower=1e-8, upper=10.0, log=True),
        UniformFloatHyperparameter("xgboost:reg_lambda", lower=1e-8, upper=10.0, log=True),
        UniformFloatHyperparameter("xgboost:gamma", lower=1e-8, upper=10.0, log=True),
    ]


def _xgboost_conditions(ctx: dict[str, Any]) -> list[Any]:
    return _all_equal_conditions(ctx["model_name"], "xgboost", ctx["params"])


def _build_xgboost(params: dict[str, Any], random_state: int | None, n_jobs: int | None, balance_classes: bool):
    estimator_params = {
        "n_estimators": int(params["n_estimators"]),
        "learning_rate": float(params["learning_rate"]),
        "max_depth": int(params["max_depth"]),
        "min_child_weight": float(params["min_child_weight"]),
        "subsample": float(params["subsample"]),
        "colsample_bytree": float(params["colsample_bytree"]),
        "reg_alpha": float(params["reg_alpha"]),
        "reg_lambda": float(params["reg_lambda"]),
        "gamma": float(params["gamma"]),
        "random_state": random_state,
        "n_jobs": n_jobs,
        "verbosity": 0,
        "eval_metric": "logloss",
    }
    if balance_classes:
        estimator_params["scale_pos_weight"] = 1.0
    return XGBClassifier(**estimator_params)


def _catboost_hyperparameters() -> list[Any]:
    return [
        UniformIntegerHyperparameter("catboost:iterations", lower=100, upper=1200),
        UniformFloatHyperparameter("catboost:learning_rate", lower=1e-3, upper=0.3, log=True),
        UniformIntegerHyperparameter("catboost:depth", lower=4, upper=10),
        UniformFloatHyperparameter("catboost:l2_leaf_reg", lower=1e-3, upper=30.0, log=True),
        UniformFloatHyperparameter("catboost:random_strength", lower=1e-3, upper=10.0, log=True),
        UniformFloatHyperparameter("catboost:bagging_temperature", lower=0.0, upper=10.0),
        CategoricalHyperparameter("catboost:auto_class_weights", choices=["none", "Balanced"]),
    ]


def _catboost_conditions(ctx: dict[str, Any]) -> list[Any]:
    return _all_equal_conditions(ctx["model_name"], "catboost", ctx["params"])


def _build_catboost(params: dict[str, Any], random_state: int | None, n_jobs: int | None, balance_classes: bool):
    estimator_params = {
        "iterations": int(params["iterations"]),
        "learning_rate": float(params["learning_rate"]),
        "depth": int(params["depth"]),
        "l2_leaf_reg": float(params["l2_leaf_reg"]),
        "random_strength": float(params["random_strength"]),
        "bagging_temperature": float(params["bagging_temperature"]),
        "random_seed": random_state,
        "verbose": False,
        "allow_writing_files": False,
    }
    if n_jobs is not None:
        estimator_params["thread_count"] = int(n_jobs)
    auto_class_weights = params.get("auto_class_weights", "none")
    if balance_classes:
        estimator_params["auto_class_weights"] = "Balanced"
    elif auto_class_weights != "none":
        estimator_params["auto_class_weights"] = auto_class_weights
    return CatBoostClassifier(**estimator_params)


def _hist_gradient_boosting_hyperparameters() -> list[Any]:
    return [
        UniformFloatHyperparameter("hist_gradient_boosting:learning_rate", lower=1e-3, upper=0.5, log=True),
        UniformIntegerHyperparameter("hist_gradient_boosting:max_iter", lower=50, upper=500),
        UniformIntegerHyperparameter("hist_gradient_boosting:max_depth", lower=2, upper=32),
        UniformIntegerHyperparameter("hist_gradient_boosting:min_samples_leaf", lower=10, upper=100),
        UniformFloatHyperparameter("hist_gradient_boosting:l2_regularization", lower=1e-10, upper=1.0, log=True),
        UniformIntegerHyperparameter("hist_gradient_boosting:max_bins", lower=32, upper=255),
        CategoricalHyperparameter("hist_gradient_boosting:early_stopping", choices=[False, True]),
    ]


def _hist_gradient_boosting_conditions(ctx: dict[str, Any]) -> list[Any]:
    return _all_equal_conditions(ctx["model_name"], "hist_gradient_boosting", ctx["params"])


def _build_hist_gradient_boosting(params: dict[str, Any], random_state: int | None, n_jobs: int | None, balance_classes: bool):
    return HistGradientBoostingClassifier(
        learning_rate=float(params["learning_rate"]),
        max_iter=int(params["max_iter"]),
        max_depth=int(params["max_depth"]),
        min_samples_leaf=int(params["min_samples_leaf"]),
        l2_regularization=float(params["l2_regularization"]),
        max_bins=int(params["max_bins"]),
        early_stopping=bool(params["early_stopping"]),
        random_state=random_state,
    )


def _adaboost_hyperparameters() -> list[Any]:
    return [
        UniformIntegerHyperparameter("adaboost:n_estimators", lower=25, upper=500),
        UniformFloatHyperparameter("adaboost:learning_rate", lower=1e-3, upper=2.0, log=True),
    ]


def _adaboost_conditions(ctx: dict[str, Any]) -> list[Any]:
    return _all_equal_conditions(ctx["model_name"], "adaboost", ctx["params"])


def _build_adaboost(params: dict[str, Any], random_state: int | None, n_jobs: int | None, balance_classes: bool):
    return AdaBoostClassifier(
        n_estimators=int(params["n_estimators"]),
        learning_rate=float(params["learning_rate"]),
        random_state=random_state,
    )


def _gaussian_nb_hyperparameters() -> list[Any]:
    return [
        UniformFloatHyperparameter("gaussian_nb:var_smoothing", lower=1e-12, upper=1e-6, log=True),
    ]


def _gaussian_nb_conditions(ctx: dict[str, Any]) -> list[Any]:
    return _all_equal_conditions(ctx["model_name"], "gaussian_nb", ctx["params"])


def _build_gaussian_nb(params: dict[str, Any], random_state: int | None, n_jobs: int | None, balance_classes: bool):
    return GaussianNB(var_smoothing=float(params["var_smoothing"]))


def _lda_hyperparameters() -> list[Any]:
    return [
        CategoricalHyperparameter("lda:solver", choices=["svd", "lsqr", "eigen"]),
        CategoricalHyperparameter("lda:shrinkage", choices=[None, "auto"]),
    ]


def _lda_conditions(ctx: dict[str, Any]) -> list[Any]:
    params = ctx["params"]
    conditions = _all_equal_conditions(ctx["model_name"], "lda", params)
    conditions = [
        cond
        for cond in conditions
        if cond.child.name != "lda:shrinkage"
    ]
    conditions.append(
        AndConjunction(
            EqualsCondition(params["lda:shrinkage"], ctx["model_name"], "lda"),
            InCondition(params["lda:shrinkage"], params["lda:solver"], ["lsqr", "eigen"]),
        )
    )
    return conditions


def _build_lda(params: dict[str, Any], random_state: int | None, n_jobs: int | None, balance_classes: bool):
    estimator_params = {"solver": params["solver"]}
    if params["solver"] in {"lsqr", "eigen"}:
        estimator_params["shrinkage"] = params.get("shrinkage")
    return LinearDiscriminantAnalysis(**estimator_params)


def _decision_tree_hyperparameters() -> list[Any]:
    return [
        CategoricalHyperparameter("decision_tree:criterion", choices=["gini", "entropy"]),
        UniformIntegerHyperparameter("decision_tree:max_depth", lower=1, upper=64),
        UniformIntegerHyperparameter("decision_tree:min_samples_split", lower=2, upper=20),
        UniformIntegerHyperparameter("decision_tree:min_samples_leaf", lower=1, upper=20),
        CategoricalHyperparameter("decision_tree:max_features", choices=["sqrt", "log2", None]),
        CategoricalHyperparameter("decision_tree:class_weight", choices=["none", "balanced"]),
    ]


def _decision_tree_conditions(ctx: dict[str, Any]) -> list[Any]:
    return _all_equal_conditions(ctx["model_name"], "decision_tree", ctx["params"])


def _build_decision_tree(params: dict[str, Any], random_state: int | None, n_jobs: int | None, balance_classes: bool):
    estimator_params = {
        "criterion": params["criterion"],
        "max_depth": int(params["max_depth"]),
        "min_samples_split": int(params["min_samples_split"]),
        "min_samples_leaf": int(params["min_samples_leaf"]),
        "max_features": params.get("max_features"),
        "random_state": random_state,
    }
    _set_optional_class_weight(estimator_params, params, balance_classes=balance_classes)
    return DecisionTreeClassifier(**estimator_params)


def _sgd_hyperparameters() -> list[Any]:
    return [
        CategoricalHyperparameter("sgd:loss", choices=["hinge", "log_loss", "modified_huber"]),
        CategoricalHyperparameter("sgd:penalty", choices=["l2", "l1", "elasticnet"]),
        UniformFloatHyperparameter("sgd:alpha", lower=1e-6, upper=1e-1, log=True),
        CategoricalHyperparameter("sgd:learning_rate", choices=["optimal", "invscaling", "adaptive"]),
        UniformFloatHyperparameter("sgd:eta0", lower=1e-4, upper=1e-1, log=True),
        CategoricalHyperparameter("sgd:average", choices=[False, True]),
        CategoricalHyperparameter("sgd:class_weight", choices=["none", "balanced"]),
    ]


def _sgd_conditions(ctx: dict[str, Any]) -> list[Any]:
    return _all_equal_conditions(ctx["model_name"], "sgd", ctx["params"])


def _build_sgd(params: dict[str, Any], random_state: int | None, n_jobs: int | None, balance_classes: bool):
    estimator_params = {
        "loss": params["loss"],
        "penalty": params["penalty"],
        "alpha": float(params["alpha"]),
        "learning_rate": params["learning_rate"],
        "eta0": float(params["eta0"]),
        "average": bool(params["average"]),
        "random_state": random_state,
    }
    _set_optional_class_weight(estimator_params, params, balance_classes=balance_classes)
    return SGDClassifier(**estimator_params)


def _passive_aggressive_hyperparameters() -> list[Any]:
    return [
        UniformFloatHyperparameter("passive_aggressive:C", lower=1e-4, upper=10.0, log=True),
        CategoricalHyperparameter("passive_aggressive:loss", choices=["hinge", "squared_hinge"]),
        CategoricalHyperparameter("passive_aggressive:average", choices=[False, True]),
        CategoricalHyperparameter("passive_aggressive:class_weight", choices=["none", "balanced"]),
    ]


def _passive_aggressive_conditions(ctx: dict[str, Any]) -> list[Any]:
    return _all_equal_conditions(ctx["model_name"], "passive_aggressive", ctx["params"])


def _build_passive_aggressive(params: dict[str, Any], random_state: int | None, n_jobs: int | None, balance_classes: bool):
    estimator_params = {
        "C": float(params["C"]),
        "loss": params["loss"],
        "average": bool(params["average"]),
        "random_state": random_state,
    }
    _set_optional_class_weight(estimator_params, params, balance_classes=balance_classes)
    return PassiveAggressiveClassifier(**estimator_params)


def _qda_hyperparameters() -> list[Any]:
    return [
        UniformFloatHyperparameter("qda:reg_param", lower=0.0, upper=1.0),
    ]


def _qda_conditions(ctx: dict[str, Any]) -> list[Any]:
    return _all_equal_conditions(ctx["model_name"], "qda", ctx["params"])


def _build_qda(params: dict[str, Any], random_state: int | None, n_jobs: int | None, balance_classes: bool):
    return QuadraticDiscriminantAnalysis(reg_param=float(params["reg_param"]))


def _liblinear_svc_hyperparameters() -> list[Any]:
    return [
        UniformFloatHyperparameter("liblinear_svc:C", lower=1e-4, upper=1e4, log=True),
        CategoricalHyperparameter("liblinear_svc:loss", choices=["hinge", "squared_hinge"]),
        CategoricalHyperparameter("liblinear_svc:class_weight", choices=["none", "balanced"]),
    ]


def _liblinear_svc_conditions(ctx: dict[str, Any]) -> list[Any]:
    return _all_equal_conditions(ctx["model_name"], "liblinear_svc", ctx["params"])


def _build_liblinear_svc(params: dict[str, Any], random_state: int | None, n_jobs: int | None, balance_classes: bool):
    estimator_params = {
        "C": float(params["C"]),
        "loss": params["loss"],
        "random_state": random_state,
    }
    _set_optional_class_weight(estimator_params, params, balance_classes=balance_classes)
    return LinearSVC(**estimator_params)


def _bernoulli_nb_hyperparameters() -> list[Any]:
    return [
        UniformFloatHyperparameter("bernoulli_nb:alpha", lower=1e-3, upper=100.0, log=True),
        CategoricalHyperparameter("bernoulli_nb:fit_prior", choices=[False, True]),
    ]


def _bernoulli_nb_conditions(ctx: dict[str, Any]) -> list[Any]:
    return _all_equal_conditions(ctx["model_name"], "bernoulli_nb", ctx["params"])


def _build_bernoulli_nb(params: dict[str, Any], random_state: int | None, n_jobs: int | None, balance_classes: bool):
    return BernoulliNB(
        alpha=float(params["alpha"]),
        fit_prior=bool(params["fit_prior"]),
    )


def _multinomial_nb_hyperparameters() -> list[Any]:
    return [
        UniformFloatHyperparameter("multinomial_nb:alpha", lower=1e-3, upper=100.0, log=True),
        CategoricalHyperparameter("multinomial_nb:fit_prior", choices=[False, True]),
    ]


def _multinomial_nb_conditions(ctx: dict[str, Any]) -> list[Any]:
    return _all_equal_conditions(ctx["model_name"], "multinomial_nb", ctx["params"])


def _build_multinomial_nb(params: dict[str, Any], random_state: int | None, n_jobs: int | None, balance_classes: bool):
    return MultinomialNB(
        alpha=float(params["alpha"]),
        fit_prior=bool(params["fit_prior"]),
    )


def _mlp_hyperparameters() -> list[Any]:
    return [
        UniformIntegerHyperparameter("mlp:hidden_layer_sizes", lower=32, upper=256),
        UniformFloatHyperparameter("mlp:alpha", lower=1e-6, upper=1e-1, log=True),
        UniformFloatHyperparameter("mlp:learning_rate_init", lower=1e-4, upper=1e-1, log=True),
        CategoricalHyperparameter("mlp:activation", choices=["relu", "tanh", "logistic"]),
        CategoricalHyperparameter("mlp:solver", choices=["adam", "sgd"]),
    ]


def _mlp_conditions(ctx: dict[str, Any]) -> list[Any]:
    return _all_equal_conditions(ctx["model_name"], "mlp", ctx["params"])


def _build_mlp(params: dict[str, Any], random_state: int | None, n_jobs: int | None, balance_classes: bool):
    return MLPClassifier(
        hidden_layer_sizes=(int(params["hidden_layer_sizes"]),),
        alpha=float(params["alpha"]),
        learning_rate_init=float(params["learning_rate_init"]),
        activation=params["activation"],
        solver=params["solver"],
        max_iter=400,
        early_stopping=True,
        random_state=random_state,
    )


def _gaussian_process_hyperparameters() -> list[Any]:
    return [
        UniformFloatHyperparameter("gaussian_process:length_scale", lower=1e-2, upper=1e2, log=True),
        UniformIntegerHyperparameter("gaussian_process:max_iter_predict", lower=20, upper=200),
    ]


def _gaussian_process_conditions(ctx: dict[str, Any]) -> list[Any]:
    return _all_equal_conditions(ctx["model_name"], "gaussian_process", ctx["params"])


def _build_gaussian_process(params: dict[str, Any], random_state: int | None, n_jobs: int | None, balance_classes: bool):
    return GaussianProcessClassifier(
        kernel=1.0 * RBF(length_scale=float(params["length_scale"])),
        max_iter_predict=int(params["max_iter_predict"]),
        random_state=random_state,
    )


def _ridge_classifier_hyperparameters() -> list[Any]:
    return [
        UniformFloatHyperparameter("ridge_classifier:alpha", lower=1e-4, upper=1e3, log=True),
        CategoricalHyperparameter("ridge_classifier:class_weight", choices=["none", "balanced"]),
        CategoricalHyperparameter("ridge_classifier:solver", choices=["auto", "svd", "cholesky", "lsqr", "sag"]),
    ]


def _ridge_classifier_conditions(ctx: dict[str, Any]) -> list[Any]:
    return _all_equal_conditions(ctx["model_name"], "ridge_classifier", ctx["params"])


def _build_ridge_classifier(params: dict[str, Any], random_state: int | None, n_jobs: int | None, balance_classes: bool):
    estimator_params = {
        "alpha": float(params["alpha"]),
        "solver": params["solver"],
        "random_state": random_state,
    }
    _set_optional_class_weight(estimator_params, params, balance_classes=balance_classes)
    return RidgeClassifier(**estimator_params)


CLASSIFICATION_COMPONENTS: dict[str, ClassificationComponent] = {
    "adaboost": ClassificationComponent("adaboost", _adaboost_hyperparameters, _adaboost_conditions, _build_adaboost),
    "bernoulli_nb": ClassificationComponent("bernoulli_nb", _bernoulli_nb_hyperparameters, _bernoulli_nb_conditions, _build_bernoulli_nb),
    "catboost": ClassificationComponent("catboost", _catboost_hyperparameters, _catboost_conditions, _build_catboost),
    "decision_tree": ClassificationComponent("decision_tree", _decision_tree_hyperparameters, _decision_tree_conditions, _build_decision_tree),
    "extra_trees": ClassificationComponent("extra_trees", _extra_trees_hyperparameters, _extra_trees_conditions, _build_extra_trees),
    "gaussian_nb": ClassificationComponent("gaussian_nb", _gaussian_nb_hyperparameters, _gaussian_nb_conditions, _build_gaussian_nb),
    "gradient_boosting": ClassificationComponent("gradient_boosting", _gradient_boosting_hyperparameters, _gradient_boosting_conditions, _build_gradient_boosting),
    "hist_gradient_boosting": ClassificationComponent("hist_gradient_boosting", _hist_gradient_boosting_hyperparameters, _hist_gradient_boosting_conditions, _build_hist_gradient_boosting),
    "knn": ClassificationComponent("knn", _knn_hyperparameters, _knn_conditions, _build_knn),
    "lda": ClassificationComponent("lda", _lda_hyperparameters, _lda_conditions, _build_lda),
    "lightgbm": ClassificationComponent("lightgbm", _lightgbm_hyperparameters, _lightgbm_conditions, _build_lightgbm),
    "liblinear_svc": ClassificationComponent("liblinear_svc", _liblinear_svc_hyperparameters, _liblinear_svc_conditions, _build_liblinear_svc),
    "logistic_regression": ClassificationComponent("logistic_regression", _logistic_regression_hyperparameters, _logistic_regression_conditions, _build_logistic_regression),
    "multinomial_nb": ClassificationComponent("multinomial_nb", _multinomial_nb_hyperparameters, _multinomial_nb_conditions, _build_multinomial_nb),
    "mlp": ClassificationComponent("mlp", _mlp_hyperparameters, _mlp_conditions, _build_mlp),
    "passive_aggressive": ClassificationComponent("passive_aggressive", _passive_aggressive_hyperparameters, _passive_aggressive_conditions, _build_passive_aggressive),
    "qda": ClassificationComponent("qda", _qda_hyperparameters, _qda_conditions, _build_qda),
    "gaussian_process": ClassificationComponent("gaussian_process", _gaussian_process_hyperparameters, _gaussian_process_conditions, _build_gaussian_process),
    "random_forest": ClassificationComponent("random_forest", _random_forest_hyperparameters, _random_forest_conditions, _build_random_forest),
    "ridge_classifier": ClassificationComponent("ridge_classifier", _ridge_classifier_hyperparameters, _ridge_classifier_conditions, _build_ridge_classifier),
    "sgd": ClassificationComponent("sgd", _sgd_hyperparameters, _sgd_conditions, _build_sgd),
    "svc": ClassificationComponent("svc", _svc_hyperparameters, _svc_conditions, _build_svc),
    "xgboost": ClassificationComponent("xgboost", _xgboost_hyperparameters, _xgboost_conditions, _build_xgboost),
}


def get_classification_components(allowed_models: list[str] | None = None) -> dict[str, ClassificationComponent]:
    if allowed_models is None:
        return dict(CLASSIFICATION_COMPONENTS)

    selected_models = sorted(set(allowed_models))
    unknown = sorted(set(selected_models) - set(CLASSIFICATION_COMPONENTS))
    if unknown:
        raise ValueError(f"Unknown classification models requested: {unknown}")
    if not selected_models:
        raise ValueError("At least one classification model must be selected.")
    return {name: CLASSIFICATION_COMPONENTS[name] for name in selected_models}
