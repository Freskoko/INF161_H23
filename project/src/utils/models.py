import json
from math import sqrt
from pathlib import Path

import pandas as pd
import plotly.express as px
from loguru import logger
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import ElasticNet, Lasso, LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeRegressor

RANDOM_STATE = 2
PWD = Path().absolute()

directory = f"{str(PWD)}/out"

training_df = pd.read_csv(f"{directory}/main_training_data.csv")
test_df = pd.read_csv(f"{directory}/main_test_data.csv")
validation_df = pd.read_csv(f"{directory}/main_validation_data.csv")


def train_models(split_dict: dict) -> dict:
    """
    Trains a variety of models on test data, and checks their MSE on validation data
    """

    X_train = split_dict["x_train"]
    y_train = split_dict["y_train"]
    X_val = split_dict["x_val"]
    y_val = split_dict["y_val"]

    # models are saved as dicts in a list
    models = [
        {"model_type": DummyRegressor, "settings": {}},
        {"model_type": Lasso, "settings": {"alpha": 100, "random_state": RANDOM_STATE}},
        {
            "model_type": RandomForestRegressor,
            "settings": {"n_estimators": 100, "random_state": RANDOM_STATE},
        },
        {
            "model_type": ElasticNet,
            "settings": {"alpha": 100, "random_state": RANDOM_STATE},
        },
        {"model_type": SVR, "settings": {"degree": 2}},
        {
            "model_type": GradientBoostingRegressor,
            "settings": {
                "n_estimators": 50,
                "learning_rate": 0.1,
                "random_state": RANDOM_STATE,
            },
        },
        {
            "model_type": RandomForestRegressor,
            "settings": {"n_estimators": 50, "random_state": RANDOM_STATE},
        },
        {
            "model_type": RandomForestRegressor,
            "settings": {"n_estimators": 15, "random_state": RANDOM_STATE},
        },
        {
            "model_type": RandomForestRegressor,
            "settings": {"n_estimators": 200, "random_state": RANDOM_STATE},
        },
        {"model_type": KNeighborsRegressor, "settings": {"n_neighbors": 5}},
        {"model_type": KNeighborsRegressor, "settings": {"n_neighbors": 50}},
        {"model_type": KNeighborsRegressor, "settings": {"n_neighbors": 100}},
        {"model_type": DecisionTreeRegressor, "settings": {"max_depth": 12}},
        {"model_type": DecisionTreeRegressor, "settings": {"max_depth": 50}},
        {"model_type": DecisionTreeRegressor, "settings": {"max_depth": 100}},
    ]

    model_strings = []
    mse_values_models = []
    clf_vals = []

    for mod in models:
        name = mod["model_type"].__name__
        settings = mod["settings"]

        logger.info(f"Training model type: {name}_{settings}")
        clf = mod["model_type"](
            **mod["settings"]
        )  # henter ut settings her med unpacking
        clf.fit(X_train, y_train)

        y_predicted = clf.predict(X_val)

        pf_mse = mean_squared_error(y_val, y_predicted, squared=True)

        mse_values_models.append(pf_mse)
        model_strings.append(f"{name}_{settings}")
        clf_vals.append(clf)

    data_models = pd.DataFrame(
        {
            "model_name": model_strings,
            "mse_values": [sqrt(i) for i in mse_values_models],
        }
    )

    fig = px.bar(
        data_models,
        x="model_name",
        y="mse_values",
        title="MSE values for different models",
        labels={"x": "Model", "y": "Mean Error"},
        text=data_models["mse_values"].round(3),
    )
    fig.update_traces(textposition="auto")

    fig.write_image(f"{PWD}/figs/MANYMODELS_MSE.png")

    model_dict = dict(zip(model_strings, clf_vals))

    logger.info("Done training a variety of models!")

    return model_dict


def find_hyper_param(split_dict: dict) -> dict:
    """
    Trains a single model (testing multiple hyperparameters) on test data, finds its MSE on validation data
    """

    X_train = split_dict["x_train"]
    y_train = split_dict["y_train"]
    X_val = split_dict["x_val"]
    y_val = split_dict["y_val"]

    models = []

    for i in range(1, 351, 50):
        if i == 0:
            i = 1
        model = {
            "model_type": RandomForestRegressor,
            "settings": {"n_estimators": i, "random_state": RANDOM_STATE},
        }
        models.append(model)

    model_strings = []
    mse_values_models = []
    clf_vals = []

    for mod in models:
        name = mod["model_type"].__name__
        settings = mod["settings"]

        logger.info(f"Training model type: {name}_{settings}")
        clf = mod["model_type"](**mod["settings"])  # henter ut settings her
        clf.fit(X_train, y_train)

        y_predicted = clf.predict(X_val)

        # finn mse
        pf_mse = mean_squared_error(y_val, y_predicted, squared=True)

        mse_values_models.append(pf_mse)
        model_strings.append(f"{name}_{settings}")
        clf_vals.append(clf)

    data_models = pd.DataFrame(
        {
            "model_name": model_strings,
            "mse_values": [sqrt(i) for i in mse_values_models],
        }
    )

    print(data_models.sort_values(by="mse_values"))

    fig = px.bar(
        data_models,
        x="model_name",
        y="mse_values",
        title="MSE values for different models",
        labels={"x": "Model", "y": "Mean Error"},
        text=data_models["mse_values"].round(3),
    )
    fig.update_traces(textposition="auto")

    fig.write_image(f"{PWD}/figs/MSE_hyperparam_models_V2.png")

    logger.info("Done training hyperparameter models!")

    model_dict = dict(zip(model_strings, clf_vals))

    return model_dict


def train_best_model(split_dict: dict, test_data: bool) -> None:
    """
    Trains the model that performed best on validation/test data
    """

    if test_data:
        X_chosen = split_dict["x_test"]
        y_chosen = split_dict["y_test"]
    else:
        X_chosen = split_dict["x_val"]
        y_chosen = split_dict["y_val"]

    X_train = split_dict["x_train"]
    y_train = split_dict["y_train"]

    # BEST MODEL:
    best_model = RandomForestRegressor(n_estimators=250, random_state=RANDOM_STATE)

    best_model.fit(X_train, y_train)

    y_test_predicted = best_model.predict(X_chosen)

    test_mse = mean_squared_error(y_chosen, y_test_predicted)
    test_rmse = sqrt(test_mse)

    print(f"Model for test data = {test_data}")
    print("MSE:", test_mse)
    print("RMSE:", test_rmse)

    importance_df = pd.DataFrame(
        {"Feature": X_train.columns, "Importance": best_model.feature_importances_}
    )

    print(importance_df.sort_values(by="Importance", ascending=False))
