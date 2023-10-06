import json
from math import sqrt
from pathlib import Path

import pandas as pd
import plotly.express as px
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
# from sklearn.metrics import accuracy_score, log_loss, mezan_squared_error
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeRegressor

RANDOM_STATE = 2
PWD = Path().absolute()

directory = f"{str(PWD)}/out"

training_df = pd.read_csv(f"{directory}/main_training_data.csv")
test_df = pd.read_csv(f"{directory}/main_test_data.csv")
validation_df = pd.read_csv(f"{directory}/main_validation_data.csv")
# split_dict = json.loads(f"{directory}/split_dict.json")


def train_models(split_dict: dict):

    X_train = split_dict["x_train"]
    y_train = split_dict["y_train"]
    X_test = split_dict["x_test"]
    y_test = split_dict["y_test"]
    X_val = split_dict["x_val"]
    y_val = split_dict["y_val"]

    # tren forskjellige modeller

    # her prøver jeg på litt fancy greie ved å kun lage settings ett sted.
    # det er kult, men kanskje litt vel overkomplisert

    models = [

        {
                    "model_type": DummyRegressor,
            "settings": {},
        },
        # {"model_type": GaussianProcessRegressor, "settings": {"alpha": 300, "random_state":RANDOM_STATE}},
        # {"model_type": MultinomialNB,"settings": {}},
        # {"model_type": LogisticRegression, "settings": {}},
        # {"model_type": Lasso, "settings": {"alpha": 300, "random_state": RANDOM_STATE}},
        # {
        #     "model_type": RandomForestRegressor,
        #     "settings": {"n_estimators": 100, "random_state": RANDOM_STATE},
        # }, 
        # {
        #     "model_type": ElasticNet,
        #     "settings": {"alpha": 1500, "random_state": RANDOM_STATE},
        # },
        # {"model_type": SVR, "settings": {"degree": 2}},
        # {"model_type": SVC, "settings": {}},
        # {"model_type": GradientBoostingRegressor, "settings": {"n_estimators": 100, "learning_rate": 0.1, "random_state": RANDOM_STATE}},
    # {"model_type": XGBRegressor, "settings": {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 5, "random_state": RANDOM_STATE}},

        {
            "model_type": RandomForestRegressor,
            "settings": {"n_estimators": 151, "random_state": RANDOM_STATE},
        },

        {
            "model_type": RandomForestRegressor,
            "settings": {"n_estimators": 50, "random_state": RANDOM_STATE},
        },

        {
            "model_type": RandomForestRegressor,
            "settings": {"n_estimators": 200, "random_state": RANDOM_STATE},
        },
        
        {"model_type": KNeighborsRegressor, "settings": {"n_neighbors": 5}},

        {"model_type": KNeighborsRegressor, "settings": {"n_neighbors": 50}},

        {"model_type": KNeighborsRegressor, "settings": {"n_neighbors": 100}},

        {"model_type":DecisionTreeRegressor,"settings":{"max_depth":12}},

        {"model_type":DecisionTreeRegressor,"settings":{"max_depth":50}},

        {"model_type":DecisionTreeRegressor,"settings":{"max_depth":100}},

       
    ]

    model_strings = []
    mse_values_models = []
    clf_vals = []

    for mod in models:
        print(f"Training model type: {mod['model_type'].__name__}_{mod['settings']}")
        clf = mod["model_type"](**mod["settings"])  # henter ut settings her
        clf.fit(X_train, y_train)

        y_predicted = clf.predict(X_val)

        # finn mse
        pf_mse = mean_squared_error(y_val, y_predicted, squared=True)

        mse_values_models.append(pf_mse)
        model_strings.append(f"{mod['model_type'].__name__}_{mod['settings']}")
        clf_vals.append(clf)


    data_models = pd.DataFrame(
        {
            "model_name": model_strings,
            "mse_values": [sqrt(i) for i in mse_values_models]
        }
    )


    fig = px.bar(
        data_models,
        x="model_name",
        y="mse_values",
        title="MSE values for different models",
        labels={"x": "Model", "y": "Mean Error"},
        text=data_models['mse_values'].round(3)
    )
    fig.update_traces(textposition='auto')

    fig.write_image(f"{PWD}/figs/MSE_models_V9.png")

    print("Done training!")

    model_dict = dict(zip(model_strings, clf_vals))

    return model_dict

def train_models_loop(split_dict: dict):

    X_train = split_dict["x_train"]
    y_train = split_dict["y_train"]
    X_test = split_dict["x_test"]
    y_test = split_dict["y_test"]
    X_val = split_dict["x_val"]
    y_val = split_dict["y_val"]


    models = []

    for i in range(130,251,10):
        model = {
        "model_type": RandomForestRegressor,
        "settings": {"n_estimators": i, "random_state": RANDOM_STATE},
        }
        models.append(model)


    model_strings = []
    mse_values_models = []
    clf_vals = []

    for mod in models:
        print(f"Training model type: {mod['model_type'].__name__}_{mod['settings']}")
        clf = mod["model_type"](**mod["settings"])  # henter ut settings her
        clf.fit(X_train, y_train)

        y_predicted = clf.predict(X_val)

        # finn mse
        pf_mse = mean_squared_error(y_val, y_predicted, squared=True)

        mse_values_models.append(pf_mse)
        model_strings.append(f"{mod['model_type'].__name__}_{mod['settings']}")
        clf_vals.append(clf)


    data_models = pd.DataFrame(
        {
            "model_name": model_strings,
            "mse_values": [sqrt(i) for i in mse_values_models]
        }
    )

    print(data_models.sort_values(by='mse_values'))


    fig = px.bar(
        data_models,
        x="model_name",
        y="mse_values",
        title="MSE values for different models",
        labels={"x": "Model", "y": "Mean Error"},
        text=data_models['mse_values'].round(3)
    )
    fig.update_traces(textposition='auto')

    fig.write_image(f"{PWD}/figs/MSE_models_V9.png")

    print("Done training!")

    model_dict = dict(zip(model_strings, clf_vals))

    return model_dict



def train_best_model(split_dict: dict):
    # best model = RandomForestRegressor

    X_train = split_dict["x_train"]
    y_train = split_dict["y_train"]
    X_test = split_dict["x_test"]
    y_test = split_dict["y_test"]
    X_val = split_dict["x_val"]
    y_val = split_dict["y_val"]
    #151
    best_model = RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE)
    best_model.fit(X_train, y_train)

    y_test_predicted = best_model.predict(X_test)

    test_mse = mean_squared_error(y_test, y_test_predicted)

    test_rmse = sqrt(test_mse)

    print("Test MSE:", test_mse)
    print("Test RMSE:", test_rmse)

    # out_dict[f"i"] = test_rmse

    # bst = sorted(out_dict.items(), key=lambda x:x[0])
    # print(json.dumps(bst,indent=2))

    importance_df = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': best_model.feature_importances_
    })

    print(importance_df.sort_values(by='Importance', ascending=False))






## TODO:

# A few suggestions to improve your code:

# 1. **Hyperparameter tuning**: You can improve your machine learning models by tuning their hyperparameters. For instance, you can adjust the "n_estimators" and "max_depth" parameters in the `RandomForestRegressor`. The optimal parameters often depend on your specific dataset, so you can use approaches like grid search or random search to experiment with different combinations and find the best ones.
#"n_estimators" and "max_depth"

# 2. **Feature importance**: Random forest provides feature importances which can show you which features are most influential in predicting your target variable. This can help you to simplify your model by excluding features that don't contribute much to the prediction. You can get the feature importances from a trained random forest model by accessing its `feature_importances_` attribute.

# 3. **Use other regression models**: You may also consider experimenting with other regression models. For instance, gradient boosting regressors (like XGBoost or LightGBM) often perform very well on various datasets.
#XGBoost or LightGBM

# 4. **Cross-validation**: Use cross-validation for more robust results. K-fold cross-validation splits the data into K subsets and then trains the model K times, each time using a different subset as the test set. This can lead to a more robust estimation of the model's performance.

# 5. **Scaling the data**: Some algorithms, especially those that use a form of gradient descent to optimize their parameters, assume all features are centered around zero and have a similar variance. Features in your dataset like 'Lufttrykk' and 'Globalstraling' may have different scales that can negatively impact the performance of these models. Consider transforming your features to have a mean of 0 and a standard deviation of 1 using techniques such as StandardScaler.
#StandardScaler

# 6. **Handling Imbalanced Data**: If the number of cycles across the bridge is heavily imbalanced in your dataset (as in very few events of people cycling), you can consider techniques such as SMOTE or ADASYN for over-sampling the minority class, or use algorithms that handle imbalance internally, like gradient boosting.

# Remember, while these strategies can potentially improve your results, they can also increase the complexity and running time of your code. Therefore, consider the trade-off between prediction accuracy and time efficiency based on your specific use case.








#-----------------

