from math import sqrt
from pathlib import Path
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC, SVR
from sklearn.gaussian_process import GaussianProcessRegressor
import plotly.express as px

RANDOM_STATE = 2

PWD = Path().absolute()

directory = f"{str(PWD)}/out"

training_df     = pd.read_csv(f"{directory}/main_training_data.csv")
test_df         = pd.read_csv(f"{directory}/main_test_data.csv")
validation_df   = pd.read_csv(f"{directory}/main_validation_data.csv")
# split_dict = json.loads(f"{directory}/split_dict.json")

def train_models(split_dict: dict):

    X_train = split_dict["x_train"]
    y_train = split_dict["y_train"]
    X_test  = split_dict["x_test"]
    y_test  = split_dict["y_test"]
    X_val   = split_dict["x_val"]
    y_val   = split_dict["y_val"]

    # tren forskjellige modeller

    #her prøver jeg på litt fancy greie ved å kun lage settings ett sted.
    #det er kult, men kanskje litt vel overkomplisert

    models = [
        # {"model_type": GaussianProcessRegressor, "settings": {"alpha": 300, "random_state":RANDOM_STATE}},
        # {"model_type": MultinomialNB,"settings": {}},
        {"model_type" : LogisticRegression,"settings": {}},
        {"model_type": Lasso, "settings": {"alpha": 300, "random_state":RANDOM_STATE}},
        {"model_type": RandomForestRegressor, "settings": {"n_estimators":500, "random_state":RANDOM_STATE}}, #hadde n_estimators på 1000, da funker an bra men tar sykt lang tid
        {"model_type": ElasticNet, "settings": {"alpha": 1500, "random_state":RANDOM_STATE}}, 
        {"model_type": SVR, "settings": {"degree":2}},
        {"model_type": SVC, "settings": {}}
    ]

    model_strings = []
    mse_values_models = []
    clf_vals = []

    for mod in models:
        print(f"Training model type: {str(mod['model_type'])}")
        clf = mod["model_type"](**mod["settings"]) #henter ut settings her
        clf.fit(X_train, y_train) 

        y_predicted = clf.predict(X_val)

        #finn mse
        pf_mse = mean_squared_error(y_val, y_predicted, squared=True)

        mse_values_models.append(pf_mse)
        model_strings.append(str(mod["model_type"]))
        clf_vals.append(clf)

    data_models = pd.DataFrame({
        'model_name': model_strings,
        'mse_values': mse_values_models
    })

    fig = px.bar(data_models, x='model_name', y='mse_values', 
                title='MSE values for different models', 
                labels={'x':'Model', 'y':'Mean Squared Error'})


    fig.write_image(f"{PWD}/MSE_models.png")

    data_models = pd.DataFrame({
        'model_name': model_strings,
        'mse_values': [sqrt(i) for i in mse_values_models]
    })

    fig = px.bar(data_models, x='model_name', y='mse_values', 
                title='ME values for different models', 
                labels={'x':'Model', 'y':'Mean Error'})

    fig.write_image(f"{PWD}/ME_models.png")

    print("Done training!")

    model_dict = dict(zip(model_strings,clf_vals))

    return model_dict


def find_accuracy_logloss(split_dict:dict, model_dict:dict):

    X_train = split_dict["x_train"]
    y_train = split_dict["y_train"]
    X_test  = split_dict["x_test"]
    y_test  = split_dict["y_test"]
    X_val   = split_dict["x_val"]
    y_val   = split_dict["y_val"]

    accuracies = [accuracy_score(y_val,model.predict(X_val)) for model in model_dict.values()]
    accuracy_dict = dict(zip(model_dict.keys(),accuracies))

    data_models = pd.DataFrame({
        'model_name': accuracy_dict.keys(),
        'accuracy_values': accuracy_dict.values()
    })

    fig = px.bar(data_models, x='model_name', y='accuracy_values', 
                title='accruacy values for different models', 
                labels={'x':'Model', 'y':'accuracy_values'})

    fig.write_image(f"{PWD}/ACCURACY_models.png")

    #-------------

    logs = [log_loss(y_val,model.predict(X_val)) for model in model_dict.values()]
    accuracy_dict = dict(zip(model_dict.keys(),logs))

    data_models = pd.DataFrame({
        'model_name': accuracy_dict.keys(),
        'log_values': accuracy_dict.values()
    })

    fig = px.bar(data_models, x='model_name', y='log_values', 
                title='accruacy values for different models', 
                labels={'x':'Model', 'y':'log_values'})

    fig.write_image(f"{PWD}/ALOGLOSS_models.png")

    print("Done finding accuracy and log loss!")

    return