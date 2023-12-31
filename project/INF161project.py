import json
import os
import time
from datetime import datetime
from io import StringIO
from math import sqrt
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.stats import spearmanr
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import KNNImputer
from sklearn.linear_model import ElasticNet, Lasso
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

# get current filepath to use when opening/saving files
PWD = Path().absolute()
DEBUG = False
GRAPHING = False
TRAIN_MANY = False
FINAL_RUN = False
RANDOM_STATE = 2


def train_models(split_dict: dict) -> None:
    """
    Input:
        split_dict : split_dict containing x/y train/val/test
    Trains a variety of models on training data, and checks their MSE on validation data
    """

    # grab train and validation data
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

    # intilaize
    model_strings = []
    mse_values_models = []
    clf_vals = []

    # loop over models, train and add values to list
    for mod in models:
        name = str(mod["model_type"].__name__)[0:8]
        settings = mod["settings"]

        print(f"MODELS : Training model type: {name}_{settings}")
        clf = mod["model_type"](
            **mod["settings"]  # henter ut settings her med unpacking
        )
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

    # sorter etter mse
    data_models.sort_values(by="mse_values")

    fig = px.bar(
        data_models,
        x="model_name",
        y="mse_values",
        title="MSE values for different models",
        labels={"x": "Model", "y": "Mean Error"},
        text=data_models["mse_values"].round(3),
    )
    fig.update_layout(
        autosize=False,
        width=1200,
        height=1200,
    )
    fig.update_traces(textposition="auto")

    fig.write_image(f"{PWD}/figs/MANYMODELS_MSE.png")

    print("MODELS : Done training a variety of models!")


def find_hyper_param(split_dict: dict) -> None:
    """
    Input:
        split_dict : split_dict containing x/y train/val/test
    Trains RandomForestRegressor with multiple hyperparameters on training data, finds its MSE on validation data
    """

    X_train = split_dict["x_train"]
    y_train = split_dict["y_train"]
    X_val = split_dict["x_val"]
    y_val = split_dict["y_val"]

    models = []

    for i in range(1, 252, 50):
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
        name = str(mod["model_type"].__name__)[0:8]
        settings = mod["settings"]

        print(f"MODELS : Training model type: {name}_{settings}")
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

    # sort model by mse
    data_models.sort_values(by="mse_values")

    print(data_models)

    fig = px.bar(
        data_models,
        x="model_name",
        y="mse_values",
        title="MSE values for RandomForestRegressor",
        labels={"x": "Model", "y": "Mean Error"},
        text=data_models["mse_values"].round(3),
    )

    fig.update_traces(textposition="auto")

    fig.write_image(f"{PWD}/figs/MSE_hyperparam_models_V3.png")

    print("MODELS : Done training hyperparameter models!")


def find_hyper_param_further(split_dict: dict) -> None:
    """
    Input:
        split_dict : split_dict containing x/y train/val/test
    Trains a single model (testing multiple hyperparameters) on test data, finds its MSE on validation data
    """

    X_train = split_dict["x_train"]
    y_train = split_dict["y_train"]
    X_val = split_dict["x_val"]
    y_val = split_dict["y_val"]

    models = []

    for i in range(151, 252, 30):
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
        name = str(mod["model_type"].__name__)[0:8]
        settings = mod["settings"]

        print(f"MODELS : Training model type: {name}_{settings}")
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

    # sort models by mse
    data_models.sort_values(by="mse_values")

    print(data_models)

    fig = px.bar(
        data_models,
        x="model_name",
        y="mse_values",
        title="MSE values for RandomForestRegressor",
        labels={"x": "Model", "y": "Mean Error"},
        text=data_models["mse_values"].round(3),
    )

    fig.update_traces(textposition="auto")

    fig.write_image(f"{PWD}/figs/MSE_hyperparam_models_further.png")

    print("MODELS : Done training hyperparameter models even further!")

    return


def train_best_model(split_dict: dict, test_data: bool) -> None:
    """
    Trains the model that performed (RandomForestRegressor) best on validation/test data
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
    best_model = RandomForestRegressor(n_estimators=181, random_state=RANDOM_STATE)

    best_model.fit(X_train, y_train)

    y_test_predicted = best_model.predict(X_chosen)

    test_mse = mean_squared_error(y_chosen, y_test_predicted)
    test_rmse = sqrt(test_mse)

    print(f"MODELS : Model for test data = {test_data}")
    print("MSE:", test_mse)
    print("RMSE:", test_rmse)

    importance_df = pd.DataFrame(
        {"Feature": X_train.columns, "Importance": best_model.feature_importances_}
    )

    print(importance_df.sort_values(by="Importance", ascending=False))


def graph_hour_variance(df: pd.DataFrame) -> None:
    """
    Given a dataframe with its index as date in the format of "2023-01-01 14:00:00",
    this function graphs the variance of traffic values in "Total_trafikk" for each hour.
    The variance is calculated from the total traffic values for each hour.
    """

    df.index = pd.to_datetime(df.index)
    df["hour"] = df.index.hour

    # find hourly variance
    hourly_variance = df.groupby("hour")["Total_trafikk"].var()

    plt.figure(figsize=[10, 5])
    plt.bar(hourly_variance.index, hourly_variance.values)
    plt.xlabel("Hour of the Day")
    plt.ylabel("Variance in Total Traffic")
    plt.suptitle("Variance in Total Traffic by Hour of the Day")
    plt.title("(After removing traffic in the 99th quantile)")
    plt.xticks(range(0, 24))
    plt.savefig(f"{PWD}/figs/hour_variance_99")
    plt.clf()


def graph_hour_diff(df: pd.DataFrame) -> None:
    """
    Given a dataframe with its index as date in the format of "2023-01-01 14:00:00",
    this function calculates and graphs the difference between the maximum and minimum
    traffic values in "Total_trafikk" for each hour.
    """

    df["hour"] = df.index.hour
    df_grouped = df.groupby("hour")["Total_trafikk"]
    hourly_max = df_grouped.max()
    hourly_min = df_grouped.min()

    # Calculate the difference
    hourly_diff = hourly_max - hourly_min

    # Plot the difference
    plt.figure(figsize=[10, 5])
    plt.bar(hourly_diff.index, hourly_diff.values)
    plt.xlabel("Hour of the Day")
    plt.ylabel("Difference in Total Traffic")
    plt.title("Difference in Total Traffic by Hour of the Day")
    plt.xticks(range(0, 24))
    plt.savefig(f"{PWD}/figs/traffic_diff_perhour")
    plt.clf()


def graph_total_traffic_overtime(df: pd.DataFrame, VERSION: str) -> None:
    """
    Given a dataframe with its index as date in the format of "2023-01-01 14:00:00",
    Graphs total traffic over time. Traffic is found in the "Total_trafikk" col.

    VERSION is a string to help indicate if graphing is pre or post processing
    """
    plt.clf()
    plt.figure(figsize=(15, 7))
    plt.plot(
        df.index,
        df["Total_trafikk"],
        label="Total Traffic",
    )

    plt.xlabel("Time")
    plt.ylabel("Traffic")
    plt.suptitle(f"Time vs Traffic_{VERSION}")

    plt.grid(True)
    plt.legend()
    plt.savefig(f"{PWD}/figs/timeVStraffic_{VERSION}.png")


def graph_weekly_amounts(df: pd.DataFrame) -> None:
    """
    Given a dataframe with its index as date in the format of "2023-01-01 14:00:00",
    Graphs average traffic per day. Traffic is found in the "Total_trafikk" col.
    """

    days = [
        "d_Monday",
        "d_Tuesday",
        "d_Wednesday",
        "d_Thursday",
        "d_Friday",
        "d_Saturday",
        "d_Sunday",
    ]

    avg_traffic = [df[df[day] == 1]["Total_trafikk"].mean() for day in days]

    avg_traffic_df = pd.DataFrame({"Day": days, "Average_Traffic": avg_traffic})

    fig = px.bar(
        avg_traffic_df,
        x="Day",
        y="Average_Traffic",
        title="Average Traffic per Day of the Week",
    )
    fig.write_image(f"{PWD}/figs/weekly_traffic.png")


def graph_monthly_amounts(df: pd.DataFrame):
    months = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ]

    avg_traffic_monthly = df.groupby("month")["Total_trafikk"].mean()

    avg_traffic_monthly_df = pd.DataFrame(
        {"Month": months, "Average_Traffic": avg_traffic_monthly}
    )

    fig = px.bar(
        avg_traffic_monthly_df,
        x="Month",
        y="Average_Traffic",
        title="Average Traffic per Month",
    )
    fig.write_image(f"{PWD}/figs/monthly_traffic.png")


def graph_a_vs_b(
    titletext: str, df: pd.DataFrame, a: str, b: str, alabel: str, blabel: str
) -> None:
    """
    General function to plot two items in a dataframe against eachother

    Also calculates spearmann and pearson correlation between the two values

    Inputs:
        titletext: the title of the plot
        df: df contaning the two items needing to be graphed
        a: item 1 to be graphed
        b: item 2 to be graphed
        alabel: label to be used for this variable in the graph
        blabel: label to be used for this variable in the graph
    """

    # see limits on data
    print(f"GRAPHING : working on graphing '{a} vs {b}'")
    print(f"GRAPHING : {a} looks like :")
    print(f"GRAPHING : max {a} is {max(df[a])}")
    print(f"GRAPHING : min {a} is {min(df[a])}")

    # Set limits on x - axis to limits + 5 in order to see range of data
    start_time = time.time()
    plt.xlim([min(df[a]) - 5, max(df[a]) + 5])  # x axis limits
    plt.figure(figsize=(15, 7))
    plt.bar(df[a], df[b])

    plt.xlabel(f"{a} ({alabel})")
    plt.ylabel(f"{b} ({blabel})")
    plt.suptitle(f"{a} vs {b} ")

    # Calculate spearmann/pearson correlation to see if trends observed visually also can be seen statistically
    pear = round(pearson_r_corr(df[a], df[b]), 4)
    spear = round(spearman_rho_corr(df[a], df[b]), 4)

    plt.title(f"""pearson_corr = {pear} spearmann_corr = {spear} {titletext}""")
    plt.grid(True)
    plt.savefig(f"{PWD}/figs/{a}VS{b}_{titletext}")

    # :warning: clear fig is very important as not clearing will cause many figs to be created ontop of eachother
    plt.clf()
    print(f"GRAPHING : saved fig '{a} VS {b}' in figs")
    print(f"GRAPHING : --- Graph took: {round(time.time() - start_time,2)} seconds ---")


def pearson_r_corr(a: float, b: float) -> float:
    """
    Helper function to generate pearson correlation
    """
    corr = np.corrcoef(a, b)[0, 1]
    return corr


def spearman_rho_corr(a: float, b: float) -> float:
    """
    Helper function to generate spearmann corr
    """
    corr, _ = spearmanr(a, b)
    return corr


def create_df_matrix(titletext: str, df: pd.DataFrame) -> None:
    """
    Function to create a covariance and correlation matrix for values in a dataframe

    Inputs:
        titletext: Text for graph
        df : the dataframe to create a covarience and correlation matrix from
    """

    # drop uneeded cols:
    df = df.drop(
        labels=[
            "d_Monday",
            "d_Tuesday",
            "d_Wednesday",
            "d_Thursday",
            "d_Friday",
            "d_Saturday",
            "d_Sunday",
            "public_holiday",
            "raining",
            "summer",
            "winter",
            "rush_hour",
            "sleeptime",
            "weekend",
            "month",
            "weekend",
            "hour",
            "Vindretning_radians",
        ],
        axis=1,
        inplace=False,
        errors="ignore",
    )

    # calculate the covariance matrix
    cov_matrix = df.cov()

    # normalizing values between 0 and 1
    cov_matrix_normalized = (cov_matrix - cov_matrix.min().min()) / (
        cov_matrix.max().max() - cov_matrix.min().min()
    )

    plt.figure(figsize=(16, 16))
    sns.heatmap(
        cov_matrix_normalized, annot=True, cmap="RdBu", vmin=-1, vmax=1, center=0
    )
    plt.title(f"Covariance Matrix Heatmap {titletext}")
    plt.savefig(f"{PWD}/figs/covv_matrix_{titletext}.png")

    plt.clf()

    # calculate the correlation matrix
    corr_matrix = df.corr()

    plt.figure(figsize=(16, 16))
    sns.heatmap(corr_matrix, annot=True, cmap="RdBu", vmin=-1, vmax=1, center=0)
    plt.title(f"Correlation Matrix Heatmap {titletext}")
    plt.savefig(f"{PWD}/figs/corr_matrix_{titletext}.png")


def graph_all_models(main_df: pd.DataFrame, pre_change: bool) -> None:
    """
    Large wrapper function to call many other graphing functions.

    Inputs:
        main_df: dataframe to perform graphing on
        pre_change boolean describing if graphing is occouring pre/post processing
    """

    if pre_change:
        titletext = "PRE_CHANGES"
    else:
        titletext = "POST_CHANGES"

    print(f"GRAPHING : Graphing all graphs... {titletext}")

    graph_total_traffic_overtime(main_df, VERSION=titletext)

    create_df_matrix(titletext, main_df)
    graph_a_vs_b(
        titletext,
        main_df,
        "Globalstraling",
        "Total_trafikk",
        "stråling",
        "antall sykler",
    )
    graph_a_vs_b(
        titletext, main_df, "Solskinstid", "Total_trafikk", "solskinn", "antall sykler"
    )
    graph_a_vs_b(
        titletext,
        main_df,
        "Lufttemperatur",
        "Total_trafikk",
        "grader celcius",
        "antall sykler",
    )

    if not pre_change:
        graph_a_vs_b(
            titletext,
            main_df,
            "Vindretning_x",
            "Total_trafikk",
            "Grader",
            "antall sykler",
        )
        graph_a_vs_b(
            titletext,
            main_df,
            "Vindretning_y",
            "Total_trafikk",
            "Grader",
            "antall sykler",
        )

    if pre_change:
        graph_a_vs_b(
            titletext,
            main_df,
            "Vindretning",
            "Total_trafikk",
            "Grader",
            "antall sykler",
        )

        graph_a_vs_b(titletext, main_df, "Vindstyrke", "Vindkast", "Styrke", "Kast")

        graph_a_vs_b(
            titletext, main_df, "Vindstyrke", "Total_trafikk", "Styrke", "Sykler"
        )

    graph_a_vs_b(
        titletext, main_df, "Lufttrykk", "Total_trafikk", "hPa", "antall sykler"
    )
    graph_a_vs_b(
        titletext, main_df, "Vindkast", "Total_trafikk", "m/s", "antall sykler"
    )

    print("GRAPHING : Finished graphing!")


def treat_florida_files(filename: str) -> pd.DataFrame:
    """
    Input:
        filename: filename of a florida weather file

    Process:
        set date to index
        change from 1 hour having 6 values, to one hour having the mean of those 6 values
        drop date and time coloumns which are now represented in the index

    Output:
        a dataframe of the csv file
    """

    df = pd.read_csv(filename, delimiter=",")

    # format date-data to be uniform, will help match data with traffic later
    df["DateFormatted"] = df.apply(
        lambda row: datetime.strptime(row["Dato"] + row["Tid"], "%Y-%m-%d%H:%M"), axis=1
    )

    # drop uneeded coloums
    df = df.drop(columns=["Dato", "Tid"])

    # change date to index, in order to
    df.set_index("DateFormatted", inplace=True)

    # combine all 6 values for a given hour into its mean
    df = df.resample("H").mean()

    return df


def treat_trafikk_files(filename: str) -> pd.DataFrame:
    """
    Input:
        filename: filename of a traffic data file

    Output:
        a dataframe of the csv file
    """

    # read file as string
    with open(filename, "r") as f:
        my_csv_text = f.read()

    # replace | with ; to get uniform delimiter, and open to StringIO to be read by pandas
    csvStringIO = StringIO(my_csv_text.replace("|", ";"))

    # now that delimiter is uniform, file can be handled
    df = pd.read_csv(csvStringIO, delimiter=";")

    # change to a uniform date -> see # Issues in README
    df["DateFormatted"] = df.apply(
        lambda row: datetime.strptime(row["Fra"], "%Y-%m-%dT%H:%M%z").strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
        axis=1,
    )

    # replace '-' with NaN and convert column into numeric
    df["Trafikkmengde"] = df["Trafikkmengde"].replace("-", np.nan).astype(float)

    # replace " " in 'Felt' values with "_" to avoid errors
    df["Felt"] = df["Felt"].str.replace(" ", "_")  # todo try without

    # dropping cols - see README on "Dropped coloumns"
    df = df.drop(
        columns=[
            "Trafikkregistreringspunkt",
            "Navn",
            "Vegreferanse",
            "Fra",
            "Til",
            "Dato",
            "Fra tidspunkt",
            "Til tidspunkt",
            "Dekningsgrad (%)",
            "Antall timer total",
            "Antall timer inkludert",
            "Antall timer ugyldig",
            "Ikke gyldig lengde",
            "Lengdekvalitetsgrad (%)",
            "< 5,6m",
            ">= 5,6m",
            "5,6m - 7,6m",
            "7,6m - 12,5m",
            "12,5m - 16,0m",
            ">= 16,0m",
            "16,0m - 24,0m",
            ">= 24,0m",
        ]
    )

    # drop all rows where the coloum "Felt" != "Totalt i retning Danmarksplass" or "Totalt i retning Florida"
    df = df[
        df["Felt"].isin(["Totalt_i_retning_Danmarksplass", "Totalt_i_retning_Florida"])
    ]

    # create empty dataframe with 'DateFormatted' as index
    result_df = pd.DataFrame(index=df["DateFormatted"].unique())

    # What this is essentially doing is transforming "Totalt_i_retning_Danmarksplass" and "Totalt_i_retning_Florida" from being
    # values in a coloumn called "Felt", to being two columns. Their values for the hours are just the same
    # The "felt" coloumn is removed and has been transformed into two different "Felt" coloumns.

    # loop through each unique felt value in the df, in this case "Totalt_i_retning_Danmarksplass" and "Totalt_i_retning_Florida"
    for felt in df["Felt"].unique():

        # filter the dataframe so where the coloumn "felt" = the felt we have chosen for this iteration
        felt_df = df[df["Felt"] == felt]

        # remove felt, since we are making new cols
        felt_df = felt_df.drop(columns="Felt")

        # the name should be different for the two felt
        felt_df = felt_df.add_suffix(f"_{felt}")
        felt_df = felt_df.set_index(f"DateFormatted_{felt}")

        # put the filtered dataframe onto the result one.
        # after doing this for both they should line up nicely
        result_df = result_df.join(felt_df)

    return result_df


def feauture_engineer(df: pd.DataFrame, data2023: bool) -> pd.DataFrame:
    """
    Input: A dataframe containing traffic and weather data with DateFormatted as the index

    Adds:

    hour: 0-24
    day_in_week: (each day has their own col)
    month: 1-12
    weekend: 0/1
    public holiday: 0/1
    raining: 0/1
    summer: 0/1
    winter: 0/1
    rush_hour: 0/1
    sleeptime: : 0/1
    Vindretning x/y : 0-1

    Returns: df with more features
    """

    # BASIC DATE FEATURES
    # hour as own coloumn 0-23
    df["hour"] = df.index.hour

    # Instead of "day_in_week" being a num 0-6, add 7 coloumns to the dataframe, monday, tuesday .. etc
    # Have the value be 0 or 1, 0 if it is not that day of the week, 1 if it is

    day_week_dict = {
        0: "Monday",
        1: "Tuesday",
        2: "Wednesday",
        3: "Thursday",
        4: "Friday",
        5: "Saturday",
        6: "Sunday",
    }

    df["d"] = df.index.weekday.map(day_week_dict)

    # make each day their own coloumn
    df = pd.get_dummies(df, columns=["d"])

    complete_days = [
        "d_Monday",
        "d_Tuesday",
        "d_Wednesday",
        "d_Thursday",
        "d_Friday",
        "d_Saturday",
        "d_Sunday",
    ]

    # loop through days
    for day in complete_days:
        # if the day is not a column in df, add it with a value of 0
        if not day in df.columns:
            df[day] = 0

    # add month
    df["month"] = df.index.month

    # add weekend
    df["weekend"] = (df.index.weekday >= 5).astype(int)

    # MORE ADVANCED FEATURES

    # add public holidays
    holidays = [
        # jul osv
        "12-24",
        "12-25",
        "01-01",
        # påske
        "04-06",
        "04-07",
        "04-08",
        "04-09",
        "04-10",
        # labour day/17 mai
        "05-01",
        "05-17",
        "05-18",
    ]

    # add public holiday
    df["public_holiday"] = df.index.strftime("%m-%d").isin(holidays).astype(int)

    # add coloumn for rain if air pressure is higher than 1050 see README
    df["raining"] = df["Lufttrykk"] <= 996

    # add seasons
    df["summer"] = (df["month"] > 5) & (df["month"] < 8)
    df["winter"] = (df["month"] >= 10) | (df["month"] <= 2)

    # add when we expect a lot of traffic
    df["rush_hour"] = (df["hour"].between(7, 9)) | (df["hour"].between(15, 17))

    # add when we do not expect a lot of traffic
    df["sleeptime"] = (df["hour"] >= 22) | (df["hour"] < 6)

    # df["Vindretning"] is full of values 0-360, transform these to points on a circle
    df["Vindretning_radians"] = np.radians(df["Vindretning"])
    df["Vindretning_x"] = np.cos(df["Vindretning_radians"])
    df["Vindretning_y"] = np.sin(df["Vindretning_radians"])

    # we cant train where there are no traffic values
    if not data2023:  # dont drop values in 2023 data!
        df = df.dropna(subset=["Total_trafikk"])

    # change all values of TRUE in all rows to 1 and FALSE to 0
    # models need NUMERIC data
    df = df.replace({True: 1, False: 0})

    return df


def merge_frames(frames: list) -> (pd.DataFrame, pd.DataFrame):
    """
    Given a list of dataframes, merges the frames to one large dataframe, given that the
    index is a date, and the same across all dataframes

    """

    # grab first df
    df_final = frames[0]

    # index to date
    df_final.index = pd.to_datetime(df_final.index)

    for frame in frames[1:]:
        frame.index = pd.to_datetime(frame.index)

        # since dataframes have the same index, -> they can be merged
        df_final = df_final.merge(frame, how="outer", left_index=True, right_index=True)

    df_final = df_final.sort_index()

    df_2023 = df_2023 = df_final.loc["2023-01-01":"2023-12-31"]
    # get where index is between 2023-01-01 00:00:00 and 2023-12-31 00:00:00 to save.

    df_final = df_final.dropna(subset=["Trafikkmengde_Totalt_i_retning_Florida"])

    # finding means of values lead to floating point errors, round to fix these
    df_final = df_final.apply(pd.to_numeric, errors="ignore").round(30)

    # combine the two traffic cols to one total trafikk col!
    df_final["Total_trafikk"] = (
        df_final["Trafikkmengde_Totalt_i_retning_Florida"]
        + df_final["Trafikkmengde_Totalt_i_retning_Danmarksplass"]
    )

    df_final = df_final.drop(
        labels=[
            "Trafikkmengde_Totalt_i_retning_Florida",
            "Trafikkmengde_Totalt_i_retning_Danmarksplass",
        ],
        axis=1,
        inplace=False,
    )

    return df_2023, df_final


def trim_transform_outliers(df: pd.DataFrame, data2023: bool) -> pd.DataFrame:
    """
    Given a dataframe, trims values in the dataframe that are considered abnormal.

    What values are considered abnormal are covered in the README under "Dropped values"
    """
    # debug dict to look at lengths
    length_dict = {"before": len(df)}
    length_dict["afterGlobal"] = len(df)

    # Transform malformed data to NaN.

    # df["Globalstraling"] values above 1000 are set to NaN
    df["Globalstraling"] = np.where(
        (df["Globalstraling"] >= 1000),
        np.nan,
        df["Globalstraling"],
    )

    # df["Solskinstid"] values above 10.01 are set to NaN
    df["Solskinstid"] = np.where(df["Solskinstid"] >= 10.01, np.nan, df["Solskinstid"])
    length_dict["afterSolskinn"] = len(df)

    # df["Lufttemperatur"] values above 50 are set to NaN
    df["Lufttemperatur"] = np.where(
        df["Lufttemperatur"] >= 50, np.nan, df["Lufttemperatur"]
    )
    length_dict["afterLufttemp"] = len(df)

    # df["Lufttrykk"] values above 1050 are set to NaN
    df["Lufttrykk"] = np.where(df["Lufttrykk"] >= 1050, np.nan, df["Lufttrykk"])
    length_dict["afterLufttrykk"] = len(df)

    # df["Vindkast"] values above 65 are set to NaN
    df["Vindkast"] = np.where(df["Vindkast"] >= 65, np.nan, df["Vindkast"])
    length_dict["afterVindkast"] = len(df)

    # df["Vindretning"] values above 360 are set to NaN
    df["Vindretning"] = np.where(df["Vindretning"] >= 361, np.nan, df["Vindretning"])
    length_dict["afterVindretning"] = len(df)

    # df["Vindstyrke"] values above 1000 and below 0 are set to NaN
    df["Vindstyrke"] = np.where(df["Vindstyrke"] < 0, np.nan, df["Vindstyrke"])
    df["Vindstyrke"] = np.where(df["Vindstyrke"] >= 1000, np.nan, df["Vindstyrke"])
    length_dict["afterVindstyrke"] = len(df)

    # replace outliers (this should be fixed above, but this is just in case)
    df = df.replace(99999, np.nan)

    # observe NaN
    num_nan = df.isna().sum()
    print(f"PARSING : Number of NaNs in each column:\n{num_nan}")

    if not data2023:
        # "Total_trafikk" column, will not used for imputation -> keep it for later
        total_traffic_series = df["Total_trafikk"]

        # "Total_trafikk" column, will not used for imputation -> remove it from df
        df_no_traffic = df.drop(columns=["Total_trafikk"])

    if data2023:
        # 2023 data does not have
        df_no_traffic = df

    # Drop "Relativ luftfuktighet" as this data only exists in 2022 and 2023.
    # errors="ignore" since pandas complains when dropping data from dataframes where it does not exist

    df_no_traffic = df_no_traffic.drop(
        columns=["Relativ luftfuktighet"], errors="ignore"
    )

    # n_neighbors = 20 is best -> see report
    imputer = KNNImputer(n_neighbors=20, weights="distance")
    df_imputed = imputer.fit_transform(df_no_traffic)

    df_fixed = pd.DataFrame(
        df_imputed, columns=df_no_traffic.columns, index=df_no_traffic.index
    )

    if not data2023:
        df_fixed = pd.concat([df_fixed, total_traffic_series], axis=1)

    return df_fixed


def normalize_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a dataframe, normalizes certain values to a 0-1 scale

    Normalized values are covered in the README under "Normalized values"
    """

    # These were ideas that were considered but dropped. This was kept in for learning purposes.

    # scaler = MinMaxScaler()
    # df[["Globalstraling", "Lufttrykk", "Solskinstid",]] = scaler.fit_transform(
    #     df[
    #         [
    #             "Globalstraling",
    #             "Lufttrykk",
    #             "Solskinstid",
    #         ]
    #     ]
    # )

    # df["Vindkast"] = df["Vindkast"]**2

    print(f"PARSING : Values pre removal of outliers: {len(df)}")

    quant = df["Total_trafikk"].quantile(0.99)
    df = df[df["Total_trafikk"] <= quant]

    print(f"PARSING : Values post removal of outliers: {len(df)}")

    return df


def drop_uneeded_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a dataframe, drops values deemed not needed, as these may just provide noise for the model,
    or their values are already represented as another coloumn

    Dropped values are covered in the README under "Dropped coloumns"
    """

    df.drop(["Vindretning", "Vindretning_radians", "Vindstyrke"], axis=1, inplace=True)

    return df


def train_test_split_process(
    df: pd.DataFrame,
) -> (dict, pd.DataFrame, pd.DataFrame, pd.DataFrame):  # fix
    """
    Given a df, data is split into training, test and validation.

    Returns:

    split_dict : dict containing x_train, y_train, etc... for all x,y vals.
    training_df : reconstructed dataframe containing only training data
    test_df : reconstructed dataframe containing only test data
    validation_df : reconstructed dataframe containing only validation data
    """

    y = df["Total_trafikk"]
    x = df.drop(["Total_trafikk"], axis=1)

    # transform 70% to training data
    x_train, x_val, y_train, y_val = train_test_split(
        x, y, shuffle=False, test_size=0.3
    )

    # split 30% which was validation into 15% val and 15% test
    x_val, x_test, y_val, y_test = train_test_split(
        x_val, y_val, shuffle=False, test_size=0.5
    )

    # data exploration is only supposed to look at training data -> use x_train/y_train
    test_df = x_test.merge(y_test, how="outer", left_index=True, right_index=True)
    validation_df = x_val.merge(y_val, how="outer", left_index=True, right_index=True)
    training_df = x_train.merge(y_train, how="outer", left_index=True, right_index=True)

    split_dict = {
        "x_train": x_train,
        "y_train": y_train,
        "x_test": x_test,
        "y_test": y_test,
        "x_val": x_val,
        "y_val": y_val,
    }

    return split_dict, training_df, test_df, validation_df


def treat_2023_file(df: pd.DataFrame, model: RandomForestRegressor) -> pd.DataFrame:
    """
    A 2023 file handler, to fill in missing values given weather data

    Inputs:
        df: A dataframe contaning 2023 data
        model: the model to use to predict cycle trafikk
    Returns:
        A dataframe much like the input, with the cycle traffic values filled in.

    """
    df = df.drop(
        columns=[
            "Trafikkmengde_Totalt_i_retning_Danmarksplass",
            "Trafikkmengde_Totalt_i_retning_Florida",
        ]
    )

    df_fixed = trim_transform_outliers(df, True)

    # add important features to help the model
    df_final = feauture_engineer(df_fixed, True)
    print("PARSING : Features engineered")

    # drop coloumns which are not needed (noise)
    df_final = drop_uneeded_cols(df_final)
    print("PARSING : Uneeded cols dropped")

    try:
        df_final["Total_trafikk"] = model.predict(df_final)
    except ValueError as e:
        print(f"WARNING: MODEL PREDICTION ERROR {e}")

    # convert time, date and prediction to wanted format
    df_final["Dato"] = pd.to_datetime(df_final.index).date
    df_final["Tid"] = pd.to_datetime(df_final.index).hour

    df_final["Prediksjon"] = df_final["Total_trafikk"]

    new_df = df_final[["Dato", "Tid", "Prediksjon"]].copy()

    # make predictions to ints as float number of cyclists makes no sense.
    new_df["Prediksjon"] = new_df["Prediksjon"].astype(int)

    new_df.reset_index()

    new_df.to_csv(f"{PWD}/out/predictions.csv")

    return df_final


def main():
    print("INFO : Starting parsing ... ")
    # loop over files in local directory
    directory = f"{str(PWD)}/raw_data"

    # multiple florida files will all be converted to df's, placed in this list, and concacted
    florida_df_list = []

    # parse files
    for filename in os.scandir(directory):
        if "Florida" in str(filename):
            florida_df = treat_florida_files(f"{str(directory)}/{filename.name}")
            florida_df_list.append(florida_df)

        if "trafikkdata" in str(filename):
            trafikk_df = treat_trafikk_files(f"{str(directory)}/{filename.name}")

    print("INFO : All files parsed!")

    # concat all the florida df's to one
    big_florida_df = pd.concat(florida_df_list, axis=0)
    print("INFO : Florida files concacted")

    # merge the dataframes
    df_2023, df_final = merge_frames([big_florida_df, trafikk_df])
    print("INFO : All files merged over")

    # divide data into training,test and validation
    split_dict_pre, training_df, test_df, validation_df = train_test_split_process(
        df_final
    )
    print("INFO : Data divided into training,validation and test")

    # average traffic per year to observe
    average_traffic_per_year = (
        training_df["Total_trafikk"].groupby(training_df.index.year).mean()
    )

    print("INFO : Average traffic per year (for training data):")
    print(average_traffic_per_year)

    if GRAPHING:

        graph_all_models(training_df, pre_change=True)
        print("INFO : Graphed all models PRE-CHANGE")

    # make dataframe dict to treat them differently
    dataframes_pre = {
        "training_df": training_df,
        "validation_df": validation_df,
        "test_df": test_df,
    }

    # loop through each data frame and transform the data
    # this is done seperatley for each df (test/train/validation)
    # so that the training data is never influced by the other data
    dataframes_post = {}

    for name, df_transforming in dataframes_pre.items():
        print(
            f"INFO : Applying KNN imputer on missing data, and removing outliers.. for {name}"
        )
        print("INFO : This could take a while...")

        # transform NaN and outliers to usable data
        df_transforming = trim_transform_outliers(df_transforming, False)
        print(f"INFO : Outliers trimmed for {name}")

        # add features to help the model
        df_transforming = feauture_engineer(df_transforming, False)
        print(f"INFO : Features engineered for {name}")

        # normalize data outliers
        df_transforming = normalize_data(df_transforming)
        print(f"INFO : Coloumns normalized for {name}")

        if GRAPHING:

            if name == "training_df":

                # graph Vindkast vs
                graph_a_vs_b(
                    "POST_CHANGES",
                    df_transforming,
                    "Vindstyrke",
                    "Vindkast",
                    "Styrke",
                    "Kast",
                )
                graph_a_vs_b(
                    "POST_CHANGES",
                    df_transforming,
                    "Vindstyrke",
                    "Total_trafikk",
                    "Retning",
                    "Sykler",
                )
                graph_a_vs_b(
                    "POST_CHANGES",
                    df_transforming,
                    "Vindretning",
                    "Total_trafikk",
                    "Retning",
                    "Sykler",
                )

        # drop coloumns which are not needed or redundant
        df_transforming = drop_uneeded_cols(df_transforming)
        print(f"INFO : Uneeded cols dropped for {name}")

        # save dataframes to use later
        dataframes_post[name] = df_transforming

    training_df = dataframes_post["training_df"]
    test_df = dataframes_post["test_df"]
    validation_df = dataframes_post["validation_df"]

    # save training data to csv to have a look
    training_df.to_csv(f"{PWD}/out/main_training_data.csv")
    print("INFO : Data saved to CSV")

    if GRAPHING:
        # Graph post data processing to visualize and analyze data
        print("GRAPHING : GRAPHING HOUR DIFF")
        graph_hour_diff(training_df)
        print("GRAPHING : GRAPHING WEEKLY AMOUNTS")
        graph_weekly_amounts(training_df)
        print("GRAPHING : GRAPHING MONTHLY AMOUNTS")
        graph_monthly_amounts(training_df)
        print("GRAPHING : GRAPHING HOUR VARIANCE")
        graph_hour_variance(training_df)

        graph_all_models(training_df, pre_change=False)
        print("INFO : Graph all models POSTCHANGE")

    split_dict_post = {
        "y_train": training_df["Total_trafikk"],
        "x_train": training_df.drop(["Total_trafikk"], axis=1),
        "y_val": validation_df["Total_trafikk"],
        "x_val": validation_df.drop(["Total_trafikk"], axis=1),
        "y_test": test_df["Total_trafikk"],
        "x_test": test_df.drop(["Total_trafikk"], axis=1),
    }

    # train models
    if TRAIN_MANY:
        train_models(split_dict_post)
        # find hyper params for the best model
        find_hyper_param(split_dict_post)
        find_hyper_param_further(split_dict_post)

    # train the best model on validation data
    print("INFO: Training model on validation data")
    train_best_model(split_dict_post, test_data=False)

    if FINAL_RUN:
        # train best model on test data
        print("INFO: Training model on test data")
        train_best_model(split_dict_post, test_data=True)

        print("INFO : Treating 2023 files")

        # use the best model to get values for 2023
        best_model = RandomForestRegressor(n_estimators=181, random_state=RANDOM_STATE)

        X_train = split_dict_post["x_train"]
        y_train = split_dict_post["y_train"]

        # the best model is used to treat 2023 files.
        best_model.fit(X_train, y_train)
        df_with_values = treat_2023_file(df_2023, best_model)

    return split_dict_post, training_df, test_df, validation_df


def create_dirs() -> None:
    """
    Helper function to create directories for saving figs and files
    """
    try:
        os.mkdir("figs")
    except FileExistsError:
        pass

    try:
        os.mkdir("out")
    except FileExistsError:
        pass


if __name__ == "__main__":
    create_dirs()
    split_dict, training_df, test_df, validation_df = main()
    print("INFO : main function complete!")
