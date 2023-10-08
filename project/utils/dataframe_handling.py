import json
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

DEBUG = True


def feauture_engineer(df: pd.DataFrame) -> pd.DataFrame:
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
    traffic: the value of the traffic in a given hour

    Returns: df with more features
    """

    # BASIC DATE FEATURES

    # hour as own coloumn 0-23
    df["hour"] = df.index.hour  # get first two values

    # Instead of "day_in_week" being a num 0-6, add 7 coloumns to the dataframe, monday, tuesday .. etc
    # And have the value being 0/1 , 0 if it is not that day, 1 if it is

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
    df = pd.get_dummies(df, columns=["d"])  # convert to True/False

    # month as own coloum 1-12
    df["month"] = df.index.month

    # MORE ADVANCED FEATURES

    # add weekend
    df["weekend"] = (df.index.weekday >= 5).astype(int)

    # THIS COULD NOT BE DONE - Check the README under "Feature engineering"
    # ------------------
    # add the hour values of the previous row, this can be a good indicator
    # df["Last_Danmarksplass"] = df["Trafikkmengde_Totalt_i_retning_Danmarksplass"].shift(
    # 1
    # )
    # df["Last_Florida"] = df["Trafikkmengde_Totalt_i_retning_Florida"].shift(1)
    # df["Last_total"] = df["Last_Danmarksplass"] + df["Last_Florida"]
    # ------------------

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
    df["sleeptime"] = df["hour"].between(22, 6)

    # df["Vindretning"] is full of values 0-360, transform these to points on a circle
    df["Vindretning_radians"] = np.radians(df["Vindretning"])
    df["Vindretning_x"] = np.cos(df["Vindretning_radians"])
    df["Vindretning_y"] = np.sin(df["Vindretning_radians"])

    # we cant train where there are no traffic values
    df = df.dropna(subset=["Trafikkmengde_Totalt_i_retning_Florida"])

    # combine the two traffic cols
    df["Total_trafikk"] = (
        df["Trafikkmengde_Totalt_i_retning_Florida"]
        + df["Trafikkmengde_Totalt_i_retning_Danmarksplass"]
    )

    df = df.drop(
        labels=[
            "Trafikkmengde_Totalt_i_retning_Florida",
            "Trafikkmengde_Totalt_i_retning_Danmarksplass",
        ],
        axis=1,
        inplace=False,
    )
    # total is found, these two are not needed

    # change all values of TRUE in all rows to 1 and FALSE to 0
    # models need NUMERIC data
    df = df.replace({True: 1, False: 0})

    return df


def merge_frames(frames: list) -> pd.DataFrame:
    """
    Given a list of dataframes, merges the frames to one large dataframe, given that the
    index is a date, and the same across all dataframes

    """

    # first DataFrame
    df_final = frames[0]

    # convert index (date) from string to datetime once only as we'll apply it to other frames
    df_final.index = pd.to_datetime(df_final.index)

    for frame in frames[1:]:
        # convert index from string to datetime
        frame.index = pd.to_datetime(frame.index)

        # since the dataframes have the same index, they can be merged!
        df_final = df_final.merge(frame, how="outer", left_index=True, right_index=True)

    df_final = df_final.sort_index()

    # remove lines where we dont have traffic information, as many missing values can cause model training
    # to overly rely on values which are there often.

    # Drops missing values
    PWD = Path().absolute()
    directory = f"{str(PWD)}/out"
    print("before drop ", len(df_final))
    df_final.to_csv(f"{directory}/before_drop.csv")
    df_final = df_final.dropna(subset=["Trafikkmengde_Totalt_i_retning_Florida"])
    print("after drop ", len(df_final))
    df_final.to_csv(f"{directory}/after_drop.csv")

    # finding means of values lead to floating point errors, round to fix these
    df_final = df_final.apply(pd.to_numeric, errors="ignore").round(30)

    return df_final


def trim_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a dataframe, trims values in the dataframe that are considered abnormal.

    What values are considered abnormal are covered in the README under "Dropped values"
    """
    length_dict = {"before": len(df)}

    # dette er innanfor grensene funnet her
    # https://veret.gfi.uib.no/
    df = df[df["Globalstraling"] < 1000]
    # df = df[df["Globalstraling"] >= 0]
    length_dict["afterGlobal"] = len(df)

    # må være fra 0-10
    df = df[df["Solskinstid"] < 10.01]
    length_dict["afterSolskinn"] = len(df)

    # må være under 50
    df = df[df["Lufttemperatur"] < 50]
    length_dict["afterLufttemp"] = len(df)

    # må være mellom 935 og 1050, det er max og min
    # verdien of all time
    df = df[(df["Lufttrykk"] < 1050)]
    length_dict["afterLufttrykk"] = len(df)

    # må være mindre en 65
    df = df[(df["Vindkast"] < 65)]
    length_dict["afterVindkast"] = len(df)

    df = df[df["Vindretning"] < 361]
    length_dict["afterVindretning"] = len(df)

    if DEBUG == True:
        print(json.dumps(length_dict, indent=2))

    return df


def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a daaframe, normalizes certain values to a 0-1 scale
    Other values are normalized differently, to an exponential scale

    Normalized values are covered in the README under "Normalized values"
    """

    # normalize between 0-1,
    scaler = MinMaxScaler()
    df[["Globalstraling", "Lufttrykk", "Solskinstid",]] = scaler.fit_transform(
        df[
            [
                "Globalstraling",
                "Lufttrykk",
                "Solskinstid",
            ]
        ]
    )

    # change vindkast to be an exponential scale
    df["Vindkast"] = df["Vindkast"] ** 2

    return df


def drop_uneeded_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a dataframe, drops values deemed not needed, as these may just provide noise for the model,
    or their values are already represented as another coloumn

    Dropped values are covered in the README under "Dropped coloumns"
    """

    df.drop(["Vindstyrke", "Vindretning", "Vindretning_radians"], axis=1, inplace=True)

    # Drop "Relativ luftfuktighet" as this data only exists in 2022 and 2023.
    # errors="ignore" as most of the data (back to 2015) will not have this coloumn
    df = df.drop(columns=["Relativ luftfuktighet"], errors="ignore")

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

    df = df.reset_index()
    df = df.drop(["DateFormatted"], axis=1)

    y = df["Total_trafikk"]
    x = df.drop(["Total_trafikk"], axis=1)

    # vi gjør at 70% blir treningsdata
    x_train, x_val, y_train, y_val = train_test_split(
        x, y, shuffle=False, test_size=0.3
    )

    # deler opp 30% som var validation til 15% val og 15% test
    x_val, x_test, y_val, y_test = train_test_split(
        x_val, y_val, shuffle=False, test_size=0.5
    )

    # utforskende anaylse ser kun på trenignsdata -> bruk x_train/y_train
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
