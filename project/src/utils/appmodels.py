import os
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from dataframe_handling import (
    drop_uneeded_cols,
    feauture_engineer,
    merge_frames,
    normalize_data,
    train_test_split_process,
    trim_transform_outliers,
)
from file_parsing import treat_florida_files, treat_trafikk_files
from loguru import logger
from sklearn.ensemble import RandomForestRegressor

PWD = Path().absolute()
GRAPHING = False
TRAIN_MANY = False
FINAL_RUN = False
RANDOM_STATE = 2


def load_best_model() -> RandomForestRegressor:
    """
    Loads the best model
    """

    # if the model already exists as pickle, return it
    try:
        pickled_model = pickle.load(open("model.pkl", "rb"))
        return pickled_model
    except FileNotFoundError as e:
        pass

    logger.info("Starting parsing on loading best model ... ")
    # loop over files in local directory
    directory = f"{str(PWD)}/src/raw_data"  # change

    # multiple florida files will all be converted to df's, placed in this list, and concacted
    florida_df_list = []

    for filename in os.scandir(directory):

        if "Florida" in str(filename):
            florida_df = treat_florida_files(f"{str(directory)}/{filename.name}")
            florida_df_list.append(florida_df)

        if "trafikkdata" in str(filename):
            trafikk_df = treat_trafikk_files(f"{str(directory)}/{filename.name}")

    logger.info("All files parsed!")

    # concat all the florida df's to one
    big_florida_df = pd.concat(florida_df_list, axis=0)
    logger.info("Florida files concacted")

    # merge the dataframes
    df_2023, df_final = merge_frames([big_florida_df, trafikk_df])
    logger.info("All files looped over")

    # divide data into training,test and validation
    split_dict_pre, training_df, test_df, validation_df = train_test_split_process(
        df_final
    )
    logger.info("Data divided into training,validation and test")

    dataframes_pre = {
        "training_df": training_df,
    }

    # loop through each data frame and transform the data
    # this is done seperatley for each df (test/train/validation)
    # so that the training data is never influced by the other data
    dataframes_post = {}

    for name, df_transforming in dataframes_pre.items():
        logger.info(
            f"Applying KNN imputer on missing data, and removing outliers.. for {name}"
        )
        logger.info("This could take a while...")

        # transform NaN and outliers to usable data
        df_transforming = trim_transform_outliers(df_transforming, False)
        logger.info(f"Outliers trimmed for {name}")

        # add important features to help the model
        df_transforming = feauture_engineer(df_transforming, False)
        logger.info(f"Features engineered for {name}")

        # normalize data outliers
        df_transforming = normalize_data(df_transforming)
        logger.info(f"Coloumns normalized for {name}")

        # drop coloumns which are not needed or redundant
        df_transforming = drop_uneeded_cols(df_transforming)
        logger.info(f"Uneeded cols dropped for {name}")

        dataframes_post[name] = df_transforming

    training_df = dataframes_post["training_df"]

    split_dict_post = {
        "y_train": training_df["Total_trafikk"],
        "x_train": training_df.drop(["Total_trafikk"], axis=1),
    }

    X_train = split_dict_post["x_train"]
    y_train = split_dict_post["y_train"]

    # BEST MODEL:
    logger.info("Training best model")
    best_model = RandomForestRegressor(n_estimators=181, random_state=2)
    best_model.fit(X_train, y_train)

    # save model as pickle
    pickle.dump(best_model, open("model.pkl", "wb"))

    return best_model


def prep_data_from_user(input_dict):
    """
    Loads the best model
    """
    logger.info("Starting prep data from user ... ")

    col_keys = [
        "DateFormatted",
        "Globalstraling",
        "Solskinstid",
        "Lufttemperatur",
        "Vindretning",
        "Vindstyrke",
        "Lufttrykk",
        "Vindkast",
    ]

    df_dict = {}

    date_format = "%Y-%m-%d %H:%M:%S"

    for col in col_keys:
        if col in input_dict.keys():
            if col == "DateFormatted":
                df_dict[col] = [datetime.strptime(input_dict[col], date_format)]
            else:
                df_dict[col] = [float(input_dict[col])]
        else:
            df_dict[col] = np.nan

        if df_dict[col] == "":
            df_dict[col] = np.nan

    print(df_dict)

    df = pd.DataFrame(df_dict)  # Converts input data into dataframe

    df["DateFormatted"] = pd.to_datetime(df["DateFormatted"])
    df.set_index("DateFormatted", inplace=True)

    print("halfway df ----------------------")
    print(df)

    name = "userinp"

    logger.info(
        f"Applying KNN imputer on missing data, and removing outliers.. for {name}"
    )
    logger.info("This could take a while...")

    # transform NaN and outliers to usable data
    df = trim_transform_outliers(df, True)
    logger.info(f"Outliers trimmed for {name}")

    # add important features to help the model
    df = feauture_engineer(df, True)
    logger.info(f"Features engineered for {name}")

    # drop coloumns which are not needed or redundant
    df = drop_uneeded_cols(df)
    logger.info(f"Uneeded cols dropped for {name}")

    logger.info("re-aranging df to fit")

    # df["DateFormatted"] = df.index

    df = df[
        [
            "Globalstraling",
            "Solskinstid",
            "Lufttemperatur",
            "Lufttrykk",
            "Vindkast",
            "hour",
            "d_Friday",
            "d_Monday",
            "d_Saturday",
            "d_Sunday",
            "d_Thursday",
            "d_Tuesday",
            "d_Wednesday",
            "month",
            "weekend",
            "public_holiday",
            "raining",
            "summer",
            "winter",
            "rush_hour",
            "sleeptime",
            "Vindretning_x",
            "Vindretning_y",
        ]
    ]
    return df


if __name__ == "__main__":

    best_model = load_best_model()

    input_dict = {
        "DateFormatted": "2023-01-01 08:00:00",
        "Globalstraling": "-0.3166666666666667",
        "Solskinstid": "0.0",
        "Lufttemperatur": "-2.566666666666667",
        "Vindretning": "320",
        "Vindstyrke": "10",
        "Lufttrykk": "910",
        "Vindkast": "12",
    }

    df = prep_data_from_user(input_dict)
    df.to_csv("trainthis.csv")
    print(best_model.predict(df))
