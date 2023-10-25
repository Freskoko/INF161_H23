import os
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor

from utils.dataframe_handling import (
    drop_uneeded_cols,
    feauture_engineer,
    merge_frames,
    normalize_data,
    train_test_split_process,
    treat_2023_file,
    trim_transform_outliers,
)
from utils.file_parsing import treat_florida_files, treat_trafikk_files
from utils.graphing import (
    graph_a_vs_b,
    graph_all_models,
    graph_df,
    graph_hour_diff,
    graph_hour_traffic_peryear,
    graph_hour_variance,
    graph_monthly_amounts,
    graph_weekly_amounts,
)
from utils.models import (
    find_hyper_param,
    find_hyper_param_further,
    train_best_model,
    train_models,
)

# get current filepath to use when opening/saving files
PWD = Path().absolute()
GRAPHING = False
TRAIN_MANY = False
FINAL_RUN = False
RANDOM_STATE = 2


def main():
    logger.info("Starting parsing ... ")
    # loop over files in local directory
    directory = f"{str(PWD)}/src/raw_data"

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

    # Calculate the average traffic for each year
    average_traffic_per_year = (
        training_df["Total_trafikk"].groupby(training_df.index.year).mean()
    )

    # Print the average traffic for each year
    print("AVERAGE TRAFFIC TRAINING DATA =")
    print(average_traffic_per_year)  # TODO SAVE THIS INTO A FILE

    if GRAPHING:
        graph_all_models(training_df, pre_change=True)
        logger.info("Graphed all models PRECHANGE")

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
    test_df = dataframes_post["test_df"]
    validation_df = dataframes_post["validation_df"]

    training_df.to_csv(f"{directory}/main_training_data.csv")
    logger.info("Data saved to CSV")

    if GRAPHING:
        # Graph post data processing to visualize and analyze data
        graph_hour_diff(training_df)
        graph_all_models(training_df, pre_change=False)
        graph_weekly_amounts(training_df)
        graph_monthly_amounts(training_df)
        graph_hour_variance(training_df)
        graph_hour_traffic_peryear(training_df)
        logger.info("Graph all models POSTCHANGE")

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
    train_best_model(split_dict_post, test_data=False)

    if FINAL_RUN:
        # train best model on test data
        train_best_model(split_dict_post, test_data=True)

        logger.info("Treating 2023 files")

        # use the best model to get values for 2023
        best_model = RandomForestRegressor(n_estimators=181, random_state=RANDOM_STATE)

        X_train = split_dict_post["x_train"]
        y_train = split_dict_post["y_train"]

        best_model.fit(X_train, y_train)
        df_with_values = treat_2023_file(df_2023, best_model)

    return split_dict_post, training_df, test_df, validation_df


if __name__ == "__main__":
    split_dict, training_df, test_df, validation_df = main()
