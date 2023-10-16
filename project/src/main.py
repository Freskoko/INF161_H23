import os
from pathlib import Path

import pandas as pd
from loguru import logger
from sklearn.ensemble import RandomForestRegressor

from utils.dataframe_handling import (drop_uneeded_cols, feauture_engineer,
                                      merge_frames, normalize_cols,
                                      train_test_split_process,
                                      treat_2023_file, trim_transform_outliers)
from utils.file_parsing import treat_florida_files, treat_trafikk_files
from utils.graphing import graph_all_models, graph_df
from utils.models import find_hyper_param, train_best_model, train_models

# get current filepath to use when opening/saving files
PWD = Path().absolute()
GRAPHING = False
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

    # --------- DO RESEARCH HERE, THEN ADD STUFF AND GRAPH AFTER AGAIN

    split_dict, training_df, test_df, validation_df = train_test_split_process(df_final)
    logger.info("Data divided into training,validation and test")

    if GRAPHING:
        graph_all_models(training_df, pre_change=True)
        logger.info("Graphed all models PRECHANGE")

    # --------- DONE RESEARCH

    # remove weird outliers like 2000 globalstråling

    logger.info("Applying KNN imputer on missing data, and removing outliers..")
    logger.info("This could take a while...")
    df_final = trim_transform_outliers(df_final, False)
    logger.info("Outliers trimmed")

    # add important features to help the model
    df_final = feauture_engineer(df_final, False)
    logger.info("Features engineered")

    # normalize coloumns from 0-1 or square coloumns^2
    df_final = normalize_cols(df_final)
    logger.info("Coloumns normalized")

    # drop coloumns which are not needed (noise)
    df_final = drop_uneeded_cols(df_final)
    logger.info("Uneeded cols dropped")

    # divide data into training,test and validation
    split_dict, training_df, test_df, validation_df = train_test_split_process(df_final)
    logger.info("Data divided into training,validation and test")

    training_df.to_csv(f"{directory}/main_training_data.csv")
    test_df.to_csv(f"{directory}/main_test_data.csv")
    validation_df.to_csv(f"{directory}/main_validation_data.csv")
    logger.info("Data saved to CSV")

    if GRAPHING:
        graph_all_models(training_df, pre_change=False)
        logger.info("Graph all models POSTCHANGE")

    model_dict = train_models(split_dict)
    best_model = find_hyper_param(split_dict)
    train_best_model(split_dict, test_data=False)

    train_best_model(split_dict, test_data=True)

    logger.info("Treating 2023 files")

    # BEST MODEL:
    best_model = RandomForestRegressor(n_estimators=250, random_state=RANDOM_STATE)

    X_train = split_dict["x_train"]
    y_train = split_dict["y_train"]

    best_model.fit(X_train, y_train)
    df_with_values = treat_2023_file(df_2023, best_model)

    return split_dict, training_df, test_df, validation_df


if __name__ == "__main__":
    split_dict, training_df, test_df, validation_df = main()
