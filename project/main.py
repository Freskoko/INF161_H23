import os
from pathlib import Path
import pandas as pd
from utils.dataframe_handling import (
    drop_uneeded_cols,
    feauture_engineer,
    merge_frames,
    normalize_cols,
    train_test_split_process,
    trim_outliers,
)
from utils.file_parsing import treat_florida_files, treat_trafikk_files
from utils.graphing import graph_all_models, graph_df
from utils.models import train_best_model, train_models, find_hyper_param
from loguru import logger

# get current filepath to use when opening/saving files
PWD = Path().absolute()


def main():
    # loop over files in local directory
    directory = f"{str(PWD)}/raw_data"

    # multiple florida files will all be converted to df's, placed in this list, and concacted
    florida_df_list = []

    for filename in os.scandir(directory):

        if "Florida" in str(filename):
            florida_df = treat_florida_files(f"{str(directory)}/{filename.name}")
            florida_df_list.append(florida_df)

        if "trafikkdata" in str(filename):
            trafikk_df = treat_trafikk_files(f"{str(directory)}/{filename.name}")

    logger.info("All files looped over")

    # concat all the florida df's to one
    big_florida_df = pd.concat(florida_df_list, axis=0)
    logger.info("Florida files concacted")

    # merge the dataframes
    df_final = merge_frames([big_florida_df, trafikk_df])
    logger.info("All files looped over")

    # remove weird outliers like 2000 globalstråling
    df_final = trim_outliers(df_final)
    logger.info("Outliers trimmed")

    # add important features to help the model
    df_final = feauture_engineer(df_final)
    logger.info("Features engineered")

    #normalize coloumns from 0-1 or square coloumns^2
    df_final = normalize_cols(df_final)
    logger.info("Coloumns normalized")

    #drop coloumns which are not needed (noise)
    df_final = drop_uneeded_cols(df_final) 
    logger.info("Uneeded cols dropped")

    # divide data into training,test and validation
    split_dict, training_df, test_df, validation_df = train_test_split_process(df_final)
    logger.info("Data divided into training,validation and test")

    return split_dict, training_df, test_df, validation_df


if __name__ == "__main__":
    split_dict, training_df, test_df, validation_df = main()

    directory = f"{str(PWD)}/out" 
    training_df.to_csv(f"{directory}/main_training_data.csv")
    test_df.to_csv(f"{directory}/main_test_data.csv")
    validation_df.to_csv(f"{directory}/main_validation_data.csv")

    model_dict = find_hyper_param(split_dict)
    assert False

    #:warning: GRAPHING TAKES A WHILE!
    graph_all_models(training_df)
    
    model_dict = train_models_loop(split_dict)
    model_dict = train_models(split_dict)
    train_best_model(split_dict)

    
    


