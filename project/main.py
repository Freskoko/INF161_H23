# ----TASK----
# make a dataframe which can be used in ML model
# Data splitting, description, visualisation, and feature engineering

# ----TODO----
# change code, make easier to read, some can 100% be refactored
# document better
# MAYBE CHANGE CUTOFF DATA?
# handle empty values - > have a look
# replace 99999 with NaN
# is it a weekend or not?

# make src and utils formatting
# miniconda or env to easy work

import os
from pathlib import Path

import pandas as pd
from utils.dataframe_handling import feauture_engineer, merge_frames
from utils.file_parsing import (treat_bysykkel_files, treat_florida_files,
                                treat_trafikk_files)
from utils.graphing import graph_df

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

        if "bysykkel" in str(filename):
            bysykkel_df = treat_bysykkel_files(f"{str(directory)}/{filename.name}")

    # concat all the florida df's to one
    big_florida_df = pd.concat(florida_df_list, axis=0)

    # merge the dataframes
    df_final = merge_frames([big_florida_df, trafikk_df])

    # add important features to help the model
    df_final = feauture_engineer(df_final)

    print(df_final)

    return df_final


if __name__ == "__main__":
    main_df = main()

    graph_df(main_df)

    directory = f"{str(PWD)}/out"
    main_df.to_csv(f"{directory}/check_main.csv")
