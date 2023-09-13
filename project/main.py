# ----TASK----
# make a dataframe which can be used in ML model
# Data splitting, description, visualisation, and feature engineering

import os
from pathlib import Path
import pandas as pd
from utils.dataframe_handling import (drop_uneeded_rows, feauture_engineer,
                                      merge_frames, train_test_split_process,
                                      trim_outliers)
from utils.file_parsing import treat_florida_files, treat_trafikk_files
from utils.graphing import graph_a_vs_b, graph_df

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

    # concat all the florida df's to one
    big_florida_df = pd.concat(florida_df_list, axis=0)

    # merge the dataframes
    df_final = merge_frames([big_florida_df, trafikk_df])

    # add important features to help the model
    df_final = feauture_engineer(df_final)

    # remove weird outliers like 2000 globalstråling
    df_final = trim_outliers(df_final)

    df_final = drop_uneeded_rows(df_final)

    # divide data into training,test and validation
    training_df,test_df,validation_df = train_test_split_process(df_final)

    print(training_df)

    return training_df


if __name__ == "__main__":
    main_df = main()

    directory = f"{str(PWD)}/out"
    main_df.to_csv(f"{directory}/main_training_data.csv")

    #:warning: GRAPHING TAKES A WHILE!

    graph_a_vs_b(main_df, "Globalstraling", "Total_trafikk","stråling"      , "antall sykler")
    graph_a_vs_b(main_df, "Solskinstid", "Total_trafikk"   ,"solskinn"      , "antall sykler")         
    graph_a_vs_b(main_df, "Lufttemperatur", "Total_trafikk","grader celcius", "antall sykler")
    graph_a_vs_b(main_df, "Vindretning", "Total_trafikk"   , "Grader"       , "antall sykler")         
    graph_a_vs_b(main_df, "Vindstyrke", "Total_trafikk"    , "Vind"         , "antall sykler")       
    graph_a_vs_b(main_df, "Lufttrykk", "Total_trafikk"     ,"hPa"           , "antall sykler")      
    graph_a_vs_b(main_df, "Vindkast", "Total_trafikk"      ,"m/s"           , "antall sykler")      
    graph_df(main_df)
