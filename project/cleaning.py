# TODO make a dataframe which can be used in ML model
# TODO Data splitting, description, visualisation, and feature engineering
# TODO make src and utils formatting
# TODO miniconda or env to easy work

import os
from datetime import datetime
from io import StringIO
from pathlib import Path
import numpy as np

import pandas as pd

# get current filepath to use when opening/saving files
PWD = Path().absolute()


def treat_florida_files(filename):
    df = pd.read_csv(filename, delimiter=",")

    # this time is in CEST
    df["DateFormatted"] = df.apply(
        lambda row: datetime.strptime(row["Dato"] + row["Tid"], "%Y-%m-%d%H:%M"), axis=1
    )

    #TODO LOOK AT THIS -> only in 2022 and 2023
    # error = ignore so files without this col wont throw KeyError
    df = df.drop(columns=["Relativ luftfuktighet"], errors="ignore")

    #lets combine the 6 values for an hour period into new rows containg the mean of the values
    
    #drop these cols to avoid trying to get the mean's of strings
    df = df.drop(columns=['Dato', 'Tid'])  # drop 'Dato' and 'Tid' columns
    
    df.set_index('DateFormatted', inplace=True)

    df = df.resample('H').mean()

    print(df)

    return df


def treat_trafikk_files(filename):

    with open(filename, "r") as f:
        my_csv_text = f.read()

    # replace | with ; to get uniform delimiter, and open to StringIO to be read by pandas
    csvStringIO = StringIO(my_csv_text.replace("|", ";"))

    df = pd.read_csv(csvStringIO, delimiter=";")


    # this time is in UTC +02:00, example format -> 2015-07-16T15:00+02:00
    # we wish to transform it to the same time format as in the florida files
    df["DateFormatted"] = df.apply(
        lambda row: datetime.strptime(row["Fra"], "%Y-%m-%dT%H:%M%z").strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
        axis=1,
    )

    #certain cols are missing values and are "-", this makes it diffiuclt to work with
    #converting to NaN helps
    df['Trafikkmengde'] = df['Trafikkmengde'].replace('-', np.nan).astype(float)

    #TODO
    #pivot the dataframe? 
    #basically i want there to only need to be one value of dateformatted,
    #not 4 values like now, because 4 values makes the data hard to line up with the previous data

    # Pivot the table
    pivot_df = pd.pivot_table(df, values='Trafikkmengde', index='DateFormatted', columns='Felt')
    # pivot_df.fillna(value, inplace=True)
    print(pivot_df)

        # save to have a look at the data
        
    #---------------------------

    directory = f"{str(PWD)}/out"
    pivot_df.to_csv(
        f"{directory}/check_pivot.csv")

      # :warning: when saving, delimiter becomes comma
    #---------------------------

    return pivot_df




    # we care about totalt i retning danmarksplass, and florida
    #

    


def main():
    directory = f"{str(PWD)}/raw_data"

    florida_df_list = []

    for filename in os.scandir(directory):

        if "Florida" in str(filename):
            florida_df = treat_florida_files(f"{str(directory)}/{filename.name}")
            florida_df_list.append(florida_df)

        if "trafikkdata" in str(filename):
            trafikk_df = treat_trafikk_files(f"{str(directory)}/{filename.name}")

    big_florida_df = pd.concat(florida_df_list, axis=0)

    df_final = pd.merge(
        big_florida_df, trafikk_df, how="inner", left_index=True, right_index=True
    )

    return df_final


if __name__ == "__main__":
    # out = main()
    main_df = (main())

    directory = f"{str(PWD)}/out"
    main_df.to_csv(
        f"{directory}/check_main.csv")
