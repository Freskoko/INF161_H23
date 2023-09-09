#----TASK----
# make a dataframe which can be used in ML model
# Data splitting, description, visualisation, and feature engineering

#----TODO----
# change code, make easier to read, some can 10% be refactored
# document better
# make src and utils formatting
# miniconda or env to easy work

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

    df["DateFormatted"] = df.apply(
        lambda row: datetime.strptime(row["Fra"], "%Y-%m-%dT%H:%M%z").strftime("%Y-%m-%d %H:%M:%S"), axis=1)

    # Replace '-' with null and convert column into numeric
    df['Trafikkmengde'] = df['Trafikkmengde'].replace('-', np.nan).astype(float)

    # Replace " " (spaces) in 'Felt' values with "_" (underscores)
    df['Felt'] = df['Felt'].str.replace(" ", "_")

    df = df.drop(columns=
                [
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
                #is this data important????   
                "< 5,6m",             
                ">= 5,6m",
                "5,6m - 7,6m",
                "7,6m - 12,5m",
                "12,5m - 16,0m",
                ">= 16,0m",
                "16,0m - 24,0m",
                ">= 24,0m",
                ] 
                   
                )# stuff we dont need
    
    print(df)

    #lets drop all rows where the coloum "Felt" != 
    # "Totalt i retning Danmarksplass" or "Totalt i retning Florida"
    df = df[df['Felt'].isin(["Totalt_i_retning_Danmarksplass", "Totalt_i_retning_Florida"])]

    # Create empty dataframe with 'DateFormatted' as index
    result_df = pd.DataFrame(index=df['DateFormatted'].unique())

    # Loop through unique 'Felt' values, filter original dataframe by 'Felt', drop 'Felt' column and join to the result dataframe
    for felt in df['Felt'].unique():
        felt_df = (df[df['Felt'] == felt]
                   .drop(columns='Felt')
                   .add_suffix(f'_{felt}')  # add suffix to column names to distinguish them when joining
                   .set_index('DateFormatted_{0}'.format(felt)))
        
        result_df = result_df.join(felt_df)

    print(result_df)

    directory = f"{str(PWD)}/out"
    result_df.to_csv(
        f"{directory}/check_traffic.csv")

    return result_df




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

    #turn into datetime instead of string to we can use .sort_index 
    big_florida_df.index = pd.to_datetime(big_florida_df.index)
    trafikk_df.index = pd.to_datetime(trafikk_df.index)

    df_final = big_florida_df.merge(trafikk_df, left_index=True, right_index=True, how='outer')
    df_final = df_final.sort_index()

    #remove lines where 

    df_final = df_final.dropna(subset=['Trafikkmengde_Totalt_i_retning_Florida'])

    return df_final


if __name__ == "__main__":
    # out = main()
    main_df = (main())

    directory = f"{str(PWD)}/out"
    main_df.to_csv(
        f"{directory}/check_main.csv")
