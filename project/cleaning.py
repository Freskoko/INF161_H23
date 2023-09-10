#----TASK----
# make a dataframe which can be used in ML model
# Data splitting, description, visualisation, and feature engineering

#----TODO----
# change code, make easier to read, some can 100% be refactored
# document better
# MAYBE CHANGE CUTOFF DATA?
# handle empty values - > have a look 
# replace 99999 with NaN
#is it a weekend or not?

# make src and utils formatting
# miniconda or env to easy work

import os
import datetime as dt
from datetime import datetime
from io import StringIO
from pathlib import Path
import numpy as np

import pandas as pd

# get current filepath to use when opening/saving files
PWD = Path().absolute()


def treat_florida_files(filename):
    """
    Input: filename of a florida weather file

    Process:
    set date to index
    change from 1 hour having 6 values, to one hour having the mean of those 6 values

    Output:
    a dataframe of the csv file, made more uniform
    
    """


    df = pd.read_csv(filename, delimiter=",")

    # format date-data to be uniform, will help match data with traffic later
    df["DateFormatted"] = df.apply(
        lambda row: datetime.strptime(row["Dato"] + row["Tid"], "%Y-%m-%d%H:%M"), axis=1
    )

    # relativ luftfuktighet only exists in 2022 and 2023, bhut we have data all 
    # the way back to 2015, they must be dropped in order to use the whole dataset
    # `error = ignore` so files without this coloum (not 2022 or 2023) wont throw KeyError
    df = df.drop(columns=["Relativ luftfuktighet"], errors="ignore")

    #drop uneeded coloums
    df = df.drop(columns=['Dato', 'Tid'])
    
    #change date to index, in order to 
    df.set_index('DateFormatted', inplace=True)

    #combine all 6 values for a given hour into its mean
    #see read me section #TODO for more info
    df = df.resample('H').mean()

    print(df)

    return df


def treat_trafikk_files(filename):
    """
    Input: filename of a traffic data file

    Process:
    set date to index

    Output:
    a dataframe of the csv file, made more uniform
    """

    #read file as string
    with open(filename, "r") as f:
        my_csv_text = f.read()

    # replace | with ; to get uniform delimiter, and open to StringIO to be read by pandas
    csvStringIO = StringIO(my_csv_text.replace("|", ";"))

    #now that delimiter is uniform, file can be handled

    df = pd.read_csv(csvStringIO, delimiter=";")

    #change to a uniform date -> see #TODO in readme for info (multiple hour cols)
    df["DateFormatted"] = df.apply(
        lambda row: datetime.strptime(row["Fra"], "%Y-%m-%dT%H:%M%z").strftime("%Y-%m-%d %H:%M:%S"), axis=1)

    #replace '-' with null and convert column into numeric
    df['Trafikkmengde'] = df['Trafikkmengde'].replace('-', np.nan).astype(float)

    #replace " " in 'Felt' values with "_" to avoid errors
    df['Felt'] = df['Felt'].str.replace(" ", "_")

    #most of this data is not useful for this project, 
    #see #TODO in readme for info (dropping many cols, length is useless?, all is one hour)
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

    #drop all rows where the coloum "Felt" != "Totalt i retning Danmarksplass" or "Totalt i retning Florida"
    #the two other values for felt are "1" and "2" and are the same as the "Totalt ... Danmarkplass" and  "Totalt ... Florida"
    df = df[df['Felt'].isin(["Totalt_i_retning_Danmarksplass", "Totalt_i_retning_Florida"])]

    #create empty dataframe with 'DateFormatted' as index
    result_df = pd.DataFrame(index=df['DateFormatted'].unique())

    #loop through unique 'Felt' values, filter original dataframe by 'Felt', 
    #drop 'Felt' column and join to the result dataframe

    #so basically we sort of pivot the "Totalt i retning Danmarksplass" and "Totalt i retning Florida" 
    #from being values, to them being coloums contaning the values in "trafikkmengde"
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

def feauture_engineer(df):
    """
    Input: A dataframe containing traffic and weather data with DateFormatted as the index

    Adds:
    hour: 0-24 
    day: 0-31 day in month -> is this needed?
    day_in_week: 0-7
    month: 1-12
    weekend: True/False
    public holiday: True/False
    Last hour traffic florida: the value of the last row's florida traffic
    Last hour traffic danmarksplass: the value of the last row's danmarksplass traffic

    Returns: the inputted df with more features
    """

    #BASIC DATE FEATURES

    #hour as own coloum 0-24
    df["hour"] = df.index.hour #get first two values

    #day as own coloum 0-31
    df["day"] = df.index.day

    #day in the week 0-7
    df["day_in_week"] = df.index.weekday

    #month as own coloum 1-12
    df["month"] = df.index.month


    #MORE ADVANCED FEATURES

    #add weekend
    df["weekend"] = df.index.weekday >= 5

    #add last hour of each row, can be a good indicator
    df["Last_Danmarksplass"] = df["Trafikkmengde_Totalt_i_retning_Danmarksplass"].shift(1)
    df["Last_Florida"] = df["Trafikkmengde_Totalt_i_retning_Florida"].shift(1)

    #add public holidays 
    holidays = [
                #jul osv
                "12-24", 
                "12-25", 
                "01-01",
                #påske
                "04-06",
                "04-07",
                "04-08",
                "04-09",
                "04-10",
                #labour day/17 mai
                "05-01",
                "05-17",
                "05-18",
                ]
    
    df['public_holiday'] = df.index.strftime('%m-%d').isin(holidays)

    return df

    

def main():
    #loop over files in local directory
    directory = f"{str(PWD)}/raw_data"

    #multiple florida files will all be converted to df's, put in this list, and concacted
    florida_df_list = []

    for filename in os.scandir(directory):

        if "Florida" in str(filename):
            florida_df = treat_florida_files(f"{str(directory)}/{filename.name}")
            florida_df_list.append(florida_df)

        if "trafikkdata" in str(filename):
            trafikk_df = treat_trafikk_files(f"{str(directory)}/{filename.name}")

    big_florida_df = pd.concat(florida_df_list, axis=0)

    #turn index (date) from string to datetime to use .sort_index to get ordered items
    big_florida_df.index = pd.to_datetime(big_florida_df.index)
    trafikk_df.index = pd.to_datetime(trafikk_df.index)

    #since the dataframes have the same index, they can be merged
    df_final = big_florida_df.merge(trafikk_df, left_index=True, right_index=True, how='outer')
    df_final = df_final.sort_index()

    #remove lines where we dont have traffic information, as many missing values can cause model training
    #to overly rely on values which are there often.
    # :warning: this causes a loss in data -> se readme for more
    df_final = df_final.dropna(subset=['Trafikkmengde_Totalt_i_retning_Florida'])

    #finding means of values lead to floating point errors, round to fix these
    df_final = df_final.apply(pd.to_numeric, errors='ignore').round(1)
    print(df_final)

    #add important features to help the model 
    df_final = feauture_engineer(df_final)

    return df_final


if __name__ == "__main__":
    # out = main()
    main_df = (main())

    directory = f"{str(PWD)}/out"
    main_df.to_csv(
        f"{directory}/check_main.csv")
