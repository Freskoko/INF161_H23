#TODO make a dataframe which can be used in ML model
#TODO Data splitting, description, visualisation, and feature engineering

import os
import pandas as pd
from pathlib import Path
from datetime import datetime
import re
from io import StringIO

PWD = Path().absolute()

def treat_florida_files(filename):
    df = pd.read_csv(filename,delimiter=",")

    #this time is in CEST
    df["DateFormatted"] = df.apply(
        lambda row: datetime.strptime(row["Dato"] + row["Tid"], '%Y-%m-%d%H:%M'), axis=1)

    return df

def treat_trafikk_files(filename):

    #open file
    with open(filename, 'r') as f:
        my_csv_text = f.read()

    #replace | with ; to get uniform delimiter, and open to StringIO to be read by od
    csvStringIO = StringIO(my_csv_text.replace("|",";"))

    df = pd.read_csv(csvStringIO,delimiter=";",header=None) 

    directory = f"{str(PWD)}/out"
    df.to_csv(f"{directory}/check.csv") # :warning: when saving, delimiter becomes comma

   
    #this time is in 2015-07-16T15:00+02:00 so UTC +02:00
    return



def main():
    directory = f"{str(PWD)}/raw_data"

    florida_df_list = []

    for filename in os.scandir(directory):

        if "Florida fefw " in str(filename):
            florida_df = treat_florida_files(f"{str(directory)}/{filename.name}")
            florida_df_list.append(florida_df)
        
        if "trafikkdata" in str(filename):
            trafikk_df = treat_trafikk_files(f"{str(directory)}/{filename.name}")

    big_florida_df = pd.concat(florida_df_list, axis = 0)
    combined = pd.concat([big_florida_df,trafikk_df], axis = 1)
    return combined
        
if __name__ == "__main__":
    main()
    # print(main())