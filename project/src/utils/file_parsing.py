from datetime import datetime
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd

# get current filepath to use when opening/saving files
PWD = Path().absolute()


def treat_florida_files(filename: str) -> pd.DataFrame:
    """
    Input: filename of a florida weather file

    Process:
    set date to index
    change from 1 hour having 6 values, to one hour having the mean of those 6 values
    drop date and time coloumns which are now represented in the index

    Output:
    a dataframe of the csv file
    """

    df = pd.read_csv(filename, delimiter=",")

    # format date-data to be uniform, will help match data with traffic later
    df["DateFormatted"] = df.apply(
        lambda row: datetime.strptime(row["Dato"] + row["Tid"], "%Y-%m-%d%H:%M"), axis=1
    )

    # drop uneeded coloums
    df = df.drop(columns=["Dato", "Tid"])

    # change date to index, in order to
    df.set_index("DateFormatted", inplace=True)

    # combine all 6 values for a given hour into its mean
    df = df.resample("H").mean()

    return df


def treat_trafikk_files(filename: str) -> pd.DataFrame:
    """
    Input: filename of a traffic data file

    Process:
    set date to index

    Output:
    a dataframe of the csv file
    """

    # read file as string
    with open(filename, "r") as f:
        my_csv_text = f.read()

    # replace | with ; to get uniform delimiter, and open to StringIO to be read by pandas
    csvStringIO = StringIO(my_csv_text.replace("|", ";"))

    # now that delimiter is uniform, file can be handled

    df = pd.read_csv(csvStringIO, delimiter=";")

    # change to a uniform date -> see # Issues in README
    df["DateFormatted"] = df.apply(
        lambda row: datetime.strptime(row["Fra"], "%Y-%m-%dT%H:%M%z").strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
        axis=1,
    )

    # replace '-' with NaN and convert column into numeric
    df["Trafikkmengde"] = df["Trafikkmengde"].replace("-", np.nan).astype(float)

    # replace " " in 'Felt' values with "_" to avoid errors
    df["Felt"] = df["Felt"].str.replace(" ", "_")

    # dropping cols - see README on "Dropped coloumns"
    df = df.drop(
        columns=[
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
            "< 5,6m",
            ">= 5,6m",
            "5,6m - 7,6m",
            "7,6m - 12,5m",
            "12,5m - 16,0m",
            ">= 16,0m",
            "16,0m - 24,0m",
            ">= 24,0m",
        ]
    )

    # drop all rows where the coloum "Felt" != "Totalt i retning Danmarksplass" or "Totalt i retning Florida"
    # the two other values for felt are "1" and "2" and are the same as the "Totalt ... Danmarkplass" and  "Totalt ... Florida"
    df = df[
        df["Felt"].isin(["Totalt_i_retning_Danmarksplass", "Totalt_i_retning_Florida"])
    ]

    # create empty dataframe with 'DateFormatted' as index
    result_df = pd.DataFrame(index=df["DateFormatted"].unique())

    # loop through unique 'Felt' values, filter original dataframe by 'Felt',
    # drop 'Felt' column and join to the result dataframe

    # so basically we sort of pivot the "Totalt i retning Danmarksplass" and "Totalt i retning Florida"
    # from being values, to them being coloums contaning the values in "trafikkmengde" (since we dropped all other cols)
    for felt in df["Felt"].unique():
        felt_df = (
            df[df["Felt"] == felt]
            .drop(columns="Felt")
            .add_suffix(
                f"_{felt}"
            )  # add suffix to column names to distinguish them when joining
            .set_index("DateFormatted_{0}".format(felt))
        )

        result_df = result_df.join(felt_df)

    # save to csv
    directory = f"{str(PWD)}/src/out"
    result_df.to_csv(f"{directory}/check_traffic.csv")

    return result_df
