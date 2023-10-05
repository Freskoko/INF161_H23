import pandas as pd
from matplotlib import pyplot as plt
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


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

    # TODO
    # forandre luftrykk skala?

    # BASIC DATE FEATURES

    # hour as own coloumn 0-24
    df["hour"] = df.index.hour  # get first two values

    # Instead of "day_in_week" being a num 0-6, add 7 coloumns to the dataframe, monday, tuesday .. etc
    # And have the value being 0/1 , 0 if it is not that day, 1 if it is

    day_week_dict = {
        0: "Monday",
        1: "Tuesday",
        2: "Wednesday",
        3: "Thursday",
        4: "Friday",
        5: "Saturday",
        6: "Sunday",
    }

    df["d"] = df.index.weekday.map(day_week_dict)

    # make each day their own coloumn
    df = pd.get_dummies(df, columns=["d"])  # convert to 1/0

    # month as own coloum 1-12
    df["month"] = df.index.month

    # MORE ADVANCED FEATURES

    # add weekend
    df["weekend"] = (df.index.weekday >= 5).astype(int)

    # THIS COULD NOT BE DONE - CHECK README
    # ------------------
    # add the hour values of the previous row, this can be a good indicator
    # df["Last_Danmarksplass"] = df["Trafikkmengde_Totalt_i_retning_Danmarksplass"].shift(
    # 1
    # )
    # df["Last_Florida"] = df["Trafikkmengde_Totalt_i_retning_Florida"].shift(1)
    # df["Last_total"] = df["Last_Danmarksplass"] + df["Last_Florida"]
    # ------------------

    # add public holidays
    holidays = [
        # jul osv
        "12-24",
        "12-25",
        "01-01",
        # påske
        "04-06",
        "04-07",
        "04-08",
        "04-09",
        "04-10",
        # labour day/17 mai
        "05-01",
        "05-17",
        "05-18",
    ]

    #add coloumn for rain if air pressure is higher than 1050
    #https://geo.libretexts.org/Bookshelves/Oceanography/Oceanography_101_(Miracosta)/08%3A_Atmospheric_Circulation/8.08%3A_How_Does_Air_Pressure_Relate_to_Weather
    df["raining"] = df["Lufttrykk"] <= 992

    # change public holiday to int
    df["public_holiday"] = df.index.strftime("%m-%d").isin(holidays).astype(int)

    # feature engineering (the shift to get data from last row)
    # leads to some missing values,
    # ex first row which has nothing to shift from, so we remove this.

    # drop of like 2 rows where there is no .shift value
    # (row 0 cant get a value from row -1)
    df = df.dropna(subset=["Trafikkmengde_Totalt_i_retning_Florida"])

    # add combo of total trafikk

    df["Total_trafikk"] = (
        df["Trafikkmengde_Totalt_i_retning_Florida"]
        + df["Trafikkmengde_Totalt_i_retning_Danmarksplass"]
    )

    df = df.drop(
        labels=[
            "Trafikkmengde_Totalt_i_retning_Florida",
            "Trafikkmengde_Totalt_i_retning_Danmarksplass",
        ],
        axis=1,
        inplace=False,
    )
    # total is found, these two are not needed

    # change all values of TRUE in all rows to 1 and FALSE to 0
    df = df.replace({True: 1, False: 0})

    return df


def merge_frames(frames: list):

    # initialize first DataFrame
    df_final = frames[0]

    # convert index (date) from string to datetime once only as we'll apply it to other frames
    df_final.index = pd.to_datetime(df_final.index)

    for frame in frames[1:]:
        # convert index from string to datetime
        frame.index = pd.to_datetime(frame.index)

        # since the dataframes have the same index, they can be merged!
        df_final = df_final.merge(frame, how="outer", left_index=True, right_index=True)

    df_final = df_final.sort_index()

    # remove lines where we dont have traffic information, as many missing values can cause model training
    # to overly rely on values which are there often.

    # Drops missing values
    df_final = df_final.dropna(subset=["Trafikkmengde_Totalt_i_retning_Florida"])

    # finding means of values lead to floating point errors, round to fix these
    df_final = df_final.apply(pd.to_numeric, errors="ignore").round(1)

    return df_final


def trim_outliers(df):
    # return df

    print(len(df))
    # dette er innanfor grensene funnet her
    # https://veret.gfi.uib.no/?prod=3&action=today#
    df = df[df["Globalstraling"] < 1000]  # g
    print(len(df))

    # må være fra 0-10, alt over er feil
    df = df[df["Solskinstid"] < 10.01]  # unsure
    print(len(df))

    # må være under 50, alt over er feil
    df = df[df["Lufttemperatur"] < 50]  # g
    print(len(df))

    # må være mellom 935 og 1050, det er max og min
    # verdien of all time
    # https://veret.gfi.uib.no/?prod=7&action=today#
    df = df[(df["Lufttrykk"] < 1050)]
    print(len(df))

    # må være mindre en 30
    # https://veret.gfi.uib.no/?prod=7&action=today#
    df = df[(df["Vindkast"] < 30)]
    print(len(df))

    # må være under 15
    df = df[df["Vindretning"] < 361]
    print(len(df))

    # må være under 15
    df = df[df["Vindstyrke"] < 15]
    print(len(df))

    return df

def normalize_cols(df):
    #normalize between 0-1, 
    #"Lufttrykk"
    #"Globalstraling"
    
    scaler = MinMaxScaler()
    df[["Globalstraling", 
        "Lufttrykk",
        "Solskinstid",
        "Vindkast"

        ]] = scaler.fit_transform(df[[
        "Globalstraling", 
        "Lufttrykk",
        "Solskinstid",
        "Vindkast"
        ]])

    #change vindkast to be an expoential scale
    df["Vindkast"] = df["Vindkast"]**2

    return df



def drop_uneeded_cols(df):
    # drop

    df.drop(["Vindstyrke", "Vindretning"], axis=1, inplace=True)

    return df


def train_test_split_process(df):

    print(df)

    df = df.reset_index()
    df = df.drop(["DateFormatted"], axis=1)

    print(df)

    y = df["Total_trafikk"]
    x = df.drop(["Total_trafikk"], axis=1)
    # vi gjør at 70% blir treningsdata

    x_train, x_val, y_train, y_val = train_test_split(
        x, y, shuffle=False, test_size=0.3
    )

    # vi deler val data opp i val data og test data
    # vi bruker val data for å sjekke hvilekn ML model er best
    # (data ikke har sett flr)
    # BESTE ML -> bruker vi TEST DATA

    x_val, x_test, y_val, y_test = train_test_split(
        x_val, y_val, shuffle=False, test_size=0.5
    )

    # utforskende anaylse ser kun på trenignsdata -> bruk x_train/y_train
    test_df = x_test.merge(y_test, how="outer", left_index=True, right_index=True)
    validation_df = x_val.merge(y_val, how="outer", left_index=True, right_index=True)
    training_df = x_train.merge(y_train, how="outer", left_index=True, right_index=True)

    print(training_df)

    split_dict = {
        "x_train": x_train,
        "y_train": y_train,
        "x_test": x_test,
        "y_test": y_test,
        "x_val": x_val,
        "y_val": y_val,
    }

    return split_dict, training_df, test_df, validation_df
