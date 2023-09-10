import pandas as pd


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

    # BASIC DATE FEATURES

    # hour as own coloum 0-24
    df["hour"] = df.index.hour  # get first two values

    # day as own coloum 0-31
    df["day"] = df.index.day

    # day in the week 0-7
    df["day_in_week"] = df.index.weekday

    # month as own coloum 1-12
    df["month"] = df.index.month

    # MORE ADVANCED FEATURES

    # add weekend
    df["weekend"] = df.index.weekday >= 5

    # add last hour of each row, can be a good indicator
    df["Last_Danmarksplass"] = df["Trafikkmengde_Totalt_i_retning_Danmarksplass"].shift(
        1
    )
    df["Last_Florida"] = df["Trafikkmengde_Totalt_i_retning_Florida"].shift(1)

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

    df["public_holiday"] = df.index.strftime("%m-%d").isin(holidays)

    # feature engineering (the shift to get data from last row)
    # leads to some missing values,
    # ex first row which has nothing to shift from, so we remove this.

    # drop of like 2 rows
    df = df.dropna(subset=["Last_Florida"])

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
    # :warning: this causes a loss in data -> se #TODO readme for more

    # drops like two rows
    df_final = df_final.dropna(subset=["Trafikkmengde_Totalt_i_retning_Florida"])

    # finding means of values lead to floating point errors, round to fix these
    df_final = df_final.apply(pd.to_numeric, errors="ignore").round(1)

    return df_final
