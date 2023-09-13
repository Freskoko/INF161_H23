import pandas as pd
from sklearn.model_selection import train_test_split


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

    #TODO
    #forandre luftrykk skala?
    #change True/False to 1/0
  

    # BASIC DATE FEATURES

    # hour as own coloum 0-24
    df["hour"] = df.index.hour  # get first two values


    #Instead of "day_in_week" being a num 0-6, add 7 coloumns to the dataframe, monday, tuesday .. etc
    #And have the value being 0/1 , 0 if it is not that day, 1 if it is

    day_week_dict ={0: 'Monday', 
                    1: 'Tuesday', 
                    2: 'Wednesday', 
                    3: 'Thursday',
                    4: 'Friday', 
                    5: 'Saturday', 
                    6: 'Sunday'}

    df['d'] = df.index.weekday.map(day_week_dict)

    df = pd.get_dummies(df, columns=['d'])



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

    df["Last_total"] = df["Last_Danmarksplass"] + df["Last_Florida"]

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

    df["public_holiday"] = df.index.strftime("%m-%d").isin(holidays).astype(int)

    # feature engineering (the shift to get data from last row)
    # leads to some missing values,
    # ex first row which has nothing to shift from, so we remove this.

    # drop of like 2 rows
    df = df.dropna(subset=["Last_Florida"])


    #add combo of total trafikk

    df["Total_trafikk"] = df["Trafikkmengde_Totalt_i_retning_Florida"] + df["Trafikkmengde_Totalt_i_retning_Danmarksplass"]

    # #in the dataframe change all False to 0, and True to 1
    # df["weekend"] = df["weekend"].astype(int)
    # df["public_holiday"] = df["public_holiday"].astype(int)

    # df["d_Monday"] = df["d_Monday"].astype(int)
    # df["d_Tuesday"] = df["d_Tuesday"].astype(int)
    # df["d_Wednsday"] = df["d_Wednsday"].astype(int)
    # df["d_Thi"] = df["d_"].astype(int)
    # df["d_"] = df["d_"].astype(int)
    # df["d_"] = df["d_"].astype(int)
    # df["d_"] = df["d_"].astype(int)    

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


def trim_outliers(df):
    # return df

    # TODO
    # Trimme outliers 
    #- globalstrålng 2000 maks
    #- luftrykk endre maks til 2000, og skala
    #- vindkast maks til 2000
    #- vindstyrke maks til 2000

    # dette er innanfor grensene funnet her
    #https://veret.gfi.uib.no/?prod=3&action=today#
    df = df[df["Globalstraling"] < 1000] #g

    #må være under 1000, en på 1750 ble funnet
    #men det må være feil #TODO check
    df = df[df["Solskinstid"] < 1000]#unsure

    #må være under 1000, en på 1750 ble funnet
    #men det må være feil #TODO check
    df = df[df["Lufttemperatur"] < 1000]#g
    
    #må være mellom 935 og 1050 
    #https://veret.gfi.uib.no/?prod=7&action=today#
    df = df[(df["Lufttrykk"] < 1050)] #g
    # df = df[(df["Lufttrykk"] > 935)]

    #må være mellom 935 og 1050 
    #https://veret.gfi.uib.no/?prod=7&action=today#
    df = df[(df["Vindkast"] < 4050)]
    # df = df[(df["Vindkast"] > 935)]

    #må være under 1000, en på 1750 ble funnet
    #men det må være feil #TODO check
    df = df[df["Vindstyrke"] < 4000]

    #må være under 1000, en på 1750 ble funnet
    #men det må være feil #TODO check
    df = df[df["Vindretning"] < 1000]#g

    return df


def train_test_split_stuff(df):
    y = df ["Total_trafikk"]
    print("--- here is the supa cool dat aframe ------------------")
    print(df)
    x = df.drop(["Total_trafikk"], axis=1)
    # vi gjør at 70% blir treningsdata

    print(x)
    print(y)

    x_train, x_val, y_train, y_val = train_test_split(
                            x,y, 
                            shuffle=False, 
                            test_size=0.3)
    
    #vi deler val data opp i val data og test data

    #vi bruker val data for å sjekke hvilekn ML moddel er best 
    # (data ikke har sett flr)

    # BESTE ML -> bruker vi TEST DATA

    x_val,x_test,y_val,y_test = train_test_split(
                            x_val,y_val, 
                            shuffle=False, 
                            test_size=0.5)
    
    #utforskende anaylse se kun på trenignsdata

    #VI VIL BARE SE PÅ X_train og Y_train

    print("----------XTRAIN")
    print(x_train)
    

    print("----------YTRAIN")
    print(y_train)

    out_df = x_train.merge(y_train, how="outer", left_index=True, right_index=True)
    print(out_df)
    
    return out_df 