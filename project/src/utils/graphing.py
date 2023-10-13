from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger
from scipy.stats import spearmanr

# get current filepath to use when opening/saving files
PWD = Path().absolute()
PWD = f"{PWD}/src"

def graph_a_vs_b(titletext:str, df: pd.DataFrame, a: str, b: str, alabel: str, blabel: str) -> None:
    """
    General function to plot two items in a dataframe against eachother

    Also calculates spearmann and pearson correlation between the two values
    """

    # see limits on data
    logger.info(f" fig working on graphing '{a} vs {b}'")
    print(f"{a} looks like :")
    print(f"max {a} is {max(df[a])}")
    print(f"min {a} is {min(df[a])}")

    # Set limits on x - axis to limits + 5 in order to see range of data
    start_time = time.time()
    plt.xlim([min(df[a]) - 5, max(df[a]) + 5])  # x axis limits
    plt.figure(figsize=(15, 7))
    plt.bar(df[a], df[b])

    plt.xlabel(f"{a} ({alabel})")
    plt.ylabel(f"{b} ({blabel})")
    plt.suptitle(f"{a} vs {b} ")

    # Calculate spearmann/pearson correlation to see if trends observed visually also can be seen statistically
    pear = round(pearson_r_corr(df[a], df[b]), 4)
    spear = round(spearman_rho_corr(df[a], df[b]), 4)

    plt.title(f"""pearson_corr = {pear} spearmann_corr = {spear} {titletext}""")
    plt.grid(True)
    plt.savefig(f"{PWD}/figs_new/{a}VS{b}_{titletext}")

    # :warning: clear fig is very important as not clearing will cause many figs to be created ontop of eachother
    plt.clf()
    logger.info(f"saved fig '{a} VS {b}' in figs_new")
    logger.info(f"--- Graph took: {round(time.time() - start_time,2)} seconds ---")

    return


def pearson_r_corr(a: float, b: float) -> float:
    corr = np.corrcoef(a, b)[0, 1]
    return corr


def spearman_rho_corr(a: float, b: float) -> float:
    corr, _ = spearmanr(a, b)
    return corr


def graph_df(df: pd.DataFrame) -> None:
    """
    Graphs Trafikkmengde_Totalt_i_retning_Danmarksplass vs Traffic towards Danmarksplass.

    This is to visualize how well the two lanes correlate.

    """
    plt.clf()
    plt.figure(figsize=(15, 7))
    plt.plot(
        df.index,
        df["Trafikkmengde_Totalt_i_retning_Danmarksplass"],
        label="Traffic towards Danmarksplass",
    )
    plt.plot(
        df.index,
        df["Trafikkmengde_Totalt_i_retning_Florida"],
        label="Traffic towards Florida",
    )
    plt.xlabel("Time")
    plt.ylabel("Traffic")
    plt.suptitle("Time vs Traffic")
    a = df["Trafikkmengde_Totalt_i_retning_Danmarksplass"]
    b = df["Trafikkmengde_Totalt_i_retning_Florida"]
    plt.title(
        f"""pearson_corr = {round(pearson_r_corr(a,b),4)} spearmann_corr = {round(spearman_rho_corr(a,b),4)}"""
    )
    plt.grid(True)
    plt.legend()
    plt.savefig(f"{PWD}/figs_new/timeVStrafficBoth.png")
    return None


def create_df_matrix(titletext: str, df: pd.DataFrame) -> None:
    """
    Function to create a covariance and correlation matrix for values in a dataframe
    """

    # drop uneeded cols: #TODO
    df = df.drop(
        labels=[
            "d_Monday",
            "d_Tuesday",
            "d_Wednesday",
            "d_Thursday",
            "d_Friday",
            "d_Saturday",
            "d_Sunday",
            "public_holiday",
            "raining",
            "summer",
            "winter",
            "rush_hour",
            "sleeptime",
            "weekend",
            "month",
            "weekend",
            "hour",
            "Vindretning_radians",
        ],
        axis=1,
        inplace=False,
        errors="ignore",
    )

    # calculate the covariance matrix
    cov_matrix = df.cov()

    # normalizing values between 0 and 1
    cov_matrix_normalized = (cov_matrix - cov_matrix.min().min()) / (
        cov_matrix.max().max() - cov_matrix.min().min()
    )

    plt.figure(figsize=(16, 16))
    sns.heatmap(
        cov_matrix_normalized, annot=True, cmap="RdBu", vmin=-1, vmax=1, center=0
    )
    plt.title(f"Covariance Matrix Heatmap {titletext}")
    plt.savefig(f"{PWD}/figs_new/covv_matrix_{titletext}.png")

    plt.clf()

    # calculate the correlation matrix
    corr_matrix = df.corr()

    plt.figure(figsize=(16, 16))
    sns.heatmap(corr_matrix, annot=True, cmap="RdBu", vmin=-1, vmax=1, center=0)
    plt.title(f"Correlation Matrix Heatmap {titletext}")
    plt.savefig(f"{PWD}/figs_new/corr_matrix_{titletext}.png")

    return


# removed vinkast ** 2
# add quantile


def graph_all_models(main_df: pd.DataFrame, pre_change: bool) -> None:

    logger.info("Graphing all graphs...")

    if pre_change:
        titletext="PRE_CHANGES"
    else:
        titletext="POST_CHANGES"

    create_df_matrix(titletext,main_df)
    graph_a_vs_b(
        titletext, main_df, "Globalstraling", "Total_trafikk", "stråling", "antall sykler"
    )
    graph_a_vs_b(titletext,main_df, "Solskinstid", "Total_trafikk", "solskinn", "antall sykler")
    graph_a_vs_b(
        titletext, main_df, "Lufttemperatur", "Total_trafikk", "grader celcius", "antall sykler"
    )

    if not pre_change:
        graph_a_vs_b(titletext, main_df, "Vindretning_x", "Total_trafikk", "Grader", "antall sykler")
        graph_a_vs_b(titletext, main_df, "Vindretning_y", "Total_trafikk", "Grader", "antall sykler")
    if pre_change:
        graph_a_vs_b(titletext, main_df, "Vindretning", "Total_trafikk", "Grader", "antall sykler")
        graph_a_vs_b(titletext, main_df, "Vindstyrke", "Total_trafikk", "Vind", "antall sykler")
    graph_a_vs_b(titletext, main_df, "Lufttrykk", "Total_trafikk", "hPa", "antall sykler")
    graph_a_vs_b(titletext, main_df, "Vindkast", "Total_trafikk", "m/s", "antall sykler")
    # graph_df(main_df)

    logger.info("Finished graphing!")
    return
