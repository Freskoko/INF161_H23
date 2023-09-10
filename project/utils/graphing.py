import matplotlib.pyplot as plt
import pandas as pd


def graph_df(df):
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
    plt.title("Time vs Traffic")
    plt.grid(True)
    plt.legend()
    plt.show()
