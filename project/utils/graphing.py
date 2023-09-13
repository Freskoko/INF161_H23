from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import numpy as np
import time

# visualisere vær vs trafikk -> se hva som trengs


def graph_a_vs_b(df, a, b, alabel, blabel):
    print(f" fig working on graphing '{a} vs {b}'")
    print(f"{a} looks like :")
    print(f"max {a} is {max(df[a])}")
    print(f"min {a} is {min(df[a])}")

    start_time = time.time()
    plt.xlim([min(df[a]) - 5, max(df[a]) + 5])  # x axis limits
    plt.figure(figsize=(15, 7))
    plt.bar(df[a], df[b])

    plt.xlabel(f"{a} ({alabel})")
    plt.ylabel(f"{b} ({blabel})")
    plt.suptitle(f"{a} vs {b} ")
    plt.title(f"""pearson_corr = {round(pearson_r_corr(df[a],df[b]),4)} spearmann_corr = {round(spearman_rho_corr(df[a],df[b]),4)}""")
    plt.grid(True)
    plt.savefig(f"figs/{a}VS{b}")
    plt.clf()
    print(f"saved fig '{a} VS {b}' in figs")
    print(f"--- Graph took: {time.time() - start_time} seconds ---")

def pearson_r_corr(a, b):
    corr = np.corrcoef(a, b)[0,1]
    return corr

def spearman_rho_corr(a, b):
    corr, _ = spearmanr(a, b)
    return corr


def graph_df(df):
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
    plt.title(f"""pearson_corr = {round(pearson_r_corr(a,b),4)} spearmann_corr = {round(spearman_rho_corr(a,b),4)}""")
    plt.grid(True)
    plt.legend()
    plt.savefig(f"figs/timeVStrafficBoth.png")
