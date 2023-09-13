import time
import matplotlib.pyplot as plt

#visualisere vær vs trafikk -> se hva som trengs

def graph_a_vs_b(df,a,b):
    print(f" fig working on graphing '{a} vs {b}'")
    print(f"{a} looks like ")
    print(f"max {a} is {max(df[a])}")
    print(f"min {a} is {min(df[a])}")
    print(df[a])
    
    # print(f"{b} looks like ")
    # print(df[b])

    start_time = time.time()
    plt.xlim([min(df[a])-5, max(df[a])+5])  #x axis limits
    plt.bar(df[a], df[b])

    plt.xlabel(a)
    plt.ylabel(b)
    plt.title(f"{a} vs {b}")
    plt.grid(True)
    plt.savefig(f"figs/{a}VS{b}")
    plt.clf()
    print(f"saved fig '{a} VS {b}' in figs")
    print(f"--- Graph took: {time.time() - start_time} seconds ---")
    # plt.show()


    
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
