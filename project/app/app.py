from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)


def load_data():
    data = pd.read_csv("/home/henrik/INF161_H23/project/app/trafikkdata.csv")
    data["date"] = pd.to_datetime(data["DateFormatted"])
    data.set_index("date", inplace=True)
    return data


def create_image(num_cyclists):
    x = np.ones((num_cyclists,))
    y = np.array(range(num_cyclists))

    fig, ax = plt.subplots()
    size = 80

    ax.scatter(x, y, c="b", s=size, alpha=0.5, edgecolors="r")

    ax.axis("off")

    fig.savefig("static/cyclist_image.png")


@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        data = load_data()
        input_date_str = request.form["date"]
        input_hour_str = request.form["hour"]
        input_datetime = datetime.strptime(
            input_date_str + " " + input_hour_str, "%Y-%m-%d %H"
        )
        try:
            traf_data = int(
                data.loc[data.index == input_datetime, "Total_trafikk"].values[0]
            )

            create_image(int(traf_data))
            traffic_image = True
            # round to int because cant have 0.1 drivers
        except IndexError:
            traf_data = "No data found for the selected date and hour!"

        return render_template(
            "home.html", traffic_data=traf_data, traffic_image=traffic_image
        )
    return render_template("home.html")


if __name__ == "__main__":
    app.run(debug=True)
