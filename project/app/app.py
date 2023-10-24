from datetime import datetime
from flask import Flask, render_template, request
import pandas as pd
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Now you can use absolute imports as if you're at the top-level directory.
from src.main import load_best_model

app = Flask(__name__)

global predictor
global data
global prepared_data

def load_data():
    data = pd.read_csv("/home/henrik/INF161_H23/project/app/trafikkdata.csv")
    data["date"] = pd.to_datetime(data["DateFormatted"])
    data.set_index("date", inplace=True)
    return data

def init():
    predictor = load_best_model()
    data = load_data()


def find_or_predict_value(input_datetime):
    # we are gonna fix this
    if input_datetime in data.index():
        traf_data = int(
            data.loc[data.index == input_datetime, "Total_trafikk"].values[0]
        )
    else:
        predicted_data = predictor.predict(prepared_data)
        traf_data = int(predicted_data[input_datetime])
    return traf_data


@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        data = load_data()
        input_date_str = request.form["date"]
        input_hour_str = request.form["hour"]
        input_datetime = datetime.strptime(
            input_date_str + " " + input_hour_str, "%Y-%m-%d %H"
        )
        traf_data = find_or_predict_value(input_datetime)
        return render_template(
            "home.html", traffic_data=traf_data
        )
    return render_template("home.html")


if __name__ == "__main__":
    init()
    app.run(debug=True)