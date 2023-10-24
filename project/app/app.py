from flask import Flask, render_template, request
import pandas as pd
# from main import load_best_model, prepare_data_for_model

app = Flask(__name__)

global predictor
global df_2023
# predictor = load_best_model()

def create_df_from_user():
    pass

def load_df_2023():
    pass

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        input_dict = request.form.to_dict()
        print(input_dict)
        prepped_data = prep_data_from_user(input_dict)
        traf_data = predictor.predict(prepped_data)
        
        return render_template("home.html", traffic_data=int(traf_data[0]))  # Cast prediction to int for display
    else:
        return render_template("home.html")

if __name__ == "__main__":
    app.run(debug=True)