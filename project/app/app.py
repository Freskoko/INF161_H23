from flask import Flask, render_template, request

from main import load_best_model, prep_data_from_user

app = Flask(__name__)

global predictor


@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        input_dict = request.form.to_dict()
        print(input_dict)
        prepped_data = prep_data_from_user(input_dict)
        traf_data = predictor.predict(prepped_data)

        return render_template("home.html", traffic_data=int(traf_data[0]))

    else:
        return render_template("home.html")


if __name__ == "__main__":
    app.run(debug=True)

    predictor = load_best_model()
