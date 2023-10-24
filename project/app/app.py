from flask import Flask, flash, render_template, request

# from main import load_best_model, prep_data_from_user

app = Flask(__name__)
app.secret_key = "A_in_INF161?_:P"

global predictor

# todo after


@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        input_dict = request.form.to_dict()
        print(input_dict)
        prepped_data = prep_data_from_user(input_dict)

        if isinstance(prepped_data, Exception):
            flash(
                f"ERROR: Ensure all data types are numbers, and the date is in the proper format"
            )
            return render_template("home.html")

        # traf_data = predictor.predict(prepped_data)
        traf_data = [123, 125]  # fix

        return render_template("home.html", traffic_data=int(traf_data[0]))

    else:
        return render_template("home.html")


if __name__ == "__main__":
    app.run(debug=True)

    print("Starting app...")

    predictor = load_best_model()
