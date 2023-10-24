from flask import Flask, flash, render_template, request

# from main import load_best_model, prep_data_from_user

app = Flask(__name__)
app.secret_key = "A_in_INF161?_:P"

global predictor


# TEMP - CHANGE THIS
def prep_data_from_user(inp):
    return [3, 1]


@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        input_dict = request.form.to_dict()
        print(input_dict)
        prepped_data = prep_data_from_user(input_dict)

        trafficamount = int(prepped_data[0])

        cyclist = """
                    __o
                 _ |/<_
                (_)| (_)

                """

        cyclist_row = " " * 20  # Specifies the distance between each cyclist
        cycle_art = (cyclist + cyclist_row) * trafficamount

        if isinstance(prepped_data, Exception):
            flash(
                f"ERROR: Ensure all inputs types are numbers, and the date is in the proper format"
            )
            return render_template("home.html")

        # traf_data = predictor.predict(prepped_data)

        return render_template(
            "home.html", traffic_data=trafficamount, cycle_art=cycle_art
        )

    else:
        return render_template("home.html")


if __name__ == "__main__":
    app.run(debug=True)

    print("Starting app...")

    predictor = load_best_model()
