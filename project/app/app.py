from appmodels import load_best_model, prep_data_from_user
from flask import Flask, flash, render_template, request

app = Flask(__name__)
app.secret_key = "A_in_INF161?_:P"

global predictor


@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        input_dict = request.form.to_dict()
        print(input_dict)
        prepped_data = prep_data_from_user(input_dict)

        print(prepped_data)
        if isinstance(prepped_data, str):
            flash(
                f"ERROR: Ensure all inputs types are numbers, and the date is in the proper format"
            )
            return render_template("home.html")

        trafficamount = predictor.predict(prepped_data)

        # bruk floor her TODO
        trafficamount = int(trafficamount[0] - 1)

        cyclist = f"""     
                            _______________   
                          / Hei Nello :^)     
                         <  I am one of {trafficamount}!   >
                    __o   \_________________/
                 _ |/<_
                (_)| (_)

                """

        cyclist_row = " " * 20  # Specifies the distance between each cyclist
        cycle_art = (cyclist + cyclist_row) * trafficamount

        # traf_data = predictor.predict(prepped_data)

        return render_template(
            "home.html", traffic_data=trafficamount, cycle_art=cycle_art
        )

    else:
        return render_template("home.html")


if __name__ == "__main__":
    predictor = load_best_model()
    print("Starting app...")
    app.run(debug=True, port="8080")
