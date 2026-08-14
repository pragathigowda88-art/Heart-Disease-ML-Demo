from flask import Flask, render_template, request
import os
import joblib
import numpy as np

app = Flask(__name__)


# ============================================================
# PROJECT PATH
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "heart_model.joblib"
)


# ============================================================
# FEATURES
# IMPORTANT:
# Each feature has 6 values because index.html expects 6 values
# ============================================================

FEATURES = [

    (
        "age",
        "Age",
        "number",
        "20",
        "100",
        "Age in years"
    ),

    (
        "sex",
        "Sex",
        "select",
        "",
        "",
        "0 = Female, 1 = Male"
    ),

    (
        "cp",
        "Chest Pain Type",
        "select",
        "",
        "",
        "0 - 3"
    ),

    (
        "trestbps",
        "Resting Blood Pressure",
        "number",
        "80",
        "220",
        "Resting blood pressure in mm Hg"
    ),

    (
        "chol",
        "Cholesterol",
        "number",
        "100",
        "600",
        "Cholesterol in mg/dl"
    ),

    (
        "fbs",
        "Fasting Blood Sugar",
        "select",
        "",
        "",
        "0 = No, 1 = Yes"
    ),

    (
        "restecg",
        "Resting ECG",
        "select",
        "",
        "",
        "0 - 2"
    ),

    (
        "thalach",
        "Maximum Heart Rate",
        "number",
        "60",
        "220",
        "Maximum heart rate achieved"
    ),

    (
        "exang",
        "Exercise Induced Angina",
        "select",
        "",
        "",
        "0 = No, 1 = Yes"
    ),

    (
        "oldpeak",
        "ST Depression",
        "number",
        "0",
        "7",
        "ST depression value"
    ),

    (
        "slope",
        "ST Slope",
        "select",
        "",
        "",
        "0 - 2"
    ),

    (
        "ca",
        "Major Vessels",
        "select",
        "",
        "",
        "0 - 4"
    ),

    (
        "thal",
        "Thalassemia",
        "select",
        "",
        "",
        "0 - 3"
    )

]


# ============================================================
# SELECT OPTIONS
# ============================================================

SELECT_OPTIONS = {

    "sex": [
        ("0", "Female"),
        ("1", "Male")
    ],

    "cp": [
        ("0", "Typical Angina"),
        ("1", "Atypical Angina"),
        ("2", "Non-anginal Pain"),
        ("3", "Asymptomatic")
    ],

    "fbs": [
        ("0", "No"),
        ("1", "Yes")
    ],

    "restecg": [
        ("0", "Normal"),
        ("1", "ST-T Abnormality"),
        ("2", "LV Hypertrophy")
    ],

    "exang": [
        ("0", "No"),
        ("1", "Yes")
    ],

    "slope": [
        ("0", "Upsloping"),
        ("1", "Flat"),
        ("2", "Downsloping")
    ],

    "ca": [
        ("0", "0"),
        ("1", "1"),
        ("2", "2"),
        ("3", "3"),
        ("4", "4")
    ],

    "thal": [
        ("0", "Normal"),
        ("1", "Fixed Defect"),
        ("2", "Reversible Defect"),
        ("3", "Other")
    ]

}


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def home():

    model_ready = os.path.exists(
        MODEL_PATH
    )

    print("\n==============================")
    print("CARDIOPREDICT")
    print("==============================")

    print("Model path:")
    print(MODEL_PATH)

    print(
        "Model available:",
        model_ready
    )

    return render_template(
        "index.html",
        features=FEATURES,
        select_options=SELECT_OPTIONS,
        model_ready=model_ready
    )


# ============================================================
# PREDICTION
# ============================================================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    # --------------------------------------------------------
    # Check model
    # --------------------------------------------------------

    if not os.path.exists(MODEL_PATH):

        return render_template(
            "result.html",
            error=(
                "The trained model was not found. "
                "Please run python train_model.py first."
            )
        )


    try:

        # ----------------------------------------------------
        # Load trained model
        # ----------------------------------------------------

        model = joblib.load(
            MODEL_PATH
        )


        # ----------------------------------------------------
        # Get form values
        # ----------------------------------------------------

        values = []


        for feature in FEATURES:

            name = feature[0]

            value = request.form.get(
                name
            )


            if value is None or value == "":

                return render_template(
                    "result.html",
                    error=(
                        f"Please enter a value for {name}."
                    )
                )


            values.append(
                float(value)
            )


        # ----------------------------------------------------
        # Convert to NumPy array
        # ----------------------------------------------------

        X = np.array(
            values,
            dtype=float
        ).reshape(
            1,
            -1
        )


        print("\nInput values:")
        print(values)


        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        prediction = int(
            model.predict(X)[0]
        )


        # ----------------------------------------------------
        # Probability
        # ----------------------------------------------------

        probability = None

        if hasattr(
            model,
            "predict_proba"
        ):

            probability = (
                model.predict_proba(X)[0][1]
                * 100
            )


        # ----------------------------------------------------
        # Result
        # ----------------------------------------------------

        if prediction == 1:

            title = (
                "Higher likelihood of heart disease"
            )

            message = (
                "The machine-learning model "
                "classified this input as class 1."
            )

            status = "warning"

        else:

            title = (
                "Lower likelihood of heart disease"
            )

            message = (
                "The machine-learning model "
                "classified this input as class 0."
            )

            status = "success"


        # ----------------------------------------------------
        # Show result
        # ----------------------------------------------------

        return render_template(

            "result.html",

            prediction=prediction,

            probability=probability,

            title=title,

            message=message,

            status=status
        )


    except Exception as e:

        print("\nPrediction error:")
        print(e)

        return render_template(
            "result.html",
            error=str(e)
        )


# ============================================================
# ABOUT PAGE
# ============================================================

@app.route("/about")
def about():

    return render_template(
        "about.html"
    )


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    print("\n====================================")
    print("       CARDIOPREDICT")
    print("====================================")

    print("\nProject:")
    print(BASE_DIR)

    print("\nModel:")
    print(MODEL_PATH)

    print(
        "\nModel exists:",
        os.path.exists(MODEL_PATH)
    )

    print(
        "\nOpen browser:"
    )

    print(
        "http://127.0.0.1:5000"
    )

    app.run(
        debug=True
    )