from collections.abc import Mapping
from math import isfinite
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, render_template, request


FEATURES = ("height_dm", "weight_hg", "hp", "attack", "defense")
MODEL_PATH = Path(__file__).with_name("pokemon_type_model (2).joblib")

if not MODEL_PATH.is_file():
    raise FileNotFoundError(f"Model file was not found: {MODEL_PATH}")

model = joblib.load(MODEL_PATH)
app = Flask(__name__)


def parse_features(form: Mapping[str, str]) -> tuple[dict[str, float], dict[str, str]]:
    values = {}
    errors = {}

    for feature in FEATURES:
        raw_value = form.get(feature, "").strip()
        if not raw_value:
            errors[feature] = "This field is required."
            continue

        try:
            value = float(raw_value)
        except ValueError:
            errors[feature] = "Enter a valid number."
            continue

        if not isfinite(value):
            errors[feature] = "Enter a finite number."
        elif value < 0:
            errors[feature] = "Enter zero or a positive number."
        else:
            values[feature] = value

    return values, errors


@app.route("/", methods=["GET", "POST"])
def predict():
    form_values = {feature: "" for feature in FEATURES}
    prediction = None
    errors = {}
    status_code = 200

    if request.method == "POST":
        form_values.update(
            {feature: request.form.get(feature, "") for feature in FEATURES}
        )
        values, errors = parse_features(request.form)
        if errors:
            status_code = 400
        else:
            features = pd.DataFrame(
                [[values[feature] for feature in FEATURES]],
                columns=FEATURES,
            )
            prediction = str(model.predict(features)[0])

    return (
        render_template(
            "index.html",
            features=FEATURES,
            form_values=form_values,
            prediction=prediction,
            errors=errors,
        ),
        status_code,
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)