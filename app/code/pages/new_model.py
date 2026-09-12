"""
New A2 model page — instrument-panel design, distinct from the original
scikit-learn page. Uses the from-scratch LinearRegression class's learned
`theta` (exported as a plain numpy array, not the class instance), so this
page never needs the notebook's custom class importable at load time.
Prediction is just: prep.transform(X) -> add bias column -> X @ theta.
"""

from pathlib import Path

import dash
import joblib
import numpy as np
import pandas as pd
from dash import Input, Output, State, dcc, html

dash.register_page(__name__, path="/new", name="New model")

MODEL_DIR = Path(__file__).parent.parent / "model"

bundle = joblib.load(MODEL_DIR / "car_price_model_v2.pkl")
theta = bundle["theta"]
prep = bundle["prep"]
FEATURE_COLS = bundle["feature_cols"]
RESIDUAL_STD = bundle["residual_std"]  # log-scale residual std, for a typical-range estimate

meta = joblib.load(MODEL_DIR / "model_metadata.pkl")
OWNER_SCALE = meta["owner_scale"]

Z_90 = 1.645  # ~90% interval on the log scale


def predict_price(row: dict):
    X = pd.DataFrame([row])[FEATURE_COLS]
    X_encoded = prep.transform(X)
    if hasattr(X_encoded, "toarray"):
        X_encoded = X_encoded.toarray()
    X_final = np.hstack([np.ones((X_encoded.shape[0], 1)), X_encoded])
    log_pred = (X_final @ theta)[0]
    price = float(np.exp(log_pred))
    low = float(np.exp(log_pred - Z_90 * RESIDUAL_STD))
    high = float(np.exp(log_pred + Z_90 * RESIDUAL_STD))
    return price, low, high


def field(id_, label, unit=None, placeholder=None, kind="number"):
    control = dcc.Input(
        id=id_, type="number", placeholder=placeholder,
        className="form-control", debounce=True,
    )
    return html.Div(
        [
            html.Label([label, html.Span(f" ({unit})", className="unit") if unit else None]),
            control,
        ],
        className="cv-field",
    )


def dropdown_field(id_, label, options):
    return html.Div(
        [
            html.Label(label),
            dcc.Dropdown(id=id_, options=[{"label": o, "value": o} for o in options],
                         placeholder="Select", className="dash-dropdown",
                         searchable=False, clearable=False),
        ],
        className="cv-field",
    )


def text_field(id_, label, placeholder=None):
    return html.Div(
        [
            html.Label(label),
            dcc.Input(id=id_, type="text", placeholder=placeholder, className="form-control"),
        ],
        className="cv-field",
    )


form = html.Div(
    [
        html.Div(
            [
                html.Div("Vehicle", className="cv-group-title"),
                html.Div(
                    [
                        text_field("new-brand", "Brand", placeholder="Maruti, Hyundai..."),
                        field("new-year", "Year", placeholder="2018"),
                        field("new-km_driven", "Distance", unit="km", placeholder="45000"),
                    ],
                    className="cv-field-row",
                ),
            ],
            className="cv-group",
        ),
        html.Div(
            [
                html.Div("Engine & efficiency", className="cv-group-title"),
                html.Div(
                    [
                        field("new-mileage", "Mileage", unit="kmpl", placeholder="19.5"),
                        field("new-engine", "Engine", unit="CC", placeholder="1197"),
                        field("new-max_power", "Max power", unit="bhp", placeholder="82.0"),
                    ],
                    className="cv-field-row",
                ),
            ],
            className="cv-group",
        ),
        html.Div(
            [
                html.Div("Condition & sale", className="cv-group-title"),
                html.Div(
                    [
                        field("new-seats", "Seats", placeholder="5"),
                        dropdown_field("new-fuel", "Fuel type", ["Diesel", "Petrol"]),
                        dropdown_field("new-transmission", "Transmission", ["Manual", "Automatic"]),
                    ],
                    className="cv-field-row",
                ),
                html.Div(
                    [
                        dropdown_field("new-owner", "Owner status", list(OWNER_SCALE.keys())),
                        dropdown_field(
                            "new-seller_type", "Seller type",
                            ["Individual", "Dealer", "Trustmark Dealer"],
                        ),
                    ],
                    className="cv-field-row",
                    style={"marginTop": "14px"},
                ),
            ],
            className="cv-group",
        ),
        html.Button("Estimate value", id="new-predict-btn", n_clicks=0, className="cv-btn"),
    ],
    className="cv-panel",
)

readout = html.Div(
    [
        html.Div(
            [html.I(className="fa-solid fa-flask", style={"fontSize": "0.75rem"}), " New model"],
            className="cv-badge",
        ),
        dcc.Loading(
            html.Div(
                html.Div(
                    "Fill in the spec sheet and estimate the value and the "
                    "reading will appear here.",
                    className="cv-readout-empty",
                ),
                id="new-prediction-output",
            ),
            type="dot",
        ),
        html.Div(
            "This estimate comes from a linear regression model written from "
            "scratch for Assignment 2: gradient descent with Xavier "
            "initialization, tuned by comparing model type, batch method, "
            "and learning rate across a cross-validated grid tracked in "
            "MLflow. The typical range above is a ~90% interval based on the "
            "model's residual error on the held-out test set, not a "
            "guarantee.",
            className="cv-readout-note",
        ),
    ],
    className="cv-readout",
)

layout = html.Div(
    [
        html.Div(
            [
                html.H1("Vehicle price predictor"),
                html.P(
                    "Estimate a fair resale price from the vehicle's specifications. "
                    "Missing values are imputed automatically."
                ),
            ],
            className="cv-hero",
        ),
        html.Div([form, readout], className="cv-grid"),
    ],
    className="cv-page",
)


@dash.callback(
    Output("new-prediction-output", "children"),
    Input("new-predict-btn", "n_clicks"),
    State("new-year", "value"),
    State("new-km_driven", "value"),
    State("new-mileage", "value"),
    State("new-engine", "value"),
    State("new-max_power", "value"),
    State("new-seats", "value"),
    State("new-owner", "value"),
    State("new-fuel", "value"),
    State("new-transmission", "value"),
    State("new-seller_type", "value"),
    State("new-brand", "value"),
    prevent_initial_call=True,
)
def predict_new(
    n_clicks, year, km_driven, mileage, engine, max_power, seats,
    owner_label, fuel, transmission, seller_type, brand,
):
    row = {
        "year": year, "km_driven": km_driven, "mileage": mileage, "engine": engine,
        "max_power": max_power, "seats": seats,
        "owner": OWNER_SCALE.get(owner_label, np.nan),
        "fuel": fuel, "transmission": transmission, "seller_type": seller_type, "brand": brand,
    }

    try:
        price, low, high = predict_price(row)
    except Exception as exc:
        return html.Div(
            [html.I(className="fa-solid fa-triangle-exclamation", style={"marginRight": "8px"}),
             f"Could not estimate a price: {exc}"],
            className="cv-alert-error",
        )

    return html.Div(
        [
            html.Div("Estimated market value", className="cv-readout-label"),
            html.Div(f"₹ {price:,.0f}", className="cv-readout-value"),
            html.Div(
                f"Typical range: ₹ {low:,.0f} – ₹ {high:,.0f}",
                style={"color": "var(--paper-dim)", "fontSize": "0.85rem", "marginTop": "2px"},
            ),
        ]
    )