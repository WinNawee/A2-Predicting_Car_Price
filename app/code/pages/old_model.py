"""
Original A1 model page (kept as-is at "/", just adapted to Dash Pages).
"""

from pathlib import Path

import dash
import dash_bootstrap_components as dbc
import joblib
import numpy as np
import pandas as pd
from dash import Input, Output, State, dcc, html

dash.register_page(__name__, path="/old", name="Original Model")

MODEL_DIR = Path(__file__).parent.parent / "model"
model = joblib.load(MODEL_DIR / "car_price_model.pkl")
meta = joblib.load(MODEL_DIR / "model_metadata.pkl")

NUM_COLS = meta["num_cols"]
TWO_LEVEL_COLS = meta["two_level_cols"]
MULTI_LEVEL_COLS = meta["multi_level_cols"]
OWNER_SCALE = meta["owner_scale"]
FEATURE_COLS = NUM_COLS + TWO_LEVEL_COLS + MULTI_LEVEL_COLS


def section_header(title, icon_class):
    return html.Div(
        [
            html.I(className=f"{icon_class} text-primary me-2"),
            html.Span(title, className="fw-bold text-dark text-uppercase small tracking-wide"),
            html.Hr(className="mt-1 mb-3 opacity-25"),
        ],
        className="mt-2 mb-2",
    )


def numeric_field(id_, label, icon, placeholder="—", help_text=""):
    return dbc.Col(
        [
            dbc.Label(
                [html.I(className=f"{icon} text-secondary me-1"), label],
                html_for=id_,
                className="form-label small fw-semibold text-secondary mb-1",
            ),
            dbc.Input(
                id=id_,
                type="number",
                placeholder=placeholder,
                className="form-control rounded-3 border-light-subtle shadow-sm-hover",
            ),
            html.Small(help_text, className="text-muted d-block mt-1") if help_text else None,
        ],
        xs=12,
        sm=6,
        md=4,
        className="mb-3",
    )


def dropdown_field(id_, label, icon, options):
    return dbc.Col(
        [
            dbc.Label(
                [html.I(className=f"{icon} text-secondary me-1"), label],
                html_for=id_,
                className="form-label small fw-semibold text-secondary mb-1",
            ),
            dcc.Dropdown(
                id=id_,
                options=[{"label": o, "value": o} for o in options],
                placeholder="Select...",
                className="custom-dropdown",
            ),
        ],
        xs=12,
        sm=6,
        md=4,
        className="mb-3",
    )


form_card = dbc.Card(
    dbc.CardBody(
        [
            html.Div(
                [
                    html.Div(
                        html.I(className="fa-solid fa-car-side fa-2x text-primary"),
                        className="bg-primary-subtle p-3 rounded-circle d-inline-block mb-3",
                    ),
                    html.H2("Vehicle Price Predictor", className="fw-bold text-dark mb-1"),
                    html.P(
                        "Original model from Assignment 1 (sklearn pipeline).",
                        className="text-muted small mb-4",
                    ),
                ],
                className="text-center pb-2",
            ),
            section_header("Vehicle Overview", "fa-solid fa-gauge-high"),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Label(
                                [html.I(className="fa-solid fa-copyright text-secondary me-1"), "Brand"],
                                html_for="old-brand",
                                className="form-label small fw-semibold text-secondary mb-1",
                            ),
                            dbc.Input(
                                id="old-brand",
                                type="text",
                                placeholder="e.g. Maruti, Hyundai",
                                className="form-control rounded-3 border-light-subtle",
                            ),
                        ],
                        xs=12,
                        sm=6,
                        md=4,
                        className="mb-3",
                    ),
                    numeric_field("old-year", "Year", "fa-regular fa-calendar", placeholder="e.g. 2018"),
                    numeric_field("old-km_driven", "Distance (km)", "fa-solid fa-road", placeholder="e.g. 45000"),
                ]
            ),
            section_header("Engine & Efficiency", "fa-solid fa-microchip"),
            dbc.Row(
                [
                    numeric_field("old-mileage", "Mileage (kmpl)", "fa-solid fa-gas-pump", placeholder="e.g. 19.5"),
                    numeric_field("old-engine", "Engine (CC)", "fa-solid fa-gears", placeholder="e.g. 1197"),
                    numeric_field("old-max_power", "Max Power (bhp)", "fa-solid fa-bolt", placeholder="e.g. 82.0"),
                ]
            ),
            section_header("Configuration & History", "fa-solid fa-file-contract"),
            dbc.Row(
                [
                    numeric_field("old-seats", "Seats", "fa-solid fa-chair", placeholder="e.g. 5"),
                    dropdown_field("old-fuel", "Fuel Type", "fa-solid fa-droplet", ["Diesel", "Petrol"]),
                    dropdown_field("old-transmission", "Transmission", "fa-solid fa-sliders", ["Manual", "Automatic"]),
                ]
            ),
            dbc.Row(
                [
                    dropdown_field("old-owner", "Owner Status", "fa-solid fa-user-check", list(OWNER_SCALE.keys())),
                    dropdown_field(
                        "old-seller_type", "Seller Type", "fa-solid fa-shop",
                        ["Individual", "Dealer", "Trustmark Dealer"],
                    ),
                ]
            ),
            html.Div(
                [
                    dbc.Button(
                        [html.I(className="fa-solid fa-calculator me-2"), "Estimate Value"],
                        id="old-predict-btn",
                        n_clicks=0,
                        color="primary",
                        size="lg",
                        className="w-100 py-2 rounded-3 fw-semibold shadow-sm",
                    ),
                ],
                className="mt-4 mb-2",
            ),
            dcc.Loading(
                html.Div(id="old-prediction-output", className="mt-4"),
                type="dot",
                color="#0d6efd",
            ),
        ],
        className="p-4 p-md-5",
    ),
    className="border-0 shadow-lg rounded-4 bg-white",
)

layout = dbc.Container(
    dbc.Row(dbc.Col(form_card, md=10, lg=8, xl=7, className="mx-auto"), className="py-2"),
    fluid=True,
)


@dash.callback(
    Output("old-prediction-output", "children"),
    Input("old-predict-btn", "n_clicks"),
    State("old-year", "value"),
    State("old-km_driven", "value"),
    State("old-mileage", "value"),
    State("old-engine", "value"),
    State("old-max_power", "value"),
    State("old-seats", "value"),
    State("old-owner", "value"),
    State("old-fuel", "value"),
    State("old-transmission", "value"),
    State("old-seller_type", "value"),
    State("old-brand", "value"),
    prevent_initial_call=True,
)
def predict_old(
    n_clicks, year, km_driven, mileage, engine, max_power, seats,
    owner_label, fuel, transmission, seller_type, brand,
):
    row = {
        "year": year, "km_driven": km_driven, "mileage": mileage, "engine": engine,
        "max_power": max_power, "seats": seats,
        "owner": OWNER_SCALE.get(owner_label, np.nan),
        "fuel": fuel, "transmission": transmission, "seller_type": seller_type, "brand": brand,
    }
    X = pd.DataFrame([row])[FEATURE_COLS]

    try:
        log_pred = model.predict(X)[0]
    except Exception as exc:
        return dbc.Alert(
            [html.I(className="fa-solid fa-triangle-exclamation me-2"), f"Error during prediction: {exc}"],
            color="danger",
            className="rounded-3 shadow-sm border-0",
        )

    price = float(np.exp(log_pred))
    return html.Div(
        html.Div(
            [
                html.P("ESTIMATED MARKET VALUE", className="text-uppercase fw-bold text-success mb-1 small"),
                html.H2(f"₹ {price:,.0f}", className="display-6 fw-bold text-dark mb-0"),
            ],
            className="bg-success-subtle border border-success-subtle p-4 rounded-4 text-center shadow-sm",
        )
    )