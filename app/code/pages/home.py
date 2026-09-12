"""
Home page — explains the project and how the two models compare, honestly:
the from-scratch model doesn't beat the original on raw accuracy, so this
page leads with what it actually offers instead (transparency, a fully
tracked hyperparameter search) rather than overstating it.
"""

import dash
from dash import html, dcc

dash.register_page(__name__, path="/", name="Home")

hero = html.Div(
    [
        html.H1("CarValuate"),
        html.P(
            "Two ways to estimate a used car's resale value from the same dataset: a "
            "scikit-learn pipeline from Assignment 1, and a linear regression model "
            "written from scratch for Assignment 2: gradient descent implemented by "
            "hand, with Xavier initialization, optional momentum, and a "
            "cross-validated search over 180 configurations tracked in MLflow."
        ),
    ],
    className="cv-hero",
)

model_cards = html.Div(
    [
        html.Div(
            [
                html.Div("Original model", className="cv-group-title"),
                html.Div("scikit-learn pipeline", style={"color": "var(--paper-dim)", "fontSize": "0.85rem", "marginBottom": "16px"}),
                html.Div("~0.92", className="cv-readout-value", style={"fontSize": "1.9rem", "color": "var(--paper)"}),
                html.Div("Test R²", className="cv-readout-label"),
                dcc.Link("Open original model →", href="/old", className="cv-nav-link", style={"display": "inline-block", "marginTop": "16px", "padding": "8px 0"}),
            ],
            className="cv-panel",
        ),
        html.Div(
            [
                html.Div("New model", className="cv-group-title"),
                html.Div("from-scratch gradient descent", style={"color": "var(--paper-dim)", "fontSize": "0.85rem", "marginBottom": "16px"}),
                html.Div("0.875", className="cv-readout-value", style={"fontSize": "1.9rem"}),
                html.Div("Test R²", className="cv-readout-label"),
                dcc.Link("Open new model →", href="/new", className="cv-nav-link", style={"display": "inline-block", "marginTop": "16px", "padding": "8px 0", "color": "var(--brass)"}),
            ],
            className="cv-panel",
        ),
    ],
    className="cv-grid",
    style={"marginBottom": "32px"},
)

explanation = html.Div(
    [
        html.Div("Which one should you use?", className="cv-group-title"),
        html.P(
            "Honestly: the original model is more accurate (R² ≈ 0.92 vs. 0.875). It's a "
            "scikit-learn ensemble, which naturally captures non-linear patterns in the data "
            "that a plain linear model can't. The new model isn't presented here as more "
            "accurate; what it offers instead is full transparency into how the prediction is "
            "made.",
            style={"color": "var(--paper-dim)", "lineHeight": "1.7", "marginBottom": "14px"},
        ),
        html.P(
            "Every coefficient in the new model is inspectable. Its feature-importance chart "
            "shows exactly which inputs push the price up or down, and by how much. It was "
            "also tuned by comparing 180 configurations (model type, batch method, weight "
            "initialization, learning rate, with and without momentum, including a polynomial "
            "variant) via cross-validation, with every run logged in MLflow.",
            style={"color": "var(--paper-dim)", "lineHeight": "1.7"},
        ),
    ],
    className="cv-panel",
)

layout = html.Div([hero, model_cards, explanation], className="cv-page")