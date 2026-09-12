"""
Car price predictor — Dash multi-page app.
Three pages sharing one instrument-panel-styled nav bar: a home page
explaining the comparison, the original scikit-learn model (/old), and
the from-scratch A2 model (/new).
"""

import dash
import dash_bootstrap_components as dbc
from dash import Dash, html, dcc, Input, Output

app = Dash(
    __name__,
    use_pages=True,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,  # required by the Original Model page (dbc.Card, dbc.Row, ...)
        "https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap",
        "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css",
    ],
    suppress_callback_exceptions=True,
)
app.title = "CarValuate | AI Price Predictor"
server = app.server

navbar = html.Div(
    [
        html.A(
            [html.Span(className="cv-brand-mark"), "CarValuate"],
            href="/",
            className="cv-brand",
        ),
        html.Div(
            [
                dcc.Link("Home", href="/", id="nav-home", className="cv-nav-link"),
                dcc.Link("Original model", href="/old", id="nav-old", className="cv-nav-link"),
                dcc.Link("New model", href="/new", id="nav-new", className="cv-nav-link"),
            ],
            className="cv-nav-links",
        ),
    ],
    className="cv-navbar",
)

app.layout = html.Div(
    [
        dcc.Location(id="cv-url"),
        navbar,
        dash.page_container,
    ]
)


@dash.callback(
    Output("nav-home", "className"),
    Output("nav-old", "className"),
    Output("nav-new", "className"),
    Input("cv-url", "pathname"),
)
def highlight_active_tab(pathname):
    base = "cv-nav-link"
    home_cls = f"{base} active" if pathname == "/" else base
    old_cls = f"{base} active" if pathname == "/old" else base
    new_cls = f"{base} active" if pathname == "/new" else base
    return home_cls, old_cls, new_cls


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8050)