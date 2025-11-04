from dash import Dash, dcc, html, Input, Output, callback
from app_pages.eco2mix import eco2mix_layout
from app_pages.spot import spot_layout
from app_pages.prev_spot import prev_spot_layout
import dash_bootstrap_components as dbc

app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True,
)

navbar = dbc.NavbarSimple(
    children=[
        dbc.Button("Accueil", href="/", color="secondary", outline=True, className="me-2"),
        dbc.Button("Données éco2mix 2024", href="/eco2mix", color="secondary", outline=True, className="me-2"),
        dbc.Button("Données SPOT", href="/spot", color="secondary", outline=True, className="me-2"),
        dbc.Button("Prévision de prix SPOT", href="/prev_spot", color="secondary", outline=True),
    ],
    brand="Prévisions de consommation d'électricité",
    brand_href="/",
    brand_style={"fontWeight": "bold", "fontSize": "1.5rem"},
    id="my_nvb",
    color="dark",
    dark=True,
    className="mb-2",
)

app.layout = html.Div(
    [
        dcc.Location(id="url", refresh=False),
        navbar,
        dbc.Container(id="page-content", className="mb-4", fluid=True),
        html.Footer(
            dbc.Container(
                html.Small("© 2025 - Agathe PASCAL ", className="text-muted"),
                className="text-center py-2"
            ),
            style={"background": "#f8f9fa"}
        ),
    ]
)

home_layout = dbc.Container(
    [
        dbc.Row(
            dbc.Col(
                dbc.Card(
                    [
                        dbc.CardHeader(
                            html.H1("Prévisions de consommation d'électricité", className="text-center"),
                            style={"backgroundColor": "#343a40", "color": "white"}
                        ),
                        dbc.CardBody(
                            [
                                html.P(
                                    "Cette application est composée de trois sous-pages :",
                                    className="lead"
                                ),
                                html.Ul(
                                    [
                                        html.Li("Visualisation des données éco2mix de 2024"),
                                        html.Li("Visualisation des prix SPOT historiques"),
                                        html.Li("Prévisions et visualisation des prix SPOT"),
                                    ]
                                ),
                            ]
                        ),
                    ],
                    className="shadow"
                ),
                width=8, className="mx-auto mt-4"
            )
        ),
    ],
    fluid=True
)

@callback(Output("page-content", "children"), Input("url", "pathname"))
def display_page(pathname):
    if pathname == "/":
        return home_layout
    elif pathname == "/eco2mix":
        return eco2mix_layout
    elif pathname == "/spot":
        return spot_layout
    elif pathname == "/prev_spot":
        return prev_spot_layout
    else:
        return dbc.Container(
            [
                html.H1("404: Not found", className="text-danger"),
                html.Hr(),
                html.P(f"L'url {pathname} n'est pas reconnue..."),
            ],
            className="mt-4"
        )

app.run(debug=True)
