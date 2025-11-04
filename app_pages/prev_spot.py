from __future__ import annotations
from pathlib import Path
import base64, io, pickle
from typing import Optional

import numpy as np
import pandas as pd
import requests
import plotly.express as px
from dash import dcc, html, Input, Output, State, callback, no_update
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# =============================================================================
# Constantes / Configuration
# =============================================================================
API_URL = "https://odre.opendatasoft.com/api/records/1.0/search/"
DATASET_ID = "eco2mix-national-tr"

MODELS_DIR = Path("models")
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
FRANCE_CSV = DATA_DIR / "France.csv"

MODEL_FILES = sorted([f.name for f in MODELS_DIR.glob("*.pkl")])

# Renommages des colonnes pour le modèle / affichage
DISPLAY_RENAME = {
    'heure': 'Heures',
    'date': 'Date',
    'prevision_j1': 'Prévision J-1',
    'prevision_j': 'Prévision J',
    'fioul_tac': 'Fioul - TAC',
    'nucleaire': 'Nucléaire',
    'ech_comm_angleterre': 'Ech. comm. Angleterre',
    'gaz_tac': 'Gaz - TAC',
    'bioenergies_biogaz': 'Bioénergies - Biogaz',
    'bioenergies_biomasse': 'Bioénergies - Biomasse',
    'destockage_batterie': 'Déstockage batterie',
    'charbon': 'Charbon',
    'hydraulique_step_turbinage': 'Hydraulique - STEP turbinage',
    'stockage_batterie': ' Stockage batterie',
    'pompage': 'Pompage',
    'gaz_ccg': 'Gaz - CCG',
    'hydraulique_lacs': 'Hydraulique - Lacs',
    'eolien_terrestre': 'Eolien terrestre',
    'ech_comm_allemagne_belgique': 'Ech. comm. Allemagne-Belgique',
    'fioul': 'Fioul',
    'solaire': 'Solaire',
    'ech_comm_italie': 'Ech. comm. Italie',
    'bioenergies': 'Bioénergies',
    'gaz_autres': 'Gaz - Autres',
    'fioul_cogen': 'Fioul - Cogén.',
    'gaz_cogen': 'Gaz - Cogén.',
    'hydraulique_fil_eau_eclusee': 'Hydraulique - Fil de l?eau + éclusée',
    'hydraulique': 'Hydraulique',
    'ech_comm_suisse': 'Ech. comm. Suisse',
    'eolien_offshore': 'Eolien offshore',
    'ech_comm_espagne': 'Ech. comm. Espagne',
    'taux_co2': 'Taux de Co2',
    'eolien': 'Eolien',
    'ech_physiques': 'Ech. physiques',
    'bioenergies_dechets': 'Bioénergies - Déchets',
    'consommation': 'Consommation',
    'gaz': 'Gaz',
    'fioul_autres': 'Fioul - Autres',
}

# =============================================================================
# Utils
# =============================================================================
def iso_day_bounds(start_date: Optional[str], end_date: Optional[str]) -> Optional[str]:
    """Filtre Opendatasoft: date_heure:[YYYY-MM-DDT00:00:00 TO YYYY-MM-DDT23:59:59]"""
    if not start_date and not end_date:
        return None
    start = start_date or end_date
    end = end_date or start_date
    return f"date_heure:[{start}T00:00:00 TO {end}T23:59:59]"

def change_names(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.rename(columns=DISPLAY_RENAME, inplace=True)
    return out

def decode_uploaded_file(contents: str) -> pd.DataFrame:
    """Décodage des fichiers eCO2mix (.xls en réalité TSV latin1) ; supprime la dernière ligne d’avertissement."""
    _, content_string = contents.split("base64,", 1)
    df = pd.read_csv(
        io.StringIO(base64.b64decode(content_string).decode("latin1")),
        sep="\t",
        index_col=False
    )
    # Enlève la dernière ligne non-numérique fréquente
    if len(df) and not pd.api.types.is_numeric_dtype(df.iloc[-1:].select_dtypes(include=[np.number])):
        df = df.iloc[:-1]
    return df

def preview_text(df: pd.DataFrame, title: str) -> html.Div:
    if df is None or df.empty:
        return html.Div("Aucune donnée.")
    return html.Div([
        html.H5(title),
        html.Hr(),
        html.Div(f"Nombre d’entrées : {len(df)}"),
        html.Pre(df.head().to_string(index=False))
    ])

def load_model(model_filename: str):
    with open(MODELS_DIR / model_filename, "rb") as f:
        return pickle.load(f)

def card(title: str, children):
    return html.Div(
        style={
            "background":"#fff","border":"1px solid #e5e7eb","borderRadius":"12px",
            "padding":"16px","marginBottom":"16px","boxShadow":"0 1px 2px rgba(0,0,0,0.04)"
        },
        children=[html.H3(title, style={"marginTop":0}), children]
    )

def _normalize_rows(v, default=100, minv=1, maxv=20000) -> int:
    """Nettoie la valeur de l'input (gère None, espaces, 5 000, etc.)."""
    if v is None:
        return default
    try:
        s = str(v).replace("\u202f", "").replace(" ", "")
        n = int(float(s))
        return max(minv, min(n, maxv))
    except Exception:
        return default

# =============================================================================
# Layout
# =============================================================================
prev_spot_layout = html.Div(
    style={"background":"#f3f4f6","borderRadius":"12px","padding":"20px","margin":"20px"},
    children=[
        html.H2("Prévisions de prix SPOT", style={"marginTop":0}),
        card("Source des données", html.Div([
            dcc.Tabs(id="data-source-tabs", value="tab-api", children=[
                dcc.Tab(label="API eCO2mix", value="tab-api", children=html.Div([
                    html.Div(style={"display":"flex","gap":"12px","flexWrap":"wrap","alignItems":"end"}, children=[
                        html.Div([
                            html.Label("Période", style={"fontWeight":600}),
                            dcc.DatePickerRange(
                                id="api-date-range",
                                minimum_nights=0,
                                display_format="YYYY-MM-DD",
                                start_date="2025-01-01",
                                end_date="2025-11-03",
                            ),
                        ]),
                        html.Div([
                            html.Label("Limite (lignes)", style={"fontWeight":600}),
                            dcc.Input(
                                id="api-rows",
                                type="number",
                                min=1,
                                step=1,                # step=1 pour éviter les invalidations
                                value=2000,
                                debounce=True,        # Enter / blur seulement
                                style={"width":"160px","height":38}
                            ),
                        ]),
                        html.Div(style={"display":"flex", "gap":"12px"}, children=[
                            html.Button(
                                "Charger depuis l’API", id="api-fetch-btn", n_clicks=0,
                                style={"height":38,"background":"#2563eb","color":"#fff",
                                       "border":"none","borderRadius":"8px","padding":"0 18px","fontWeight":600}
                            ),
                            html.Button(
                                "Télécharger les données", id="download-data-btn", n_clicks=0,
                                style={"height":38,"background":"#10b981","color":"#fff",
                                       "border":"none","borderRadius":"8px","padding":"0 18px","fontWeight":600}
                            ),
                        ]),
                    ]),
                    html.Div(
                        id="api-preview",
                        style={"marginTop":12,"background":"#f9fafb","borderRadius":"8px",
                               "padding":"12px","border":"1px solid #e5e7eb","fontFamily":"monospace"}
                    ),
                ])),
                dcc.Tab(label="Fichier CSV", value="tab-upload", children=html.Div([
                    dcc.Upload(
                        id="upload-data",
                        children=html.Div(["Glisser-déposer ou sélectionner un fichier eCO2mix (.xls/.csv)"]),
                        style={"width":"100%","height":"60px","lineHeight":"60px","borderWidth":"1px",
                               "borderStyle":"dashed","borderRadius":"8px","textAlign":"center","background":"#fff",
                               "cursor":"pointer","fontWeight":600},
                        multiple=False,
                    ),
                    html.Div(
                        id="upload-preview",
                        style={"marginTop":12,"background":"#f9fafb","borderRadius":"8px",
                               "padding":"12px","border":"1px solid #e5e7eb","fontFamily":"monospace"}
                    ),
                ])),
            ]),
            dcc.Store(id="data-store"),   # stocke la dernière source de données chargée
            dcc.Download(id="download-data"),
        ])),
        card("Modèle et prévisions", html.Div([
            dcc.Dropdown(
                id="model-dropdown",
                options=[{"label": f, "value": f} for f in MODEL_FILES],
                placeholder="Sélectionner un modèle (.pkl)"
            ),
            html.Div(style={"height":8}),
            html.Button("Lancer les prévisions", id="run-forecasts-button", n_clicks=0),
            html.Div(id="forecast-output", style={"marginTop":12}),
        ])),
    ],
)

# =============================================================================
# Callbacks
# =============================================================================

# Réinitialise le store et les aperçus au changement d’onglet
@callback(
    Output("data-store", "data"),
    Output("api-preview", "children"),
    Output("upload-preview", "children"),
    Input("data-source-tabs", "value"),
)
def clear_on_tab_change(tab_value: str):
    return None, html.Div(), html.Div()

@callback(
    Output("data-store", "data", allow_duplicate=True),
    Output("api-preview", "children", allow_duplicate=True),
    Input("api-fetch-btn", "n_clicks"),
    Input("api-rows", "n_submit"),
    State("api-date-range", "start_date"),
    State("api-date-range", "end_date"),
    State("api-rows", "value"),
    prevent_initial_call=True,
)
def fetch_from_api(n_clicks: int, n_submit: int,
                   start_date: Optional[str], end_date: Optional[str],
                   rows_value: Optional[int]):
    if not (n_clicks or n_submit):
        return no_update, no_update

    try:
        target = _normalize_rows(rows_value, default=100, minv=1, maxv=20000)
        q = iso_day_bounds(start_date, end_date)

        page_size = 10000 # max API
        start = 0
        all_records: list[dict] = []

        while len(all_records) < target:
            remaining = target - len(all_records)
            rows = min(page_size, remaining)

            params = {
                "dataset": DATASET_ID,
                "rows": rows,
                "start": start,
                "sort": "-date_heure",
            }
            if q:
                params["q"] = q

            r = requests.get(API_URL, params=params, timeout=30)
            r.raise_for_status()
            page = r.json().get("records", [])

            if not page:
                break

            all_records.extend(page)
            start += max(1, len(page)) # évite boucle infinie si API bug
            if len(page) < rows:       # plus de données dispo
                break

        if not all_records:
            return None, "Aucune donnée renvoyée par l’API pour cette période."

        df = pd.DataFrame([rec.get("fields", {}) for rec in all_records])
        df = change_names(df)

        store = {"source": "api", "columns": list(df.columns), "data": df.to_dict(orient="records")}
        preview = preview_text(df, "Aperçu des données API")
        return store, preview

    except Exception as e:
        return None, f"Erreur API : {e}"

@callback(
    Output("data-store", "data", allow_duplicate=True),
    Output("upload-preview", "children", allow_duplicate=True),
    Input("upload-data", "contents"),
    State("upload-data", "filename"),
    prevent_initial_call=True,
)
def handle_upload(contents: Optional[str], filename: Optional[str]):
    if not contents:
        return no_update, html.Div()
    try:
        df = decode_uploaded_file(contents)
        store = {"source": "upload", "columns": list(df.columns), "data": df.to_dict(orient="records")}
        return store, preview_text(df, f"Aperçu du fichier : {filename}")
    except Exception as e:
        return None, f"Erreur de lecture du fichier : {e}"

@callback(
    Output("download-data", "data"),
    Input("download-data-btn", "n_clicks"),
    State("data-store", "data"),
    prevent_initial_call=True,
)
def download_current(n_clicks: int, store: Optional[dict]):
    if not n_clicks or not store:
        return no_update
    df = pd.DataFrame(store["data"], columns=store.get("columns"))
    return dcc.send_data_frame(df.to_csv, "donnees.csv", index=False)

@callback(
    Output("forecast-output", "children"),
    Input("run-forecasts-button", "n_clicks"),
    State("model-dropdown", "value"),
    State("data-store", "data"),
)
def run_forecasts(n_clicks: int, model_filename: Optional[str], store: Optional[dict]):
    if not (n_clicks and model_filename):
        return html.Div()

    if not store or not store.get("data"):
        return html.Div("Aucune donnée disponible : charge un CSV ou l’API dans l’onglet courant.")

    # Chargement données
    df = pd.DataFrame(store["data"], columns=store.get("columns"))

    # Chargement modèle
    try:
        model = load_model(model_filename)
    except Exception as e:
        return html.Div([f"Erreur lors du chargement du modèle : {e}"])

    # Préparation X
    try:
        if not hasattr(model, "feature_names_in_"):
            return html.Div("Le modèle ne contient pas 'feature_names_in_' (sklearn ≥ 1.0 requis).")

        missing = [c for c in model.feature_names_in_ if c not in df.columns]
        if missing:
            return html.Div([f"Colonnes manquantes pour le modèle : {missing}"])

        X = df[model.feature_names_in_].replace("ND", np.nan).apply(pd.to_numeric, errors="coerce").dropna()
        if X.empty:
            return html.Div("Aucune ligne exploitable après nettoyage (NaN/ND).")
    except Exception as e:
        return html.Div([f"Erreur préparation des features : {e}"])

    # Prédiction
    try:
        y_pred = model.predict(X)
    except Exception as e:
        return html.Div([f"Erreur pendant la prédiction : {e}"])

    # Axe de temps
    if {"Date", "Heures"}.issubset(df.columns):
        df["Datetime"] = pd.to_datetime(df["Date"] + " " + df["Heures"], errors="coerce")
    elif "date" in df.columns:
        df["Datetime"] = pd.to_datetime(df["date"], errors="coerce")
    else:
        df["Datetime"] = pd.date_range(start="2024-01-01", periods=len(df), freq="H")

    df_pred = df.loc[X.index].copy()
    df_pred["Prix Spot Prédit"] = y_pred

    # Données réelles (France.csv)
    try:
        df_spot = pd.read_csv(FRANCE_CSV)
        df_spot["Datetime (Local)"] = pd.to_datetime(df_spot["Datetime (Local)"])
    except Exception as e:
        return html.Div([f"Impossible de lire {FRANCE_CSV} : {e}"])

    # Jointure & métrique
    df_eval = pd.merge(df_pred, df_spot, left_on="Datetime", right_on="Datetime (Local)", how="inner")
    if df_eval.empty:
        return html.Div("Aucune correspondance temporelle entre vos données et France.csv.")
    df_eval = df_eval.rename(columns={"Price (EUR/MWhe)":"Prix Spot Réel"}).sort_values("Datetime").reset_index(drop=True)

    rmse = np.sqrt(mean_squared_error(df_eval["Prix Spot Réel"], df_eval["Prix Spot Prédit"]))
    mae = np.mean(np.abs(df_eval["Prix Spot Réel"] - df_eval["Prix Spot Prédit"]))
    mape = np.mean(np.abs((df_eval["Prix Spot Réel"] - df_eval["Prix Spot Prédit"]) / df_eval["Prix Spot Réel"])) * 100
    r2 = r2_score(df_eval["Prix Spot Réel"], df_eval["Prix Spot Prédit"])

    # Figure
    fig = px.line(
        df_eval, x="Datetime", y=["Prix Spot Prédit", "Prix Spot Réel"],
        labels={"value":"Prix (EUR/MWhe)", "variable":"Série"}
    )
    fig.update_layout(xaxis_title="Date", yaxis_title="Prix")

    

    return html.Div([
        dcc.Graph(figure=fig),
        html.P(f"RMSE du modèle : {rmse:.2f}", style={"fontWeight":"bold","marginTop":"10px"}),
        html.P(f"MAE du modèle : {mae:.2f}", style={"fontWeight":"bold","marginTop":"4px"}),
        html.P(f"MAPE du modèle : {mape:.2f}%", style={"fontWeight":"bold","marginTop":"4px"}),
        html.P(f"R² du modèle : {r2:.3f}", style={"fontWeight":"bold","marginTop":"4px"}),
    ])
