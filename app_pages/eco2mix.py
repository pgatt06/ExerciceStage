# app_pages/eco2mix.py
from dash import html, dcc, Input, Output, callback
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import numpy as np


# -------------------------------
# Données
# -------------------------------
CHEMIN_DONNEES = "data/eCO2mix_RTE_En-cours-Consolide.csv"

FILIÈRES = [
    "Nucléaire", "Gaz", "Hydraulique", "Eolien", "Solaire",
    "Fioul", "Charbon", "Bioénergies", "Pompage", "Export.", "Import.",
]

# -------------------------------
# Fonction 
# -------------------------------

# Charger les données
def charger_donnees():
    df = pd.read_csv(CHEMIN_DONNEES, sep="\t", encoding="latin1", index_col=False)
    df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Heures'], errors='coerce')
    df = df.dropna()
    df.sort_values(by="Datetime", inplace=True)
    # def de import et export 
    df["Export."] = df.filter(regex="Ech. comm.").apply(lambda row: row[row < 0].sum(), axis=1)
    df["Import."] = df.filter(regex="Ech. comm.").apply(lambda row: row[row > 0].sum(), axis=1)
    return df

# Agrégation temporelle
def resample_power(dff: pd.DataFrame, freq: str, cols: list[str]):
    if freq not in ["30min", "h", "D", "W"]:
        return dff[["Datetime"] + cols].copy()

    freq_map = {
        "30min": "30min",
        "h": "h",
        "D": "D",
        "W": "W-MON"
    }
    return (dff.set_index("Datetime")[cols]
            .resample(freq_map[freq]).mean()
            .reset_index())

# -------------------------------
# Palette de couleurs par filière
# -------------------------------
PALETTE = {
    "Nucléaire": "#1f77b4",   
    "Gaz": "#ff7f0e",
    "Hydraulique": "#2ca02c",
    "Eolien": "#17becf",
    "Solaire": "#bcbd22",
    "Fioul": "#d62728",
    "Charbon": "#7f7f7f",
    "Bioénergies": "#8c564b",
    "Pompage": "#9467bd",
    "Export.": "#e377c2",
    "Import.": "#7f7f7f",
}

df = charger_donnees()

DISPO = [c for c in FILIÈRES if c in list(df.columns) + ["Export.", "Import."]]
date_min = pd.to_datetime(df["Datetime"].min())
date_max = pd.to_datetime(df["Datetime"].max())


# -------------------------------
# Mise en page
# -------------------------------
header = dbc.Card(
    dbc.CardBody([
        html.H2("Production d’électricité par filière", className="m-0"),
        html.Div("Source : éCO2mix (RTE) – Visualisation", className="text-muted"),
    ]),
    className="mb-3",
)

sidebar = dbc.Card(
    dbc.CardBody([
        html.Label("Période", className="fw-semibold"),
        dcc.DatePickerRange(
            id="eco2-date",
            min_date_allowed=date_min.date(),
            max_date_allowed=date_max.date(),
            start_date=max(date_min, date_max - pd.Timedelta(days=30)).date(),
            end_date=date_max.date(),
            display_format="DD/MM/YYYY",
            className="mb-3",
        ),
        html.Label("Granularité", className="fw-semibold"),
        dcc.Dropdown(
            id="eco2-freq",
            options=[
                {"label": "30 minutes", "value": "30min"},
                {"label": "Heure", "value": "h"},
                {"label": "Jour", "value": "D"},
                {"label": "Semaine", "value": "W"},
            ],
            value="h",
            clearable=False,
            className="mb-3",
        ),
        html.Label("Filières", className="fw-semibold"),
        dcc.Dropdown(
            id="eco2-series",
            options=[{"label": c, "value": c} for c in DISPO],
            value=[c for c in ["Nucléaire", "Eolien", "Solaire", "Hydraulique"] if c in DISPO] or DISPO[:4],
            multi=True,
            placeholder="Choisir les filières…",
            className="mb-3",
        ),
        dcc.Checklist(
            id="eco2-options",
            options=[
                {"label": " Afficher en % du total", "value": "percent"},
                {"label": " Empiler les aires", "value": "stack"},
            ],
            value=["stack"],
            className="mb-2",
        ),
        html.Span(
            "i",
            id="percent-help",
            title="Info",
            className="ms-1 me-auto d-inline-flex align-items-center justify-content-center border rounded-circle",
            style={"width":"18px","height":"18px","fontWeight":"600","cursor":"pointer","fontSize":"12px"}
        ),
        dbc.Tooltip(
            "Pour le calcul des pourcentages, les valeurs négatives (pompage, exportation) "
            "ne sont pas prises en compte dans le total.",
            target="percent-help",
            placement="right",
        ),
            ]),
            className="mb-3",
)

# -------------------------------
kpi_cards = dbc.Row([
    dbc.Col(dbc.Card(dbc.CardBody([
        html.Div([
            "Puissance moyenne",
            html.Span(" i", id="kpi-avg-help", className="ms-1 text-secondary", style={"cursor":"pointer"}),
            dbc.Tooltip(
                "Moyenne sur la période sélectionnée. Les valeurs négatives (ex : pompage, exportation) ne sont pas prises en compte.",
                target="kpi-avg-help", placement="right"
            ),
        ], className="text-muted small"),
        html.H4(id="kpi-avg", className="m-0"),
    ])), md=4),

    dbc.Col(dbc.Card(dbc.CardBody([
        html.Div([
            "Pic de puissance",
            html.Span(" i", id="kpi-max-help", className="ms-1 text-secondary", style={"cursor":"pointer"}),
            dbc.Tooltip(
                "Valeur maximale observée (avec horodatage en dessous). Les valeurs négatives (ex : pompage, exportation) ne sont pas prises en compte.",
                target="kpi-max-help", placement="right"
            ),
        ], className="text-muted small"),
        html.H4(id="kpi-max", className="m-0"),
        html.Div(id="kpi-max-ts", className="text-muted small"),
    ])), md=4),

    dbc.Col(dbc.Card(dbc.CardBody([
        html.Div([
            "Creux de puissance",
            html.Span(" i", id="kpi-min-help", className="ms-1 text-secondary", style={"cursor":"pointer"}),
            dbc.Tooltip(
                "Valeur minimale observée. Les valeurs négatives (ex : pompage, exportation) ne sont pas prises en compte.",
                target="kpi-min-help", placement="right"
            ),
        ], className="text-muted small"),
        html.H4(id="kpi-min", className="m-0"),
        html.Div(id="kpi-min-ts", className="text-muted small"),
    ])), md=4),
], className="g-3 mb-3")


# -------------------------------

main_charts = dbc.Row([
    dbc.Col(dbc.Card(dbc.CardBody([
        html.H5("Évolution par filière", className="mb-2"),
        dcc.Graph(id="eco2-area", config={"displaylogo": False}),
    ])), md=8, className="mb-3"),
    dbc.Col(dbc.Card(dbc.CardBody([
        html.H5("Mix sur la période", className="mb-2"),
        dcc.Graph(id="eco2-pie", config={"displaylogo": False}),
        html.Div(id="eco2-note", className="text-muted small"),
    ])), md=4, className="mb-3"),
])

eco2mix_layout = html.Div([
    header,
    dbc.Row([
        dbc.Col(sidebar, md=3),
        dbc.Col(html.Div([kpi_cards, main_charts]), md=9),
    ], className="g-3"),
], className="p-3")

# -------------------------------
# Callback unique (graphes + KPI)
# -------------------------------
@callback(
    Output("eco2-area", "figure"),
    Output("eco2-pie", "figure"),
    Output("eco2-note", "children"),
    Output("kpi-avg", "children"),
    Output("kpi-max", "children"),
    Output("kpi-max-ts", "children"),
    Output("kpi-min", "children"),
    Output("kpi-min-ts", "children"),
    Input("eco2-date", "start_date"),
    Input("eco2-date", "end_date"),
    Input("eco2-freq", "value"),
    Input("eco2-series", "value"),
    Input("eco2-options", "value"),
)




def maj_graphs(start_date, end_date, freq, series, options):
    options = options or []
    series = [s for s in (series or []) if s in df.columns]
    if not series:
        series = DISPO[:3]

    dff = df[(df["Datetime"] >= pd.to_datetime(start_date)) &
             (df["Datetime"] <= pd.to_datetime(end_date))].copy()

    if dff.empty:
        empty = px.line(pd.DataFrame({"Datetime": [], "val": []}), x="Datetime", y="val")
        empty.update_layout(
            template="plotly_white",
            annotations=[dict(text="Aucune donnée sur la période", x=0.5, y=0.5, xref="paper", yref="paper",
                              showarrow=False)]
        )
        return empty, empty, "", "—", "—", "", "—", ""

    # Agrégation temporelle
    agg = resample_power(dff[["Datetime"] + series], freq, series)

    # -------- KPI (total sélection) --------
    # Ne garder que les filières qui produisent de l'énergie
    filieres_prod = [f for f in series if f not in ["Export.", "Import.", "Pompage"]]
    total_series = agg[filieres_prod].sum(axis=1)
    moy = np.nanmean(total_series) if len(total_series) else np.nan
    idx_max = int(np.nanargmax(total_series)) if len(total_series) else None
    idx_min = int(np.nanargmin(total_series)) if len(total_series) else None

    def fmt_mw(x):
        if pd.isna(x):
            return "—"
        return f"{x:,.0f} MW".replace(",", " ")

    kpi_avg = fmt_mw(moy)
    kpi_max = fmt_mw(total_series.iloc[idx_max]) if idx_max is not None else "—"
    kpi_min = fmt_mw(total_series.iloc[idx_min]) if idx_min is not None else "—"
    kpi_max_ts = (agg["Datetime"].iloc[idx_max].strftime("%d/%m/%Y %H:%M")
                  if idx_max is not None else "")
    kpi_min_ts = (agg["Datetime"].iloc[idx_min].strftime("%d/%m/%Y %H:%M")
                  if idx_min is not None else "")

    # -------- PLOTS --------
    long = agg.melt(id_vars="Datetime", value_vars=series, var_name="Filière", value_name="Puissance (MW)")

    percent_mode = "percent" in options
    stack = "stack" in options

    plot_df = agg[["Datetime"] + series].copy()

    if percent_mode:
        # Ne garder que les valeurs positives pour le calcul des pourcentages
        pos_df = plot_df[series].clip(lower=0)
        total_pos = pos_df.sum(axis=1).replace(0, np.nan)
        pct_df = pos_df.div(total_pos, axis=0) * 100
        area_df = pct_df.copy()
        area_df["Datetime"] = plot_df["Datetime"]
        area_long = area_df.melt(id_vars="Datetime", value_vars=series, var_name="Filière", value_name="% du total")
        # Supprimer les lignes où "% du total" est NaN ou <= 0
        area_long = area_long[area_long["% du total"] > 0]
        fig_area = px.area(
            area_long,
            x="Datetime",
            y="% du total",
            color="Filière",
            color_discrete_map=PALETTE,
            groupnorm="percent" if stack else "",
            labels={"% du total": "% du total"},
            hover_data={"Filière": True, "% du total": ":.2f"},
        )
        if not stack:
            fig_area.update_traces(stackgroup=None)
        y_title = "% du total"
    else:
        if stack:
            # On sépare positives et négatives pour afficher les aires correctement
            area_long = long.copy()
            area_long_pos = area_long.copy()
            area_long_pos["Puissance (MW)"] = area_long_pos["Puissance (MW)"].clip(lower=0)
            area_long_pos = area_long_pos[area_long_pos["Puissance (MW)"] > 0]

            area_long_neg = area_long.copy()
            area_long_neg["Puissance (MW)"] = area_long_neg["Puissance (MW)"].clip(upper=0)
            area_long_neg = area_long_neg[area_long_neg["Puissance (MW)"] < 0]

            # Positives
            if area_long_pos.empty:
                fig_area = px.area(pd.DataFrame({"Datetime": [], "val": []}), x="Datetime", y="val")
            else:
                fig_area = px.area(
                    area_long_pos,
                    x="Datetime",
                    y="Puissance (MW)",
                    color="Filière",
                    color_discrete_map=PALETTE,
                    labels={"Puissance (MW)": "Puissance (MW)"},
                    hover_data={"Filière": True, "Puissance (MW)": ":.2f"},
                )

            # Negatives
            fig_neg = None
            if not area_long_neg.empty:
                fig_neg = px.area(
                    area_long_neg,
                    x="Datetime",
                    y="Puissance (MW)",  # valeurs < 0 -> aires sous 0
                    color="Filière",
                    color_discrete_map=PALETTE,
                    labels={"Puissance (MW)": "Puissance (MW)"},
                    hover_data={"Filière": True, "Puissance (MW)": ":.2f"},
                )
            if "stack" in options: 
                for tr in getattr(fig_area, "data", []):
                    tr.stackgroup = "pos"
                    tr.fill = "tonexty"
                if fig_neg is not None:
                    fig_area.add_traces(fig_neg.data)
    

            y_title = "Puissance (MW)"
        else:
            fig_area = px.line(
                long,
                x="Datetime",
                y="Puissance (MW)",
                color="Filière",
                color_discrete_map=PALETTE,
                labels={"Puissance (MW)": "Puissance (MW)"},
                hover_data={"Filière": True, "Puissance (MW)": ":.2f"},
            )
            
            y_title = "Puissance (MW)"

    fig_area.update_layout(
        template="plotly_white",
        hovermode="x unified",
        margin=dict(l=30, r=20, t=10, b=30),
        legend_title_text="",
        yaxis_title=y_title,
        shapes=[dict(type="line", xref="paper", x0=0, x1=1, yref="y", y0=0, y1=0,
                     line=dict(width=1, dash="dot", color="#888"))],
    )

    # -------- Camembert du mix sur la période --------
    # Calcul du mix en ne prenant que les valeurs positives
    mix_vals = agg[series].clip(lower=0).mean(numeric_only=True)
    mix_df = mix_vals.reset_index()
    mix_df.columns = ["Filière", "Puissance moyenne (MW)"]
    mix_df = mix_df[mix_df["Puissance moyenne (MW)"].fillna(0) > 0]

    if mix_df.empty:
        fig_pie = px.pie(values=[1], names=["(aucune)"])
    else:
        fig_pie = px.pie(
            mix_df, values="Puissance moyenne (MW)", names="Filière",
            color="Filière", color_discrete_map=PALETTE, hole=0.5
        )
    fig_pie.update_layout(
        template="plotly_white",
        margin=dict(l=10, r=10, t=10, b=10),
        legend_title_text="",
    )

    note = f"Période du {pd.to_datetime(start_date).strftime('%d/%m/%Y')} au {pd.to_datetime(end_date).strftime('%d/%m/%Y')} – granularité { {'30min': '30 minutes','h':'heure','D':'jour','W':'semaine'}[freq] }."

    return fig_area, fig_pie, note, kpi_avg, kpi_max, kpi_max_ts, kpi_min, kpi_min_ts




