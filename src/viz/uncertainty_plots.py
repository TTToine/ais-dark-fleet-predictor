import logging
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px

def create_interactive_map_with_uncertainty(df, lat_col="lat", lon_col="lon", prob_col="prob", lower_col="low", upper_col="high", title="Risk Map", zoom=3):
    df = df.copy()
    df["width"] = df[upper_col] - df[lower_col]
    df["size"] = df["width"] * 30 + 8
    fig = go.Figure(go.Scattermapbox(lat=df[lat_col], lon=df[lon_col], mode="markers",
        marker=dict(size=df["size"], color=df[prob_col], colorscale="RdYlGn_r", reversescale=True)))
    fig.update_layout(title=title, mapbox_style="open-street-map", height=600, margin=dict(l=0,r=0,t=50,b=0))
    return fig

def generate_uncertainty_report(df, prob_col="prob", lower_col="low", upper_col="high", output_html="report.html"):
    map_fig = create_interactive_map_with_uncertainty(df, prob_col=prob_col, lower_col=lower_col, upper_col=upper_col)
    html = f"<html><body><h1>Uncertainty Report</h1>{map_fig.to_html(full_html=False)}</body></html>"
    with open(output_html, "w") as f: f.write(html)
    return html


def plot_geographic_predictions(df: pd.DataFrame,
                                lat_col: str = 'Lat',
                                lon_col: str = 'Lon',
                                prob_col: str = 'prob_regime_sospetto',
                                mmsi_col: str = 'MMSI',
                                save_path: str = 'models/geographic_predictions.html') -> str:
    """
    Plotta su mappa interattiva (folium) le ultime posizioni note di ogni nave,
    colorate per prob_regime_sospetto con gradiente verde → giallo → rosso.
    Il raggio del cerchio scala con la probabilità.

    Fallback automatico su matplotlib statico se folium non è installato.
    Installazione: pip install folium

    Args:
        df: DataFrame con colonne Lat, Lon, prob_regime_sospetto e (opz.) MMSI, Timestamp.
        save_path: path di output (.html per folium, convertito a .png nel fallback).

    Returns:
        path del file salvato.
    """
    # Per ogni nave prendi l'ultima posizione nota
    if mmsi_col in df.columns and 'Timestamp' in df.columns:
        last_pos = (df.sort_values('Timestamp')
                    .groupby(mmsi_col, as_index=False)
                    .last())
    else:
        last_pos = df.copy()

    last_pos = last_pos.dropna(subset=[lat_col, lon_col, prob_col])

    if len(last_pos) == 0:
        logging.warning("plot_geographic_predictions: nessuna posizione valida trovata.")
        return save_path

    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)

    try:
        import folium

        center_lat = float(last_pos[lat_col].mean())
        center_lon = float(last_pos[lon_col].mean())
        m = folium.Map(location=[center_lat, center_lon], zoom_start=7,
                       tiles='CartoDB positron')

        def _prob_to_hex(p: float) -> str:
            p = max(0.0, min(1.0, p))
            r = int(min(255, p * 2 * 255))
            g = int(min(255, (1 - p) * 2 * 255))
            return f'#{r:02x}{g:02x}00'

        for _, row in last_pos.iterrows():
            prob  = float(row[prob_col])
            color = _prob_to_hex(prob)
            mmsi_label = str(int(row[mmsi_col])) if mmsi_col in row.index else 'N/A'

            folium.CircleMarker(
                location=[float(row[lat_col]), float(row[lon_col])],
                radius=5 + prob * 10,
                color=color,
                fill=True,
                fill_opacity=0.8,
                popup=folium.Popup(
                    f"<b>MMSI:</b> {mmsi_label}<br>"
                    f"<b>P(sospetto):</b> {prob:.3f}<br>"
                    f"<b>Lat:</b> {row[lat_col]:.4f}  <b>Lon:</b> {row[lon_col]:.4f}",
                    max_width=220
                ),
                tooltip=f"MMSI {mmsi_label} — P={prob:.2f}"
            ).add_to(m)

        legend_html = """
        <div style="position:fixed;bottom:30px;right:30px;z-index:9999;
             background:white;padding:10px 14px;border-radius:8px;
             border:1px solid #bbb;font-size:13px;">
          <b>P(regime sospetto)</b><br>
          <span style="color:#00ff00;font-size:18px">&#9632;</span> 0.0 — Normale<br>
          <span style="color:#ffff00;font-size:18px">&#9632;</span> 0.5 — Incerto<br>
          <span style="color:#ff0000;font-size:18px">&#9632;</span> 1.0 — Sospetto
        </div>"""
        m.get_root().html.add_child(folium.Element(legend_html))

        m.save(save_path)
        logging.info(f"🗺️  Mappa geografica interattiva salvata in {save_path}")
        return save_path

    except ImportError:
        logging.warning(
            "folium non installato: fallback su matplotlib statico. "
            "Installa con: pip install folium"
        )
        fig, ax = plt.subplots(figsize=(12, 8))
        sc = ax.scatter(
            last_pos[lon_col], last_pos[lat_col],
            c=last_pos[prob_col], cmap='RdYlGn_r',
            vmin=0, vmax=1, s=80, alpha=0.8,
            edgecolors='black', linewidths=0.5
        )
        plt.colorbar(sc, ax=ax, label='P(regime sospetto)')
        ax.set_xlabel('Longitudine')
        ax.set_ylabel('Latitudine')
        ax.set_title('Ultime posizioni note — Probabilità regime sospetto', fontsize=13)
        ax.grid(alpha=0.3)

        save_png = save_path.replace('.html', '.png')
        plt.savefig(save_png, dpi=300, bbox_inches='tight')
        logging.info(f"🗺️  Mappa statica salvata in {save_png}")
        plt.close()
        return save_png
