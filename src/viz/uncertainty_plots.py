import numpy as np
import pandas as pd
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
