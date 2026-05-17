#!/usr/bin/env python3
"""Visualizzazioni geografiche"""
import folium
import pandas as pd

def create_folium_map(df, predictions=None):
    m = folium.Map(location=[40, 10], zoom_start=6)
    return m
