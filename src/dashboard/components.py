#!/usr/bin/env python3
"""Componenti UI riutilizzabili"""
import streamlit as st

def render_header():
    st.markdown("## 🚢 AIS Dark Fleet Predictor")

def render_sidebar():
    st.sidebar.header("Filtri")
    return {}
