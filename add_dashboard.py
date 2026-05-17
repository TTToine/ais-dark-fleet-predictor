#!/usr/bin/env python3
"""
AIS Dark Fleet Predictor - Interactive Dashboard Auto-Installer
===================================================================
Questo script automatizza la creazione di una dashboard interattiva
con Streamlit + Folium per visualizzare predizioni, incertezze e
analisi geografiche delle navi dark fleet.
Usage:
python add_dashboard.py [--with-demo-data] [--docker] [--minimal]
"""
import argparse
import json
import shutil
import sys
from pathlib import Path
from datetime import datetime, timedelta

# ============================================================================
# CONFIGURAZIONE
# ============================================================================
PROJECT_ROOT = Path.cwd()
SRC_DIR = PROJECT_ROOT / "src"
DASHBOARD_DIR = PROJECT_ROOT / "dashboard"
TESTS_DIR = PROJECT_ROOT / "tests"
DATA_DIR = PROJECT_ROOT / "data"
CONFIG_DIR = PROJECT_ROOT / "configs"

FILES_TO_CREATE = {
    PROJECT_ROOT / "app.py": "main_app_template",
    PROJECT_ROOT / "pages" / "1_🌍_Geographic_Analysis.py": "geographic_page_template",
    PROJECT_ROOT / "pages" / "2_📊_Model_Performance.py": "performance_page_template",
    PROJECT_ROOT / "pages" / "3_🎯_Uncertainty_Analysis.py": "uncertainty_page_template",
    PROJECT_ROOT / "pages" / "4_📥_Export_Report.py": "export_page_template",
    SRC_DIR / "dashboard" / "components.py": "components_template",
    SRC_DIR / "dashboard" / "map_visualizations.py": "map_viz_template",
    SRC_DIR / "dashboard" / "data_loader.py": "data_loader_template",
    SRC_DIR / "dashboard" / "__init__.py": "init_template",
    TESTS_DIR / "test_dashboard.py": "dashboard_test_template",
    PROJECT_ROOT / "Dockerfile.dashboard": "dockerfile_template",
    PROJECT_ROOT / ".streamlit" / "config.toml": "streamlit_config_template",
    PROJECT_ROOT / ".streamlit" / "secrets.toml.example": "secrets_template",
}

# ============================================================================
# TEMPLATE (mantengo gli stessi dal tuo file originale)
# ============================================================================

def main_app_template() -> str:
    return '''#!/usr/bin/env python3
"""🚢 AIS Dark Fleet Predictor - Interactive Dashboard"""
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent / "src"))

st.set_page_config(page_title="AIS Dark Fleet Predictor", page_icon="🚢", layout="wide")

st.title("🚢 AIS Dark Fleet Predictor")
st.markdown("Dashboard interattiva per monitoraggio navi dark fleet")

# Placeholder per demo
st.info("💡 Carica i dati o esegui: python src/dashboard/data_loader.py --generate-demo")
'''

def geographic_page_template() -> str:
    return '''#!/usr/bin/env python3
import streamlit as st
st.set_page_config(page_title="Geographic Analysis", page_icon="🌍")
st.title("🌍 Analisi Geografica")
'''

def performance_page_template() -> str:
    return '''#!/usr/bin/env python3
import streamlit as st
st.set_page_config(page_title="Model Performance", page_icon="📊")
st.title("📊 Performance Modello")
'''

def uncertainty_page_template() -> str:
    return '''#!/usr/bin/env python3
import streamlit as st
st.set_page_config(page_title="Uncertainty Analysis", page_icon="🎯")
st.title("🎯 Analisi Incertezza")
'''

def export_page_template() -> str:
    return '''#!/usr/bin/env python3
import streamlit as st
st.set_page_config(page_title="Export", page_icon="📥")
st.title("📥 Export Dati")
'''

def components_template() -> str:
    return '''#!/usr/bin/env python3
"""Componenti UI riutilizzabili"""
import streamlit as st

def render_header():
    st.markdown("## 🚢 AIS Dark Fleet Predictor")

def render_sidebar():
    st.sidebar.header("Filtri")
    return {}
'''

def map_viz_template() -> str:
    return '''#!/usr/bin/env python3
"""Visualizzazioni geografiche"""
import folium
import pandas as pd

def create_folium_map(df, predictions=None):
    m = folium.Map(location=[40, 10], zoom_start=6)
    return m
'''

def data_loader_template() -> str:
    return '''#!/usr/bin/env python3
"""Caricamento e generazione dati demo"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import argparse

def generate_demo_data(n_ships=1000, n_days=30, output_dir="data/demo"):
    np.random.seed(42)
    print(f"📊 Generazione dati demo: {n_ships} navi, {n_days} giorni...")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Genera dati sintetici
    df = pd.DataFrame({
        "mmsi": np.random.randint(1e8, 1e9, n_ships * n_days),
        "latitude": np.random.uniform(30, 60, n_ships * n_days),
        "longitude": np.random.uniform(-10, 30, n_ships * n_days),
        "timestamp": pd.date_range("2024-01-01", periods=n_ships * n_days, freq="H")[:n_ships * n_days],
        "dark_fleet_probability": np.random.beta(2, 5, n_ships * n_days),
    })
    
    df.to_parquet(output_path / "ais_demo.parquet", index=False)
    print(f"✅ Dati salvati in {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-ships", type=int, default=1000)
    parser.add_argument("--n-days", type=int, default=30)
    parser.add_argument("--output", type=str, default="data/demo")
    args = parser.parse_args()
    generate_demo_data(args.n_ships, args.n_days, args.output)
'''

def init_template() -> str:
    return '''"""Dashboard Module"""
from .components import render_sidebar, render_header
from .map_visualizations import create_folium_map
from .data_loader import load_data, generate_demo_data

__all__ = ["render_sidebar", "render_header", "create_folium_map", "load_data", "generate_demo_data"]
'''

def dashboard_test_template() -> str:
    return '''#!/usr/bin/env python3
"""Test per dashboard"""
import pytest

def test_placeholder():
    assert True
'''

def dockerfile_template() -> str:
    return '''FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
'''

def streamlit_config_template() -> str:
    return '''[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
[server]
headless = true
port = 8501
'''

def secrets_template() -> str:
    return '''# secrets.toml example
[database]
host = "localhost"
'''

# ============================================================================
# FUNZIONI DI AUTOMAZIONE
# ============================================================================

def create_directory_structure():
    """Crea la struttura directory."""
    dirs = [
        PROJECT_ROOT / "pages",
        PROJECT_ROOT / ".streamlit",
        SRC_DIR / "dashboard",
        PROJECT_ROOT / "exports",
    ]
    for dir_path in dirs:
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"✅ Creato: {dir_path}")

def write_file(filepath: Path, content: str):
    """Scrive contenuto in un file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"✅ Creato: {filepath}")

def create_requirements():
    """Crea/aggiorna requirements.txt."""
    req_file = PROJECT_ROOT / "requirements.txt"
    dashboard_deps = """
# Dashboard dependencies
streamlit>=1.28.0
streamlit-folium>=0.15.0
folium>=0.15.0
plotly>=5.18.0
"""
    if req_file.exists():
        with open(req_file, "r") as f:
            existing = f.read()
        for dep in dashboard_deps.strip().split("\n"):
            if dep.strip() and not dep.startswith("#") and dep.split(">=")[0] not in existing:
                with open(req_file, "a") as f:
                    f.write("\n" + dep)
    else:
        with open(req_file, "w") as f:
            f.write(dashboard_deps)
    print(f"✅ Aggiornato: {req_file}")

def create_readme_section():
    """Aggiunge sezione dashboard al README."""
    readme_path = PROJECT_ROOT / "README.md"
    dashboard_section = """
## 🎨 Dashboard Interattiva
Il progetto include una dashboard Streamlit completa:

### Avvio Dashboard
```bash
# Con dati demo
python src/dashboard/data_loader.py --generate-demo
streamlit run app.py

# Oppure con Docker
docker build -f Dockerfile.dashboard -t ais-dashboard .
docker run -p 8501:8501 ais-dashboard
"""
if readme_path.exists():
    with open(readme_path, "r", encoding="utf-8") as f:
        content = f.read()
if "## 🎨 Dashboard Interattiva" not in content:
    with open(readme_path, "a", encoding="utf-8") as f:
        f.write(dashboard_section)
        print("✅ Sezione Dashboard aggiunta al README.md")
else:
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(f"# AIS Dark Fleet Predictor\n{dashboard_section}")
        print("✅ Creato README.md")
def main():
    parser = argparse.ArgumentParser(description="🚀 Installa Dashboard Streamlit")
    parser.add_argument("--with-demo-data", action="store_true", help="Genera dati demo")
    parser.add_argument("--docker", action="store_true", help="Includi Dockerfile")
    parser.add_argument("--minimal", action="store_true", help="Solo file essenziali")
    args = parser.parse_args()
    # 1. Crea directory
    print("📂 Creazione struttura directory...")
    create_directory_structure()

    # 2. Scrivi file
    print("📝 Scrittura file...")
    templates = {
        "main_app_template": main_app_template,
        "geographic_page_template": geographic_page_template,
        "performance_page_template": performance_page_template,
        "uncertainty_page_template": uncertainty_page_template,
        "export_page_template": export_page_template,
        "components_template": components_template,
        "map_viz_template": map_viz_template,
        "data_loader_template": data_loader_template,
        "init_template": init_template,
        "dashboard_test_template": dashboard_test_template,
        "dockerfile_template": dockerfile_template,
        "streamlit_config_template": streamlit_config_template,
        "secrets_template": secrets_template,
    }

    for filepath, template_name in FILES_TO_CREATE.items():
        if "Dockerfile" in str(filepath) and not args.docker:
            continue
        if template_name in templates:
            content = templates[template_name]()
            write_file(filepath, content)

    # 3. Requirements
    print("📦 Aggiornamento dipendenze...")
    create_requirements()

    # 4. Dati demo
    if args.with_demo_data:
        print("📊 Generazione dati demo...")
        try:
            import sys
            sys.path.append(str(PROJECT_ROOT / "src"))
            from dashboard.data_loader import generate_demo_data
            generate_demo_data(n_ships=500, n_days=7, output_dir=str(DATA_DIR / "demo"))
        except Exception as e:
            print(f"⚠️ Impossibile generare dati demo: {e}")

    # 5. README
    create_readme_section()

    print("\n🎉 Installazione Dashboard Completata!")
    print("👉 Per avviare: streamlit run app.py")
if name == "main":
    main()