#!/usr/bin/env python3
"""
apply_fixes.py
Script automatico cross-platform per applicare tutte le fix del progetto.
Funziona su Windows (PowerShell/CMD), macOS e Linux.
Richiede Python 3.8+ già installato.
"""
import os
import sys
import shutil
import subprocess
from pathlib import Path
from datetime import datetime

def log(msg, level="info"):
    colors = {"info": "\033[0;32m", "warn": "\033[1;33m", "error": "\033[0;31m", "reset": "\033[0m"}
    icon = {"info": "✓", "warn": "!", "error": "✗"}[level]
    print(f"{colors[level]}[{icon}] {msg}{colors['reset']}")

def safe_write(path: str, content: str):
    Path(path).write_text(content.strip() + "\n", encoding="utf-8")

def patch_file(path: str, old: str, new: str):
    p = Path(path)
    if not p.exists(): return
    txt = p.read_text(encoding="utf-8")
    if old not in txt: return
    p.write_text(txt.replace(old, new, 1), encoding="utf-8")

def main():
    # 1. Verifica root progetto
    if not Path("Requirements.txt").exists() and not Path("src").is_dir():
        log("Errore: esegui questo script nella root di ais-dark-fleet-predictor", "error")
        sys.exit(1)

    confirm = input("🔧 Questo script creerà/modificherà file. Continuare? [y/N] ").strip().lower()
    if confirm != "y":
        log("Annullato.", "warn")
        return

    # 2. Backup
    backup_dir = Path(f".backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    backup_dir.mkdir(exist_ok=True)
    log(f"Backup in corso: {backup_dir}", "info")
    for f in ["run_pipeline.py", "example_run.py", "Dockerfile", "README.md"]:
        if Path(f).exists():
            shutil.copy(f, backup_dir)

    # 3. Directory
    log("Creazione directory...", "info")
    for d in ["configs", "src", "tests", ".github/workflows", "logs", "data/raw", "data/processed", ".cache/hmm"]:
        Path(d).mkdir(parents=True, exist_ok=True)

    # 4. Contenuti file (sintetizzati per spazio, ma completi)
    log("Creazione file di configurazione e moduli...", "info")
    
    safe_write("configs/pipeline_config.yaml", """
project:
  name: "ais-dark-fleet-predictor"
  version: "1.0.0"
geography:
  default_bbox: { min_lat: 34.0, max_lat: 39.0, min_lon: 10.0, max_lon: 16.0 }
  regions:
    mediterranean: { bbox: { min_lat: 30.0, max_lat: 46.0, min_lon: -6.0, max_lon: 36.0 }, gap_threshold_hours: 12.0 }
data_prep: { gap_threshold_hours: 12.0, downsampling_interval_minutes: 10 }
hmm_model: { window_size: 36, caching: { enabled: true, backend: "joblib", ttl_hours: 24 } }
gb_training: { n_trials: 50, n_splits: 5, gap_hours: 24.0 }
logging: { level: "INFO", format: "%(asctime)s - %(levelname)s - %(message)s", file: "logs/pipeline.log" }
performance: { n_jobs: -1, chunk_size_ships: 100, memory_limit_gb: 14 }
""")

    safe_write("src/exceptions.py", """
from typing import Optional, Dict, Any
from datetime import datetime

class AISPipelineError(Exception):
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None, recoverable: bool = False):
        self.timestamp = datetime.utcnow()
        self.context = context or {}
        self.recoverable = recoverable
        super().__init__(message)
    def to_dict(self) -> Dict[str, Any]:
        return {"error_type": self.__class__.__name__, "message": str(self), "timestamp": self.timestamp.isoformat(), "context": self.context, "recoverable": self.recoverable}

class DataValidationError(AISPipelineError): pass
class CausalLeakageError(AISPipelineError): pass
class ModelInferenceError(AISPipelineError): pass
class ConfigurationError(AISPipelineError): pass
class ResourceExhaustionError(AISPipelineError): pass
""")

    safe_write("src/logging_config.py", """
import logging, sys, json
from pathlib import Path
from logging.handlers import RotatingFileHandler
from .exceptions import AISPipelineError

class StructuredFormatter(logging.Formatter):
    def format(self, record):
        if getattr(record, 'json_output', False):
            return json.dumps({"ts": self.formatTime(record), "lvl": record.levelname, "msg": record.getMessage(), "ctx": getattr(record, 'context', {})})
        return super().format(record)

def setup_logging(config: dict, log_dir: str = None, json_logs: bool = False) -> logging.Logger:
    logger = logging.getLogger(config.get('project', {}).get('name', 'ais-predictor'))
    logger.setLevel(getattr(logging, config.get('logging', {}).get('level', 'INFO')))
    if logger.hasHandlers(): logger.handlers.clear()
    fmt = StructuredFormatter(datefmt='%Y-%m-%dT%H:%M:%S') if json_logs else logging.Formatter(config.get('logging', {}).get('format', '%(asctime)s - %(levelname)s - %(message)s'))
    ch = logging.StreamHandler(sys.stdout); ch.setFormatter(fmt); logger.addHandler(ch)
    if config.get('logging', {}).get('file'):
        log_path = Path(log_dir or '.') / config['logging']['file']
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = RotatingFileHandler(log_path, maxBytes=100*1024*1024, backupCount=5)
        fh.setFormatter(fmt); fh.setLevel(logging.DEBUG); logger.addHandler(fh)
    return logger
""")

    safe_write("src/hmm_cache.py", """
import hashlib, pickle, logging, joblib
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
from .exceptions import ResourceExhaustionError

logger = logging.getLogger(__name__)

class HMMCache:
    def __init__(self, enabled=True, backend="joblib", cache_dir=".cache/hmm", ttl_hours=24, memory_limit_gb=14.0):
        self.enabled, self.backend, self.ttl, self.memory_limit = enabled, backend, timedelta(hours=ttl_hours), int(memory_limit_gb * 1024**3)
        if enabled and backend == "joblib":
            self.cache_path = Path(cache_dir); self.cache_path.mkdir(parents=True, exist_ok=True)
    def _key(self, mmsi, df, cfg):
        return hashlib.sha256(f"{mmsi}:{hash(str(df.values))}:{hash(str(cfg))}".encode()).hexdigest()[:16]
    def get(self, mmsi, df, cfg) -> Optional[Dict[str, Any]]:
        if not self.enabled: return None
        try:
            f = self.cache_path / f"{self._key(mmsi, df, cfg)}.pkl"
            if f.exists() and (datetime.now() - datetime.fromtimestamp(f.stat().st_mtime)) <= self.ttl:
                return joblib.load(f)
        except: pass
        return None
    def set(self, mmsi, df, cfg, result) -> bool:
        if not self.enabled: return False
        try:
            f = self.cache_path / f"{self._key(mmsi, df, cfg)}.pkl"
            joblib.dump(result, f, compress=3); return True
        except: return False
""")

    # 5. Patch file esistenti
    log("Applicazione patch ai moduli esistenti...", "info")
    patch_file("src/hmm_model.py", "import pandas as pd", "import pandas as pd\nfrom .hmm_cache import HMMCache\nfrom .exceptions import ModelInferenceError\nfrom .logging_config import setup_logging")
    patch_file("src/hmm_model.py", "def __init__(self, window_size=36, **kwargs):", "def __init__(self, window_size=36, config=None, **kwargs):\n        self.config = config or {}\n        self.cache = HMMCache(enabled=self.config.get('hmm_model', {}).get('caching', {}).get('enabled', True))")
    patch_file("src/data_prep.py", "import pandas as pd", "import pandas as pd\nfrom .data_validation import AISDataValidator, validate_and_adapt\nfrom .exceptions import DataValidationError")

    # 6. Aggiorna Requirements
    log("Aggiornamento dipendenze...", "info")
    new_deps = "\npyyaml>=6.0\njoblib>=1.3.0\nmapie>=0.8.0\npytest>=7.4.0\npytest-cov>=4.1.0"
    req = Path("Requirements.txt")
    if new_deps.strip().split("\n")[0] not in req.read_text():
        req.write_text(req.read_text() + new_deps, encoding="utf-8")

    # 7. Verifica sintassi
    log("Verifica sintassi Python...", "info")
    res = subprocess.run([sys.executable, "-m", "py_compile", "src/exceptions.py", "src/logging_config.py", "src/hmm_cache.py"], capture_output=True)
    if res.returncode == 0:
        log("✅ Tutti i file Python sono sintatticamente validi", "info")
    else:
        log(f"⚠️ Warning di sintassi (verifica manualmente): {res.stderr.decode()}", "warn")

    log("🚀 Script completato!", "info")
    print("\n📋 Prossimi passi:")
    print("  1. pip install -e .")
    print("  2. pytest tests/ -v")
    print("  3. python run_pipeline.py --dry-run")
    print(f"  📁 Backup salvati in: {backup_dir}")

if __name__ == "__main__":
    main()