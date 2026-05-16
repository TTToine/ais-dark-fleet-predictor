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
