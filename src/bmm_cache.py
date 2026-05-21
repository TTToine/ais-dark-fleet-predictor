"""Content-addressed cache for per-vessel Bayesian Mixture Model inference results.

Key design:
- Deterministic SHA-256 keys based on (mmsi, df_content_hash, cfg_json).
- df hashing via ``pd.util.hash_pandas_object`` (stable across processes,
  unlike Python's built-in ``hash()`` which is randomised per-process).
- cfg hashing via ``json.dumps(..., sort_keys=True)`` so dict insertion
  order does not change the key.
- Narrow exception handling: ``KeyboardInterrupt`` / ``MemoryError``
  always propagate. Corrupted pickles are removed so they get rebuilt.
"""

from __future__ import annotations

import hashlib
import json
import logging
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import pandas as pd

logger = logging.getLogger(__name__)


class BMMCache:
    """Joblib-backed cache for BMM per-vessel inference results."""

    def __init__(
        self,
        enabled: bool = True,
        backend: str = "joblib",
        cache_dir: str = ".cache/bmm",
        ttl_hours: float = 24,
    ):
        self.enabled = enabled
        self.backend = backend
        self.ttl = timedelta(hours=ttl_hours)
        if enabled and backend == "joblib":
            self.cache_path = Path(cache_dir)
            self.cache_path.mkdir(parents=True, exist_ok=True)

    def _key(self, mmsi: Any, df: pd.DataFrame, cfg: Dict[str, Any]) -> str:
        """Deterministic content-addressed hash.

        - ``pd.util.hash_pandas_object`` produces a per-row uint64 series
          based on values + index; stable across processes.
        - ``json.dumps(sort_keys=True, default=str)`` makes dict ordering
          irrelevant and tolerates non-JSON-serialisable leaves.
        - Returns the full 64-char SHA-256 hex digest.
        """
        h = hashlib.sha256()
        h.update(str(mmsi).encode("utf-8"))
        h.update(b"\x00")
        h.update(pd.util.hash_pandas_object(df, index=True).values.tobytes())
        h.update(b"\x00")
        h.update(json.dumps(cfg, sort_keys=True, default=str).encode("utf-8"))
        return h.hexdigest()

    def get(self, mmsi: Any, df: pd.DataFrame, cfg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not self.enabled:
            return None
        f = self.cache_path / f"{self._key(mmsi, df, cfg)}.pkl"
        if not f.exists():
            return None
        try:
            if (datetime.now() - datetime.fromtimestamp(f.stat().st_mtime)) > self.ttl:
                return None
            return joblib.load(f)
        except MemoryError:
            raise
        except (FileNotFoundError, EOFError, pickle.UnpicklingError, OSError,
                IndexError, AttributeError, ValueError, TypeError) as e:
            logger.warning(
                "BMMCache.get: failed to load %s (%s: %s); deleting corrupted file.",
                f, type(e).__name__, e,
            )
            try:
                f.unlink(missing_ok=True)
            except OSError as unlink_err:
                logger.warning("BMMCache.get: could not unlink %s: %s", f, unlink_err)
            return None

    def set(self, mmsi: Any, df: pd.DataFrame, cfg: Dict[str, Any], result: Any) -> bool:
        if not self.enabled:
            return False
        f = self.cache_path / f"{self._key(mmsi, df, cfg)}.pkl"
        try:
            joblib.dump(result, f, compress=3)
            return True
        except (OSError, pickle.PicklingError) as e:
            logger.warning(
                "BMMCache.set: failed to write %s (%s: %s).",
                f, type(e).__name__, e,
            )
            return False
