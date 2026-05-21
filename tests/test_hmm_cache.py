"""Tests for src.bmm_cache.BMMCache.

Also asserts that the deprecation shim ``src.hmm_cache.HMMCache`` still
resolves to ``BMMCache``.
"""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.bmm_cache import BMMCache

REPO_ROOT = Path(__file__).resolve().parents[1]


def _make_df(seed: int = 0, n: int = 50) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "SOG": rng.normal(8, 2, n),
            "COG": rng.uniform(0, 360, n),
            "Timestamp": pd.date_range("2024-01-01", periods=n, freq="10min"),
        }
    )


def test_set_then_get_roundtrips_dict(tmp_path):
    cache = BMMCache(cache_dir=str(tmp_path), ttl_hours=1)
    df = _make_df()
    cfg = {"window_size": 36, "advi_steps": 300}
    payload = {"prob": np.array([0.1, 0.9]), "n": 2}

    assert cache.set("123456789", df, cfg, payload) is True
    loaded = cache.get("123456789", df, cfg)
    assert loaded is not None
    assert loaded["n"] == 2
    np.testing.assert_array_equal(loaded["prob"], payload["prob"])


def test_row_difference_changes_key(tmp_path):
    cache = BMMCache(cache_dir=str(tmp_path))
    df1 = _make_df()
    df2 = df1.copy()
    df2.iloc[0, df2.columns.get_loc("SOG")] += 0.001  # single tiny change

    cfg = {"a": 1}
    k1 = cache._key("MMSI1", df1, cfg)
    k2 = cache._key("MMSI1", df2, cfg)
    assert k1 != k2
    assert len(k1) == 64  # full SHA-256 hex


def test_cfg_key_order_does_not_change_hash(tmp_path):
    cache = BMMCache(cache_dir=str(tmp_path))
    df = _make_df()
    k1 = cache._key("X", df, {"a": 1, "b": 2, "c": 3})
    k2 = cache._key("X", df, {"c": 3, "b": 2, "a": 1})
    assert k1 == k2


def test_key_stable_across_processes(tmp_path):
    """Hash must be identical across two interpreter processes with
    different ``PYTHONHASHSEED`` values. This is the core fix vs. the
    old ``hash(str(df.values))`` implementation."""
    df = _make_df(seed=42, n=30)
    df_path = tmp_path / "df.parquet"
    df.to_parquet(df_path)

    script = textwrap.dedent(
        f"""
        import sys
        sys.path.insert(0, {str(REPO_ROOT)!r})
        import pandas as pd
        from src.bmm_cache import BMMCache
        df = pd.read_parquet({str(df_path)!r})
        c = BMMCache(cache_dir={str(tmp_path / 'cache')!r})
        print(c._key("MMSI_TEST", df, {{"a": 1, "b": [1, 2, 3]}}))
        """
    )

    def run_with_seed(seed: str) -> str:
        env = os.environ.copy()
        env["PYTHONHASHSEED"] = seed
        out = subprocess.check_output([sys.executable, "-c", script], env=env)
        return out.decode().strip()

    k_a = run_with_seed("0")
    k_b = run_with_seed("random")
    k_c = run_with_seed("12345")
    assert k_a == k_b == k_c
    assert len(k_a) == 64


def test_corrupted_pickle_is_removed_on_get(tmp_path):
    cache = BMMCache(cache_dir=str(tmp_path))
    df = _make_df()
    cfg = {"x": 1}

    key = cache._key("M", df, cfg)
    bad_file = Path(cache.cache_path) / f"{key}.pkl"
    bad_file.write_bytes(b"this is not a valid pickle stream \x00\x01\x02")

    assert bad_file.exists()
    result = cache.get("M", df, cfg)
    assert result is None
    assert not bad_file.exists(), "corrupted cache file should be removed"


def test_disabled_cache_is_noop(tmp_path):
    cache = BMMCache(enabled=False, cache_dir=str(tmp_path))
    df = _make_df()
    assert cache.set("M", df, {}, {"foo": 1}) is False
    assert cache.get("M", df, {}) is None


def test_deprecation_shim_resolves_to_bmm(tmp_path):
    with pytest.warns(DeprecationWarning):
        from src.hmm_cache import HMMCache  # noqa: F401
    from src.bmm_cache import BMMCache as Real
    assert HMMCache is Real
