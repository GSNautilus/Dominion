"""Filesystem caching helpers for DOMINION intermediate results.

Each cache file is named after a (kind, params) pair and lives next to
the source image in a sibling ``<stem>.dominion_cache`` directory. The
image's md5 is folded into the cache directory name so that re-running
on a modified image won't accidentally consume stale results.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Optional

import numpy as np


_FILE_HASH_CACHE: dict[str, str] = {}


def _image_md5(image_path: Path) -> str:
    """Md5-hash the image bytes, memoised per absolute path."""
    key = str(image_path.resolve())
    cached = _FILE_HASH_CACHE.get(key)
    if cached is not None:
        return cached
    h = hashlib.md5()
    with open(image_path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    digest = h.hexdigest()
    _FILE_HASH_CACHE[key] = digest
    return digest


def _params_hash(params: dict) -> str:
    payload = json.dumps(params, sort_keys=True, default=str).encode()
    return hashlib.md5(payload).hexdigest()[:8]


def cache_path(image_path: Path, kind: str, params: dict) -> Path:
    """Return the canonical cache file path for ``(image, kind, params)``.

    The cache directory is created if needed. The image md5 is rolled
    into the directory name so that editing the image invalidates the
    cache automatically.
    """
    image_path = Path(image_path)
    image_hash = _image_md5(image_path)[:8]
    params_hash = _params_hash(params)
    cache_dir = image_path.parent / f"{image_path.stem}.dominion_cache_{image_hash}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{kind}_{params_hash}.npz"


def load_npz(path: Path) -> Optional[dict]:
    """Load a previously cached npz into a dict of arrays, or None."""
    path = Path(path)
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def save_npz(path: Path, **arrays: np.ndarray) -> None:
    """Write arrays to an npz at ``path`` (parent dir is created)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
