"""Small I/O helpers shared by the ST-ES experiment scripts."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from pathlib import Path
from typing import Iterable, Iterator

# Generation caches and run outputs live outside the repository by default.
DEFAULT_RUN_DIR = Path(os.environ.get("ST_ES_RUN_DIR", Path(__file__).resolve().parents[4] / "st_es_runs"))


def read_jsonl(path: str | Path) -> Iterator[dict]:
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def append_jsonl(path: str | Path, record: dict) -> None:
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_json(path: str | Path, obj) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False, default=str)


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def environment_metadata() -> dict:
    import importlib

    meta = {"python": sys.version.split()[0], "platform": platform.platform()}
    for mod in ("torch", "transformers", "datasets", "math_verify", "numpy"):
        try:
            meta[mod] = importlib.import_module(mod).__version__
        except Exception:  # pragma: no cover - optional dependency missing
            meta[mod] = None
    try:
        import torch

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            meta.update({"gpu": props.name, "gpu_total_memory_mb": round(props.total_memory / 2**20),
                         "cuda_runtime": torch.version.cuda})
    except Exception:  # pragma: no cover
        pass
    return meta


def limit_gpu_memory(fraction: float = 0.85) -> None:
    """Cap PyTorch's CUDA caching allocator at ``fraction`` of device memory.

    On Windows (WDDM) an allocator that grows past dedicated memory does not
    fail: it spills into shared system memory, which is backed by host commit
    charge. In our runs this pushed one process to 16.5 GB of commit and grew
    the page file until the system drive was almost full. With a cap, PyTorch
    releases cached blocks and retries at the limit, and only raises OOM if the
    memory is really needed.
    """
    import torch

    if fraction and torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(fraction)


def rows_by_uid(records: Iterable[dict]) -> dict[str, dict]:
    return {r["uid"]: r for r in records}
