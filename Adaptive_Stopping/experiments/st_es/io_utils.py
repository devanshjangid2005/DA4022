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


def rows_by_uid(records: Iterable[dict]) -> dict[str, dict]:
    return {r["uid"]: r for r in records}
