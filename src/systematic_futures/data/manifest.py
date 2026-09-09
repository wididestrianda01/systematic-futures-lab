"""Frozen-data manifest: SHA-256 + schema + date-range per file; ingest gate."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
from pyarrow import ArrowException

MANIFEST_NAME = "manifest.json"


class ManifestMismatchError(RuntimeError):
    """Ingest gate: data drifted from the frozen manifest."""


def _file_entry(path: Path) -> dict:
    data = path.read_bytes()
    entry: dict = {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
    if path.suffix == ".parquet":
        try:
            df = pd.read_parquet(path)
        except (OSError, ArrowException):  # corrupted/unreadable: hash is the gate
            entry["schema"] = None
        else:
            entry["schema"] = {col: str(dtype) for col, dtype in df.dtypes.items()}
            date_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
            if date_cols:
                col = df[date_cols[0]]
                entry["date_range"] = [str(col.min().date()), str(col.max().date())]
    return entry


def build_manifest(data_dir: Path) -> dict:
    """Per-file SHA-256 + size (+ schema/date-range for Parquet) under data_dir."""
    data_dir = Path(data_dir)
    files = sorted(p for p in data_dir.rglob("*") if p.is_file() and p.name != MANIFEST_NAME)
    return {"files": {p.relative_to(data_dir).as_posix(): _file_entry(p) for p in files}}


def write_manifest(data_dir: Path) -> Path:
    out = Path(data_dir) / MANIFEST_NAME
    out.write_text(json.dumps(build_manifest(data_dir), indent=2, sort_keys=True) + "\n")
    return out


def verify_manifest(data_dir: Path, manifest_path: Path | None = None) -> None:
    """Re-derive and compare. Raises ManifestMismatchError naming every drift."""
    data_dir = Path(data_dir)
    manifest_path = Path(manifest_path) if manifest_path else data_dir / MANIFEST_NAME
    frozen = json.loads(manifest_path.read_text())["files"]
    current = build_manifest(data_dir)["files"]
    problems = []
    for name, entry in frozen.items():
        if name not in current:
            problems.append(f"missing: {name}")
            continue
        drifted = [k for k in entry if current[name].get(k) != entry[k]]
        if drifted:
            problems.append(f"drift in {name}: {', '.join(drifted)}")
    problems += [f"unexpected file: {name}" for name in current if name not in frozen]
    if problems:
        raise ManifestMismatchError("manifest gate failed:\n" + "\n".join(problems))


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 2 or argv[0] not in {"write", "verify"}:
        print("usage: python -m systematic_futures.data.manifest {write|verify} DATA_DIR",
              file=sys.stderr)
        return 2
    data_dir = Path(argv[1])
    if argv[0] == "write":
        print(f"wrote {write_manifest(data_dir)}")
    else:
        verify_manifest(data_dir)
        print("manifest gate: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
