"""Frozen-data manifest: SHA-256 + schema + date-range per file; ingest gate."""

from __future__ import annotations

import hashlib
import json
import sys
import warnings
from pathlib import Path

import pandas as pd
from pyarrow import ArrowException

MANIFEST_NAME = "manifest.json"
TABLE_SUFFIXES = {".parquet", ".csv"}
READ_ERRORS = (OSError, ArrowException, ValueError, UnicodeDecodeError, pd.errors.ParserError)


class ManifestMismatchError(RuntimeError):
    """Ingest gate: data drifted from the frozen manifest."""


def _read_table(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


def _date_column(df: pd.DataFrame) -> pd.Series | None:
    """The first fully-parseable date column: a native one, else parsed text."""
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            return df[col]
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            continue
        try:
            with warnings.catch_warnings():  # inference warnings are noise on non-date columns
                warnings.simplefilter("ignore", UserWarning)
                parsed = pd.to_datetime(df[col])
        except (ValueError, TypeError):
            continue
        if parsed.notna().all():
            return parsed
    return None


def _file_entry(path: Path) -> dict:
    data = path.read_bytes()
    entry: dict = {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
    if path.suffix not in TABLE_SUFFIXES:
        return entry
    try:
        df = _read_table(path)
    except READ_ERRORS:  # corrupted/unreadable: the hash is the gate
        entry["schema"] = None
        return entry
    entry["schema"] = {col: str(dtype) for col, dtype in df.dtypes.items()}
    dates = _date_column(df)
    if dates is not None and dates.notna().any():
        entry["date_range"] = [str(dates.min().date()), str(dates.max().date())]
    return entry


def build_manifest(data_dir: Path) -> dict:
    """Per-file SHA-256 + size (+ schema/date-range for Parquet and CSV) under data_dir."""
    data_dir = Path(data_dir)
    files = sorted(p for p in data_dir.rglob("*") if p.is_file() and p.name != MANIFEST_NAME)
    return {"files": {p.relative_to(data_dir).as_posix(): _file_entry(p) for p in files}}


def write_manifest(data_dir: Path, manifest_path: Path | None = None) -> Path:
    """Write the manifest to `manifest_path`, defaulting to `data_dir/manifest.json`."""
    out = Path(manifest_path) if manifest_path else Path(data_dir) / MANIFEST_NAME
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(build_manifest(data_dir), indent=2, sort_keys=True) + "\n")
    return out


def freeze(data_dir: Path, manifest_path: Path) -> None:
    """Verify the store against its committed manifest, then re-freeze it.

    Verify-then-write is the gate: writing first would compare the store against
    a manifest derived from the same bytes, so a re-run could never report drift.
    """
    if Path(manifest_path).exists():
        verify_manifest(data_dir, manifest_path)
    write_manifest(data_dir, manifest_path)


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
        print(
            "usage: python -m systematic_futures.data.manifest {write|verify} DATA_DIR",
            file=sys.stderr,
        )
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
