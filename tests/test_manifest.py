import json

import pandas as pd
import pytest

from systematic_futures.data.manifest import (
    ManifestMismatchError,
    build_manifest,
    verify_manifest,
    write_manifest,
)


def make_dir(tmp_path):
    dates = pd.bdate_range("2020-01-01", periods=10)
    for sym in ("AAA", "BBB"):
        pd.DataFrame({"date": dates, "close": range(10)}).to_parquet(tmp_path / f"{sym}.parquet")
    return tmp_path


def test_roundtrip_passes(tmp_path):
    make_dir(tmp_path)
    write_manifest(tmp_path)
    verify_manifest(tmp_path)  # no raise


def test_modified_bytes_fail(tmp_path):
    make_dir(tmp_path)
    write_manifest(tmp_path)
    p = tmp_path / "AAA.parquet"
    p.write_bytes(p.read_bytes() + b"x")
    with pytest.raises(ManifestMismatchError, match="AAA.parquet.*sha256"):
        verify_manifest(tmp_path)


def test_schema_drift_fail(tmp_path):
    make_dir(tmp_path)
    write_manifest(tmp_path)
    dates = pd.bdate_range("2020-01-01", periods=10)
    pd.DataFrame({"date": dates, "close": range(10), "extra": range(10)}).to_parquet(
        tmp_path / "AAA.parquet"
    )
    with pytest.raises(ManifestMismatchError, match="AAA.parquet.*schema"):
        verify_manifest(tmp_path)


def test_date_range_recorded_and_drift_named(tmp_path):
    make_dir(tmp_path)
    entry = build_manifest(tmp_path)["files"]["AAA.parquet"]
    assert entry["date_range"] == ["2020-01-01", "2020-01-14"]
    write_manifest(tmp_path)
    dates = pd.bdate_range("2021-01-01", periods=10)
    pd.DataFrame({"date": dates, "close": range(10)}).to_parquet(tmp_path / "AAA.parquet")
    with pytest.raises(ManifestMismatchError, match="AAA.parquet"):
        verify_manifest(tmp_path)


def test_missing_and_extra_files_fail(tmp_path):
    make_dir(tmp_path)
    write_manifest(tmp_path)
    (tmp_path / "BBB.parquet").unlink()
    with pytest.raises(ManifestMismatchError, match="missing: BBB.parquet"):
        verify_manifest(tmp_path)
    make_dir(tmp_path)
    (tmp_path / "ZZZ.parquet").touch()
    with pytest.raises(ManifestMismatchError, match="unexpected file: ZZZ.parquet"):
        verify_manifest(tmp_path)


def test_manifest_json_structure(tmp_path):
    make_dir(tmp_path)
    out = write_manifest(tmp_path)
    doc = json.loads(out.read_text())["files"]
    assert set(doc) == {"AAA.parquet", "BBB.parquet"}
    assert len(doc["AAA.parquet"]["sha256"]) == 64
