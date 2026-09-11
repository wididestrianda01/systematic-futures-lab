import json

import pandas as pd
import pytest

from systematic_futures.data.manifest import (
    ManifestMismatchError,
    build_manifest,
    freeze,
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


def test_csv_store_gets_schema_and_date_range_too(tmp_path):
    """The raw store is CSV: it must be gated on more than a hash, exactly like Parquet."""
    dates = pd.bdate_range("2020-01-01", periods=3)
    pd.DataFrame(
        {"DATETIME": dates.strftime("%Y-%m-%d %H:%M:%S"), "PRICE": [1.0, 2.0, 3.0]}
    ).to_csv(tmp_path / "AAA.csv", index=False)
    entry = build_manifest(tmp_path)["files"]["AAA.csv"]
    assert set(entry["schema"]) == {"DATETIME", "PRICE"}  # column-level, like Parquet
    assert entry["schema"]["PRICE"] == "float64"
    assert entry["date_range"] == ["2020-01-01", "2020-01-03"]
    write_manifest(tmp_path)
    pd.DataFrame(
        {"DATETIME": dates.strftime("%Y-%m-%d %H:%M:%S"), "PRICE": [1.0, 2.0, 4.0]}
    ).to_csv(tmp_path / "AAA.csv", index=False)
    with pytest.raises(ManifestMismatchError, match="AAA.csv.*sha256"):
        verify_manifest(tmp_path)


def test_freeze_refuses_drift_before_refreezing(tmp_path, tmp_path_factory):
    """Verify-then-write: a store edited out of band must fail the freeze, not be
    absorbed into a fresh manifest."""
    make_dir(tmp_path)
    manifest = tmp_path_factory.mktemp("manifests") / "frozen.json"
    freeze(tmp_path, manifest)
    verify_manifest(tmp_path, manifest)  # first freeze wrote what it found

    dates = pd.bdate_range("2022-01-01", periods=10)
    pd.DataFrame({"date": dates, "close": range(10)}).to_parquet(tmp_path / "AAA.parquet")
    with pytest.raises(ManifestMismatchError, match="AAA.parquet"):
        freeze(tmp_path, manifest)


def test_manifest_json_structure(tmp_path):
    make_dir(tmp_path)
    out = write_manifest(tmp_path)
    doc = json.loads(out.read_text())["files"]
    assert set(doc) == {"AAA.parquet", "BBB.parquet"}
    assert len(doc["AAA.parquet"]["sha256"]) == 64
