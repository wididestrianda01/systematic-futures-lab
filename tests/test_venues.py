import pytest

from systematic_futures.data.venues import DROPPED, INSTRUMENTS, instrument_for


def test_universe_complete():
    assert len(INSTRUMENTS) == 16


def test_dropped_roots_recorded():
    assert set(DROPPED) == {"KC", "SB", "CC", "BZ"}
    assert all("no free feed" in why or "develop window" in why for why in DROPPED.values())


def test_instrument_mapping():
    assert instrument_for("ES") == "SP500"
    assert instrument_for("6B") == "GBP"


def test_lookup_hard_fails():
    for root in ("XX", "KC", "BZ"):  # unknown AND dropped roots hard-fail
        with pytest.raises(KeyError):
            instrument_for(root)
