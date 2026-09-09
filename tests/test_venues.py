from systematic_futures.data.venues import DROPPED, UNIVERSE, dataset_for


def test_universe_complete():
    assert len(UNIVERSE) == 17


def test_softs_dropped_with_reason():
    assert set(DROPPED) == {"KC", "SB", "CC"}
    assert all("no free ICE EOD" in why for why in DROPPED.values())


def test_dataset_lookup_hard_fails_on_unknown():
    assert dataset_for("ES") == "CME-FREE-EOD"
    for symbol in ("XX", "KC"):  # unknown AND dropped softs hard-fail
        try:
            dataset_for(symbol)
        except KeyError:
            pass
        else:
            raise AssertionError(f"{symbol} must hard-fail")
