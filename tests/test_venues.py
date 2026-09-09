from systematic_futures.data.venues import DATASETS, UNIVERSE, dataset_for, roots_for


def test_universe_complete():
    assert len(UNIVERSE) == 20


def test_softs_on_ice_rest_on_cme():
    assert set(roots_for("IFUS.ICE")) == {"KC", "SB", "CC"}
    assert set(roots_for("GLBX.MDP3")) == set(UNIVERSE) - {"KC", "SB", "CC"}


def test_dataset_lookup_hard_fails_on_unknown():
    assert dataset_for("ES") == DATASETS["CME"]
    try:
        dataset_for("XX")
    except KeyError:
        pass
    else:
        raise AssertionError("unknown symbol must hard-fail")
