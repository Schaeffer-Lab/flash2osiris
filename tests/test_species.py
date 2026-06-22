"""Unit tests for the species-mask logic embedded in flash_osiris/yt_plugin.py.

The plugin file is normally exec'd by yt's plugin loader, which injects the names
``derived_field``, ``units`` and ``load``.  To test the pure species helpers
(``molar_weight`` / ``species_mask_map`` / ``_mass_thresholds``) without importing
yt, we exec the file here with no-op stubs for those injected names -- exactly the
contract yt provides -- and then exercise the functions from the resulting namespace.

This keeps ALL species knowledge in the single plugin file (no separate module) while
still being CI-testable with nothing but the stdlib + pytest.
"""

import os

import pytest

PLUGIN = os.path.join(os.path.dirname(__file__), "..", "flash_osiris", "yt_plugin.py")


def _load_plugin_namespace():
    """Exec the plugin file with stubs for the yt-injected names and return its globals."""
    ns = {
        "derived_field": lambda *a, **k: (lambda fn: fn),  # no-op decorator
        "units": object(),                                  # unused by the helpers
        "load": lambda *a, **k: None,                       # unused by the helpers
    }
    with open(PLUGIN) as f:
        exec(compile(f.read(), PLUGIN, "exec"), ns)
    return ns


PLUGIN_NS = _load_plugin_namespace()
molar_weight = PLUGIN_NS["molar_weight"]
species_mask_map = PLUGIN_NS["species_mask_map"]
_mass_thresholds = PLUGIN_NS["_mass_thresholds"]


def test_molar_weight_case_insensitive():
    assert molar_weight("al") == pytest.approx(26.982)
    assert molar_weight("Al") == pytest.approx(26.982)
    assert molar_weight("AL") == pytest.approx(26.982)


def test_molar_weight_unknown_raises():
    with pytest.raises(KeyError):
        molar_weight("unobtanium")


def test_mask_map_matches_original_alsi_convention():
    # Lightest ion -> mask 1, independent of input order. Al (26.98) < Si (28.09).
    assert species_mask_map(["al", "si"]) == {"al": 1, "si": 2}
    assert species_mask_map(["si", "al"]) == {"al": 1, "si": 2}


def test_mask_map_three_species_ordered_by_mass():
    # C (12) < Al (27) < Si (28)
    assert species_mask_map(["si", "al", "c"]) == {"c": 1, "al": 2, "si": 3}


def test_thresholds_are_adjacent_midpoints():
    # Single threshold at the Al/Si midpoint (~27.5), matching the old hard-coded value.
    (t,) = _mass_thresholds(["al", "si"])
    assert t == pytest.approx((26.982 + 28.085) / 2)
    # Two thresholds for three species, ascending.
    thresholds = _mass_thresholds(["si", "al", "c"])
    assert thresholds == sorted(thresholds)
    assert len(thresholds) == 2


def test_single_species_has_no_thresholds():
    assert _mass_thresholds(["al"]) == []
    assert species_mask_map(["al"]) == {"al": 1}
