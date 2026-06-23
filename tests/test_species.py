"""Unit tests for the population/species logic embedded in flash_osiris/yt_plugin.py.

OSIRIS ion populations are separated by FLASH **material** (target vs chamber/...),
not by ion mass. The pure helpers tested here are:

  * ``parse_flash_species`` -- pull the ordered ``species=...`` material list out of a
    FLASH ``sim info`` setup string,
  * ``resolve_populations`` -- map the (vacuum-excluded) materials to OSIRIS species
    names, honouring an optional ``{flash_material: osiris_name}`` rename and returning
    the dominant-material index for each.

The plugin file is normally exec'd by yt's plugin loader, which injects the names
``derived_field``, ``units`` and ``load``. To test the pure helpers without importing
yt, we exec the file here with no-op stubs for those injected names -- exactly the
contract yt provides -- and exercise the functions from the resulting namespace. This
keeps ALL population knowledge in the single plugin file while staying CI-testable with
nothing but the stdlib + pytest.
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
parse_flash_species = PLUGIN_NS["parse_flash_species"]
resolve_populations = PLUGIN_NS["resolve_populations"]
FLASH_VACUUM_NAMES = PLUGIN_NS["FLASH_VACUUM_NAMES"]

# A realistic FLASH setup string (as stored in the HDF5 "sim info").
ALSI_SIM_INFO = (
    "setup.py -auto MagShockZ_3D +pm4dev -3d +cartesian +hdf5typeio "
    "species=cham,targ,vac +mtmmmt +laser +usm3t +mgd mgd_meshgroups=1"
)
CH_SIM_INFO = (
    "setup.py -auto THSC -2d -nxb=16 -nyb=16 +cartesian +hdf5typeio "
    "species=cham,targ,vac +mtmmmt +laser +usm3t +mgd mgd_meshgroups=1"
)


def test_parse_species_preserves_order_and_lowercases():
    assert parse_flash_species(ALSI_SIM_INFO) == ["cham", "targ", "vac"]
    assert parse_flash_species("setup.py species=Targ,Cham") == ["targ", "cham"]


def test_parse_species_absent_returns_empty():
    assert parse_flash_species("setup.py -auto Foo +cartesian") == []
    assert parse_flash_species("") == []
    assert parse_flash_species(None) == []


def test_vacuum_is_a_known_name():
    # The generator/plugin drops these before calling resolve_populations.
    assert "vac" in FLASH_VACUUM_NAMES


def test_resolve_defaults_to_material_names():
    # No rename: OSIRIS species names default to the FLASH material field names, and the
    # dominant-material index matches the input order.
    pops = resolve_populations(["cham", "targ"])
    assert pops == [("cham", "cham", 0), ("targ", "targ", 1)]


def test_resolve_rename_reproduces_alsi_convention():
    # cham=Al chamber, targ=Si target (from the EOS tables). The rename gives back the
    # historical al/si OSIRIS species names while keeping the material-based separation.
    pops = resolve_populations(["cham", "targ"], {"cham": "al", "targ": "si"})
    assert pops == [("al", "cham", 0), ("si", "targ", 1)]


def test_resolve_rename_is_case_insensitive_on_keys():
    pops = resolve_populations(["cham", "targ"], {"Cham": "background", "TARG": "piston"})
    assert pops == [("background", "cham", 0), ("piston", "targ", 1)]


def test_resolve_partial_rename_keeps_field_name_for_the_rest():
    pops = resolve_populations(["cham", "targ"], {"targ": "piston"})
    assert pops == [("cham", "cham", 0), ("piston", "targ", 1)]


def test_resolve_unknown_rename_key_raises():
    with pytest.raises(ValueError):
        resolve_populations(["cham", "targ"], {"foo": "bar"})


def test_resolve_empty_materials_raises():
    with pytest.raises(ValueError):
        resolve_populations([])


def test_resolve_single_material():
    pops = resolve_populations(["targ"], {"targ": "piston"})
    assert pops == [("piston", "targ", 0)]
