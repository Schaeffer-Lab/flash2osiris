### USAGE INSTRUCTIONS ###
# 1. LINK THIS FILE TO ~/.config/yt/my_plugins.py:
#       ln -s /absolute/path/to/this/file ~/.config/yt/my_plugins.py
# 2. ADD `yt.enable_plugins()` TO YOUR SCRIPT (the generator does this for you).
# 3. CALL `yt.load_for_osiris(flash_dump, rqm_factor=..., species=['al', 'si'])`.
#
# This file is the SINGLE place that knows about ion species.  To run a FLASH dump
# with different ions, you only need to:
#   * make sure each element is in MOLAR_WEIGHTS below, and
#   * pass its name in the `species` list (these are the `charge_states` keys in the
#     run.yaml; the generator forwards them here automatically).
# Everything else -- the per-species density / thermal-velocity fields and the
# species mask that separates them -- is built generically from that list.


# This file is a plugin for yt that adds OSIRIS-relevant fields to FLASH datasets.

import numpy as np


# ---------------------------------------------------------------------------
# Species table.  Standard atomic weights (u).  Add a row to support a new ion.
# `species` names passed to load_for_osiris are matched here case-insensitively
# ('al' -> 'Al'), so the run.yaml can keep using lowercase OSIRIS species names.
# ---------------------------------------------------------------------------
MOLAR_WEIGHTS = {
    "H": 1.0078,  "He": 4.0026, "Li": 6.94,   "Be": 9.0122, "B": 10.81,
    "C": 12.011,  "N": 14.007,  "O": 15.999,  "F": 18.998,  "Ne": 20.180,
    "Na": 22.990, "Mg": 24.305, "Al": 26.982, "Si": 28.085, "P": 30.974,
    "S": 32.06,   "Cl": 35.45,  "Ar": 39.948, "K": 39.098,  "Ca": 40.078,
    "Fe": 55.845, "Cu": 63.546, "Xe": 131.29,
}


def molar_weight(name):
    """Atomic weight (u) for an element named like the run.yaml charge_states keys
    (case-insensitive: 'al' and 'Al' both resolve to aluminium)."""
    try:
        return MOLAR_WEIGHTS[name.capitalize()]
    except KeyError:
        raise KeyError(
            f"No molar weight for element '{name}'. Add '{name.capitalize()}' to "
            f"MOLAR_WEIGHTS in this file (it drives the species-mass mask)."
        )


def species_mask_map(species):
    """Map each species name to its integer mask value.

    Species are ordered by ASCENDING atomic mass and `np.digitize` of the mean ion
    mass (1/sumy) against the midpoint thresholds assigns mask 1 to the lightest
    ion, 2 to the next, etc.  For ['al', 'si'] this gives {'al': 1, 'si': 2} with a
    single threshold at ~27.5 u -- identical to the original hard-coded MagShockZ
    convention.  The generator reads this map back off the dataset, so the two sides
    can never disagree about which mask integer is which ion.
    """
    ordered = sorted(species, key=molar_weight)
    return {name: i + 1 for i, name in enumerate(ordered)}


def _mass_thresholds(species):
    """Bin edges (u) between adjacent species (midpoints of their atomic weights),
    for np.digitize(1/sumy, [-inf] + thresholds + [inf]).  Empty for one species."""
    weights = [molar_weight(n) for n in sorted(species, key=molar_weight)]
    return [(weights[i] + weights[i + 1]) / 2.0 for i in range(len(weights) - 1)]


@derived_field(name=("flash", "idens"), sampling_type="cell", units="1/code_length**3")
def _ion_number_density(field, data):
    return data["gas", "ion_number_density"]


@derived_field(name=("flash", "edens"), sampling_type="cell", units="1/code_length**3")
def _electron_number_density(field, data):
    return data["gas", "El_number_density"]


@derived_field(name=("flash", "Ex"), sampling_type="cell", units="code_magnetic")
def _Ex(field, data):
    return (data["flash", "velz"] * data["flash", "magy"] - data["flash", "vely"] * data["flash", "magz"]) / units.speed_of_light


@derived_field(name=("flash", "Ey"), sampling_type="cell", units="code_magnetic")
def _Ey(field, data):
    return (data["flash", "velx"] * data["flash", "magz"] - data["flash", "velz"] * data["flash", "magx"]) / units.speed_of_light


@derived_field(name=("flash", "Ez"), sampling_type="cell", units="code_magnetic")
def _Ez(field, data):
    return (data["flash", "vely"] * data["flash", "magx"] - data["flash", "velx"] * data["flash", "magy"]) / units.speed_of_light


def load_for_osiris(filename: str, rqm_factor: float = 1, species=None):
    """Load a FLASH dump and register the OSIRIS-relevant derived fields.

    A wrapper around ``yt.load()`` that adds:
      * E-field components (v x B / c),
      * a ``species_mask`` separating ion species by mean ion mass,
      * a charge-density field ``<name>dens`` and thermal-velocity field
        ``vth<name>`` for each requested species,
      * electron thermal velocity ``vthele``,
      * current density (Ampere's law) and ion/electron drift velocities.

    Parameters
    ----------
    filename : str
        Path to the FLASH HDF5 plot file.
    rqm_factor : float
        rqm reduction factor (only the current-driven part of the drift velocities
        is divided by sqrt(rqm_factor) so it survives the later velocity rescaling).
    species : list[str]
        Ion species names (the run.yaml ``charge_states`` keys, e.g. ['al', 'si']).
        Each must appear in :data:`MOLAR_WEIGHTS`.  The species->mask map is attached
        to the returned dataset as ``ds.osiris_species_masks`` for the generator.
    """
    if not species:
        raise ValueError(
            "load_for_osiris needs a non-empty `species` list (the run.yaml "
            "charge_states keys, e.g. ['al', 'si'])."
        )

    ds = load(filename)

    c = units.speed_of_light_cgs
    e = units.electron_charge_cgs
    m_e = units.mass_electron_cgs

    # --- species mask: bin FLASH cells by mean ion mass (1/sumy, in proton masses) ---
    mask_map = species_mask_map(species)        # {name: mask_int}, 1 = lightest ion
    thresholds = _mass_thresholds(species)
    ds.osiris_species_masks = mask_map           # read back by the generator

    def make_species_mask(field, data):
        """Assign every cell a mask value 1..N by mean ion mass.  With thresholds
        [t1, t2, ...] (ascending), np.digitize gives 1 below t1, 2 between t1 and
        t2, etc., so mask i corresponds to the i-th lightest species."""
        bins = [-np.inf] + thresholds + [np.inf]
        return np.digitize(1 / data["flash", "sumy"], bins)

    ds.add_field(("flash", "species_mask"), function=make_species_mask,
                 units="", sampling_type="cell", force_override=False)

    # --- per-species charge density and thermal velocity (built generically) ---
    # Ion thermal velocity uses the FULL ion mass m_i = m_p/sumy (sumy = ions per
    # baryon = 1/A), NOT the mass-per-charge m_p/ye.  This loads the macro-ions at the
    # real ion thermal speed sqrt(k T_ion/m_i), so the ion pressure rho*vth^2 = n_i k
    # T_ion is physical and the electron/ion PRESSURE ratio (and beta_i) is conserved.
    # The charge state Z = ye/sumy enters only implicitly through the FLASH fields, so
    # no explicit Z is needed.  (Trade-off: each macro-ion is charge-equivalent and
    # carries 1/Z of a real ion's thermal energy, so the per-particle T_e/T_i reads
    # ~Z x -- we deliberately conserve the pressure ratio, not the temperature ratio.)
    def make_vthion(field, data):
        vthion = np.sqrt(data["flash", "tion"] * units.boltzmann_constant / (units.proton_mass / data["flash", "sumy"]))
        return vthion.to("code_velocity")

    for name, mval in mask_map.items():
        def make_density(field, data, _mval=mval):
            return data["flash", "edens"] * (data["flash", "species_mask"] == _mval)

        ds.add_field(("flash", f"{name}dens"), function=make_density,
                     units="1/code_length**3", sampling_type="cell", force_override=False)
        ds.add_field(("flash", f"vth{name}"), function=make_vthion,
                     units="code_velocity", sampling_type="cell", force_override=False)

    def make_vthele(field, data):
        vthele = np.sqrt(data["flash", "tele"] * units.boltzmann_constant / units.electron_mass_cgs)
        return vthele.to("code_velocity")

    ds.add_field(("flash", "vthele"), function=make_vthele,
                 units="code_velocity", sampling_type="cell", force_override=False)
    # Total ion thermal velocity (mask-independent), kept for inspection/back-compat.
    ds.add_field(("flash", "vthion"), function=make_vthion,
                 units="code_velocity", sampling_type="cell", force_override=False)

    # We need the gradients in order to calculate Ampere's law
    ds.add_gradient_fields(("flash", "magx"))
    ds.add_gradient_fields(("flash", "magy"))
    ds.add_gradient_fields(("flash", "magz"))

    def make_Jx(field, data):
        return c / (4 * np.pi) * (data["flash", "magz_gradient_y"] - data["flash", "magy_gradient_z"])

    def make_Jy(field, data):
        return c / (4 * np.pi) * (data["flash", "magx_gradient_z"] - data["flash", "magz_gradient_x"])

    def make_Jz(field, data):
        return c / (4 * np.pi) * (data["flash", "magy_gradient_x"] - data["flash", "magx_gradient_y"])

    ds.add_field(("flash", "Jx"), function=make_Jx, units="code_magnetic/code_time", sampling_type="cell")
    ds.add_field(("flash", "Jy"), function=make_Jy, units="code_magnetic/code_time", sampling_type="cell")
    ds.add_field(("flash", "Jz"), function=make_Jz, units="code_magnetic/code_time", sampling_type="cell")

    # NOTE: The current-driven contribution to the velocities (Ampere's law) must be
    # exempt from the velocity scaling applied later to preserve Mach number.  We
    # therefore divide the current part by sqrt(rqm_factor) now, so it is unchanged
    # after the later rescaling.
    def _v_ix(field, data):
        return ((m_e * data["flash", "Jx"] / (data["flash", "dens"] * e)) + data["flash", "velx"])

    def _v_iy(field, data):
        return ((m_e * data["flash", "Jy"] / (data["flash", "dens"] * e)) + data["flash", "vely"])

    def _v_iz(field, data):
        return ((m_e * data["flash", "Jz"] / (data["flash", "dens"] * e)) + data["flash", "velz"])

    def _v_ex(field, data):
        return ((m_e * data["flash", "Jx"] / (data["flash", "dens"] * e) - data["flash", "Jx"] / (data["flash", "edens"] * e)) / np.sqrt(rqm_factor) + data["flash", "velx"])

    def _v_ey(field, data):
        return ((m_e * data["flash", "Jy"] / (data["flash", "dens"] * e) - data["flash", "Jy"] / (data["flash", "edens"] * e)) / np.sqrt(rqm_factor) + data["flash", "vely"])

    def _v_ez(field, data):
        return ((m_e * data["flash", "Jz"] / (data["flash", "dens"] * e) - data["flash", "Jz"] / (data["flash", "edens"] * e)) / np.sqrt(rqm_factor) + data["flash", "velz"])

    ds.add_field(("flash", "v_ix"), function=_v_ix, units="code_velocity", sampling_type="cell")
    ds.add_field(("flash", "v_iy"), function=_v_iy, units="code_velocity", sampling_type="cell")
    ds.add_field(("flash", "v_iz"), function=_v_iz, units="code_velocity", sampling_type="cell")
    ds.add_field(("flash", "v_ex"), function=_v_ex, units="code_velocity", sampling_type="cell")
    ds.add_field(("flash", "v_ey"), function=_v_ey, units="code_velocity", sampling_type="cell")
    ds.add_field(("flash", "v_ez"), function=_v_ez, units="code_velocity", sampling_type="cell")

    return ds
