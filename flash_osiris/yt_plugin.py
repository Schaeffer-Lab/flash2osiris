### USAGE INSTRUCTIONS ###
# 1. LINK THIS FILE TO ~/.config/yt/my_plugins.py:
#       ln -s /absolute/path/to/this/file ~/.config/yt/my_plugins.py
# 2. ADD `yt.enable_plugins()` TO YOUR SCRIPT (the generator does this for you).
# 3. CALL `yt.load_for_osiris(flash_dump, rqm_factor=..., species_names={...})`.
#
# This file is the SINGLE place that knows how OSIRIS ion populations are defined
# from a FLASH dump.  Populations are separated by FLASH **material** (the laser
# target vs the chamber/background gas), NOT by ion mass: FLASH +mtmmmt runs carry a
# mass-fraction field per material (e.g. `targ`, `cham`, declared by `species=...` in
# the setup call), and a cell belongs to whichever material dominates it.  This works
# even when target and chamber are the *same* element mix (e.g. a CH plasma where the
# only difference is piston vs background), which a mass-based split cannot do.
#
# To run a FLASH dump with different populations you usually need to change NOTHING:
# the materials are auto-detected from the dump.  Optionally rename the OSIRIS species
# via `species_names = {<flash_material>: <osiris_name>}` (the run.yaml `species_names`
# key), e.g. {'targ': 'piston', 'cham': 'background'} or {'cham': 'al', 'targ': 'si'}.

# This file is a plugin for yt that adds OSIRIS-relevant fields to FLASH datasets.

import numpy as np


# FLASH "materials" that are vacuum/ambient fill and should NOT become OSIRIS ion
# populations (their mass-fraction field is often not even dumped).
FLASH_VACUUM_NAMES = ("vac", "vacu", "vacuum")


def parse_flash_species(sim_info: str):
    """Ordered list of FLASH material names declared in a dump's setup call.

    FLASH multi-material setups name their materials in the ``setup.py ... species=a,b,c``
    argument, which is recorded verbatim in the HDF5 ``sim info`` string.  Returns the
    materials in declaration order, lower-cased (e.g. ``['cham', 'targ', 'vac']``).  The
    vacuum entry is kept here; callers drop it via :data:`FLASH_VACUUM_NAMES`.
    """
    import re
    m = re.search(r"species=([A-Za-z0-9_,]+)", sim_info or "")
    if not m:
        return []
    return [s.strip().lower() for s in m.group(1).split(",") if s.strip()]


def resolve_populations(ion_materials, species_names=None):
    """Map each (vacuum-excluded, present) FLASH material to an OSIRIS ion population.

    Parameters
    ----------
    ion_materials : list[str]
        Material field names, in a fixed order, with vacuum already removed and only
        the ones actually present in the dump kept (e.g. ``['cham', 'targ']``).  This
        order defines the integer used by the ``dominant_material`` argmax field.
    species_names : dict[str, str] | None
        Optional ``{flash_material: osiris_species_name}`` rename (the run.yaml
        ``species_names`` key, e.g. ``{'cham': 'al', 'targ': 'si'}``).  Keys are matched
        case-insensitively; materials not listed keep their FLASH field name.

    Returns
    -------
    list[tuple[str, str, int]]
        ``(osiris_name, flash_material, index)`` per population, where ``index`` is the
        material's position in ``ion_materials`` (i.e. the value the ``dominant_material``
        field takes for that population).
    """
    rename = {k.lower(): v for k, v in (species_names or {}).items()}
    unknown = set(rename) - {m.lower() for m in ion_materials}
    if unknown:
        raise ValueError(
            f"species_names refers to material(s) {sorted(unknown)} that are not "
            f"non-vacuum FLASH materials in this dump (have {ion_materials})."
        )
    pops = [(rename.get(m.lower(), m), m, i) for i, m in enumerate(ion_materials)]
    if not pops:
        raise ValueError(
            "No non-vacuum FLASH materials found -- cannot define any OSIRIS ion "
            "population.  Check the dump's `species=` setup (sim info)."
        )
    return pops


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


def _read_sim_info(filename: str) -> str:
    """The FLASH ``sim info`` setup string (carries the ``species=...`` material list)."""
    import h5py
    with h5py.File(filename, "r") as f:
        return f["sim info"][:].tobytes().decode("latin1")


def load_for_osiris(filename: str, rqm_factor: float = 1, species_names=None):
    """Load a FLASH dump and register the OSIRIS-relevant derived fields.

    A wrapper around ``yt.load()`` that adds:
      * E-field components (v x B / c),
      * a ``dominant_material`` field (argmax over the FLASH material mass fractions),
      * a charge-density field ``<name>dens`` (electron density where that material
        dominates) and a thermal-velocity field ``vth<name>`` for each ion population,
      * electron thermal velocity ``vthele``,
      * current density (Ampere's law) and ion/electron drift velocities.

    Populations are separated by FLASH **material**, not ion mass (see the module
    header).  The materials are auto-detected from the dump's ``species=`` setup; the
    vacuum material is excluded.  ``species_names`` optionally renames the OSIRIS
    species (``{flash_material: osiris_name}``).  The resulting name->material map and
    name->dominant-index map are attached to the returned dataset
    (``ds.osiris_species_materials`` / ``ds.osiris_dominant_index``) so the generator
    and this plugin can never disagree about which population is which.
    """
    ds = load(filename)

    c = units.speed_of_light_cgs
    e = units.electron_charge_cgs
    m_e = units.mass_electron_cgs

    # --- ion populations from FLASH materials (target / chamber / ...), not from mass ---
    declared = parse_flash_species(_read_sim_info(filename))      # incl. vacuum
    flash_fields = {name for ftype, name in ds.field_list if ftype == "flash"}
    ion_materials = [m for m in declared
                     if m not in FLASH_VACUUM_NAMES and m in flash_fields]
    if not ion_materials:
        raise ValueError(
            f"No non-vacuum FLASH material fields found in {filename} (declared "
            f"species={declared}, present flash fields include "
            f"{sorted(flash_fields & set(declared))}).  Cannot define ion populations."
        )
    pops = resolve_populations(ion_materials, species_names)

    # dominant_material: index (into ion_materials) of the largest material fraction in
    # each cell.  Vacuum is not in the stack, so vacuum cells fall to whichever ion
    # material is (barely) larger -- harmless because their electron density ~ 0 and the
    # density fields/rqm averages are electron-density weighted.
    def make_dominant_material(field, data):
        stack = np.stack([np.asarray(data["flash", m]) for m in ion_materials], axis=0)
        return np.argmax(stack, axis=0)

    ds.add_field(("flash", "dominant_material"), function=make_dominant_material,
                 units="", sampling_type="cell", force_override=False)

    # --- per-population charge density and thermal velocity ---
    # Ion thermal velocity uses the FULL ion mass m_i = m_p/sumy (sumy = ions per
    # baryon = 1/A), NOT the mass-per-charge m_p/ye.  This loads the macro-ions at the
    # real ion thermal speed sqrt(k T_ion/m_i), so the ion pressure rho*vth^2 = n_i k
    # T_ion is physical and the electron/ion PRESSURE ratio (and beta_i) is conserved.
    # For a compound material (e.g. a CH plasma) m_i is the FLASH mass-fraction-weighted
    # mean ion mass, i.e. the effective mass of that fluid -- exactly what we want when
    # the population is a "piston"/"background" rather than a single element.
    def make_vthion(field, data):
        vthion = np.sqrt(data["flash", "tion"] * units.boltzmann_constant / (units.proton_mass / data["flash", "sumy"]))
        return vthion.to("code_velocity")

    osiris_species_materials = {}   # osiris_name -> flash material field
    osiris_dominant_index = {}      # osiris_name -> dominant_material integer
    for name, material, idx in pops:
        def make_density(field, data, _idx=idx):
            return data["flash", "edens"] * (data["flash", "dominant_material"] == _idx)

        ds.add_field(("flash", f"{name}dens"), function=make_density,
                     units="1/code_length**3", sampling_type="cell", force_override=False)
        ds.add_field(("flash", f"vth{name}"), function=make_vthion,
                     units="code_velocity", sampling_type="cell", force_override=False)
        osiris_species_materials[name] = material
        osiris_dominant_index[name] = idx

    ds.osiris_species_materials = osiris_species_materials   # read back by the generator
    ds.osiris_dominant_index = osiris_dominant_index

    def make_vthele(field, data):
        vthele = np.sqrt(data["flash", "tele"] * units.boltzmann_constant / units.electron_mass_cgs)
        return vthele.to("code_velocity")

    ds.add_field(("flash", "vthele"), function=make_vthele,
                 units="code_velocity", sampling_type="cell", force_override=False)
    # Total ion thermal velocity (population-independent), kept for inspection/back-compat.
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
