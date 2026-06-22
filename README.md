# flash2osiris

Convert **FLASH** MHD simulation output into initialized **OSIRIS** PIC input decks.

Given a FLASH HDF5 plot file and a small `run.yaml`, this tool:

1. reads the FLASH fields through a custom **yt plugin**,
2. derives the OSIRIS normalizations and per-species `rqm` (mass-per-charge) from the
   FLASH ionization fields,
3. saves the field/density/thermal-velocity slices as `.npy` grids, and
4. renders the OSIRIS input **deck** and the **python-init script** (which interpolates
   those slices onto the OSIRIS grid at runtime).

It works with **arbitrary ion species** — you only name the ions in the run spec.

---

## Layout

```
flash2osiris/
├── flash_osiris/
│   ├── yt_plugin.py     # yt plugin: ALL species knowledge lives here (see below)
│   ├── generator.py     # the generator CLI (reads run.yaml, writes the deck)
│   └── __init__.py
├── templates/
│   ├── osiris_deck_TEMPLATE.jinja   # OSIRIS input deck (species-generic)
│   └── py_init_TEMPLATE.jinja       # OSIRIS python-init script (species-generic)
├── examples/
│   ├── perlmutter_1d.run.yaml             # 1D lineout (Al/Si)
│   ├── perlmutter_2d.run.yaml             # 2D box (Al/Si)
│   └── example_carbon_hydrogen.run.yaml   # different ions, to show plug-and-play
├── tests/test_species.py   # unit tests for the species-mask logic
├── environment.yml         # conda env (yt, jinja2, plasmapy, ...)
├── setup_plugin.sh         # links the yt plugin into ~/.config/yt/
└── runme.sh                # thin wrapper: python -m flash_osiris.generator --config ...
```

---

## Setup (one time)

```bash
conda env create -f environment.yml
conda activate flash2osiris

# Link the yt plugin so yt.enable_plugins() can find load_for_osiris:
./setup_plugin.sh          # ln -s flash_osiris/yt_plugin.py ~/.config/yt/my_plugins.py
```

## Run

```bash
conda activate flash2osiris
python -m flash_osiris.generator --config examples/perlmutter_1d.run.yaml
# or:  ./runme.sh examples/perlmutter_1d.run.yaml
```

Outputs are written under `./input_files/<inputfile_name>.<dim>d/`:

```
input_files/<name>.1d/
├── <name>.1d              # OSIRIS input deck
├── <name>.1d.py           # OSIRIS python-init script
├── interp/*.npy           # field/density/vth slices the py-script interpolates
├── figures/               # lineout + FLASH-vs-OSIRIS conservation diagnostics
├── run.yaml               # frozen, re-runnable copy of the resolved run spec
└── run_manifest.yaml      # provenance (git commit, omega_pe, gyrotime, ...)
```

The CLI has **no hidden defaults**: every parameter must come from the `--config`
run.yaml or an explicit flag (CLI flags override the config). See
`python -m flash_osiris.generator --help`.

---

## Configuring ion species (the important part)

**Species are named once, as the keys of `charge_states` in the run.yaml:**

```yaml
charge_states: {al: 13, si: 14}    # ions = al, si; values are physical charge states Z
```

To run a different plasma, just change those keys — e.g. `{c: 6, h: 1}` (see
`examples/example_carbon_hydrogen.run.yaml`). Nothing else needs editing. From that
list the generator automatically builds, for each ion:

- a **species mask** that separates the ions by mean ion mass in the FLASH data,
- a charge-density field `<name>dens` and thermal-velocity field `vth<name>`,
- the per-species OSIRIS `rqm = 1836/ye` (averaged over the OSIRIS domain per ion),
- the OSIRIS species blocks in the deck and the matching init functions in the py-script.

### The one file to know: `flash_osiris/yt_plugin.py`

All species knowledge is consolidated here. It holds a `MOLAR_WEIGHTS` table (atomic
weights) and the mask logic. Each ion you name must be in that table; the lookup is
case-insensitive (`al` → `Al`). Ions are ordered by **ascending atomic mass**, and
mask `1` is the lightest. For `{al, si}` this reproduces the original convention
exactly (`al → 1`, `si → 2`, one mass threshold at ≈27.5 u).

**To add a new element:** add one row to `MOLAR_WEIGHTS` (if it isn't already there)
and list its name in `charge_states`. No other code changes.

The generator reads the resulting name→mask mapping straight off the dataset
(`ds.osiris_species_masks`), so the plugin and the generator can never disagree about
which mask integer is which ion.

> Charge state `Z` is **not** passed to OSIRIS explicitly: ions are loaded
> charge-equivalently (density = `n_e`, `q = +1`) with mass-per-charge `rqm = m_i/(Z
> m_e)`, and `Z` folds in implicitly through the FLASH `ye`/`sumy` fields. The
> `charge_states` values are carried through to downstream analysis, which needs `Z`.

---

## Required FLASH fields

The plugin (`load_for_osiris`) reads these FLASH fields, so your dump must provide
them (standard FLASH MHD + ionization output):

| purpose                 | fields |
|-------------------------|--------|
| densities              | `dens`, `gas: El_number_density`, `gas: ion_number_density` |
| magnetic field         | `magx`, `magy`, `magz` |
| bulk velocity          | `velx`, `vely`, `velz` |
| temperatures           | `tele` (electron), `tion` (ion) |
| ionization / mean mass | `ye` (electrons per baryon), `sumy` (ions per baryon = 1/Ā) |

From these it derives the E-field (`v×B/c`), current density (Ampère's law), ion/
electron drift velocities, the species mask, and the per-species densities and thermal
velocities. The species mask uses `1/sumy` (mean ion mass in proton masses) to
separate ions — so the FLASH run must actually contain the ions you list.

---

## Tests

```bash
python -m pytest tests/        # species-mask logic (no yt needed; stdlib + pytest)
```

The tests exec `yt_plugin.py` with stubs for yt's injected names, so the pure species
helpers stay covered in CI without the heavy yt stack.

---

## Notes / conventions

- **OSIRIS normalization** is to the electron plasma frequency `ω_pe` and reference
  density `n0`. Lengths are in `c/ω_pe`, time in `1/ω_pe`, fields/densities/energies as
  in OSIRIS' Gaussian-based normalization.
- **1D** runs are a lineout (cm endpoints → `c/ω_pe`); **2D** runs are a box in
  `c/ω_pe`. `x2` is the shock-normal axis in 2D.
- `dt` is set from the CFL condition (`dx·0.95/√dims`).
- Ion thermal velocity uses the **full** ion mass `m_i = m_p/sumy` (not the
  mass-per-charge), so the ion pressure `n_i k T_ion` and the electron/ion pressure
  ratio are physical and conserved through the FLASH→OSIRIS map.
