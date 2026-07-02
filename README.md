# flash2osiris

Convert **FLASH** MHD simulation output into initialized **OSIRIS** PIC input decks.

Given a FLASH HDF5 plot file and a small `run.yaml`, this tool:

1. reads the FLASH fields through a custom **yt plugin**,
2. derives the OSIRIS normalizations and per-species `rqm` (mass-per-charge) from the
   FLASH ionization fields,
3. saves the field/density/thermal-velocity slices as `.npy` grids, and
4. renders the OSIRIS input **deck** and the **python-init script** (which interpolates
   those slices onto the OSIRIS grid at runtime).

It works with **arbitrary plasmas** — ion populations are auto-detected from the FLASH
materials (laser target vs. background gas), so you usually configure nothing.

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
│   ├── example_1d.run.yaml                  # 1D lineout (Al chamber / Si target)
│   ├── example_2d.run.yaml                  # 2D box (Al chamber / Si target)
│   └── example_magshock2019_ch.run.yaml     # CH piston/background, to show plug-and-play
├── tests/test_species.py   # unit tests for the population (material) logic
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
python -m flash_osiris.generator --config examples/example_1d.run.yaml
# or:  ./runme.sh examples/example_1d.run.yaml
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

## Configuring ion populations (the important part)

OSIRIS ion populations are separated by **FLASH material**, not by ion mass. FLASH
laser-HEDP runs (`+mtmmmt`, `species=...` in the setup) carry a mass-fraction field per
material — typically `targ` (the laser **target**/piston) and `cham` (the **chamber**/
background gas), plus a `vac` vacuum that is dropped. A cell belongs to whichever
material dominates it. **This is auto-detected from the dump, so the default config is
nothing at all.**

Why material and not mass? Because the populations you care about (piston vs background)
can be the *same element mix*. In the CH example both `targ` and `cham` are a
carbon-hydrogen plasma — identical `sumy`, so a mass-based split physically cannot tell
them apart, but the material fields can.

Optionally rename the OSIRIS species with `species_names` (keys are FLASH material
fields, values are the names you want):

```yaml
species_names: {targ: piston, cham: background}   # name populations by role
# or, to reproduce the historical Al/Si run (cham=Al chamber, targ=Si target):
species_names: {cham: al, targ: si}
```

From the materials the generator automatically builds, for each population:

- a `dominant_material` field and a charge-density field `<name>dens` (the electron
  density where that material dominates) and thermal-velocity field `vth<name>`,
- the per-population OSIRIS `rqm = 1836/ye` and effective ion mass `m_i/m_e = 1836/sumy`,
  both **electron-density-weighted** over that population's cells in the OSIRIS domain —
  correct even for a compound (CH) material, with **no atomic-weight table needed**,
- the OSIRIS species blocks in the deck and the matching init functions in the py-script.

### The one file to know: `flash_osiris/yt_plugin.py`

All population knowledge is consolidated here: `parse_flash_species` (read the
`species=...` material list from the dump), `resolve_populations` (drop vacuum, apply
the `species_names` rename, assign each a dominant-material index), and the
`dominant_material`/`<name>dens` field registration. The generator reads the resulting
name→material and name→index maps straight off the dataset
(`ds.osiris_species_materials` / `ds.osiris_dominant_index`), so the plugin and the
generator can never disagree about which population is which.

> Charge state `Z` is **not** passed to OSIRIS explicitly: ions are loaded
> charge-equivalently (density = `n_e`, `q = +1`) with mass-per-charge `rqm = 1836/ye =
> m_i/(Z m_e)`, and `Z` folds in implicitly through the FLASH `ye`/`sumy` fields. The
> optional `charge_states` map in the run.yaml is analysis metadata for the
> collisionless deck (the generator otherwise never reads it) — **except** that with
> collisions enabled it also supplies each ion's `q_real` (see below).

---

## Collisions (optional, CPU-only)

OSIRIS Monte Carlo (binary) collisions run **only in the standard CPU solver** — the
`cuda` and `tiles` modes do not implement them. Add an optional `collisions:` block to
the run.yaml and set `algorithm: cpu` (the generator asserts this):

```yaml
algorithm: cpu
charge_states: {al: 13, si: 14}   # also supplies each ion's q_real (Z); electrons = 1
collisions:
  enabled: true
  n_collide: 1                    # cadence: collide every n_collide steps
  model: perez                    # perez (default) | sentoku | takizuka | isotropic
  nx_collision_cells: 1           # cells grouped per collision cell (per dim)
  coulomb_log: 10.0               # "auto" for physical ln(Lambda), or a fixed value
  species: [e, al, si]            # species that collide
  like_collide: [e, al, si]       # subset that also self-collides
```

When enabled the generator (1) emits `n0 = reference_density` in `nl_simulation` (the
collision module reads `n0` and aborts if it is unset), (2) adds `q_real`/`if_collide`/
`if_like_collide` to each species, sourcing `q_real` from `charge_states`, and (3)
appends the `collisions` namelist. You do **not** specify a collision frequency — OSIRIS
computes the physical rate from the plasma state and `n0`; `n_collide` is only the
cadence.

**Reduced-mass caveat:** OSIRIS builds `m_real = q_real * rqm` from the *deck* `rqm`
(reduced by `rqm_factor`), and the collision rate scales as `(q_a q_b / mu)^2`. So at
`rqm_factor != 1` the ion collisionality is distorted by ~`rqm_factor^2`; `coulomb_log`
is the single scalar knob to recalibrate the overall rate (the generator warns when
`coulomb_log: auto` is used with `rqm_factor != 1`). For a faithful collisional run
prefer `rqm_factor: 1`.

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
| materials              | one mass-fraction field per `species=` material (e.g. `targ`, `cham`) |

From these it derives the E-field (`v×B/c`), current density (Ampère's law), ion/
electron drift velocities, the `dominant_material` field, and the per-population
densities and thermal velocities. The material list is read from the dump's `species=`
setup string (in `sim info`); the vacuum material is dropped automatically.

> **Heads-up:** the OSIRIS init fundamentally needs `velx/vely/velz` and the full
> `magx/magy/magz`. Some FLASH plot files are written with a reduced variable list that
> omits them (e.g. `MagShock2019_hdf5_plt_cnt_0008` ships only `magx/magy`, no
> velocities) — use a checkpoint or a re-dump that includes them.

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
