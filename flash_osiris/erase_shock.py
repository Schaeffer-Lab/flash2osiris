"""Erase the (physically suspect) MHD shock ahead of the piston from an already
generated FLASH->OSIRIS 1D run, so OSIRIS sees a clean piston driving into the
ambient instead of a spurious shock.

The conversion pipeline (generator.py) is left completely untouched.  This is a
standalone post-processing step: it reads an existing run directory's normalized
``interp/*.npy`` slices and rewrites a band of them.

Why this is a 1D problem
------------------------
A 1D OSIRIS run samples the FLASH slices *only along the oblique lineout*
``FLASH = origin + s * (cos t, sin t)`` for arc-length ``s in [0, flash_distance]``
(see py_init's ``_flash_xy`` / generator's ``_lineout_points``).  So "erasing the
shock" is a 1D operation in ``s``: we only have to make the *sampled lineout*
correct.  We do that by overwriting, in every slice, the diagonal slab of cells
whose arc-length ``s(x, y)`` falls in ``[s1, s2]`` with the value the lineout has
just upstream of the shock (at ``s2``).  Because ``s``-target depends only on the
arc-length, the slab is constant along the perpendicular direction, so the
on-line interpolation reproduces the intended 1D profile exactly.

Maxwell / continuity care
-------------------------
* ``E = -v x B`` and the drift velocities were all computed upstream and stored as
  independent arrays, so filling the band with the same upstream lineout values
  keeps ``E = -v x B`` self-consistent by construction (exact for a hard replace;
  approximate only inside the narrow taper band).
* In 1D ``div B = d B_par / d s`` where ``B_par = magx cos t + magy sin t`` is the
  in-plane component *along* the lineout.  ``B_par`` must stay ~constant to keep
  ``div B = 0``.  We therefore hold ``B_par`` at its upstream constant across the
  whole band (no taper); only the tangential parts (``B_perp``, ``magz``) and the
  other fields are erased/tapered.  ``magx/magy`` are recomposed from the edited
  ``(B_par, B_perp)``.

Usage
-----
    python -m flash_osiris.erase_shock --run <run_dir> --s1 <c/wpe> --s2 <c/wpe>
        [--taper <c/wpe>] [--write]

Without ``--write`` it only produces ``<run_dir>/shock_erase_preview.png`` so you
can iterate on ``s1/s2/taper``.  The first invocation copies ``interp/`` to
``interp_raw/`` (once); every run derives its edit from that pristine backup, so
re-running with new bounds never compounds a previous edit.
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

import numpy as np
import yaml


# --------------------------------------------------------------------------- #
# Geometry: read what OSIRIS actually uses, don't re-derive it.
# --------------------------------------------------------------------------- #
class Geometry:
    """Lineout geometry + interpolators for a run dir, mirroring py_init's
    ``_flash_xy`` / ``_interp`` so the preview is exactly what OSIRIS samples."""

    def __init__(self, run_dir: Path, src_dir: Path):
        self.run_dir = Path(run_dir)
        self.src_dir = Path(src_dir)          # where the pristine .npy live (interp_raw)

        manifest = yaml.safe_load((self.run_dir / "run_manifest.yaml").read_text())
        d = manifest["derived"]
        if d.get("theta_rad") is None:
            raise ValueError(f"{run_dir} is not a 1D run (no theta_rad in manifest); "
                             "this tool only supports 1D lineouts.")
        self.x0 = float(d["xmin"])            # lineout origin = arc-length 0
        self.y0 = float(d["ymin"])
        self.theta = float(d["theta_rad"])
        # flash_distance = arc-length where the FLASH data ends (before any held-constant
        # extension buffer). Older runs predate the extension feature and the manifest key,
        # so fall back to the total domain length (no extension => flash_distance == distance).
        fd = d.get("flash_distance_c_over_wpe")
        if fd is None:
            fd = d["distance_c_over_wpe"]
        self.flash_distance = float(fd)
        self.cos, self.sin = np.cos(self.theta), np.sin(self.theta)

        # box_bounds (covering-grid extent) lives only in the rendered py-script;
        # it is the exact extent OSIRIS builds its interpolator over.
        self.box_bounds = _parse_box_bounds(self.run_dir)

        self._interp_cache: dict[str, object] = {}

    # -- coordinate helpers (mirror py_init) -------------------------------- #
    def line_xy(self, s):
        """FLASH (x, y) at arc-length s along the lineout."""
        s = np.asarray(s, dtype=float)
        return self.x0 + self.cos * s, self.y0 + self.sin * s

    def arc_length(self, X, Y):
        """Arc-length coordinate s of FLASH points (X, Y)."""
        return (X - self.x0) * self.cos + (Y - self.y0) * self.sin

    def cell_grid(self, shape):
        """(X, Y) FLASH coords for every cell of an [i_x, i_y] array of `shape`."""
        bb = self.box_bounds
        xs = np.linspace(bb["xmin"], bb["xmax"], shape[0])
        ys = np.linspace(bb["ymin"], bb["ymax"], shape[1])
        return np.meshgrid(xs, ys, indexing="ij")

    # -- field access ------------------------------------------------------- #
    def load(self, field):
        return np.load(self.src_dir / f"{field}.npy")

    def interp(self, field):
        from scipy.interpolate import RegularGridInterpolator
        if field not in self._interp_cache:
            g = self.load(field)
            bb = self.box_bounds
            self._interp_cache[field] = RegularGridInterpolator(
                (np.linspace(bb["xmin"], bb["xmax"], g.shape[0]),
                 np.linspace(bb["ymin"], bb["ymax"], g.shape[1])),
                g, method="linear", bounds_error=True, fill_value=0.0)
        return self._interp_cache[field]

    def sample_line(self, field, s):
        """Field values along the lineout at arc-length(s) s."""
        x, y = self.line_xy(s)
        return self.interp(field)(np.column_stack([np.atleast_1d(x), np.atleast_1d(y)]))

    def scalar_at(self, field, s):
        """Single scalar value of `field` on the lineout at arc-length s."""
        return float(self.sample_line(field, np.array([s]))[0])


def _parse_box_bounds(run_dir: Path) -> dict:
    py = next(run_dir.glob("py-script-*.py"), None)
    if py is None:
        raise FileNotFoundError(f"no py-script-*.py in {run_dir}")
    text = py.read_text()
    m = re.search(r'box_bounds\s*=\s*\{([^}]*)\}', text)
    if not m:
        raise ValueError(f"could not find box_bounds in {py}")
    vals = dict(re.findall(r'"(xmin|xmax|ymin|ymax)"\s*:\s*([-\d.eE+]+)', m.group(1)))
    return {k: float(vals[k]) for k in ("xmin", "xmax", "ymin", "ymax")}


# --------------------------------------------------------------------------- #
# The erase profile (1D in arc-length).
# --------------------------------------------------------------------------- #
def _taper_weight(s, s1, taper):
    """Cosine ramp 0->1 over [s1, s1+taper]; 1 above; 0 below s1."""
    if taper <= 0:
        return np.where(np.asarray(s) >= s1, 1.0, 0.0)
    w = 0.5 * (1.0 - np.cos(np.pi * np.clip((np.asarray(s) - s1) / taper, 0.0, 1.0)))
    return w


def scalar_target(s, f_s1, f_up, s1, taper):
    """Edited value of a scalar field at arc-length s within the band [s1, s2]:
    ramp from the piston-side value f_s1 up to the upstream value f_up over the
    taper, flat at f_up above it.  (taper=0 -> hard replace with f_up.)"""
    w = _taper_weight(s, s1, taper)
    return (1.0 - w) * f_s1 + w * f_up


# --------------------------------------------------------------------------- #
# Apply the erase to the 2D slices.
# --------------------------------------------------------------------------- #
def erase_run(run_dir, s1, s2, taper, write):
    run_dir = Path(run_dir)
    interp = run_dir / "interp"
    raw = run_dir / "interp_raw"

    # One-time pristine backup; all edits derive from it (idempotent re-runs).
    if not raw.exists():
        shutil.copytree(interp, raw)
        print(f"[backup] copied {interp} -> {raw}")

    geo = Geometry(run_dir, src_dir=raw)
    if not (0.0 <= s1 < s2 <= geo.flash_distance):
        raise ValueError(f"require 0 <= s1 < s2 <= flash_distance "
                         f"({geo.flash_distance:.1f}); got s1={s1}, s2={s2}")

    fields = sorted(p.stem for p in raw.glob("*.npy"))
    has_B = "magx" in fields and "magy" in fields
    scalar_fields = [f for f in fields if f not in ("magx", "magy")] if has_B else fields

    edited = {}   # field -> new array (only if changed)

    # -- scalar / free fields (edens, *dens, vth*, v_*, magz, E*, and lone B) -- #
    for field in scalar_fields:
        g = geo.load(field)
        X, Y = geo.cell_grid(g.shape)
        s_cell = geo.arc_length(X, Y)
        band = (s_cell >= s1) & (s_cell <= s2)
        f_s1 = geo.scalar_at(field, s1)
        f_up = geo.scalar_at(field, s2)
        gn = g.copy()
        gn[band] = scalar_target(s_cell[band], f_s1, f_up, s1, taper)
        edited[field] = gn

    # -- magnetic field: hold B_par constant (div B), taper B_perp ---------- #
    if has_B:
        c, sn = geo.cos, geo.sin
        gx, gy = geo.load("magx"), geo.load("magy")
        X, Y = geo.cell_grid(gx.shape)
        s_cell = geo.arc_length(X, Y)
        band = (s_cell >= s1) & (s_cell <= s2)

        # endpoint (par, perp) decompositions on the lineout
        def par_perp_at(s):
            bx, by = geo.scalar_at("magx", s), geo.scalar_at("magy", s)
            return bx * c + by * sn, -bx * sn + by * c
        par_up, perp_up = par_perp_at(s2)     # upstream
        _, perp_s1 = par_perp_at(s1)          # piston-side perp

        s_band = s_cell[band]
        par_new = np.full(s_band.shape, par_up)                          # constant, no taper
        perp_new = scalar_target(s_band, perp_s1, perp_up, s1, taper)    # tapered
        gxn, gyn = gx.copy(), gy.copy()
        gxn[band] = par_new * c - perp_new * sn
        gyn[band] = par_new * sn + perp_new * c
        edited["magx"], edited["magy"] = gxn, gyn

    make_preview(geo, s1, s2, taper, run_dir / "shock_erase_preview.png")

    if write:
        for field, gn in edited.items():
            np.save(interp / f"{field}.npy", gn)
        print(f"[write] rewrote {len(edited)} slices in {interp}")
    else:
        print(f"[dry-run] preview only; no slices written. Add --write to commit.")
    return geo


# --------------------------------------------------------------------------- #
# Preview PNG: original vs edited lineouts (the iteration loop).
# --------------------------------------------------------------------------- #
def _edited_line(geo, field, s, s1, s2, taper):
    orig = geo.sample_line(field, s)
    out = orig.copy()
    band = (s >= s1) & (s <= s2)
    out[band] = scalar_target(s[band], geo.scalar_at(field, s1),
                              geo.scalar_at(field, s2), s1, taper)
    return orig, out


def _edited_B_line(geo, s, s1, s2, taper):
    """Return (par, perp, magz) original and edited along the lineout."""
    c, sn = geo.cos, geo.sin
    bx, by = geo.sample_line("magx", s), geo.sample_line("magy", s)
    par, perp = bx * c + by * sn, -bx * sn + by * c
    band = (s >= s1) & (s <= s2)
    par_e, perp_e = par.copy(), perp.copy()
    par_e[band] = geo.scalar_at("magx", s2) * c + geo.scalar_at("magy", s2) * sn
    perp_s1 = -geo.scalar_at("magx", s1) * sn + geo.scalar_at("magy", s1) * c
    perp_up = -geo.scalar_at("magx", s2) * sn + geo.scalar_at("magy", s2) * c
    perp_e[band] = scalar_target(s[band], perp_s1, perp_up, s1, taper)
    return (par, perp), (par_e, perp_e)


def make_preview(geo, s1, s2, taper, out_path, n=1600):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    s = np.linspace(0.0, geo.flash_distance, n)
    fields = sorted(p.stem for p in geo.src_dir.glob("*.npy"))
    dens = [f for f in fields if f.endswith("dens")]
    vfields = [f for f in ("v_ix", "v_iy", "v_iz") if f in fields]
    efields = [f for f in ("Ex", "Ey", "Ez") if f in fields]
    has_B = "magx" in fields and "magy" in fields

    fig, axes = plt.subplots(4, 1, figsize=(11, 13), sharex=True)

    def mark(ax):
        ax.axvline(s1, color="k", ls="--", lw=0.8)
        ax.axvline(s2, color="k", ls="--", lw=0.8)
        if taper > 0:
            ax.axvspan(s1, s1 + taper, color="orange", alpha=0.12, label="taper")

    # 1) densities (log) -- place s1 just outside the piston.
    ax = axes[0]
    dens_lines = {}                       # field -> (orig, edit) for the inset
    for f in dens:
        o, e = _edited_line(geo, f, s, s1, s2, taper)
        dens_lines[f] = (o, e)
        line, = ax.plot(s, np.maximum(o, 1e-30), lw=1.0, alpha=0.45, label=f"{f} orig")
        ax.plot(s, np.maximum(e, 1e-30), lw=1.6, ls="--", color=line.get_color(),
                label=f"{f} edit")
    ax.set_yscale("log"); ax.set_ylabel("density [n0]"); mark(ax)
    ax.legend(fontsize=7, ncol=2, loc="upper right")
    ax.set_title("Densities — put s1 just outside the piston")

    # Zoomed inset around the chosen [s1, s2] so the bounds can be placed precisely.
    band = s2 - s1
    pad = max(0.75 * band, 3.0 * (geo.box_bounds["xmax"] - geo.box_bounds["xmin"])
              / (geo.load("edens").shape[0] - 1))          # >= a few covering-grid cells
    zlo, zhi = max(0.0, s1 - pad), min(geo.flash_distance, s2 + pad)
    ax.axvspan(zlo, zhi, color="gray", alpha=0.08)          # show the zoom window on the parent
    axins = ax.inset_axes([0.30, 0.40, 0.66, 0.55])
    win = (s >= zlo) & (s <= zhi)
    ymin, ymax = np.inf, -np.inf
    for f, (o, e) in dens_lines.items():
        (li,) = axins.plot(s[win], np.maximum(o[win], 1e-30), lw=1.2, alpha=0.5)
        axins.plot(s[win], np.maximum(e[win], 1e-30), lw=1.8, ls="--", color=li.get_color())
        vals = np.concatenate([o[win], e[win]])
        vals = vals[vals > 1e-6]
        if vals.size:
            ymin, ymax = min(ymin, vals.min()), max(ymax, vals.max())
    axins.set_yscale("log")
    if np.isfinite(ymin):
        axins.set_ylim(0.5 * ymin, 2.0 * ymin if ymax <= ymin else 2.0 * ymax)
    axins.axvline(s1, color="k", ls="--", lw=0.8)
    axins.axvline(s2, color="k", ls="--", lw=0.8)
    if taper > 0:
        axins.axvspan(s1, s1 + taper, color="orange", alpha=0.15)
    axins.set_xlim(zlo, zhi)
    axins.tick_params(labelsize=7)
    axins.set_title("zoom: density around [s1, s2]", fontsize=8)

    # 2) magnetic field: |B|, B_par (div-B check, should stay flat), B_perp, magz
    ax = axes[1]
    if has_B:
        (par, perp), (par_e, perp_e) = _edited_B_line(geo, s, s1, s2, taper)
        mo, mz_e = _edited_line(geo, "magz", s, s1, s2, taper)
        bmag = np.sqrt(par**2 + perp**2 + mo**2)
        bmag_e = np.sqrt(par_e**2 + perp_e**2 + mz_e**2)
        ax.plot(s, bmag, lw=1.0, alpha=0.45, label="|B| orig")
        ax.plot(s, bmag_e, lw=1.6, ls="--", label="|B| edit")
        ax.plot(s, par, lw=1.0, alpha=0.45, label="B∥ orig (div-B)")
        ax.plot(s, par_e, lw=1.6, ls="--", label="B∥ edit (should stay flat)")
        ax.plot(s, perp, lw=1.0, alpha=0.45, label="B⊥ orig")
        ax.plot(s, perp_e, lw=1.6, ls="--", label="B⊥ edit")
    ax.set_ylabel("B [norm]"); mark(ax); ax.legend(fontsize=7, ncol=2)
    ax.set_title("Magnetic field — B∥ is the ∇·B check (must stay flat)")

    # 3) ion fluid velocity
    ax = axes[2]
    for f in vfields:
        o, e = _edited_line(geo, f, s, s1, s2, taper)
        line, = ax.plot(s, o, lw=1.0, alpha=0.45, label=f"{f} orig")
        ax.plot(s, e, lw=1.6, ls="--", color=line.get_color(), label=f"{f} edit")
    ax.set_ylabel("v_i [norm]"); mark(ax); ax.legend(fontsize=7, ncol=2)
    ax.set_title("Ion fluid velocity")

    # 4) electric field
    ax = axes[3]
    for f in efields:
        o, e = _edited_line(geo, f, s, s1, s2, taper)
        line, = ax.plot(s, o, lw=1.0, alpha=0.45, label=f"{f} orig")
        ax.plot(s, e, lw=1.6, ls="--", color=line.get_color(), label=f"{f} edit")
    ax.set_ylabel("E [norm]"); mark(ax); ax.legend(fontsize=7, ncol=2)
    ax.set_title("Electric field (E=-v×B; exact for hard replace, approx in taper)")

    axes[-1].set_xlabel(r"arc-length along lineout  s  [$c/\omega_{pe}$]")
    fig.suptitle(f"Shock erase preview:  s1={s1:g}, s2={s2:g}, taper={taper:g} "
                 f"[c/ωpe]   (θ={geo.theta:.3f} rad)", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"[preview] wrote {out_path}")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Erase the MHD shock ahead of the piston "
                                            "from a 1D FLASH->OSIRIS run.")
    p.add_argument("--run", required=True, type=Path,
                   help="run directory (contains interp/, run_manifest.yaml, py-script-1d.py)")
    p.add_argument("--s1", required=True, type=float,
                   help="piston-side edge of the shock band [c/wpe arc-length]")
    p.add_argument("--s2", required=True, type=float,
                   help="upstream edge (values copied from here) [c/wpe arc-length]")
    p.add_argument("--taper", type=float, default=None,
                   help="cosine taper width at s1 [c/wpe]; default = 5%% of (s2-s1). "
                        "Use 0 for a hard replace.")
    p.add_argument("--write", action="store_true",
                   help="commit the edit to interp/ (default: preview PNG only)")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    taper = args.taper if args.taper is not None else 0.05 * (args.s2 - args.s1)
    erase_run(args.run, args.s1, args.s2, taper, args.write)


if __name__ == "__main__":
    main()
