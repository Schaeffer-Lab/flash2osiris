import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import yt
from jinja2 import Environment, FileSystemLoader


yt.enable_plugins()

logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Jinja deck/py-script templates live in the sibling templates/ directory.
TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "templates"


class FLASH_OSIRIS_Base:
    def __init__(self, 
                 path_to_FLASH_data: str,
                 OSIRIS_inputfile_name: str,
                 reference_density_cc: float,
                 ppc: int,
                 dx: float,
                 osiris_dims: int,
                 xmin: float,
                 xmax: float,
                 ymin: float,
                 ymax: float,
                 rqm_normalization_factor: float,
                 tmax_gyroperiods: float,
                 algorithm: str,
                 deck_options: Dict,
                 species_names: Dict[str, str] = None,  # optional {flash_material: osiris_name} rename
                 theta: float = None,        # 1D only (set by FLASH_OSIRIS_1D)
                 distance: float = None,     # 1D only (set by FLASH_OSIRIS_1D)
                 extension: float = 0.0,     # 1D only: buffer length [c/wpe] beyond the FLASH endpoint
                 normalizations_override: Dict[str, float] = {}):
        # No validation here: this interface is driven only from the terminal, so
        # argument types/choices/required-ness are enforced once by argparse, plus a
        # couple of checks in main() (file existence, dim-specific geometry).

        self.osiris_dims = osiris_dims
        self.FLASH_data = Path(path_to_FLASH_data)
        self.inputfile_name = OSIRIS_inputfile_name + f".{self.osiris_dims}d"
        self.n0 = reference_density_cc * yt.units.cm**-3
        self.ppc = ppc
        self.deck = deck_options
        self.interpolation = self.deck['interpolation']
        self.dx = dx
        self.rqm_factor = rqm_normalization_factor
        # Ion populations are auto-detected from the FLASH materials by the yt plugin
        # (target / chamber / ...); species_names is an optional {flash_material:
        # osiris_name} rename. self.species_names (the OSIRIS species names) is filled in
        # from the plugin once the dump is loaded, below.
        self._species_names_override = species_names
        # species_rqms is derived from FLASH (density-weighted mean 1836/ye over each
        # population's dominant-material cells, restricted to the OSIRIS domain) once the
        # covering grid + coords exist.
        self.tmax_gyroperiods = tmax_gyroperiods
        self.algorithm = algorithm
        self.dt = self.dx * 0.95 / np.sqrt(self.osiris_dims) # CFL condition
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.theta = theta
        # 1D lineout lengths [c/wpe]: flash_distance is the arc length that stays inside the
        # FLASH domain (defines the endpoint xmax/ymax, sampled for rqm/|B|); self.distance is
        # the TOTAL OSIRIS domain = flash_distance + extension. Fields in the extension buffer
        # are held constant at the endpoint value (see _lineout_points and the py-script clamp).
        self.flash_distance = distance
        self.extension = extension
        self.distance = (distance + extension) if distance is not None else None
        self.normalizations_override = normalizations_override

        # Outputs (deck, py-script, interp slices, figures, run.yaml) are written under
        # ./input_files/<inputfile_name>/ relative to the current working directory.
        self.proj_dir = Path.cwd()
        logger.info(f"Project directory: {self.proj_dir}")
        self.output_dir = self.proj_dir / "input_files" / self.inputfile_name
        logger.info(f"Output directory: {self.output_dir}")
        self.fig_dir = self.output_dir / "figures"

        self.omega_pe = np.sqrt(self.n0 * yt.units.electron_charge_mks**2 / (yt.units.eps_0 * yt.units.electron_mass)).to('1/s')
        logger.info(f"Plasma frequency omega_pe: {np.format_float_scientific(self.omega_pe.to('1/s'), 2)} 1/s")
        logger.info(f"1000 c/w_pe is {np.format_float_scientific((1000 * yt.units.speed_of_light / self.omega_pe).to('cm'),2)} cm")

        logger.info(f"Loading FLASH data from {self.FLASH_data}")
        self.ds = yt.load_for_osiris(self.FLASH_data.as_posix(),
                                     rqm_factor=self.rqm_factor,
                                     species_names=self._species_names_override)
        # Populations the plugin built from the FLASH materials:
        #   osiris_species_materials : {osiris_name -> flash material field}
        #   osiris_dominant_index    : {osiris_name -> dominant_material integer}
        self.species_materials = self.ds.osiris_species_materials
        self.species_dominant_index = self.ds.osiris_dominant_index
        self.species_names = list(self.species_materials.keys())
        logger.info(f"Ion populations (osiris_name <- flash material): {self.species_materials}")

        level = 3 # If you are getting inexplicable crashes, you are probably running out of memory. This is probably the culprit.
        # FLASH 2D dumps are degenerate in the unused axis (domain_dimensions == 1 there);
        # refining that axis fabricates phantom cells and garbage data (e.g. an 8-deep z
        # slab built from a single real z-cell). Only refine the non-degenerate dimensions.
        refine = self.ds.refine_by ** level
        self.dims = np.where(self.ds.domain_dimensions > 1,
                             self.ds.domain_dimensions * refine,
                             self.ds.domain_dimensions).astype(int)

        logger.info(f"Creating covering grid at level {level} with dims {self.dims}")

        self.all_data = self.ds.covering_grid(
            level,
            left_edge=self.ds.domain_left_edge,
            dims=self.dims,
            num_ghost_zones=1,
        )

        logger.info("Covering grid created successfully")
        logger.info("Extracting coordinate arrays")
        self.x = self.all_data['flash', 'x'][:, 0, 0] * self.omega_pe / yt.units.speed_of_light
        self.y = self.all_data['flash', 'y'][0, :, 0] * self.omega_pe / yt.units.speed_of_light
        self.z = self.all_data['flash', 'z'][0, 0, :] * self.omega_pe / yt.units.speed_of_light

        logger.info(f"x bounds: {np.round(self.x[[0, -1]], 2)} c/w_pe")
        logger.info(f"y bounds: {np.round(self.y[[0, -1]], 2)} c/w_pe")
        logger.info(f"z bounds: {np.round(self.z[[0, -1]], 2)} c/w_pe")

        # Per-population OSIRIS rqm (mass-per-charge) straight from FLASH: rqm = 1836/ye,
        # electron-density weighted over each population's dominant-material cells in the
        # OSIRIS domain (lineout for 1D / box for 2D). No charge state is specified.
        self.species_rqms = self._compute_species_rqms()
        logger.info(f"FLASH-derived population rqms (edens-wt 1836/ye): {self.species_rqms}")


        debye_osiris = np.sqrt(
            self.all_data['flash', 'tele'][-1, -1, 0] * yt.units.boltzmann_constant / (yt.units.electron_mass * yt.units.speed_of_light**2)
        )
        
        # logger.info(f"Debye length: {debye_osiris.value} osiris units")
        logger.info(f"Background temperature: {self.all_data['flash', 'tele'][-1, -1, 0].to('K'):.3e}")
        logger.info(f"Background temperature: {(self.all_data['flash', 'tele'][-1, -1, 0] * yt.units.boltzmann_constant).to('eV'):.3e}")


        # Diagnostic only (not used downstream): at the domain corner, the OSIRIS
        # mass-per-charge rqm = 1836/ye and the true ion/electron mass ratio = 1836/sumy.
        corner_ye = self.all_data['flash', 'ye'][-1, -1, 0]
        corner_sumy = self.all_data['flash', 'sumy'][-1, -1, 0]
        self.rqm_real = 1836 / corner_ye
        logger.info(f"{'*'*10} corner mass-per-charge (1836/ye): {self.rqm_real:.1f}; "
                    f"corner mass ratio m_i/m_e (1836/sumy): {1836/corner_sumy:.1f} {'*'*10}")


        logger.info("normalizing plasma parameters")
        ######## NORMALIZATIONS ######## 
        B_norm = (self.omega_pe * yt.units.electron_mass * yt.units.speed_of_light / yt.units.elementary_charge).to('Gauss')
        v_norm = (yt.units.speed_of_light / np.sqrt(self.rqm_factor)).to('cm/s')
        E_norm = (self.omega_pe * yt.units.electron_mass * yt.units.speed_of_light / yt.units.elementary_charge / np.sqrt(self.rqm_factor)).to('statV/cm')
        # E_norm must equal v_norm * B_norm so that E = -v x B holds in normalized units
        # E_norm = v_norm * B_norm

        logger.info(f"Electric field normalization: {E_norm:.3e}")
        logger.info(f"Magnetic field normalization: {B_norm:.3e}")
        logger.info(f"Velocity normalization: {v_norm:.3e} cm/s")
        
        self.normalizations = {
            'edens': self.n0,

            'magx': B_norm, 'magy': B_norm, 'magz': B_norm,

            'Ex': E_norm, 'Ey': E_norm, 'Ez': E_norm,

            'v_ix': v_norm, 'v_iy': v_norm, 'v_iz': v_norm,
            'v_ex': v_norm, 'v_ey': v_norm, 'v_ez': v_norm,

            'vthele': yt.units.speed_of_light,
        }
        # Per-species density / thermal-velocity normalizations (one pair per ion).
        for name in self.species_names:
            self.normalizations[f'{name}dens'] = self.n0
            self.normalizations[f'vth{name}'] = v_norm

        # Calculate gyrotime and simulation duration.  The run length is set in ion
        # gyroperiods of the LIGHTEST population (smallest effective ion mass m_i =
        # 1836/sumy), matching the original Al/Si behaviour (which used 'al', the
        # lighter ion).  The gyroperiod scales with that population's mass-per-charge rqm.
        ref_species = min(self.species_eff_mass, key=self.species_eff_mass.get)
        logger.info(f"Gyrotime reference population (lightest): {ref_species} "
                    f"(eff. mass m_i/m_e = {self.species_eff_mass[ref_species]:.1f})")
        # Gyrofrequency uses the TOTAL field magnitude |B| = sqrt(magx^2+magy^2+magz^2),
        # not a single component: the magnetizing field here is in-plane, so a magz-only
        # gyrofreq is wrong for this geometry. Sample |B| over the OSIRIS domain (median)
        # instead of one full-box corner that can land in field-free vacuum (-> inf).
        B_norm_domain = self._domain_B_magnitude() / self.normalizations['magz']
        logger.info(f"Domain-median |B|: {float(B_norm_domain):.4e} OSIRIS units")
        self.gyrotime = self.species_rqms[ref_species] / self.rqm_factor / B_norm_domain
        self.tmax = int(self.gyrotime * self.tmax_gyroperiods)

        # Verify. From the OSIRIS website we know that
        B_test = 1e5 * yt.units.Gauss / B_norm.to('Gauss')
        B_gauss = 5.681e-8 * B_test * self.omega_pe
        logger.info(f"Test magnetic field value: {B_test:.3e} OSIRIS units, which corresponds to {B_gauss:.3e} Gauss")
        # Do it another way to make sure
        B_gauss = 3.204e-3 * B_test * self.n0**0.5
        logger.info(f"Test magnetic field value: {B_test:.3e} OSIRIS units, which corresponds to {B_gauss:.3e} Gauss")

        n_species = 3
        if self.osiris_dims == 1:
            n_particles = self.distance / self.dx  * n_species * self.ppc
        elif self.osiris_dims == 2:
            n_particles = (self.xmax-self.xmin) * (self.ymax - self.ymin) / self.dx**2  * n_species * self.ppc**2
        n_bytes_particles = n_particles* 2 * 70 # maria says ~70 bytes per particle. I don't know if this is single or double precision, we also need to allocate for twice as many particles

        mem_per_GPU = 40e9
        max_bytes_per_GPU = mem_per_GPU * .8 # 80% of 40GB (A100)
        print("Number of particles: ", np.format_float_scientific(n_particles,3))

        logger.info(f"Recommended number of GPUs: {np.ceil(n_bytes_particles/max_bytes_per_GPU)}")
        logger.info(f"Recommended number of nodes: {np.ceil(n_bytes_particles/max_bytes_per_GPU/4)}")

        # __init__ materialized several covering-grid fields (x/y/z, tele, ye, magz);
        # coordinate arrays are already copied out, so drop the cache before slicing.
        self.all_data.clear_data()


    def _compute_species_rqms(self):
        """Per-population OSIRIS rqm (mass-per-charge, = 1836/ye) and effective ion mass
        (m_i/m_e = 1836/sumy), straight from FLASH and restricted to the OSIRIS domain
        (lineout for 1D, box for 2D).

        Each cell is attributed to a population by the plugin's ``dominant_material``
        field (the FLASH material that dominates the cell), so populations are separated
        by material -- never by ion mass and never hard-coded here.  The per-population
        rqm and effective mass are **electron-density weighted** averages over that
        population's cells, so near-vacuum cells (which can be force-assigned by the
        argmax) carry ~no weight.  Sets ``self.species_eff_mass`` and returns the rqms."""
        from scipy.interpolate import RegularGridInterpolator
        mid = self.dims[2] // 2
        if self.osiris_dims == 1:
            pts = np.column_stack([np.linspace(self.xmin, self.xmax, 4000),
                                   np.linspace(self.ymin, self.ymax, 4000)])
        else:
            gx, gy = np.meshgrid(np.linspace(self.xmin, self.xmax, 128),
                                 np.linspace(self.ymin, self.ymax, 128))
            pts = np.column_stack([gx.ravel(), gy.ravel()])

        def sample(field, method='linear'):
            arr = np.asarray(self.all_data['flash', field][:, :, mid])
            self.all_data.clear_data()
            return RegularGridInterpolator((self.x, self.y), arr, method=method,
                                           bounds_error=True)(pts)

        ye_line = sample('ye')
        sumy_line = sample('sumy')
        edens_line = sample('edens')
        dom_line = sample('dominant_material', method='nearest')
        rqm_line = PROTON_ELECTRON_MASS_RATIO / ye_line
        mass_line = PROTON_ELECTRON_MASS_RATIO / sumy_line

        rqms = {}
        self.species_eff_mass = {}
        for name, didx in self.species_dominant_index.items():
            sel = np.round(dom_line).astype(int) == didx
            w = edens_line * sel
            if sel.any() and w.sum() > 0:
                rqms[name] = float(np.average(rqm_line[sel], weights=edens_line[sel]))
                self.species_eff_mass[name] = float(np.average(mass_line[sel], weights=edens_line[sel]))
            else:
                rqms[name] = float(np.average(rqm_line, weights=edens_line))
                self.species_eff_mass[name] = float(np.average(mass_line, weights=edens_line))
                logger.warning(f"Population '{name}' (material "
                               f"'{self.species_materials[name]}') has no electron density "
                               f"in the OSIRIS domain; using domain-mean rqm "
                               f"{rqms[name]:.1f} as placeholder.")
        return rqms

    def _domain_B_magnitude(self):
        """Median total magnetic-field magnitude |B| over the OSIRIS domain (lineout for
        1D, box grid for 2D), returned as a YTQuantity in the B-normalization units.

        Used to set the ion gyroperiod. The relevant field is the magnitude |B| =
        sqrt(magx^2 + magy^2 + magz^2) -- never a single component (which is geometry
        dependent: the in-plane field dominates here, the out-of-plane field in the
        original MagShockZ setup). A domain median is robust to the field-free vacuum
        corners that a single-point sample can hit."""
        from scipy.interpolate import RegularGridInterpolator
        mid = self.dims[2] // 2
        if self.osiris_dims == 1:
            pts = np.column_stack([np.linspace(self.xmin, self.xmax, 4000),
                                   np.linspace(self.ymin, self.ymax, 4000)])
        else:
            gx, gy = np.meshgrid(np.linspace(self.xmin, self.xmax, 128),
                                 np.linspace(self.ymin, self.ymax, 128))
            pts = np.column_stack([gx.ravel(), gy.ravel()])

        b_unit = self.normalizations['magz'].units
        b2 = np.zeros(len(pts))
        for comp in ('magx', 'magy', 'magz'):
            arr = np.asarray(self.all_data['flash', comp][:, :, mid].to(b_unit))
            self.all_data.clear_data()
            b2 = b2 + RegularGridInterpolator((self.x, self.y), arr, method='linear',
                                              bounds_error=True)(pts) ** 2
        return float(np.median(np.sqrt(b2))) * b_unit

    def save_slices(self, normal_axis="z"):
        """
        Process and save field data slices for OSIRIS.
        
        Parameters
        ----------
        normal_axis : str, optional
            Axis normal to the slice plane ('x', 'y', or 'z'). Default is 'z'. # TODO allow for other normal axes
            
        Note:
            - Every field (densities included) is saved as a raw .npy grid; the
              py-script and the readers here rebuild the interpolator on load, so
              nothing depends on a pickled scipy object surviving a version change.
            - Densities additionally feed OSIRIS' own built-in interpolator.
        """
        interp_dir = self.output_dir / "interp"
        if not interp_dir.exists():
            interp_dir.mkdir(parents=True)
            
        # Validate normal_axis
        axis_map = {"x": 0, "y": 1, "z": 2}
        if normal_axis not in axis_map.keys():
            raise ValueError("normal_axis must be one of 'x', 'y', or 'z'")
        normal = axis_map[normal_axis]
        
        middle_index = self.dims[normal] // 2
        chunk_size = 32 
        
        for field, normalization in self.normalizations.items():
            if field in self.normalizations_override.keys():
                normalization = normalization * self.normalizations_override[field]
                logger.info(f"{field} is normalized by additional factor of {np.format_float_scientific(self.normalizations_override[field],3)}")
            logger.info(f"Processing {field} with normalization {np.format_float_scientific(normalization, 3)}")
            self._save_field(field, normalization, middle_index, chunk_size, interp_dir)
            # The covering grid caches every field it materializes (~1 GB each at
            # level 3). Drop the cache after each field so peak memory stays ~1 field
            # instead of growing to ~N_fields GB (which trips the login-node limit).
            self.all_data.clear_data()

    def _save_field(self, field, normalization, middle_index, chunk_size, interp_dir):
        """Save a field slice as a raw .npy grid indexed [i_x, i_y] over (self.x,
        self.y).  Every field (densities included) is stored the same way; the
        py-script and the generator's own readers rebuild the scipy interpolator
        on load, so nothing depends on the pickle format of a given scipy version."""
        field_data = np.zeros(self.all_data['flash', field][:, :, middle_index].shape)

        # Process data in chunks to save memory
        for i in range(0, self.all_data['flash', field].shape[0], chunk_size):
            end = min(i + chunk_size, self.all_data['flash', field].shape[0])
            chunk = self.all_data['flash', field][i:end, :, middle_index]
            field_data[i:end, :] = chunk / normalization

        # Densities: zero out negligible values so the vacuum is empty.
        if field.endswith('dens'):
            field_data[field_data < 0.001] = 0

        np.save(f"{interp_dir}/{field}.npy", field_data)

        # Tried to include memory cleanup here, but it just doesn't work

    def _field_interp(self, field):
        """Rebuild the (x_flash, y_flash) linear interpolator for a saved .npy
        slice -- the single reader used by every plotting/diagnostic path here,
        mirroring the py-script's _interp (no pickled scipy objects)."""
        from scipy.interpolate import RegularGridInterpolator
        data = np.load(self.output_dir / f'interp/{field}.npy')
        return RegularGridInterpolator(
            (np.asarray(self.x), np.asarray(self.y)), data,
            method='linear', bounds_error=True, fill_value=0)

    def write_input_file(self):
        """
        Generate and write OSIRIS input file using Jinja2 templates.
        
        Reads thermal velocity bounds and generates the input file
        with appropriate parameters for all species.
        """
        
        # Read thermal velocity bounds for all species
        thermal_bounds = self._read_thermal_bounds()
        
        # Prepare the context dictionary for Jinja2 template rendering
        if self.osiris_dims == 1:
            nx = int(self.distance / self.dx)
            ny = None
            xmin, xmax = 0, self.distance 
            ymin, ymax = None, None
        else:
            nx = int((self.xmax - self.xmin) / self.dx)
            ny = int((self.ymax - self.ymin) / self.dx)
            xmin, xmax = self.xmin, self.xmax
            ymin, ymax = self.ymin, self.ymax
            
        n_tiles_x, n_tiles_y = self._calculate_tile_numbers()
        
        species_list = self._prepare_species_list(thermal_bounds) # Type: List[Dictionary]

        d = self.deck  # tunable deck options (defaults merged + resolved in __init__)

        if self.osiris_dims == 1:
            num_par_max = int(self.ppc*nx/n_tiles_x/4) # Factor of 4 is random. Otherwise it's way too large idk
        else:
            num_par_max = int(nx*ny/(n_tiles_x*n_tiles_y)*self.ppc**2/4)

        context = {
            'dims': self.osiris_dims,
            'inputfile_name': self.inputfile_name,
            'algorithm': self.algorithm,
            'nx': nx,
            'ny': ny,
            'xmin': xmin,
            'xmax': xmax,
            'ymin': ymin,
            'ymax': ymax,
            'interpolation': self.interpolation,
            'ppc': self.ppc,
            'num_par_max': num_par_max,
            'dt': np.format_float_scientific(self.dt, 4),
            'ndump': int(self.tmax / (d['n_dump_total'] * self.dt)),
            'tmax': self.tmax,
            'tile_numbers': [n_tiles_x, n_tiles_y],
            'num_species': len(self.species_rqms) + 1,
            'species_list': species_list,
            # parallel / node configuration
            'node_number': d['node_number'],
            'num_threads': d['num_threads'],
            # restart
            'if_restart': '.true.' if d['restart'] else '.false.',
            # field solver / boundaries
            'vpml_bnd_size': d['vpml_bnd_size'],
            'emf_boundary': d['emf_boundary'],
            'part_boundary': d['part_boundary'],
            'smooth_type': d['smooth_type'],
            'smooth_order': d['smooth_order'],
            # diagnostics
            'n_ave': d['n_ave'],
            'emf_reports': d['emf_reports'],
            'reports': d['reports'],
            'rep_udist': d['rep_udist'],
            'phasespaces': d['phasespaces'],
            'e_ps_pmin': d['e_ps_pmin'],
            'e_ps_pmax': d['e_ps_pmax'],
            'i_ps_pmin': d['i_ps_pmin'],
            'i_ps_pmax': d['i_ps_pmax'],
            'ps_np': d['ps_np'],
            'ps_ngamma': d['ps_ngamma'],
            'ps_gammamax': d['ps_gammamax'],
            'ps_nx': d['ps_nx'],
            'ps_ny': d['ps_ny'],
        }

        # Load and render template
        env = Environment(loader=FileSystemLoader(TEMPLATE_DIR))
        template = env.get_template('osiris_deck_TEMPLATE.jinja')
        content = template.render(**context)
        
        # Write the actual input file
        input_file_path = self.output_dir / self.inputfile_name
        logger.info(f"Writing OSIRIS input file to {input_file_path}")
        
        with open(input_file_path, "w") as f:
            f.write(content)
        
        logger.info(f"OSIRIS input file written successfully")
   
    def _calculate_tile_numbers(self):
        precision = 8 # bytes per number (double precision)
        match self.interpolation:
            case 'linear':
                interp = 1
            case 'quadratic':
                interp = 2
            case 'cubic':
                interp = 3
            case 'quartic':
                interp = 4
            case _:
                raise ValueError("Unsupported interpolation type")
        ### FOR PERLMUTTER:
        shmemsize = 163e3 * 0.8 # 80% of shared memory per block in bytes

        ### FOR LOCAL MACHINES:
        # TBD: adjust shared memory size accordingly
        max_tile_size = int((shmemsize/(2*3*precision))**(1/self.osiris_dims) - (2 * interp - 1))
        print("Max tile size per dimension: ", max_tile_size)
        if self.osiris_dims == 1:
            # 1D domain length is the lineout distance (xmax-xmin is ~0 for a
            # theta=pi/2 lineout). n_tiles_y is unused in 1D but returned for a
            # consistent signature.
            i = 0
            n_tiles_x = 2**i
            n_tiles_y = 1
            tile_size_x = self.distance / self.dx / n_tiles_x
            while tile_size_x > max_tile_size:
                i += 1
                n_tiles_x = 2**i
                tile_size_x = self.distance / self.dx / n_tiles_x
        if self.osiris_dims == 2:
            i, j = 0, 0
            n_tiles_x, n_tiles_y = 2**i, 2**j
            tile_size_x = (self.xmax - self.xmin) / self.dx / n_tiles_x
            tile_size_y = (self.ymax - self.ymin) / self.dx / n_tiles_y
            while tile_size_x > max_tile_size:
                i += 1
                n_tiles_x = 2**i
                tile_size_x = (self.xmax - self.xmin) / self.dx / n_tiles_x
            while tile_size_y > max_tile_size:
                j += 1
                n_tiles_y = 2**j
                tile_size_y = (self.ymax - self.ymin) / self.dx / n_tiles_y
        logger.info(f"Calculated tile numbers: n_tiles_x = {n_tiles_x}, n_tiles_y = {n_tiles_y}")
        return n_tiles_x, n_tiles_y
    
    def _prepare_species_list(self, thermal_bounds):
        """Prepare species data for template rendering."""
        species_list = []
        
        # Add electrons
        electron_config = self._get_species_config('e', thermal_bounds['electron'], is_electron=True)
        species_list.append(electron_config)
        
        # Add ions
        for ion, bounds in thermal_bounds['ions'].items():
            ion_config = self._get_species_config(ion, bounds, is_electron=False)
            species_list.append(ion_config)
        
        return species_list
    
    def _get_species_config(self, species_name, thermal_bounds, is_electron=False):
        """Get configuration dictionary for a single species."""
        # Phase-space momentum bounds come from the deck options (e_ps_*/i_ps_*),
        # rendered at the template's diag_species level -- not per species here.
        config = {
            'name': species_name,
            'rqm': -1.0 if is_electron else int(self.species_rqms[species_name] / self.rqm_factor),
        }
        
        # Thermal velocity bounds (dimension-dependent)
        if self.osiris_dims == 1:
            vth_start, vth_end = thermal_bounds
            config['vth_start'] = np.format_float_scientific(vth_start, 4)
            config['vth_end'] = np.format_float_scientific(vth_end, 4)
        else:
            # x (direction 1) is periodic in 2D, so only the y-face thermal bounds are
            # used by the deck (spe_bound uth_bnd(...,2)).
            config['vth_y_start'] = np.format_float_scientific(thermal_bounds['y'][0], 4)
            config['vth_y_end'] = np.format_float_scientific(thermal_bounds['y'][1], 4)
        
        return config

    def _read_thermal_bounds(self) -> Dict:
        """
        Read thermal velocity bounds for electrons and all ion species.
        
        Parameters
        ----------
        end_point : list
            Endpoint coordinates [x, y]
            
        Returns
        -------
        dict
            Thermal velocity bounds for electrons and all ion species
        """
        if self.osiris_dims == 1:
            bounds = {
            'electron': None,
            'ions': {}
            }
            # Sample vth at the two lineout endpoints, in (x, y) order.
            vthele = self._field_interp("vthele")
            bounds['electron'] = [
                vthele((self.xmin, self.ymin)),
                vthele((self.xmax, self.ymax))
            ]
            logger.info(f"Electron thermal velocity bounds: {bounds['electron']}")

            for ion in self.species_rqms.keys():
                vthion = self._field_interp(f"vth{ion}")
                bounds['ions'][ion] = [
                    vthion((self.xmin, self.ymin)),
                    vthion((self.xmax, self.ymax))
                ]
                logger.info(f"{ion} thermal velocity bounds: {bounds['ions'][ion]}")
        elif self.osiris_dims == 2:
            bounds = {
                'electron': {},
                'ions': {}
            }
            # x (direction 1) is periodic in 2D, so only the y-face thermal bounds are
            # needed (sampled across x at the ymin/ymax faces).
            num_samples = 16  # Number of points to sample
            x_samples = np.linspace(self.xmin, self.xmax, num_samples)
            vthele = self._field_interp("vthele")
            bounds['electron']['y'] = [
                np.mean([vthele((x, self.ymin)) for x in x_samples]),
                np.mean([vthele((x, self.ymax)) for x in x_samples])
            ]
            logger.info(f"Electron thermal velocity bounds: {bounds['electron']}")

            for ion in self.species_rqms.keys():
                vthion = self._field_interp(f"vth{ion}")
                bounds['ions'][ion] = {'y': [
                    np.mean([vthion((x, self.ymin)) for x in x_samples]),
                    np.mean([vthion((x, self.ymax)) for x in x_samples])
                ]}
                logger.info(f"{ion} thermal velocity bounds: {bounds['ions'][ion]}")
        return bounds
    
    def write_python_file(self):
        """
        Generate and write OSIRIS python initialization file using Jinja2 templates.
        """
        # Prepare the context dictionary for Jinja2 template rendering
        context = {
            "dims": self.osiris_dims,
            "start_point": [self.xmin, self.ymin],
            "distance": self.distance,
            "flash_distance": self.flash_distance,
            "theta": self.theta,
            "xmin": self.xmin,
            "xmax": self.xmax,
            "ymin": self.ymin,
            "ymax": self.ymax,
            "species_list": list(self.species_rqms.keys()),
            "box_bounds": {
                "xmin": np.min(self.x).value,
                "xmax": np.max(self.x).value,
                "ymin": np.min(self.y).value,
                "ymax": np.max(self.y).value,
            },
        }
        
        # Load and render template
        env = Environment(loader=FileSystemLoader(TEMPLATE_DIR))
        template = env.get_template('py_init_TEMPLATE.jinja')
        content = template.render(**context)
        
        # Write the actual python initialization file
        python_file_path = self.output_dir / f"py-script-{self.osiris_dims}d.py"
        logger.info(f"Writing OSIRIS python initialization file to {python_file_path}")
        
        with open(python_file_path, "w") as f:
            f.write(content)
        
        logger.info(f"OSIRIS python initialization file written successfully")
    
    def write_manifest(self, cli_args=None):
        """
        Write run_manifest.yaml into the output directory so a scratch directory is
        self-describing even when the folder name is ambiguous. To avoid redundancy
        with the runme, it records only provenance (git commit, UTC timestamp, the
        exact CLI command) and quantities derived here that are NOT in the runme
        (omega_pe, c/omega_pe, gyrotime, tmax, and the normalized-unit geometry).
        """
        import subprocess
        import datetime
        import yaml

        try:
            git_hash = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=str(self.proj_dir),
                stderr=subprocess.DEVNULL).decode().strip()
        except Exception:
            git_hash = "unknown"

        c_over_wpe_cm = float((yt.units.speed_of_light / self.omega_pe).to('cm').value)

        # The runme/CLI command is the full record of *inputs*; the manifest stores
        # only provenance plus quantities *derived* here (not present in the runme),
        # to avoid duplicating values that already live in the runme.
        manifest = {
            "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "git_commit": git_hash,
            "cli_command": cli_args,
            "derived": {
                "omega_pe_rad_s": float(self.omega_pe.to('1/s').value),
                "c_over_wpe_cm": c_over_wpe_cm,
                "c_over_wpe_um": c_over_wpe_cm * 1e4,
                "gyrotime_wpe_inv": float(self.gyrotime),
                "tmax_wpe_inv": float(self.tmax),
                # 1D lineout endpoints are given in cm in the runme; record the
                # normalized-unit geometry the deck actually uses.
                "xmin": float(self.xmin), "xmax": float(self.xmax),
                "ymin": float(self.ymin), "ymax": float(self.ymax),
                "theta_rad": (None if self.theta is None else float(self.theta)),
                # distance_c_over_wpe is the TOTAL OSIRIS domain (flash lineout + extension);
                # flash/extension split recorded separately so analysis knows where the
                # physical FLASH data ends and the held-constant buffer begins.
                "distance_c_over_wpe": (None if self.distance is None else float(self.distance)),
                "flash_distance_c_over_wpe": (None if self.flash_distance is None else float(self.flash_distance)),
                "extension_c_over_wpe": float(self.extension),
            },
        }

        out_path = self.output_dir / "run_manifest.yaml"
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)
        with open(out_path, "w") as f:
            yaml.safe_dump(manifest, f, default_flow_style=False, sort_keys=False)
        logger.info(f"Run manifest written to {out_path}")
        return out_path

    def _lineout_points(self, n):
        """Sample points along the 1D OSIRIS lineout, mirroring the py-script.

        Returns ``(pts, dist)`` where ``dist = linspace(0, self.distance, n)`` spans the FULL
        OSIRIS domain and ``pts`` are the FLASH (x, y) for arc-length ``clip(dist, 0,
        flash_distance)`` -- i.e. the extension buffer (dist > flash_distance) is held constant
        at the endpoint (xmax, ymax), exactly like the runtime clamp in py_init's _flash_xy. So
        the extension shows up as a flat segment instead of stretching the FLASH profile."""
        dist = np.linspace(0, self.distance, n)
        s = np.clip(dist, 0.0, self.flash_distance)
        x_points = self.xmin + np.cos(self.theta) * s
        y_points = self.ymin + np.sin(self.theta) * s
        return np.column_stack([x_points, y_points]), dist

    def plot1D(self, fields):
        import matplotlib.pyplot as plt

        self.fig_dir.mkdir(parents=True, exist_ok=True)
        if isinstance(fields, str):
            fields = [fields]  # Convert to a list with a single element
        else:
            fields = fields  # Use as is

        for field in fields:
            f = self._field_interp(field)
            data = np.load(self.output_dir / f'interp/{field}.npy')   # (i_x, i_y)

            # Create figure with two subplots side by side
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

            # Left subplot: 2D plane with lineout. Arrays are stored (i_x, i_y);
            # transpose to (i_y, i_x) so imshow's (rows=y, cols=x) matches the extent.
            img = np.log(data).T if field.endswith('dens') else data.T
            im = ax1.imshow(img, origin='lower',
                        extent=[self.x[0], self.x[-1], self.y[0], self.y[-1]])
            ax1.plot([self.xmin, self.xmax], [self.ymin, self.ymax], color='r', linewidth=2, label='Lineout')
            ax1.set_xlabel(r'x [$c/\omega_{pe}$]')
            ax1.set_ylabel(r'y [$c/\omega_{pe}$]')
            ax1.set_title(f'{field} - 2D plane')
            ax1.legend()
            plt.colorbar(im, ax=ax1)

            # Sample the lineout over the FULL OSIRIS domain; the extension buffer (beyond
            # flash_distance) is held constant at the endpoint, matching the runtime clamp.
            n_points = 10000
            pts, dist = self._lineout_points(n_points)
            data_line = f(pts)

            # Right subplot: 1D lineout
            ax2.plot(dist, data_line, color='r', linewidth=2)
            ax2.set_xlabel(r'Distance along lineout [$c/\omega_{pe}$]')
            ax2.set_ylabel(field)
            ax2.set_title(f'{field} - Lineout')
            ax2.grid(True)
            
            plt.tight_layout()
            plt.savefig(f'{self.fig_dir}/{field}_1D.png', dpi=150)
            plt.close()

    def _sample_conservation(self, pts):
        """Compute the FLASH (physical) and OSIRIS-init (normalized) dimensionless
        conservation quantities at sample points pts (N x 2, in c/wpe). Returns
        (flash, osiris) dicts keyed by quantity. Shared by the 1D (lineout) and 2D
        (box-grid) diagnostics. FLASH side: physical covering-grid fields (formulas
        from flash_utils.mach_numbers). OSIRIS side: the saved normalized interp
        slices (formulas from scripts/dimensionless_params.py). Requires
        save_slices() to have run first."""
        from scipy.interpolate import RegularGridInterpolator

        kB = 1.380649e-16        # erg/K
        c_cgs = 2.99792458e10    # cm/s
        mid = self.dims[2] // 2

        def fl(field, unit=None, ftype='flash'):
            sl = self.all_data[ftype, field][:, :, mid]
            arr = np.asarray(sl if unit is None else sl.to(unit))
            self.all_data.clear_data()  # cap covering-grid memory (one field at a time)
            return RegularGridInterpolator((np.asarray(self.x), np.asarray(self.y)), arr,
                                           method='linear', bounds_error=True)(pts)

        def osa(field):
            return self._field_interp(field)(pts)

        # ---- FLASH physical (CGS) ----
        B = np.sqrt(sum(fl(f, 'G')**2 for f in ('magx', 'magy', 'magz')))
        rho = fl('dens', 'g/cm**3')
        ne = fl('El_number_density', 'cm**-3', ftype='gas')
        nion = fl('ion_number_density', 'cm**-3', ftype='gas')
        Te = fl('tele', 'K'); Ti = fl('tion', 'K')
        vmag = np.sqrt(sum(fl(f, 'cm/s')**2 for f in ('v_ix', 'v_iy', 'v_iz')))
        vA_f = B / np.sqrt(4.0 * np.pi * rho)
        PB_f = B**2 / (8.0 * np.pi)
        flash = {
            'pressure_ratio':    (ne * kB * Te) / (nion * kB * Ti),  # P_e/P_i
            'mach_alfven':       vmag / vA_f,
            'beta_e':            ne * kB * Te / PB_f,
            'beta_i':            nion * kB * Ti / PB_f,
            'magnetization':     (vA_f / c_cgs)**2,
        }

        # ---- OSIRIS-init normalized ----
        # Everything in OSIRIS-normalized units: density in n0, velocity in c, B in
        # B_norm = sqrt(4 pi n0 m_e c^2).  Hence the magnetic pressure B^2/8pi -> B2/2 and
        # all pressures/energies are in n0 m_e c^2.  edens / <name>dens are CHARGE
        # densities [n0 e]; species rqm = m_i/(Z m_e) (mass-per-charge), so rqm*charge
        # density is the ion MASS density -- Z folds in implicitly via FLASH ye/sumy, so no
        # explicit charge state is needed (only rqm_factor).
        B2 = sum(osa(f)**2 for f in ('magx', 'magy', 'magz'))
        edens_s = osa('edens')
        # Sum the ion mass density and ion pressure over every species.  vth<name> uses
        # the FULL ion mass m_i = m_p/sumy (see yt_plugin.py), so the ions are loaded at
        # the real ion thermal speed and the ion pressure
        #   P_i = sum_s rho_s * vth_s^2 = n_i k T_ion   (physical, Z implicit)
        # is conserved.  We therefore conserve the electron/ion PRESSURE ratio, not the
        # per-particle T_e/T_i (which reads ~Z x: each macro-ion carries 1/Z of a real
        # ion's thermal energy).  Magnetic pressure B^2/8pi -> B2/2 in these units.
        rho_sim = 0.0   # ion mass density [n0 m_e]
        P_i = 0.0       # ion pressure [n0 m_e c^2]
        for name in self.species_names:
            rqm_s = self.species_rqms[name] / self.rqm_factor
            dens_s = osa(f'{name}dens')
            rho_sim = rho_sim + rqm_s * dens_s
            P_i = P_i + rqm_s * dens_s * osa(f'vth{name}')**2
        P_e = edens_s * osa('vthele')**2                                    # [n0 m_e c^2]
        PB  = B2 / 2.0                                                      # B^2/8pi
        vmag_s = np.sqrt(sum(osa(f)**2 for f in ('v_ix', 'v_iy', 'v_iz')))
        osiris = {
            'pressure_ratio': P_e / P_i,
            'mach_alfven':    vmag_s / np.sqrt(B2 / rho_sim),   # v_A = sqrt(sigma)
            'beta_e':         P_e / PB,
            'beta_i':         P_i / PB,
            'magnetization':  B2 / rho_sim,
        }
        return flash, osiris

    def _overlay_plot(self, fname, dist, flash_y, osiris_y, ylabel, title,
                      flash_label, osiris_label, annotation=None, logy=False):
        """Overlay a FLASH (physical) and OSIRIS-init (normalized) profile along the
        lineout, with a relative-deviation panel below (temperature-ratio style)."""
        import matplotlib.pyplot as plt

        with np.errstate(divide='ignore', invalid='ignore'):
            rel_dev = np.where(flash_y != 0,
                               np.abs(osiris_y - flash_y) / np.abs(flash_y), np.nan)
        max_dev = np.nanmax(rel_dev)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True,
                                       gridspec_kw={'height_ratios': [3, 1]})
        ax1.plot(dist, flash_y, color='k', lw=2, label=flash_label)
        ax1.plot(dist, osiris_y, color='r', lw=2, ls='--', label=osiris_label)
        if logy:
            # symlog (not log): these profiles span ~5 decades and hit ~0 in the far
            # upstream / between features, where pure log would silently drop points.
            # linthresh is a robust low percentile of the nonzero |values|, so the
            # physically-interesting O(1-10) region expands while the near-zero
            # baseline stays visible (degrades to ~log when there are no zeros).
            both = np.concatenate([np.asarray(flash_y, float).ravel(),
                                   np.asarray(osiris_y, float).ravel()])
            pos = np.abs(both[np.isfinite(both) & (both != 0)])
            linthresh = float(np.percentile(pos, 10)) if pos.size else 1e-12
            ax1.set_yscale('symlog', linthresh=linthresh)
        ax1.set_ylabel(ylabel)
        ax1.set_title(title)
        ax1.legend()
        ax1.grid(True)
        if annotation:
            ax1.text(0.02, 0.97, annotation, transform=ax1.transAxes, va='top',
                     bbox=dict(boxstyle='round', fc='w', alpha=0.85))
        ax2.plot(dist, rel_dev, color='b', lw=1.5)
        ax2.set_xlabel(r'Distance along lineout [$c/\omega_{pe}$]')
        ax2.set_ylabel('rel. dev.')
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig(self.fig_dir / fname, dpi=150)
        plt.close()
        logger.info(f"{title}: max relative deviation {max_dev:.3e} -> {self.fig_dir / fname}")
        return max_dev

    def _overlay_plot_2d(self, fname, extent, flash2d, osiris2d, title, label,
                         log=False, annotation=None):
        """Three-panel 2D comparison over the OSIRIS box: FLASH | OSIRIS-init |
        relative residual (OSIRIS-FLASH)/|FLASH|. flash2d/osiris2d are (ny, nx)."""
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm

        with np.errstate(divide='ignore', invalid='ignore'):
            resid = np.where(flash2d != 0, (osiris2d - flash2d) / np.abs(flash2d), np.nan)
        max_dev = np.nanmax(np.abs(resid))

        both = np.concatenate([flash2d[np.isfinite(flash2d)], osiris2d[np.isfinite(osiris2d)]])
        if log:
            pos = both[both > 0]
            norm = LogNorm(vmin=np.percentile(pos, 1), vmax=np.percentile(pos, 99)) if pos.size else None
            kw = dict(norm=norm)
        else:
            kw = dict(vmin=np.percentile(both, 1), vmax=np.percentile(both, 99))

        fig, axes = plt.subplots(1, 3, figsize=(19, 5.5))
        for ax, data, sub in ((axes[0], flash2d, 'FLASH'), (axes[1], osiris2d, 'OSIRIS-init')):
            im = ax.imshow(data, origin='lower', extent=extent, aspect='auto', **kw)
            ax.set_title(f'{sub}  {label}')
            ax.set_xlabel(r'x [$c/\omega_{pe}$]')
            fig.colorbar(im, ax=ax)
        axes[0].set_ylabel(r'y [$c/\omega_{pe}$]')
        vlim = np.nanpercentile(np.abs(resid), 99) or 1.0
        im = axes[2].imshow(resid, origin='lower', extent=extent, aspect='auto',
                            cmap='RdBu_r', vmin=-vlim, vmax=vlim)
        axes[2].set_title('relative residual (OSIRIS-FLASH)/|FLASH|')
        axes[2].set_xlabel(r'x [$c/\omega_{pe}$]')
        fig.colorbar(im, ax=axes[2])
        if annotation:
            axes[1].text(0.02, 0.97, annotation, transform=axes[1].transAxes, va='top',
                         color='w', bbox=dict(boxstyle='round', fc='k', alpha=0.4))
        fig.suptitle(title)
        plt.tight_layout()
        plt.savefig(self.fig_dir / fname, dpi=150)
        plt.close()
        logger.info(f"{title}: max |relative residual| {max_dev:.3e} -> {self.fig_dir / fname}")
        return max_dev

    def plot_conservation_diagnostics(self, n_points=10000):
        """FLASH-vs-OSIRIS-init conservation check for the dimensionless parameters
        (P_e/P_i, Alfvenic Mach number, electron beta, ion beta, magnetization).

        1D: line overlay + relative-deviation panel along the lineout.
        2D: three-panel imshow (FLASH | OSIRIS | relative residual) over the box.

        plot+warn: always writes the figures and logs the max deviation. Ions are loaded
        charge-equivalently (density = n_e, charge q=+1) with mass-per-charge rqm =
        m_i/(Z m_e), and the per-species vth<name> use the FULL ion mass m_i = m_p/sumy
        (see yt_plugin.py), so the macro-ions move at the real ion thermal speed. The charge
        state Z = ye/sumy enters only implicitly through the FLASH fields -- no explicit Z
        is needed, only rqm_factor. Conservation (at t=0):
          - P_e/P_i, M_A, beta_e, beta_i : conserved. beta_i works because the ion
                          pressure rho_sim*vth_i^2 = n_i k T_ion is physical (mass-per-
                          charge rqm gives the right mass density; full-mass vth gives the
                          right thermal speed).
          - sigma    : ~rqm_factor x physical (sim ions lighter); converges at
                          rqm_factor = 1.
        NB: the per-particle T_e/T_i is NOT conserved (reads ~Z x) -- each macro-ion
        carries 1/Z of a real ion's thermal energy -- so we conserve the *pressure* ratio
        instead.
        """
        self.fig_dir.mkdir(parents=True, exist_ok=True)
        # key -> (ylabel, title, FLASH label, OSIRIS label, log)
        specs = [
            ('pressure_ratio', r'$P_e/P_i$', 'Electron/ion pressure ratio at t=0',
             r'FLASH  $n_e T_e/(n_i T_i)$', r'OSIRIS  $P_e/P_i$', True),
            ('mach_alfven', r'$M_A$', 'Alfvenic Mach number at t=0',
             r'FLASH  $|v|/v_A$', r'OSIRIS  $|v|/\sqrt{\sigma}$', True),
            ('beta_e', r'$\beta_e$', 'Electron beta at t=0',
             r'FLASH  $n_e k_B T_e/(B^2/8\pi)$', r'OSIRIS  $n_e T_e/(B^2/2)$', True),
            ('beta_i', r'$\beta_i$', 'Ion beta at t=0',
             r'FLASH  $n_i k_B T_i/(B^2/8\pi)$', r'OSIRIS  $\rho_{sim}\,v_{th,i}^2/(B^2/2)$', True),
            ('magnetization', r'$\sigma$', 'Magnetization at t=0',
             r'FLASH  $B^2/(4\pi\rho c^2)$', r'OSIRIS  $B^2/(\mathrm{rqm}_i n_e)$', True),
        ]
        if self.rqm_factor == 1:
            sigma_note = 'rqm_factor=1: $\\sigma$ converges to physical'
        else:
            sigma_note = f'sim ions rqm_factor={self.rqm_factor:g}x lighter:\n$\\sigma_{{sim}}\\approx$ rqm_factor$\\,\\sigma_{{phys}}$'
        notes = {'magnetization': sigma_note}

        if self.osiris_dims == 1:
            # Full OSIRIS domain; extension buffer held constant at the endpoint (flat segment).
            pts, dist = self._lineout_points(n_points)
            flash, osiris = self._sample_conservation(pts)
            for key, ylabel, title, fl_lbl, os_lbl, logy in specs:
                self._overlay_plot(f'{key}_1D.png', dist, flash[key], osiris[key],
                                   ylabel, title, fl_lbl, os_lbl, logy=logy,
                                   annotation=notes.get(key))
        else:
            nx = min(400, max(64, int((self.xmax - self.xmin) / self.dx)))
            ny = min(400, max(64, int((self.ymax - self.ymin) / self.dx)))
            XX, YY = np.meshgrid(np.linspace(self.xmin, self.xmax, nx),
                                 np.linspace(self.ymin, self.ymax, ny))   # (ny, nx)
            pts = np.column_stack([XX.ravel(), YY.ravel()])
            flash, osiris = self._sample_conservation(pts)
            extent = [self.xmin, self.xmax, self.ymin, self.ymax]
            for key, ylabel, title, fl_lbl, os_lbl, logy in specs:
                self._overlay_plot_2d(f'{key}_2D.png', extent,
                                      flash[key].reshape(ny, nx), osiris[key].reshape(ny, nx),
                                      title, ylabel, log=logy, annotation=notes.get(key))

    def plot2D(self, fields):
        import matplotlib.pyplot as plt
        self.fig_dir.mkdir(parents=True, exist_ok=True)
        output = {}
        if isinstance(fields, str):
            fields = [fields]  # Convert to a list with a single element
        else:
            fields = fields  # Use as is
        for field in fields:
            f = self._field_interp(field)
            data = np.load(self.output_dir / f'interp/{field}.npy')   # (i_x, i_y)

            # Create figure with two subplots side by side
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))


            # plt.imshow(data.T, origin='lower',
            #             extent=[self.x[0], self.x[-1], self.y[0], self.y[-1]],vmax = 10) # TODO add some logic for vmax

            ax1.plot([self.xmin, self.xmax], [self.ymin, self.ymin], color='r')
            ax1.plot([self.xmin, self.xmax], [self.ymax, self.ymax], color='r')
            ax1.plot([self.xmin, self.xmin], [self.ymin, self.ymax], color='r')
            ax1.plot([self.xmax, self.xmax], [self.ymin, self.ymax], color='r')

            # Left subplot: 2D plane with lineout. Stored (i_x, i_y) -> transpose.
            img = np.log(data).T if field.endswith('dens') else data.T
            im = ax1.imshow(img, origin='lower',
                    extent=[self.x[0], self.x[-1], self.y[0], self.y[-1]])
            ax1.set_xlabel(r'x [$c/\omega_{pe}$]')
            ax1.set_ylabel(r'y [$c/\omega_{pe}$]')
            ax1.set_title(f'{field} - 2D plane')
            ax1.legend()
            fig.colorbar(im, ax=ax1)

            # Create points along the lineout from (xmin, ymin) to (xmax, ymax)
            n_points = 10000
            
            bottom_line_x = np.linspace(self.xmin, self.xmax, n_points)
            bottom_line_y = self.ymin * np.ones_like(bottom_line_x)
            bottom_line = np.column_stack([bottom_line_x, bottom_line_y])
            bottom_data = f(bottom_line)

            top_line_x = np.linspace(self.xmin, self.xmax, n_points)
            top_line_y = self.ymax * np.ones_like(top_line_x)
            top_line = np.column_stack([top_line_x, top_line_y])
            top_data = f(top_line)

            left_line_x = self.xmin * np.ones_like(bottom_line_x)
            left_line_y = np.linspace(self.ymin, self.ymax, n_points)
            left_line = np.column_stack([left_line_x, left_line_y])
            left_data = f(left_line)

            right_line_x = self.xmax * np.ones_like(bottom_line_x)
            right_line_y = np.linspace(self.ymin, self.ymax, n_points)
            right_line = np.column_stack([right_line_x, right_line_y])
            right_data = f(right_line)
            
            # Right subplot: 1D lineout
            bottom_distance = np.linspace(0, self.xmax - self.xmin, n_points)
            top_distance = np.linspace(0, self.xmax - self.xmin, n_points)
            left_distance = np.linspace(0, self.ymax - self.ymin, n_points)
            right_distance = np.linspace(0, self.ymax - self.ymin, n_points)
            
            ax2.plot(bottom_distance, bottom_data, color='r', linewidth=2, label = 'Bottom edge')
            ax2.plot(top_distance, top_data, color='b', linewidth=2, label = 'Top edge')
            ax2.plot(left_distance, left_data, color='g', linewidth=2, label = 'Left edge')
            ax2.plot(right_distance, right_data, color='m', linewidth=2, label = 'Right edge')
            ax2.set_xlabel(r'[$c/\omega_{pe}$]')
            ax2.set_ylabel(field)
            ax2.set_title(f'{field} - Lineout')
            ax2.grid(True)
            ax2.legend()
            
            plt.tight_layout()
            plt.savefig(f'{self.fig_dir}/{field}_2D.png', dpi=150)
            plt.close()

            output[field] = [bottom_data, top_data, left_data, right_data]

        return output
    
class FLASH_OSIRIS_1D(FLASH_OSIRIS_Base):
    
    def __init__(self,
                 start_point: List[float],
                 distance: float,
                 theta: float,
                 extension: float = 0.0,
                 **kwargs):
        # xmax/ymax are the FLASH endpoint (arc-length = distance), kept INSIDE the FLASH
        # domain so rqm/|B| sampling is unchanged. The optional extension buffer is added to
        # the total domain in the base class (self.distance), not to the endpoint.
        xmax = distance * np.cos(theta) + start_point[0] # Have it match the form of 2D setup
        ymax = distance * np.sin(theta) + start_point[1]
         # CFL condition
        super().__init__(osiris_dims=1, theta = theta, distance=distance, extension=extension,
                         xmax=xmax, ymax=ymax,
                         ymin=start_point[1], xmin=start_point[0], **kwargs)


        logger.info(str(self))
        
    def __str__(self):
        lines = [
            "=" * 50,
            "FLASH-OSIRIS INTERFACE",
            f"FLASH data: {self.FLASH_data}",
            f"Input file: {self.inputfile_name}",
            f"Reference density: {self.n0}",
            f"Species rqms: {self.species_rqms}",
            f"RQM normalization factor: {self.rqm_factor}",
            f"OSIRIS dimensions: 1D",
            f"Particles per cell: {self.ppc}",
            f"Start point: [{self.xmin}, {self.ymin}] [c/ωpe]",
            f"Ray angle: {self.theta} rad",
            f"FLASH lineout distance: {self.flash_distance} [c/ωpe]",
            f"Extension (held-constant buffer): {self.extension} [c/ωpe]",
            f"Total OSIRIS domain: {self.distance} [c/ωpe]",
            f"Output directory: {self.output_dir}",
            "=" * 50
        ]
        return "\n".join(lines)
    

class FLASH_OSIRIS_2D(FLASH_OSIRIS_Base):
    """2D box configuration."""
    
    osiris_dims = 2
    
    def __init__(self,
                 xmin: float,
                 xmax: float,
                 ymin: float,
                 ymax: float,
                 **kwargs):
        super().__init__(osiris_dims=2, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, **kwargs)
        logger.info("\n" + str(self))

    def __str__(self):
        lines = [
            "=" * 50,
            "FLASH-OSIRIS INTERFACE",
            f"FLASH data: {self.FLASH_data}",
            f"Input file: {self.inputfile_name}",
            f"Reference density: {self.n0:.2e} cm^-3",
            f"Species rqms: {self.species_rqms}",
            f"RQM normalization factor: {self.rqm_factor}",
            f"OSIRIS dimensions: 2D",
            f"Particles per cell: {self.ppc}",
            f"X range: {self.xmin} to {self.xmax} [c/ωpe]",
            f"Y range: {self.ymin} to {self.ymax} [c/ωpe]",
            f"Output directory: {self.output_dir}",
            "=" * 50
        ]
        return "\n".join(lines)
    

# Fields plotted as lineouts (1D) or edge profiles (2D) after the slices are saved.
# Built per run so the per-species density / thermal-velocity fields follow whatever
# ions are in the run.yaml.
def build_plot_fields(species_names):
    return (['edens']
            + [f'{n}dens' for n in species_names]
            + ['magx', 'magy', 'magz', 'Ex', 'Ey', 'Ez', 'vthele']
            + [f'vth{n}' for n in species_names]
            + ['v_ix', 'v_iy', 'v_ey'])

# Proton/electron mass ratio, used with the FLASH ionization fields to form the
# OSIRIS rqm = 1836/ye (mass-per-charge) and the ion mass m_i/m_e = 1836/sumy.
PROTON_ELECTRON_MASS_RATIO = 1836


def _str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("true", "t", "1", "yes"):
        return True
    if v.lower() in ("false", "f", "0", "no"):
        return False
    raise argparse.ArgumentTypeError("expected true/false")


# Top-level run.yaml keys that are dict-valued and must survive into the argparse
# namespace WHOLE (not flattened): species_names is the optional {flash_material:
# osiris_name} population rename; charge_states is analysis-only metadata.
_METADATA_KEYS = {"charge_states", "species_names"}


def load_run_config(path):
    """Flatten a run.yaml into a ``dest -> value`` dict for argparse defaults.

    Logical groups (``geometry``/``solver``/``diagnostics``) are one level of
    nesting purely for readability; their sub-keys map directly to argparse dests
    and are flattened here.  Metadata blocks in :data:`_METADATA_KEYS` (e.g.
    ``charge_states``) are passed through unflattened.
    """
    import yaml
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    flat = {}
    for k, v in cfg.items():
        if isinstance(v, dict) and k not in _METADATA_KEYS:
            flat.update(v)
        else:
            flat[k] = v
    return flat


def build_run_spec(args):
    """Build the canonical (nested, re-runnable) run.yaml dict from resolved args.

    The result is both the frozen provenance record written into the run dir and a
    valid ``--config`` input (see :func:`load_run_config`).  ``species_names`` (the
    optional population rename) and ``charge_states`` (analysis-only metadata) are
    carried through; the generator derives rqm/mass from FLASH and uses neither value.
    """
    spec = {
        "data_path": str(args.data_path),
        "dim": args.dim,
        "inputfile_name": args.inputfile_name,
        "reference_density": args.reference_density,
        "rqm_factor": args.rqm_factor,
        "dx": args.dx,
        "ppc": args.ppc,
        "tmax_gyroperiods": args.tmax_gyroperiods,
        "algorithm": args.algorithm,
        "node_number": args.node_number,
        "num_threads": args.num_threads,
        "n_dump_total": args.n_dump_total,
        "restart": args.restart,
    }
    species_names = getattr(args, "species_names", None)
    if species_names is not None:
        spec["species_names"] = species_names
    charge_states = getattr(args, "charge_states", None)
    if charge_states is not None:
        spec["charge_states"] = charge_states
    if args.dim == 1:
        spec["geometry"] = {"start_point": args.start_point, "end_point": args.end_point,
                            "extension": getattr(args, "extension", 0.0)}
    else:
        spec["geometry"] = {"xmin": args.xmin, "xmax": args.xmax,
                            "ymin": args.ymin, "ymax": args.ymax}
    spec["solver"] = {
        "vpml_bnd_size": args.vpml_bnd_size,
        "emf_boundary": args.emf_boundary,
        "part_boundary": args.part_boundary,
        "interpolation": args.interpolation,
        "smooth_type": args.smooth_type,
        "smooth_order": args.smooth_order,
    }
    spec["diagnostics"] = {
        "n_ave": args.n_ave,
        "emf_reports": args.emf_reports,
        "reports": args.reports,
        "rep_udist": args.rep_udist,
        "phasespaces": args.phasespaces,
        "e_ps_pmin": args.e_ps_pmin,
        "e_ps_pmax": args.e_ps_pmax,
        "i_ps_pmin": args.i_ps_pmin,
        "i_ps_pmax": args.i_ps_pmax,
        "ps_np": args.ps_np,
        "ps_ngamma": args.ps_ngamma,
        "ps_gammamax": args.ps_gammamax,
        "ps_nx": args.ps_nx,
        "ps_ny": args.ps_ny,
    }
    return spec


def freeze_run_yaml(out_dir, args):
    """Write the resolved run spec to ``<out_dir>/run.yaml`` (the analysis source of
    truth: reference_density, rqm_factor, dx, ppc, charge_states, ...)."""
    import yaml
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "run.yaml"
    with open(out_path, "w") as f:
        yaml.safe_dump(build_run_spec(args), f, default_flow_style=False, sort_keys=False)
    logger.info(f"Frozen run spec written to {out_path}")
    return out_path


def main(args):
    # The only checks argparse can't express (types/choices/required are handled
    # by the parser); dim-specific geometry is asserted in the branches below.
    assert Path(args.data_path).exists(), f"FLASH data not found: {args.data_path}"

    # Ion populations are auto-detected from the FLASH materials (target / chamber / ...)
    # by the yt plugin; species_names is an optional {flash_material: osiris_name} rename
    # from the run.yaml (e.g. {cham: al, targ: si} or {targ: piston, cham: background}).
    # Per-population rqm/mass are derived from FLASH (1836/ye, 1836/sumy) inside the class.
    species_names = getattr(args, "species_names", None)
    logger.info(f"Population rename (species_names): {species_names}")

    # All deck options come from the CLI -- nothing is defaulted here.
    deck_options = {
        "node_number": args.node_number,
        "num_threads": args.num_threads,
        "n_dump_total": args.n_dump_total,
        "restart": args.restart,
        "vpml_bnd_size": args.vpml_bnd_size,
        "emf_boundary": args.emf_boundary,
        "part_boundary": args.part_boundary,
        "interpolation": args.interpolation,
        "smooth_type": args.smooth_type,
        "smooth_order": args.smooth_order,
        "n_ave": args.n_ave,
        "emf_reports": args.emf_reports,
        "reports": args.reports,
        "rep_udist": args.rep_udist,
        "phasespaces": args.phasespaces,
        "e_ps_pmin": args.e_ps_pmin,
        "e_ps_pmax": args.e_ps_pmax,
        "i_ps_pmin": args.i_ps_pmin,
        "i_ps_pmax": args.i_ps_pmax,
        "ps_np": args.ps_np,
        "ps_ngamma": args.ps_ngamma,
        "ps_gammamax": args.ps_gammamax,
        "ps_nx": args.ps_nx,
        "ps_ny": args.ps_ny,
    }

    common = dict(
        path_to_FLASH_data=Path(args.data_path),
        OSIRIS_inputfile_name=args.inputfile_name,
        reference_density_cc=args.reference_density,
        ppc=args.ppc,
        dx=args.dx,
        rqm_normalization_factor=args.rqm_factor,
        tmax_gyroperiods=args.tmax_gyroperiods,
        algorithm=args.algorithm,
        deck_options=deck_options,
        species_names=species_names,
    )

    if args.dim == 1:
        assert args.start_point is not None and args.end_point is not None, \
            "--start_point and --end_point (cm, x y z) are required for --dim 1"
        # Convert the cm lineout endpoints (x, y of the z-midplane) to c/omega_pe.
        # c/omega_pe is the electron inertial length (plasmapy).
        import astropy.units as u
        from plasmapy.formulary.lengths import inertial_length
        cwpe = inertial_length(args.reference_density * u.cm**-3, 'e-').to('cm').value
        x0, y0 = args.start_point[0] / cwpe, args.start_point[1] / cwpe
        x1, y1 = args.end_point[0] / cwpe, args.end_point[1] / cwpe
        distance = float(np.hypot(x1 - x0, y1 - y0))
        theta = float(np.arctan2(y1 - y0, x1 - x0))
        # Optional extension buffer beyond the endpoint, given in cm (consistent with
        # start_point/end_point), converted to c/wpe. Fields are held constant there.
        extension = float(getattr(args, "extension", 0.0) or 0.0) / cwpe
        logger.info(f"c/omega_pe = {cwpe*1e4:.3f} um; lineout start=({x0:.1f},{y0:.1f}) "
                    f"c/wpe, distance={distance:.1f} c/wpe, theta={theta:.4f} rad, "
                    f"extension={extension:.1f} c/wpe")

        sim = FLASH_OSIRIS_1D(start_point=[x0, y0], distance=distance, theta=theta,
                              extension=extension, **common)
        plot = lambda: sim.plot1D(build_plot_fields(sim.species_names))
    else:
        for name in ("xmin", "xmax", "ymin", "ymax"):
            assert getattr(args, name) is not None, f"--{name} (c/wpe) is required for --dim 2"
        sim = FLASH_OSIRIS_2D(xmin=args.xmin, xmax=args.xmax,
                              ymin=args.ymin, ymax=args.ymax, **common)
        plot = lambda: sim.plot2D(build_plot_fields(sim.species_names))

    # Write the essential artifacts (slices, deck, py-script) and provenance FIRST,
    # so a failure in the optional diagnostic plots below never costs the run spec.
    sim.save_slices()
    sim.write_input_file()
    sim.write_python_file()
    # run.yaml is the analysis source of truth (params); the manifest holds only
    # provenance + derived quantities (git, omega_pe, gyrotime, ...).
    freeze_run_yaml(sim.output_dir, args)
    sim.write_manifest(cli_args=" ".join(sys.argv))

    # Diagnostics last (lineouts + FLASH-vs-OSIRIS-init conservation overlays).
    plot()
    sim.plot_conservation_diagnostics()


def parse_args(argv=None):
    """Build the CLI, resolve an optional ``--config`` run.yaml, and return args.

    Precedence: a ``--config`` run.yaml supplies values for every argument, and
    explicit CLI flags override it. ``required`` is relaxed only for keys the config
    actually provides, so anything left unspecified still errors (no hidden defaults).
    """
    _pre = argparse.ArgumentParser(add_help=False)
    _pre.add_argument('--config', type=str, default=None)
    _pre_args, _ = _pre.parse_known_args(argv)
    _config_defaults = load_run_config(_pre_args.config) if _pre_args.config else {}

    p = argparse.ArgumentParser(
        description="Generate a python-init OSIRIS deck + py-script + interp slices + "
                    "lineout plots from a FLASH dump. Provide either a --config run.yaml "
                    "(the single source of truth) or every flag explicitly (no hidden "
                    "defaults); CLI flags override the config.")

    p.add_argument('--config', type=str, default=None,
                   help="YAML run spec providing values for all arguments (CLI overrides)")

    # --- physics / core (all required) ---
    p.add_argument('--data_path', type=str, required=True, help="Path to the FLASH dump")
    p.add_argument('--dim', type=int, choices=[1, 2], required=True, help="OSIRIS dimensionality")
    p.add_argument('--inputfile_name', type=str, required=True,
                   help="OSIRIS input file / output folder name (a '.<dim>d' suffix is appended)")
    p.add_argument('--reference_density', type=float, required=True, help="Normalization density [cm^-3]")
    p.add_argument('--rqm_factor', type=float, required=True, help="rqm reduction factor")
    p.add_argument('--dx', type=float, required=True, help="Cell size [c/omega_pe]")
    p.add_argument('--ppc', type=int, required=True, help="Particles per cell (per dimension)")
    p.add_argument('--tmax_gyroperiods', type=float, required=True, help="Run length [ion gyroperiods]")
    p.add_argument('--algorithm', type=str, choices=["cpu", "cuda", "tiles"], required=True)
    # Ion populations are auto-detected from the FLASH materials; the optional
    # {flash_material: osiris_name} rename lives in the run.yaml `species_names` key
    # (dict-valued, so it is config-only -- not a CLI flag).

    # --- geometry: 1D lineout (cm) OR 2D box (c/omega_pe) ---
    p.add_argument('--start_point', type=float, nargs=3, help="[1D] lineout start (x y z) [cm]")
    p.add_argument('--end_point', type=float, nargs=3, help="[1D] lineout end (x y z) [cm]")
    p.add_argument('--extension', type=float, default=0.0,
                   help="[1D] extra distance beyond end_point; fields held constant at the endpoint [cm]")
    p.add_argument('--xmin', type=float, help="[2D] box xmin [c/omega_pe]")
    p.add_argument('--xmax', type=float, help="[2D] box xmax [c/omega_pe]")
    p.add_argument('--ymin', type=float, help="[2D] box ymin [c/omega_pe]")
    p.add_argument('--ymax', type=float, help="[2D] box ymax [c/omega_pe]")

    # --- parallel / time / dumps (all required) ---
    p.add_argument('--node_number', type=int, nargs='+', required=True, help="Nodes (1 val 1D, 2 vals 2D)")
    p.add_argument('--num_threads', type=int, required=True, help="OpenMP threads per node")
    p.add_argument('--n_dump_total', type=int, required=True, help="Total dumps over the run (sets ndump)")
    p.add_argument('--restart', type=_str2bool, required=True, help="if_restart (true/false)")

    # --- field solver / boundaries / smoothing (all required) ---
    p.add_argument('--vpml_bnd_size', type=int, required=True)
    p.add_argument('--emf_boundary', type=str, nargs=2, required=True, help="EMF boundary (lower upper)")
    p.add_argument('--part_boundary', type=str, nargs=2, required=True, help="Particle boundary (lower upper)")
    p.add_argument('--interpolation', type=str, choices=["linear", "quadratic", "cubic", "quartic"], required=True)
    p.add_argument('--smooth_type', type=str, required=True)
    p.add_argument('--smooth_order', type=int, required=True)

    # --- diagnostics (all required) ---
    p.add_argument('--n_ave', type=int, required=True, help="Cells averaged per dim in diagnostics")
    p.add_argument('--emf_reports', type=str, nargs='+', required=True)
    p.add_argument('--reports', type=str, nargs='+', required=True, help="Per-species reports")
    p.add_argument('--rep_udist', type=str, nargs='+', required=True)
    p.add_argument('--phasespaces', type=str, nargs='+', required=True)
    p.add_argument('--e_ps_pmin', type=float, nargs=3, required=True)
    p.add_argument('--e_ps_pmax', type=float, nargs=3, required=True)
    p.add_argument('--i_ps_pmin', type=float, nargs=3, required=True)
    p.add_argument('--i_ps_pmax', type=float, nargs=3, required=True)
    p.add_argument('--ps_np', type=int, nargs=3, required=True, help="Momentum bins (px py pz)")
    p.add_argument('--ps_ngamma', type=int, required=True)
    p.add_argument('--ps_gammamax', type=float, required=True)
    p.add_argument('--ps_nx', type=int, required=True, help="Spatial bins along x1 for phase space")
    p.add_argument('--ps_ny', type=int, required=True, help="Spatial bins along x2 (2D only, still required)")

    # Apply the run.yaml as defaults: relax `required` for anything it supplies
    # (CLI still overrides), and inject metadata (e.g. charge_states) into the
    # namespace. Anything neither in the config nor on the CLI still errors out.
    if _config_defaults:
        for action in p._actions:
            if action.dest in _config_defaults:
                action.required = False
        p.set_defaults(**_config_defaults)

    return p.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    logger.info(args)
    main(args)
