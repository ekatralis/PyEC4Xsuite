import xtrack as xt
import xfields as xf
import xobjects as xo
from typing import Literal
from PyECLOUD import buildup_simulation as bsim
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c

class XsuiteUniformBinSlicer:

    def __init__(self, n_slices: int, mode: Literal["percentile", "minmax"] = "minmax",
                 percentile: float = 0.001, particles: xt.Particles = None):
        self.n_slices = n_slices
        self.n_bins = self.n_slices + 1
        self.mode = mode
        self.percentile = percentile
        self.z_min = None
        self.z_max = None
        self.bins = None
        if particles is not None:
            self.set_bins(particles)
            self.particles = particles
        else:
            self.particles = None

    def _get_beam_edges(self, particles: xt.Particles):
        if self.mode == "percentile":
            self.z_min = np.percentile(particles.zeta, self.percentile)
            self.z_max = np.percentile(particles.zeta, 100-self.percentile)
        elif self.mode == "minmax":
            self.z_min = np.min(particles.zeta)
            self.z_max = np.max(particles.zeta)
        else:
            raise Exception(f"{self.mode} mode not supported")
        
    def _validate_input(self,particles) -> xt.Particles:
        if particles is None:
            particles = self.particles
        if particles is None:
            raise Exception(f"Please provide particles to be binned")
        if self.bins is None:
            raise Exception(f"Please use the set_bins function or reinitialize with particles")
        return particles

    def set_bins(self, particles: xt.Particles) -> np.ndarray:
        self._get_beam_edges(particles)
        self.bins = np.linspace(self.z_min,self.z_max,self.n_bins)
        self.dz = self.bins[1] - self.bins[0]
        return self.bins

    def particles_in_slice(self, slice_num: int, particles: xt.Particles = None) -> np.ndarray:
        if self.bins is None:
            self.set_bins(particles)
        cond = None
        particles = self._validate_input(particles)
        if slice_num > self.n_slices-1:
            raise Exception(f"Slice {slice_num} is out of bounds for number of slices: {self.n_slices}. Slices are 0 indexed")
        inside = (self.bins[slice_num] < particles.zeta) & (particles.zeta <= self.bins[slice_num+1])
        if slice_num == 0:
            cond = inside | (particles.zeta <= self.bins[slice_num]) 
        elif slice_num == self.n_slices:
            cond = inside | (particles.zeta > self.bins[slice_num+1])
        else:
            cond = inside
        particles_idx = particles.particle_id[np.where(cond)[0]]
        return particles_idx

    def get_slice(self, slice_num: int, particles: xt.Particles = None) -> dict:
        particles = self._validate_input(particles)
        idx_particle_slice = self.particles_in_slice(slice_num, particles = particles)
        beta = np.mean(particles.beta0[idx_particle_slice])
        gamma = 1/(np.sqrt(1-beta**2))
        dt = self.dz / (beta * c)
        slice_dict = {
            "particle_idx" : idx_particle_slice,
            "beta"         : beta,
            "gamma"        : gamma,
            "dt"           : dt
        }
        return slice_dict
        
extra_allowed_kwargs = {
    "x_beam_offset",
    "y_beam_offset",
    "probes_position",
    "enable_kick_x",
    "enable_kick_y",
}

class xEcloud:
    needs_cpu = True
    def __init__(self,
                 L_ecloud,
                 slicer,
                 Dt_ref,
                 pyecl_input_folder="./",
                 flag_clean_slices = False,
                 space_charge_obj = None,
                 verbose = False,
                 save_pyecl_outp_as = None,
                 **kwargs
                ):
        self.slicer = slicer
        self.Dt_ref = Dt_ref
        self.L_ecloud = L_ecloud
        self.verbose = verbose
        self.flag_clean_slices = flag_clean_slices

        # Initialize E-Cloud
        self.cloudsim = bsim.BuildupSimulation(
            pyecl_input_folder=pyecl_input_folder,
            skip_beam=True,
            spacech_ele=space_charge_obj,
            ignore_kwargs=extra_allowed_kwargs,
            skip_pyeclsaver=(save_pyecl_outp_as is None),
            filen_main_outp=save_pyecl_outp_as,
            **self.kwargs
        )
        
        if self.cloudsim.config_dict["track_method"] == "Boris":
            pass
        elif self.cloudsim.config_dict["track_method"] == "BorisMultipole":
            pass
        else:
            raise ValueError(
                """track_method should be 'Boris' or 'BorisMultipole' - others are not implemented in the PyEC4XS module"""
            )
        
        self.x_beam_offset = 0.0
        self.y_beam_offset = 0.0
        if "x_beam_offset" in kwargs:
            self.x_beam_offset = kwargs["x_beam_offset"]
        if "y_beam_offset" in kwargs:
            self.y_beam_offset = kwargs["y_beam_offset"]

        self.enable_kick_x = True
        self.enable_kick_y = True
        if "enable_kick_x" in kwargs:
            self.enable_kick_x = kwargs["enable_kick_x"]
        if not self.enable_kick_x:
            print("Horizontal kick on the beam is disabled!")
        if "enable_kick_y" in kwargs:
            self.enable_kick_y = kwargs["enable_kick_y"]
        if not self.enable_kick_y:
            print("Vertical kick on the beam is disabled!")
        self.i_reinit = 0
        self.t_sim = 0.0
        self.i_curr_bunch = -1

    def track(self, particles):
        # particles.x += 10
        pass


