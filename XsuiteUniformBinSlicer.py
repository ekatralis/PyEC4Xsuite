import xtrack as xt
from typing import Literal
from scipy.constants import c
import numpy as np

class XsuiteUniformBinSlicer:

    def __init__(self, particles: xt.Particles, n_slices: int = 10, 
                 mode: Literal["percentile", "minmax"] = "minmax", percentile_pct: float = 1e-4,
                 iter_mode: Literal["LeftToRight", "RightToLeft"] = "LeftToRight"):
        self.n_slices = n_slices
        self.n_bins = self.n_slices + 1
        slice_list = list(range(n_slices))
        if iter_mode == "LeftToRight":
            pass
        elif iter_mode == "RightToLeft":
            slice_list.reverse()
        else:
            raise ValueError("Invalid iter_mode")
        self.slice_list = slice_list
        self.mode = mode
        self.percentile = percentile_pct
        self.z_min = None
        self.z_max = None
        self.bins = None
        self.particles = particles
        self._set_active_particle_idx()
        self._set_bins()
        self.beta_avg = np.nan_to_num(np.mean(self.particles.beta0[self.active_particles]))
        self.gamma_avg = np.nan_to_num(np.mean(self.particles.gamma0[self.active_particles]))
    
    def _set_active_particle_idx(self):
        self.active_particles = np.where(self.particles.state>0)[0]
        self.active_particles_set = set(self.active_particles)

    def _get_beam_edges(self):
        if self.mode == "percentile":
            self.z_min = np.percentile(self.particles.zeta[self.active_particles], self.percentile)
            self.z_max = np.percentile(self.particles.zeta[self.active_particles], 100-self.percentile)
        elif self.mode == "minmax":
            self.z_min = np.min(self.particles.zeta[self.active_particles])
            self.z_max = np.max(self.particles.zeta[self.active_particles])
        else:
            raise Exception(f"{self.mode} mode not supported")

    def _set_bins(self) -> np.ndarray:
        self._get_beam_edges()
        self.bins = np.linspace(self.z_min,self.z_max,self.n_bins)
        self.dz = self.bins[1] - self.bins[0]
        return self.bins

    def _particles_in_slice(self, slice_num: int) -> np.ndarray:
        cond = None
        if slice_num > self.n_slices-1:
            raise Exception(f"Slice {slice_num} is out of bounds for number of slices: {self.n_slices}. Slices are 0 indexed")
        inside = (self.bins[slice_num] < self.particles.zeta) & (self.particles.zeta <= self.bins[slice_num+1])
        if slice_num == 0:
            cond = inside | (self.particles.zeta <= self.bins[slice_num]) 
        elif slice_num == self.n_slices-1:
            cond = inside | (self.particles.zeta > self.bins[slice_num+1])
        else:
            cond = inside
        particles_ids_in_bin = set(np.where(cond)[0])
        particles_idx = list(self.active_particles_set.intersection(particles_ids_in_bin))
        return particles_idx

    def _get_slice(self, slice_num: int) -> dict:
        idx_particle_slice = self._particles_in_slice(slice_num)
        active_particles = len(idx_particle_slice)
        if active_particles > 0:
            beta = np.mean(self.particles.beta0[idx_particle_slice])
            gamma = 1/(np.sqrt(1-beta**2))
        else:
            beta = self.beta_avg
            gamma = self.gamma_avg
        dt = self.dz / (beta * c)
        z_center = (self.bins[slice_num+1] + self.bins[slice_num]) / 2
        slice_dict = {
            "num_active"   : active_particles,
            "particle_idx" : idx_particle_slice,
            "beta"         : beta,
            "zeta"         : z_center,
            "gamma"        : gamma,
            "dz"           : self.dz,
            "dt"           : dt, 
            "slice_num"   : f"{slice_num+1}/{self.n_slices}"
        }
        return slice_dict
    
    def __getitem__(self,key):
        if key < 0 or key>self.n_slices-1:
            raise Exception(f"Slice {key} is out of bounds for number of slices: {self.n_slices}. Slices are 0 indexed")
        return self._get_slice(slice_num = self.slice_list[key])
    
    def __next__(self):
        self.idx += 1
        if self.idx < self.n_slices:
            return self.__getitem__(self.idx)
        else:
            raise StopIteration

    def __iter__(self):
        self.idx = -1
        return self