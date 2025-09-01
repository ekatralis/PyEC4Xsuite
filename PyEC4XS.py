import xtrack as xt
import xfields as xf
import xobjects as xo
from typing import Literal
from PyECLOUD import buildup_simulation as bsim
import numpy as np
import matplotlib.pyplot as plt
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

    def set_bins(self, particles: xt.Particles) -> np.ndarray:
        self._get_beam_edges(particles)
        self.bins = np.linspace(self.z_min,self.z_max,self.n_bins)
        return self.bins

    def particles_in_bin(self, bin_num: int, particles: xt.Particles = None) -> np.ndarray:
        if self.bins is None:
            self.set_bins(particles)
        cond = None
        if particles is None:
            particles = self.particles
        if particles is None:
            raise Exception(f"Please provide particles to be binned")
        if self.bins is None:
            raise Exception(f"Please use the set_bins function or reinitialize with particles")
        if bin_num > self.n_slices-1:
            raise Exception(f"Bin {bin_num} is out of bounds for number of slices: {self.n_slices}. Slices are 0 indexed")
        inside = (self.bins[bin_num] < particles.zeta) & (particles.zeta <= self.bins[bin_num+1])
        if bin_num == 0:
            cond = inside | (particles.zeta <= self.bins[bin_num]) 
        elif bin_num == self.n_slices:
            cond = inside | (particles.zeta > self.bins[bin_num+1])
        else:
            cond = inside
        particles_idx = np.where(cond)[0]
        return particles_idx

class xEcloud:
    needs_cpu = True
    def __init__(self):
        pass
    
    def track(self, particles):
        pass

