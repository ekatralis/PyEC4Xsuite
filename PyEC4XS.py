import xtrack as xt
import xfields as xf
import xobjects as xo
from typing import Literal
from PyECLOUD import buildup_simulation as bsim
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c
# from . import myfilemanager as mfm
import myfilemanager as mfm # Change for testing

class XsuiteUniformBinSlicer:

    def __init__(self, particles: xt.Particles, n_slices: int = 10, 
                 mode: Literal["percentile", "minmax"] = "minmax", percentile: float = 0.001,
                 iter_mode: Literal["LeftToRight", "RightToLeft"] = "RightToLeft"):
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
        self.percentile = percentile
        self.z_min = None
        self.z_max = None
        self.bins = None
        self.particles = particles
        self._set_bins()

    def _get_beam_edges(self):
        if self.mode == "percentile":
            self.z_min = np.percentile(self.particles.zeta, self.percentile)
            self.z_max = np.percentile(self.particles.zeta, 100-self.percentile)
        elif self.mode == "minmax":
            self.z_min = np.min(self.particles.zeta)
            self.z_max = np.max(self.particles.zeta)
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
        elif slice_num == self.n_slices:
            cond = inside | (self.particles.zeta > self.bins[slice_num+1])
        else:
            cond = inside
        particles_idx = self.particles.particle_id[np.where(cond)[0]]
        return particles_idx

    def _get_slice(self, slice_num: int) -> dict:
        idx_particle_slice = self._particles_in_slice(slice_num)
        beta = np.mean(self.particles.beta0[idx_particle_slice])
        gamma = 1/(np.sqrt(1-beta**2))
        dt = self.dz / (beta * c)
        z_mean = np.mean(self.particles.zeta[idx_particle_slice])
        slice_dict = {
            "particle_idx" : idx_particle_slice,
            "beta"         : beta,
            "zeta"         : z_mean,
            "gamma"        : gamma,
            "dt"           : dt, 
            "slice_info"   : f"{slice_num+1}/{self.n_slices}"
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
                 slicerKwargs,
                 Dt_ref,
                 pyecl_input_folder="./",
                 flag_clean_slices = False,
                 space_charge_obj = None,
                 verbose = False,
                 save_pyecl_outp_as = None,
                 enable_diagnostics = False,
                 **kwargs
                ):
        self.slicer = slicer
        self.slicerKwargs = slicerKwargs
        self.Dt_ref = Dt_ref
        self.L_ecloud = L_ecloud
        self.verbose = verbose
        self.flag_clean_slices = flag_clean_slices
        self.kwargs = kwargs
        self.enable_diagnostics = enable_diagnostics

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
        
        self.initial_MP_e_clouds = [
            cl.MP_e.extract_dict() for cl in self.cloudsim.cloud_list
        ]

        self.beam_PyPIC_state = self.cloudsim.spacech_ele.PyPICobj.get_state_object()
        
        if self.enable_diagnostics:
            self._diagnostics_init()

        self.track_only_first_time = False

        self.N_tracks = 0
        self.i_reinit = 0
        self.t_sim = 0.0
        self.i_curr_bunch = -1

    def track(self, particles: xt.Particles):
        self._reinitialize()
        # particles.x += 10
        slices = self.slicer(**self.slicerKwargs,particles = particles)
        for slice in slices:
            self._track_single_slice(particles=particles,slice=slice)
        self._finalize()
        pass

    def _reinitialize(self):
        cc = mfm.obj_from_dict(self.cloudsim.config_dict)

        for thiscloud, initdict in zip(
            self.cloudsim.cloud_list, self.initial_MP_e_clouds
        ):
            thiscloud.MP_e.init_from_dict(initdict)
            thiscloud.MP_e.set_nel_mp_ref(thiscloud.MP_e.nel_mp_ref_0)
            thisconf = thiscloud.config_dict

            if thiscloud.pyeclsaver is not None:
                thiscloud.pyeclsaver.extract_sey = False
                thiscloud.pyeclsaver.start_observing(
                    cc.Dt,
                    thiscloud.MP_e,
                    None,
                    thiscloud.impact_man,
                    thisconf.r_center,
                    thisconf.Dt_En_hist,
                    thisconf.logfile_path,
                    thisconf.progress_path,
                    flag_detailed_MP_info=thisconf.flag_detailed_MP_info,
                    flag_movie=thisconf.flag_movie,
                    flag_sc_movie=thisconf.flag_sc_movie,
                    save_mp_state_time_file=thisconf.save_mp_state_time_file,
                    flag_presence_sec_beams=self.cloudsim.flag_presence_sec_beams,
                    sec_beams_list=self.cloudsim.sec_beams_list,
                    dec_fac_secbeam_prof=thisconf.dec_fac_secbeam_prof,
                    el_density_probes=thisconf.el_density_probes,
                    save_simulation_state_time_file=thisconf.save_simulation_state_time_file,
                    x_min_hist_det=thisconf.x_min_hist_det,
                    x_max_hist_det=thisconf.x_max_hist_det,
                    y_min_hist_det=thisconf.y_min_hist_det,
                    y_max_hist_det=thisconf.y_max_hist_det,
                    Dx_hist_det=thisconf.Dx_hist_det,
                    dec_fact_out=cc.dec_fact_out,
                    stopfile=cc.stopfile,
                    filen_main_outp=thisconf.filen_main_outp,
                    flag_cos_angle_hist=thisconf.flag_cos_angle_hist,
                    cos_angle_width=thisconf.cos_angle_width,
                    flag_multiple_clouds=(len(self.cloudsim.cloud_list) > 1),
                    cloud_name=thisconf.cloud_name,
                    flag_last_cloud=(thiscloud is self.cloudsim.cloud_list[-1]),
                )
                thiscloud.pyeclsaver.filen_main_outp = (
                        thiscloud.pyeclsaver.filen_main_outp.split(".mat")[0].split(
                            "__iter"
                        )[0]
                        + "__iter%d.mat" % self.i_reinit
                    )
        if self.enable_diagnostics:
            self._diagnostics_reinit()
        self.i_curr_bunch = -1
        self.i_reinit += 1
    
    def _finalize(self):
        if self.enable_diagnostics:
            self._diagnostics_finalize()
    
    def _track_single_slice(self, particles: xt.Particles, slice: dict):
        return

    def _diagnostics_init(self):
        self.save_ele_field_probes = False
        self.x_probes = -1
        self.y_probes = -1
        self.Ex_ele_last_track_at_probes = -1
        self.Ey_ele_last_track_at_probes = -1
        if "probes_position" in list(self.kwargs.keys()):
            self.save_ele_field_probes = True
            self.probes_position = self.kwargs["probes_position"]
            self.N_probes = len(self.probes_position)
            self.x_probes = []
            self.y_probes = []
            for ii_probe in range(self.N_probes):
                self.x_probes.append(self.probes_position[ii_probe]["x"])
                self.y_probes.append(self.probes_position[ii_probe]["y"])

            self.x_probes = np.array(self.x_probes)
            self.y_probes = np.array(self.y_probes)
        self.save_ele_distributions_last_track = False
        self.save_ele_potential_and_field = False
        self.save_ele_potential = False
        self.save_ele_field = False
        self.save_ele_MP_position = False
        self.save_ele_MP_velocity = False
        self.save_ele_MP_size = False

        self.save_beam_distributions_last_track = False
        self.save_beam_potential_and_field = False
        self.save_beam_potential = False
        self.save_beam_field = False
    
    def _diagnostics_reinit(self):
        
        if self.save_ele_distributions_last_track:
            self.rho_ele_last_track = []

        if self.save_ele_potential_and_field:
            self.save_ele_potential = True
            self.save_ele_field = True

        if self.save_ele_potential:
            self.phi_ele_last_track = []

        if self.save_ele_field:
            self.Ex_ele_last_track = []
            self.Ey_ele_last_track = []

        if self.save_beam_distributions_last_track:
            self.rho_beam_last_track = []

        if self.save_beam_potential_and_field:
            self.save_beam_potential = True
            self.save_beam_field = True

        if self.save_beam_potential:
            self.phi_beam_last_track = []

        if self.save_beam_field:
            self.Ex_beam_last_track = []
            self.Ey_beam_last_track = []

        if self.save_ele_MP_position:
            self.x_MP_last_track = []
            self.y_MP_last_track = []

        if self.save_ele_MP_velocity:
            self.vx_MP_last_track = []
            self.vy_MP_last_track = []

        if self.save_ele_MP_size:
            self.nel_MP_last_track = []

        if (
            self.save_ele_MP_position
            or self.save_ele_MP_velocity
            or self.save_ele_MP_size
        ):
            self.N_MP_last_track = []

        if self.save_ele_field_probes:
            self.Ex_ele_last_track_at_probes = []
            self.Ey_ele_last_track_at_probes = []
    
    def _diagnostics_finalize(self):
        if self.save_ele_distributions_last_track:
            self.rho_ele_last_track = np.array(self.rho_ele_last_track[::-1])

        if self.save_ele_potential:
            self.phi_ele_last_track = np.array(self.phi_ele_last_track[::-1])

        if self.save_ele_field:
            self.Ex_ele_last_track = np.array(self.Ex_ele_last_track[::-1])
            self.Ey_ele_last_track = np.array(self.Ey_ele_last_track[::-1])

        if self.save_beam_distributions_last_track:
            self.rho_beam_last_track = np.array(self.rho_beam_last_track[::-1])

        if self.save_beam_potential:
            self.phi_beam_last_track = np.array(self.phi_beam_last_track[::-1])

        if self.save_beam_field:
            self.Ex_beam_last_track = np.array(self.Ex_beam_last_track[::-1])
            self.Ey_beam_last_track = np.array(self.Ey_beam_last_track[::-1])

        if self.save_ele_MP_position:
            self.x_MP_last_track = np.array(self.x_MP_last_track[::-1])
            self.y_MP_last_track = np.array(self.y_MP_last_track[::-1])

        if self.save_ele_MP_velocity:
            self.vx_MP_last_track = np.array(self.vx_MP_last_track[::-1])
            self.vy_MP_last_track = np.array(self.vy_MP_last_track[::-1])

        if self.save_ele_MP_size:
            self.nel_MP_last_track = np.array(self.nel_MP_last_track[::-1])

        if (
            self.save_ele_MP_position
            or self.save_ele_MP_velocity
            or self.save_ele_MP_size
        ):
            self.N_MP_last_track = np.array(self.N_MP_last_track[::-1])

        if self.save_ele_field_probes:
            self.Ex_ele_last_track_at_probes = np.array(
                self.Ex_ele_last_track_at_probes[::-1]
            )
            self.Ey_ele_last_track_at_probes = np.array(
                self.Ey_ele_last_track_at_probes[::-1]
            )
