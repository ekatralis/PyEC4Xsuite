import xtrack as xt
import xfields as xf
import xobjects as xo
from typing import Literal
from PyECLOUD import buildup_simulation as bsim
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c
import time
# from . import myfilemanager as mfm
import myfilemanager as mfm # Change for testing
from scipy.constants import e as qe

class IterableSlices:

    def __init__(self, particles: xt.Particles, slicer: xf.TempSlicer,
                 iter_mode = "Default"):
        assert iter_mode in ["Default", "Reverse"]
        self.slicer = slicer
        self.n_slices = slicer.num_slices
        self.slice_particle_list = []
        slicer.assign_slices(particles=particles)
        for num_slice in range(self.n_slices):
            self.slice_particle_list.append(np.where(particles.slice==num_slice)[0])
        self.n_bins = self.n_slices + 1
        self.slice_list = list(range(self.n_slices))
        if iter_mode == "Default":
            pass
        elif iter_mode == "Reverse":
            self.slice_list.reverse()
        else:
            raise ValueError("Invalid iter_mode")
        assert (particles.beta0 == particles.beta0[0]).all()
        assert (particles.gamma0 == particles.gamma0[0]).all()
        self.beta = particles.beta0[0]
        self.gamma = particles.gamma0[0]

    def _get_slice(self, slice_num: int) -> dict:
        idx_particle_slice = self.slice_particle_list[slice_num]
        active_particles = len(idx_particle_slice)
        dz = self.slicer.bin_widths_beamstrahlung[slice_num]
        dt = dz / (self.beta * c)
        z_center = self.slicer.bin_centers[slice_num]
        slice_dict = {
            "num_active"   : active_particles,
            "particle_idx" : idx_particle_slice,
            "beta"         : self.beta,
            "zeta"         : z_center,
            "gamma"        : self.gamma,
            "dz"           : dz,
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
        
extra_allowed_kwargs = {
    "x_beam_offset",
    "y_beam_offset",
    "probes_position",
    "enable_kick_x",
    "enable_kick_y",
}

class DummyBeamTim(object):
    """Dummy beam-timing class to interface with buildup simulation"""

    def __init__(self, PyPIC_state):
        self.PyPIC_state = PyPIC_state

        self.b_spac = 0.0
        self.pass_numb = 0
        self.N_pass_tot = 1

    def get_beam_eletric_field(self, MP_e):
        if MP_e.N_mp > 0:
            if self.PyPIC_state is None:
                Ex_n_beam = 0.0 * MP_e.x_mp[0 : MP_e.N_mp]
                Ey_n_beam = 0.0 * MP_e.y_mp[0 : MP_e.N_mp]
            else:
                # compute beam electric field
                Ex_n_beam, Ey_n_beam = self.PyPIC_state.gather(
                    MP_e.x_mp[0 : MP_e.N_mp], MP_e.y_mp[0 : MP_e.N_mp]
                )
        else:
            Ex_n_beam = 0.0
            Ey_n_beam = 0.0
        return Ex_n_beam, Ey_n_beam

class Empty():
    pass

class xEcloud:
    needs_cpu = True
    iscollective = True
    def __init__(self,
                 L_ecloud,
                 slicer,
                 Dt_ref,
                 pyecl_input_folder="./",
                 flag_clean_slices = False,
                 space_charge_obj = None,
                 verbose = False,
                 save_pyecl_outp_as = None,
                 enable_diagnostics = False,
                 kick_mode_for_beam_field=False,
                 force_interp_at_substeps_interacting_slices=False,
                 **kwargs
                ):
        self.slicer = slicer
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
            self._print("Horizontal kick on the beam is disabled!")
        if "enable_kick_y" in kwargs:
            self.enable_kick_y = kwargs["enable_kick_y"]
        if not self.enable_kick_y:
            self._print("Vertical kick on the beam is disabled!")
        
        self.initial_MP_e_clouds = [
            cl.MP_e.extract_dict() for cl in self.cloudsim.cloud_list
        ]

        self.beam_PyPIC_state = self.cloudsim.spacech_ele.PyPICobj.get_state_object()
        
        if self.enable_diagnostics:
            self._print("Diagnostics enabled")
            self._diagnostics_init()
        else:
            self._print("Diagnostics disabled. Use enable_diagnostics flag to enable")
        self.slices_iter_mode = "Default"
        self._print(f"Iterable Mode for slices is set to '{self.slices_iter_mode}'")
        self.track_only_first_time = False
        self.kick_mode_for_beam_field = kick_mode_for_beam_field
        self.force_interp_at_substeps_interacting_slices = force_interp_at_substeps_interacting_slices

        self.N_tracks = 0
        self.i_reinit = 0
        self.t_sim = 0.0
        self.i_curr_bunch = -1

    def track(self, particles: xt.Particles):
        if self.track_only_first_time:
            if self.N_tracks > 0:
                self._print("Warning: Track skipped because track_only_first_time is True.")
                return
            
        # Initial assumptions to get module up and running.
        # These assumptions will change, but require updates to PyECLOUD.
        # These updates should follow shortly after these changes are confirmed
        if np.mean(particles.charge) != particles.q0:
            raise AssertionError("Module asssumes that all particles have the same charge")

        if self.verbose:
            start_time = time.time()

        self._reinitialize()
        slices = IterableSlices(particles=particles,slicer=self.slicer,iter_mode=self.slices_iter_mode)
        force_newpass = True
        for slice in slices:
            self._track_single_slice(particles=particles,slice=slice,force_pyecl_newpass=force_newpass)
            if force_newpass:
                force_newpass = False

        self._finalize()

        if self.verbose:
            stop_time = time.time()
            self._print("Done track %d in %.3f s" % (self.N_tracks, stop_time - start_time))

        self.N_tracks += 1

    def _track_single_slice(self, particles: xt.Particles, slice: dict, force_pyecl_newpass: bool = False):
        spacech_ele = self.cloudsim.spacech_ele
        # Check if the slice interacts with the beam
        if slice["num_active"] > 0:
            if "slice_info" in list(slice.keys()):
                if "interact_with_EC" in list(slice["slice_info"].keys()):
                    interact_with_EC = slice["slice_info"]["interact_with_EC"]
                else:
                    interact_with_EC = True
            else:
                interact_with_EC = True
        else:
            interact_with_EC = False
        
        dt_slice = slice["dt"]
        ix = slice["particle_idx"]
        dz = slice["dz"]
        charge = particles.q0 * qe

        # Check if sub-slicing is needed
        if self.cloudsim.config_dict["Dt"] is not None:
            if dt_slice > self.cloudsim.config_dict["Dt"]:
                if interact_with_EC:
                    raise ValueError(
                        "Slices that interact with the cloud cannot be longer than the buildup timestep!"
                    )

                N_cloud_steps = np.int_(
                    np.ceil(dt_slice / self.cloudsim.config_dict["Dt"])
                )
                dt_cloud_step = dt_slice / N_cloud_steps
                dt_array = np.array(N_cloud_steps * [dt_cloud_step])
            else:
                dt_array = np.array([dt_slice])
        else:
            dt_array = np.array([dt_slice])
        
        # Acquire bunch passage information
        if "slice_info" in list(slice.keys()):
            if "info_parent_bunch" in list(slice["slice_info"].keys()):

                # check if first slice of first bunch
                if (
                    slice["slice_info"]["info_parent_bunch"]["i_bunch"] == 0
                    and slice["slice_info"]["i_slice"] == 0
                ):
                    self.finalize_and_reinitialize()

                # check if new passage
                if slice["slice_info"]["info_parent_bunch"]["i_bunch"] > self.i_curr_bunch:
                    self.i_curr_bunch = slice["slice_info"]["info_parent_bunch"]["i_bunch"]
                    new_pass = True
                else:
                    new_pass = force_pyecl_newpass

            else:
                new_pass = force_pyecl_newpass
                self.i_curr_bunch = 0
        else:
            new_pass = force_pyecl_newpass
            self.i_curr_bunch = 0

        # Cloud simulation
        for i_clou_step, dt in enumerate(dt_array):
            # define substep
            if dt > self.Dt_ref:
                N_sub_steps = int(np.round(dt / self.Dt_ref))
            else:
                N_sub_steps = 1

            Dt_substep = dt / N_sub_steps
            # print Dt_substep, N_sub_steps, dt

            if len(ix) == 0 or not interact_with_EC:  # no particles in the beam

                # build dummy beamtim object
                dummybeamtim = DummyBeamTim(None)

                dummybeamtim.lam_t_curr = 0.0
                dummybeamtim.sigmax = 0.0
                dummybeamtim.sigmay = 0.0
                dummybeamtim.x_beam_pos = 0.0
                dummybeamtim.y_beam_pos = 0.0
            else:
                # beam field
                self.beam_PyPIC_state.scatter(
                    x_mp=particles.x[ix] + self.x_beam_offset,
                    y_mp=particles.y[ix] + self.y_beam_offset,
                    nel_mp=particles.weight[ix] / dz,
                    charge=charge,
                )
                self.cloudsim.spacech_ele.PyPICobj.solve_states([self.beam_PyPIC_state])

                # build dummy beamtim object
                dummybeamtim = DummyBeamTim(self.beam_PyPIC_state)

                dummybeamtim.lam_t_curr = np.mean(
                    particles.weight / dz
                ) * len(ix)
                dummybeamtim.sigmax = np.std(particles.x[ix])
                dummybeamtim.sigmay = np.std(particles.y[ix])
                dummybeamtim.x_beam_pos = np.mean(particles.x[ix]) + self.x_beam_offset
                dummybeamtim.y_beam_pos = np.mean(particles.y[ix]) + self.y_beam_offset

            dummybeamtim.tt_curr = self.t_sim  # In order to have the PIC activated
            dummybeamtim.Dt_curr = dt
            dummybeamtim.pass_numb = self.i_curr_bunch
            dummybeamtim.flag_new_bunch_pass = new_pass

            # Force space charge recomputation
            force_recompute_space_charge = interact_with_EC or (
                i_clou_step == 0
            )  # we always force at the first step, as we don't know the state of the PIC

            # Disable cleanings and regenerations
            skip_MP_cleaning = interact_with_EC
            skip_MP_regen = interact_with_EC

            # print(dummybeamtim.tt_curr, dummybeamtim.flag_new_bunch_pass, force_recompute_space_charge)

            # Perform cloud simulation step
            self.cloudsim.sim_time_step(
                beamtim_obj=dummybeamtim,
                Dt_substep_custom=Dt_substep,
                N_sub_steps_custom=N_sub_steps,
                kick_mode_for_beam_field=self.kick_mode_for_beam_field,
                force_recompute_space_charge=force_recompute_space_charge,
                skip_MP_cleaning=skip_MP_cleaning,
                skip_MP_regen=skip_MP_regen,
                force_reinterp_fields_at_substeps=(interact_with_EC
                    and self.force_interp_at_substeps_interacting_slices)
            )

            if interact_with_EC:
                # Build MP_system-like object with beam coordinates
                MP_p = Empty()
                MP_p.x_mp = particles.x[ix] + self.x_beam_offset
                MP_p.y_mp = particles.y[ix] + self.y_beam_offset
                MP_p.N_mp = len(particles.x[ix])

                ## compute cloud field on beam particles
                Ex_sc_p, Ey_sc_p = spacech_ele.get_sc_eletric_field(MP_p)
                
                # Apply kick
                mass_kg = particles.mass[ix]* qe / c**2
                if self.enable_kick_x or self.enable_kick_y:
                    fact_kick = (
                        charge
                        / (mass_kg * particles.beta0[ix] * particles.beta0[ix] * particles.gamma0[ix] * c * c)
                        * self.L_ecloud
                    )
                    dd  = particles.delta[ix]
                    px0 = particles.px[ix]
                    py0 = particles.py[ix]

                    # current slopes from canonicals
                    pn   = np.sqrt((1.0 + dd)**2 - px0**2 - py0**2)     # pz/P0
                    xp   = px0 / pn
                    yp   = py0 / pn

                    # apply kick as slope increments (same as PyHEADTAIL)
                    if self.enable_kick_x:
                        xp  += fact_kick * Ex_sc_p
                    if self.enable_kick_y:
                        yp  += fact_kick * Ey_sc_p

                    # write back EXACTLY to canonicals
                    den   = np.sqrt(1.0 + xp**2 + yp**2)
                    scale = (1.0 + dd) / den
                    particles.px[ix] = scale * xp
                    particles.py[ix] = scale * yp
                # (delta unchanged if there's no Ez)

                ## kick beam particles
                # mass_kg = particles.mass[ix]* qe / c**2
                # fact_kick = (
                #     charge
                #     / (mass_kg * particles.beta0[ix] * particles.beta0[ix] * particles.gamma0[ix] * c * c)
                #     * self.L_ecloud
                # )
                # if self.enable_kick_x:
                #     particles.px[ix] += (1 + particles.delta[ix]) * fact_kick * Ex_sc_p
                # if self.enable_kick_y:
                #     particles.py[ix] += (1 + particles.delta[ix]) * fact_kick * Ey_sc_p
            
            if self.enable_diagnostics:
                self._diagnostics_save(spacech_ele)
            self.t_sim += dt
            new_pass = False  # it can be true only for the first sub-slice of a slice

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
    
    def finalize_and_reinitialize(self):
        self._print("Exec. finalize and reinitialize")
        self._finalize()
        self._reinitialize()

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
    
    def _diagnostics_save(self,spacech_ele):
        MPe_for_save = self.cloudsim.cloud_list[0].MP_e

        if self.save_ele_distributions_last_track:
            self.rho_ele_last_track.append(spacech_ele.rho.copy())

        if self.save_ele_potential:
            self.phi_ele_last_track.append(spacech_ele.phi.copy())

        if self.save_ele_field:
            self.Ex_ele_last_track.append(spacech_ele.efx.copy())
            self.Ey_ele_last_track.append(spacech_ele.efy.copy())

        if self.save_beam_distributions_last_track:
            self.rho_beam_last_track.append(self.beam_PyPIC_state.rho.copy())

        if self.save_beam_potential:
            self.phi_beam_last_track.append(self.beam_PyPIC_state.phi.copy())

        if self.save_beam_field:
            self.Ex_beam_last_track.append(self.beam_PyPIC_state.efx.copy())
            self.Ey_beam_last_track.append(self.beam_PyPIC_state.efy.copy())

        if self.save_ele_MP_position:
            self.x_MP_last_track.append(MPe_for_save.x_mp.copy())
            self.y_MP_last_track.append(MPe_for_save.y_mp.copy())

        if self.save_ele_MP_velocity:
            self.vx_MP_last_track.append(MPe_for_save.vx_mp.copy())
            self.vy_MP_last_track.append(MPe_for_save.vy_mp.copy())

        if self.save_ele_MP_size:
            self.nel_MP_last_track.append(MPe_for_save.nel_mp.copy())

        if (
            self.save_ele_MP_position
            or self.save_ele_MP_velocity
            or self.save_ele_MP_size
        ):
            self.N_MP_last_track.append(MPe_for_save.N_mp)

        if self.save_ele_field_probes:
            MP_probes = Empty()
            MP_probes.x_mp = self.x_probes
            MP_probes.y_mp = self.y_probes
            MP_probes.nel_mp = self.x_probes * 0.0 + 1.0  # fictitious charge of 1 C
            MP_probes.N_mp = len(self.x_probes)
            Ex_sc_probe, Ey_sc_probe = spacech_ele.get_sc_eletric_field(MP_probes)

            self.Ex_ele_last_track_at_probes.append(Ex_sc_probe.copy())
            self.Ey_ele_last_track_at_probes.append(Ey_sc_probe.copy())

    def _print(self, *values, sep=' ', end='\n', file=None, flush=False):
        print("[PyEC4XS]", *values, sep=sep, end=end, file=file, flush=flush)
