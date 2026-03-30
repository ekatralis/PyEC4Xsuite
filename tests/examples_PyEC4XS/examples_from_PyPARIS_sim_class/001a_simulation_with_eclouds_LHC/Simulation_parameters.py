from scipy.constants import c


check_for_resubmit = False

machine_configuration = "LHC-collision"
optics_pickle_file = None
n_segments = 3
beta_x = 400.0
beta_y = 400.0
Q_x = 62.27
Q_y = 60.295
Qp_x = 0.0
Qp_y = 0.0
octupole_knob = 0.0
V_RF = 12e6
longitudinal_mode = "non-linear"

enable_transverse_damper = False
dampingrate_x = 50.0
dampingrate_y = 100.0

bunch_from_file = None
intensity = 1.2e11
epsn_x = 2.5e-6
epsn_y = 2.5e-6
sigma_z = 1.2e-9 / 4.0 * c
x_kick_in_sigmas = 0.1
y_kick_in_sigmas = 0.1

n_slices = 64
z_cut = 2.5e-9 / 2.0 * c
macroparticles_per_slice = 500
n_macroparticles = macroparticles_per_slice * n_slices
write_buffer_every = 1

N_turns = 128
N_turns_target = 128
sim_stop_frac = 0.9
flag_check_emittance_growth = False
epsn_x_max_growth_fraction = 0.5
epsn_y_max_growth_fraction = 0.5

footprint_mode = False
n_macroparticles_for_footprint_map = 500000
n_macroparticles_for_footprint_track = 5000

chamb_type = "polyg"
x_aper = 2.300000e-02
y_aper = 1.800000e-02
filename_chm = "!folder_of_this_file!/../../../../PyPARIS_example/LHC_chm_ver.mat"
Dt_ref = 25e-12
pyecl_input_folder = "!folder_of_this_file!/../../../../PyPARIS_example/pyecloud_config"
sey = 1.30

PyPICmode = "ShortleyWeller_WithTelescopicGrids"
N_min_Dh_main = 10.0
Dh_sc_ext = 0.8e-3
f_telescope = 0.3
N_nodes_discard = 5.0
target_size_internal_grid_sigma = 10.0
target_Dh_internal_grid_sigma = 0.5
custom_target_grid_arcs = None
force_interp_at_substeps_interacting_slices = True

enable_kick_x = True
enable_kick_y = True

enable_arc_dip = True
fraction_device_dip = 0.65
init_unif_edens_flag_dip = 1
init_unif_edens_dip = 1.0e10
N_MP_ele_init_dip = 5000
N_mp_max_dip = N_MP_ele_init_dip * 4
B_multip_dip = [8.33]

enable_arc_quad = False
fraction_device_quad = 0.26
N_mp_max_quad = 2000000
B_multip_quad = [0.0, 188.2]
folder_path = "../../LHC_ecloud_distrib_quads/"
filename_state = "combined_distribution_sey%.2f_%.1fe11ppb_7tev.mat" % (
    sey,
    intensity / 1e11,
)
filename_init_MP_state_quad = folder_path + filename_state

enable_eclouds_at_kick_elements = False
path_buildup_simulations_kick_elements = "/tmp/unused_for_pyec4xs_smoke"
name_MP_state_file_kick_elements = "MP_state_9.mat"
orbit_factor = 0.625
