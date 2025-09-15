# test_pyht.py
import numpy as np
np.random.seed(42)

from PyHEADTAIL.particles.slicing import UniformBinSlicer
from PyHEADTAIL.particles.particles import Particles
from PyHEADTAIL.particles.generators import ParticleGenerator
# from PyHEADTAIL.trackers.tracker import Tracker

# PyEC4PyHT ECLOUD element
from PyECLOUD.PyEC4PyHT import Ecloud
from sim_config_manager import SimConfig
from tqdm import tqdm

# ---- Common parameters (mirrored in Xsuite script) ----
q0 = 1.602176634e-19
clight = 299792458.0

# ---- Params ----
sigx = 1.0e-3
sigy = 1.0e-3
sigz = 0.075

# Ecloud parameters
L_ec = 10.0
n_passages = 10

# Slicer parameters
n_slices = 64
# slicer = UniformBinSlicer(n_slices, z_cuts=(-4*sigz, 4*sigz))
slicer = UniformBinSlicer(n_slices)

# Generate Bunch
from machines_for_testing import SPS
N_kicks = 1
epsn_x = 2.5e-6
epsn_y = 2.5e-6
machine = SPS(n_segments=N_kicks, machine_configuration='Q20-injection', accQ_x=20., accQ_y=20.)
bunch = machine.generate_6D_Gaussian_bunch(n_macroparticles=300000, intensity=1.15e11, epsn_x=epsn_x, epsn_y=epsn_y, sigma_z=0.2)



# Export PyHEADTAIL bunch to have an identical one in Xsuite
np.savez_compressed(
    "bunch_pyht.npz",
    x=bunch.x,
    xp=bunch.xp,
    y=bunch.y,
    yp=bunch.yp,
    z=bunch.z,
    dp=bunch.dp,
    n_macrop=bunch.macroparticlenumber,
    intensity=getattr(bunch, "intensity", None),
    q0=getattr(bunch, "q0", 1.0),
)

# Also save reference energy
gamma = machine.gamma
# proton example
m0_eV = 938.2720813e6
beta = np.sqrt(1 - 1/gamma**2)
p0c_eV = beta * gamma * m0_eV
np.savez_compressed("bunch_ref.npz", p0c_eV=p0c_eV, gamma=gamma)

# Import PyECLOUD config as in PyPARIS_sim_class
pp = SimConfig("./Simulation_parameters.py")

sigma_x_smooth = sigx
sigma_y_smooth = sigy

nel_mp_ref_0 = (
                pp.init_unif_edens_dip * 4 * pp.x_aper * pp.y_aper
                / pp.N_MP_ele_init_dip
            )

if pp.custom_target_grid_arcs is not None:
    target_grid_arcs = pp.custom_target_grid_arcs
else:
    target_grid_arcs = {
        "x_min_target": -pp.target_size_internal_grid_sigma * sigma_x_smooth,
        "x_max_target": pp.target_size_internal_grid_sigma * sigma_x_smooth,
        "y_min_target": -pp.target_size_internal_grid_sigma * sigma_y_smooth,
        "y_max_target": pp.target_size_internal_grid_sigma * sigma_y_smooth,
        "Dh_target": pp.target_Dh_internal_grid_sigma * sigma_x_smooth,
    }

# ---- ECLOUD element ----
ec = Ecloud(
    L_ecloud=L_ec,
    slicer=slicer,
    Dt_ref=pp.Dt_ref,
    pyecl_input_folder=pp.pyecl_input_folder,
    chamb_type=pp.chamb_type,
    x_aper=pp.x_aper,
    y_aper=pp.y_aper,
    filename_chm=pp.filename_chm,
    PyPICmode=pp.PyPICmode,
    Dh_sc=pp.Dh_sc_ext,
    N_min_Dh_main=pp.N_min_Dh_main,
    f_telescope=pp.f_telescope,
    N_nodes_discard=pp.N_nodes_discard,
    target_grid=target_grid_arcs,
    init_unif_edens_flag=pp.init_unif_edens_flag_dip,
    init_unif_edens=pp.init_unif_edens_dip,
    N_mp_max=pp.N_mp_max_dip,
    nel_mp_ref_0=nel_mp_ref_0,
    B_multip=pp.B_multip_dip,
    enable_kick_x=pp.enable_kick_x,
    enable_kick_y=pp.enable_kick_y,
    x_beam_offset = 1e-3,
    y_beam_offset = -1e-3
)


# ---- Tracking over multiple passages ----
kicks_x = []
kicks_y = []
print("BEAM SPECS:")
print(f"XP:{np.mean(bunch.xp)}")
print(f"YP:{np.mean(bunch.yp)}")
print(f"X:{np.mean(bunch.x)}")
print(f"Y:{np.mean(bunch.y)}")

for i in tqdm(range(n_passages)):
    xp_mean_before = np.mean(bunch.xp)
    yp_mean_before = np.mean(bunch.yp)
    ec.track(bunch)
    xp_mean_after = np.mean(bunch.xp)
    yp_mean_after = np.mean(bunch.yp)
    kicks_x.append(xp_mean_after - xp_mean_before)
    kicks_y.append(yp_mean_after - yp_mean_before)

# Save for comparison
kicks_x = np.array(kicks_x, dtype=float)
np.save('x_kicks_pyht.npy', kicks_x)
print('Saved x_kicks_pyht.npy')
kicks_y = np.array(kicks_y, dtype=float)
np.save('y_kicks_pyht.npy', kicks_y)
print('Saved y_kicks_pyht.npy')