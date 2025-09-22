# test_xsuite.py
import numpy as np
np.random.seed(42)

# Xsuite basics
import xobjects as xo
import xpart as xp
import xtrack as xt

import sys
sys.path.append("../")
import xfields as xf
from PyEC4XS import xEcloud   # ensure it exposes same knobs as below
from sim_config_manager import SimConfig
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import shutil

if os.path.exists("./xsout"):
    shutil.rmtree("./xsout")
os.makedirs("./xsout")
os.makedirs("./xsout/phi_ele")
os.makedirs("./xsout/phi_beam")
os.makedirs("./xsout/phi_comb")


# Define conversion functions between Xsuite and PyHEADTAIL coordinates
def slopes_to_pxpy(xp, yp, delta):
    den = np.sqrt(1.0 + xp**2 + yp**2)              # = |p| / p_s
    px = (1.0 + delta) * xp / den                   # px = p_x / p0
    py = (1.0 + delta) * yp / den
    return px, py

def pxpy_to_slopes(px, py, delta):
    ps_over_p0 = np.sqrt((1.0 + delta)**2 - px**2 - py**2)
    xp = px / ps_over_p0
    yp = py / ps_over_p0
    return xp, yp

# ---- Common parameters ----
q0 = 1.602176634e-19
clight = 299792458.0

sigx = 1.0e-3
sigy = 1.0e-3
sigz = 0.075

# Ecloud parameters
L_ec = 10.0
n_passages = 10

# Slicer parameters
n_slices = 64

slicer=xf.TempSlicer(n_slices=n_slices,sigma_z=sigz)
# slicer_config = dict(n_slices=n_slices, mode="percentile", percentile_pct=1e-4)
# slicer_config = dict(n_slices=n_slices, mode="minmax")

# Catch warning to fix bug
# import warnings
# import numpy as np

# # 1) Turn Python RuntimeWarnings into exceptions
# warnings.filterwarnings("error", category=RuntimeWarning)

# # 2) Make NumPy raise on invalid/divide issues (instead of just warning)
# np.seterr(divide="raise", invalid="raise")


# ---- Particles ----
context = xo.ContextCpu()
mass0_eV = xp.PROTON_MASS_EV  # proton rest mass in eV

# Load PyHEADTAIL bunch
dat = np.load("bunch_pyht.npz")
ref = np.load("bunch_ref.npz")

x     = dat["x"]
xp_    = dat["xp"]     
y     = dat["y"]
yp_    = dat["yp"]
zeta  = -dat["z"]      
delta = dat["dp"]

px, py = slopes_to_pxpy(xp_, yp_, delta)


# Reference particle
p0c   = float(ref["p0c_eV"])  # [eV]
# Example for protons
parts = xp.Particles(
    p0c=p0c,                 # reference momentum * c [eV]
    mass0=xp.PROTON_MASS_EV, # change if not protons
    q0=1,                    # +1 e for protons
    x=x, px=px, y=y, py=py, zeta=zeta, delta=delta,
)

# Import rest of beam properties from np file
if "intensity" in dat.files and dat["intensity"] is not None:
    intensity = float(dat["intensity"])
    n_macrop  = int(dat["n_macrop"])
    parts.weight = np.full(n_macrop, intensity / n_macrop)

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

# ---- ECLOUD element in Xsuite ----
ec = xEcloud(
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
    y_beam_offset = -1e-3,
    enable_diagnostics=True
)
ec.save_ele_potential_and_field = True
ec.save_beam_potential_and_field = True

# Simple line: just the EC element
line = xt.Line(elements=[ec])
tracker = xt.Tracker(line=line, _context=context)

# ---- Track and record kicks over passages ----
kicks_x = []
kicks_y = []
print("BEAM SPECS:")
xp_, yp_ = pxpy_to_slopes(parts.px,parts.py,delta)
print(f"XP:{np.mean(xp_)}")
print(f"YP:{np.mean(yp_)}")
print(f"X:{np.mean(parts.x)}")
print(f"Y:{np.mean(parts.y)}")

k=0
j=0
u=0
for i in tqdm(range(n_passages)):
    xp_, yp_ = pxpy_to_slopes(parts.px,parts.py,parts.delta)
    xp_mean_before = np.mean(xp_[parts.state>0])
    yp_mean_before = np.mean(yp_[parts.state>0])
    line.track(parts, num_turns=1) 
    xp_, yp_ = pxpy_to_slopes(parts.px,parts.py,parts.delta)
    xp_mean_after = np.mean(xp_[parts.state>0])
    yp_mean_after = np.mean(yp_[parts.state>0])
    kicks_x.append(xp_mean_after - xp_mean_before)
    kicks_y.append(yp_mean_after - yp_mean_before)
    if i==0:
        vmin_ele = min(phi.min() for phi in ec.phi_ele_last_track)
        vmax_ele = max(phi.max() for phi in ec.phi_ele_last_track)
        vmin_beam = min(phi.min() for phi in ec.phi_beam_last_track)
        vmax_beam = max(phi.max() for phi in ec.phi_beam_last_track)
        k+=len(ec.phi_ele_last_track)
        j+=len(ec.phi_beam_last_track)
        u+=len(ec.phi_beam_last_track)
    else:
        for phi in ec.phi_ele_last_track:
            im = plt.imshow(phi, origin='lower', aspect='auto', interpolation='nearest',
                            cmap=plt.cm.viridis,vmin = vmin_ele, vmax=vmax_ele)
            plt.colorbar(im, label='value')
            plt.xlabel('column index')
            plt.ylabel('row index')
            plt.title(f'Electron Potential {k}')
            plt.savefig(f"./xsout/phi_ele/phi_frame_{k}.png")
            plt.clf()
            k+=1
        for phi in ec.phi_beam_last_track:
            im = plt.imshow(phi, origin='lower', aspect='auto', interpolation='nearest',
                            cmap=plt.cm.viridis, vmin=vmin_beam, vmax=vmax_beam)
            plt.colorbar(im, label='value')
            plt.xlabel('column index')
            plt.ylabel('row index')
            plt.title(f'Beam Potential {j}')
            plt.savefig(f"./xsout/phi_beam/phi_frame_{j}.png")
            plt.clf()
            j+=1
        for phi1,phi2 in zip(ec.phi_ele_last_track,ec.phi_beam_last_track):
            im = plt.imshow(phi1+phi2, origin='lower', aspect='auto', interpolation='nearest',
                            cmap=plt.cm.viridis, vmin=vmin_ele, vmax=vmax_beam)
            plt.colorbar(im, label='value')
            plt.xlabel('column index')
            plt.ylabel('row index')
            plt.title(f'Potential {u}')
            plt.savefig(f"./xsout/phi_comb/phi_frame_{u}.png")
            plt.clf()
            u+=1

# Save for comparison
kicks_x = np.array(kicks_x, dtype=float)
np.save('x_kicks_xsuite.npy', kicks_x)
print('Saved x_kicks_xsuite.npy')
kicks_y = np.array(kicks_y, dtype=float)
np.save('y_kicks_xsuite.npy', kicks_y)
print('Saved y_kicks_xsuite.npy')
