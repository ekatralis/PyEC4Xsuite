import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

_mpl_dir = Path(tempfile.gettempdir()) / "pyec4xs_mpl"
_mpl_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_dir))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.constants import c


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


OUTPUT_DIR = SCRIPT_DIR / "comparison_against_pyparis_sim_class"


def _slope_at_zero(z, values):
    i0 = len(z) // 2
    return float((values[i0 + 1] - values[i0 - 1]) / (z[i0 + 1] - z[i0 - 1]))


def main():
    import xpart as xp
    import xtrack as xt
    from PyPARIS_sim_class.LHC_custom import LHC as PyHTLHC

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pyht_machine = PyHTLHC(
        n_segments=3,
        machine_configuration="LHC-collision",
        beta_x=400.0,
        beta_y=400.0,
        accQ_x=62.27,
        accQ_y=60.295,
        Qp_x=0.0,
        Qp_y=0.0,
        octupole_knob=0.0,
        V_RF=12e6,
        longitudinal_mode="non-linear",
    )

    z_grid = np.linspace(-5e-3, 5e-3, 401)
    n_part = len(z_grid)

    pyht_beam = pyht_machine.generate_6D_Gaussian_bunch(
        n_macroparticles=n_part,
        intensity=1.0,
        epsn_x=1e-6,
        epsn_y=1e-6,
        sigma_z=1e-6,
    )
    pyht_beam.x[:] = 0.0
    pyht_beam.xp[:] = 0.0
    pyht_beam.y[:] = 0.0
    pyht_beam.yp[:] = 0.0
    pyht_beam.z[:] = z_grid
    pyht_beam.dp[:] = 0.0
    pyht_machine.longitudinal_map.track(pyht_beam)
    pyht_delta_kick = np.array(pyht_beam.dp, copy=True)

    particle_ref = xp.Particles(p0c=6800e9, mass0=xp.PROTON_MASS_EV, q0=1.0)
    beta0 = float(particle_ref.beta0[0])
    circumference = 26658.8832
    alpha = 3.225e-4
    frequency_rf = 35640 * beta0 * c / circumference

    xtrack_kicks = {}
    for lag_deg in (0.0, 180.0):
        elem = xt.LineSegmentMap(
            length=0.0,
            qx=0.0,
            qy=0.0,
            betx=1.0,
            bety=1.0,
            longitudinal_mode="nonlinear",
            momentum_compaction_factor=alpha,
            slippage_length=circumference,
            voltage_rf=[12e6],
            frequency_rf=[frequency_rf],
            lag_rf=[lag_deg],
        )
        part = xp.Particles(
            p0c=6800e9,
            mass0=xp.PROTON_MASS_EV,
            q0=1.0,
            x=np.zeros(n_part),
            px=np.zeros(n_part),
            y=np.zeros(n_part),
            py=np.zeros(n_part),
            zeta=z_grid.copy(),
            delta=np.zeros(n_part),
        )
        elem.track(part)
        xtrack_kicks[lag_deg] = np.array(part.delta, copy=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, xlim in zip(axes, [5e-3, 5e-4]):
        ax.plot(z_grid, pyht_delta_kick, label="PyHEADTAIL", color="tab:blue")
        ax.plot(
            z_grid,
            xtrack_kicks[0.0],
            label="Xsuite lag_rf=0 deg",
            color="tab:red",
            linestyle="--",
        )
        ax.plot(
            z_grid,
            xtrack_kicks[180.0],
            label="Xsuite lag_rf=180 deg",
            color="tab:green",
            linestyle="-.",
        )
        ax.set_xlim(-xlim, xlim)
        ax.set_xlabel("Initial z / zeta [m]")
        ax.grid(True)

    axes[0].set_ylabel("One-turn delta change")
    axes[0].set_title("Full Range")
    axes[1].set_title("Zoom Near 0")
    axes[1].legend(loc="best")
    fig.tight_layout()

    plot_path = OUTPUT_DIR / "rf_phase_convention_kick_comparison.png"
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)

    summary = {
        "plot": str(plot_path),
        "pyheadtail_slope_at_zero": _slope_at_zero(z_grid, pyht_delta_kick),
        "xsuite_lag_0_slope_at_zero": _slope_at_zero(z_grid, xtrack_kicks[0.0]),
        "xsuite_lag_180_slope_at_zero": _slope_at_zero(z_grid, xtrack_kicks[180.0]),
    }

    with open(OUTPUT_DIR / "rf_phase_convention_summary.json", "w", encoding="utf-8") as fid:
        json.dump(summary, fid, indent=2, sort_keys=True)

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
