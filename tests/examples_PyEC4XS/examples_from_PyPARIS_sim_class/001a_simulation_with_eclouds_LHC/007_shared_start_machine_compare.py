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


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


OUTPUT_DIR = SCRIPT_DIR / "comparison_against_pyparis_sim_class"


def _mean_and_std_pyht(beam):
    return (
        float(np.mean(beam.z)),
        float(np.mean(beam.dp)),
        float(np.std(beam.z)),
    )


def _mean_and_std_xsuite(particles):
    return (
        float(np.mean(particles.zeta)),
        float(np.mean(particles.delta)),
        float(np.std(particles.zeta)),
    )


def _run_mode(mode, n_turns=128):
    import xpart as xp
    from PyPARIS_sim_class.LHC_custom import LHC as PyHTLHC
    from PyEC4XS.LHC_custom import LHC as XsuiteLHC

    pyht = PyHTLHC(
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
        longitudinal_mode=mode,
    )
    xsuite = XsuiteLHC(
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
        longitudinal_mode=mode,
    )

    bunch = pyht.generate_6D_Gaussian_bunch_matched(
        n_macroparticles=32000,
        intensity=1.2e11,
        epsn_x=2.5e-6,
        epsn_y=2.5e-6,
        sigma_z=1.2e-9 / 4.0 * 299792458.0,
    )
    inj = pyht.transverse_map.get_injection_optics()
    sigma_x = np.sqrt(inj["beta_x"] * 2.5e-6 / pyht.betagamma)
    sigma_y = np.sqrt(inj["beta_y"] * 2.5e-6 / pyht.betagamma)
    bunch.x += 0.1 * sigma_x
    bunch.y += 0.1 * sigma_y

    particles = xp.Particles(
        p0c=float(xsuite.particle_ref.p0c[0]),
        mass0=float(xsuite.particle_ref.mass0),
        q0=float(xsuite.particle_ref.q0),
        x=np.array(bunch.x, copy=True),
        px=np.array(bunch.xp, copy=True),
        y=np.array(bunch.y, copy=True),
        py=np.array(bunch.yp, copy=True),
        zeta=np.array(bunch.z, copy=True),
        delta=np.array(bunch.dp, copy=True),
        weight=np.full(bunch.macroparticlenumber, float(bunch.particlenumber_per_mp)),
    )

    mean_z_diff = []
    mean_dp_diff = []
    sigma_z_diff = []
    for _ in range(n_turns + 1):
        pyht_mean_z, pyht_mean_dp, pyht_sigma_z = _mean_and_std_pyht(bunch)
        xs_mean_z, xs_mean_dp, xs_sigma_z = _mean_and_std_xsuite(particles)
        mean_z_diff.append(pyht_mean_z - xs_mean_z)
        mean_dp_diff.append(pyht_mean_dp - xs_mean_dp)
        sigma_z_diff.append(pyht_sigma_z - xs_sigma_z)

        for elem in pyht.one_turn_map:
            elem.track(bunch)
        for elem in xsuite.one_turn_map:
            elem.track(particles)

    return {
        "mean_z_diff": np.array(mean_z_diff),
        "mean_dp_diff": np.array(mean_dp_diff),
        "sigma_z_diff": np.array(sigma_z_diff),
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = {
        "linear": _run_mode("linear"),
        "non-linear": _run_mode("non-linear"),
    }

    turns = np.arange(len(results["linear"]["mean_z_diff"]))
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)

    axes[0, 0].plot(turns, results["linear"]["mean_z_diff"], color="tab:blue")
    axes[0, 0].set_title("Linear: mean_z diff")
    axes[0, 0].set_ylabel("PyHEADTAIL - Xsuite [m]")
    axes[0, 0].grid(True)

    axes[0, 1].plot(turns, results["linear"]["mean_dp_diff"], color="tab:orange")
    axes[0, 1].set_title("Linear: mean_dp diff")
    axes[0, 1].set_ylabel("PyHEADTAIL - Xsuite")
    axes[0, 1].grid(True)

    axes[1, 0].plot(turns, results["non-linear"]["mean_z_diff"], color="tab:blue")
    axes[1, 0].set_title("Non-linear: mean_z diff")
    axes[1, 0].set_xlabel("Turn")
    axes[1, 0].set_ylabel("PyHEADTAIL - Xsuite [m]")
    axes[1, 0].grid(True)

    axes[1, 1].plot(turns, results["non-linear"]["mean_dp_diff"], color="tab:orange")
    axes[1, 1].set_title("Non-linear: mean_dp diff")
    axes[1, 1].set_xlabel("Turn")
    axes[1, 1].set_ylabel("PyHEADTAIL - Xsuite")
    axes[1, 1].grid(True)

    fig.tight_layout()
    plot_path = OUTPUT_DIR / "shared_start_machine_compare.png"
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)

    summary = {
        "plot": str(plot_path),
        "linear_max_abs_mean_z_diff": float(np.max(np.abs(results["linear"]["mean_z_diff"]))),
        "linear_max_abs_mean_dp_diff": float(np.max(np.abs(results["linear"]["mean_dp_diff"]))),
        "linear_max_abs_sigma_z_diff": float(np.max(np.abs(results["linear"]["sigma_z_diff"]))),
        "nonlinear_max_abs_mean_z_diff": float(np.max(np.abs(results["non-linear"]["mean_z_diff"]))),
        "nonlinear_max_abs_mean_dp_diff": float(np.max(np.abs(results["non-linear"]["mean_dp_diff"]))),
        "nonlinear_max_abs_sigma_z_diff": float(np.max(np.abs(results["non-linear"]["sigma_z_diff"]))),
    }

    with open(OUTPUT_DIR / "shared_start_machine_compare_summary.json", "w", encoding="utf-8") as fid:
        json.dump(summary, fid, indent=2, sort_keys=True)

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
