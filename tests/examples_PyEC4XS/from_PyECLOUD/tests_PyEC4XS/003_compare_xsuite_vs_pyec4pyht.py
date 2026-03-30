import importlib.util
import json
import os
import tempfile
from pathlib import Path

import numpy as np

_mpl_dir = Path(tempfile.gettempdir()) / "pyec4xs_mpl"
_mpl_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_dir))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from PyECLOUD import PyEC4PyHT
from PyHEADTAIL.particles.slicing import UniformBinSlicer

from common import (
    LEGACY_TEST_ROOT,
    build_sps_ring,
    build_tracking_line,
    compute_sigmas,
    ensure_output_dir,
    generate_matched_bunch,
    inject_legacy_reference_particles,
    load_legacy_reference,
    make_pyht_compatible_slicer,
    relative_error_percent,
    set_reproducible_seed,
    snapshot_reference_particles,
    write_json,
)


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = ensure_output_dir(SCRIPT_DIR / "output" / "003_compare_xsuite_vs_pyec4pyht")
DRIFT_SIM_DIR = str((SCRIPT_DIR / "../../../../drift_sim").resolve())


def _load_legacy_machine_module():
    path = LEGACY_TEST_ROOT / "machines_for_testing.py"
    spec = importlib.util.spec_from_file_location("legacy_machines_for_testing", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


LEGACY_MACHINES = _load_legacy_machine_module()


def _snapshot_pyht_reference_particles(bunch, n_reference_particles):
    mask = np.array(bunch.id) <= int(n_reference_particles)
    particle_id = np.array(bunch.id[mask], dtype=int)
    order = np.argsort(particle_id)
    return {
        "particle_id": particle_id[order],
        "xp": np.array(bunch.xp[mask], dtype=float)[order],
        "yp": np.array(bunch.yp[mask], dtype=float)[order],
        "z": np.array(bunch.z[mask], dtype=float)[order],
    }


def _build_case_config(name):
    if name == "001_compare_kick_against_headtail":
        return {
            "name": name,
            "benchmark_file": LEGACY_TEST_ROOT
            / "headtail_for_test"
            / "test_protons"
            / "SPS_Q20_proton_check_dipole_20150212_prb.dat",
            "n_kicks": 1,
            "machine_configuration": "Q20-injection",
            "pyht_accQ_x": 20.0,
            "pyht_accQ_y": 20.0,
            "xs_accQ_x": 20.0,
            "xs_accQ_y": 20.0,
            "comparison_mode": "cumulative",
            "sigma_z": 0.2,
            "B_multip": [0.5],
            "n_reference_particles": 5000,
            "n_macroparticles": 300000,
            "intensity": 1.15e11,
            "epsn_x": 2.5e-6,
            "epsn_y": 2.5e-6,
            "init_unif_edens": 2.0e11,
            "n_mp_ele_init": 100000,
        }
    if name == "002_compare_kick_and_betatron_motion":
        return {
            "name": name,
            "benchmark_file": LEGACY_TEST_ROOT
            / "headtail_for_test"
            / "test_protons"
            / "SPS_Q20_proton_check_dipole_betatron_motion_20150212_prb.dat",
            "n_kicks": 3,
            "machine_configuration": "Q20-injection",
            "pyht_accQ_x": 20.13,
            "pyht_accQ_y": 20.18,
            "xs_accQ_x": 20.13,
            "xs_accQ_y": 20.18,
            "comparison_mode": "incremental",
            "sigma_z": 0.2,
            "B_multip": [0.5],
            "n_reference_particles": 5000,
            "n_macroparticles": 300000,
            "intensity": 1.15e11,
            "epsn_x": 2.5e-6,
            "epsn_y": 2.5e-6,
            "init_unif_edens": 2.0e11,
            "n_mp_ele_init": 100000,
        }
    raise ValueError(f"Unknown case {name}")


def _run_xsuite_case(cfg, reference):
    ring = build_sps_ring(
        cfg["machine_configuration"],
        n_segments=cfg["n_kicks"],
        accQ_x=cfg["xs_accQ_x"],
        accQ_y=cfg["xs_accQ_y"],
    )
    sigma_x, sigma_y = compute_sigmas(
        ring["injection_optics"], ring["betagamma"], cfg["epsn_x"], cfg["epsn_y"]
    )
    x_aper = 20.0 * sigma_x
    y_aper = 20.0 * sigma_y
    dh_sc = 2.0 * x_aper / 128.0 / 2.0
    nel_mp_ref_0 = (
        cfg["init_unif_edens"] * 4.0 * x_aper * y_aper / cfg["n_mp_ele_init"]
    )
    slicer = make_pyht_compatible_slicer(n_slices=64, z_cut=3.0 * cfg["sigma_z"])

    from PyEC4XS import xEcloud

    ecloud = xEcloud(
        L_ecloud=ring["circumference"] / cfg["n_kicks"],
        slicer=slicer,
        Dt_ref=25e-12,
        pyecl_input_folder=DRIFT_SIM_DIR,
        x_aper=x_aper,
        y_aper=y_aper,
        Dh_sc=dh_sc,
        init_unif_edens_flag=1,
        init_unif_edens=cfg["init_unif_edens"],
        N_mp_max=cfg["n_mp_ele_init"] * 4.0,
        nel_mp_ref_0=nel_mp_ref_0,
        B_multip=cfg["B_multip"],
    )
    line = build_tracking_line(ring, ecloud=ecloud)

    generation_ring = build_sps_ring(cfg["machine_configuration"], n_segments=1)
    bunch = generate_matched_bunch(
        ring_data=generation_ring,
        n_macroparticles=cfg["n_macroparticles"],
        intensity=cfg["intensity"],
        epsn_x=cfg["epsn_x"],
        epsn_y=cfg["epsn_y"],
        sigma_z=cfg["sigma_z"],
    )
    inject_legacy_reference_particles(bunch, reference)

    n_turns = reference["x"].shape[0]
    initial = snapshot_reference_particles(bunch, cfg["n_reference_particles"])
    previous = initial
    per_turn = []

    for turn_idx in range(n_turns - 1):
        line.track(bunch, num_turns=1)
        after = snapshot_reference_particles(bunch, cfg["n_reference_particles"])
        if cfg["comparison_mode"] == "cumulative":
            dx = after["px"] - initial["px"]
            dy = after["py"] - initial["py"]
            ref_dx = reference["xp"][turn_idx + 1, :] - reference["xp"][0, :]
            ref_dy = reference["yp"][turn_idx + 1, :] - reference["yp"][0, :]
        else:
            dx = after["px"] - previous["px"]
            dy = after["py"] - previous["py"]
            ref_dx = reference["xp"][turn_idx + 1, :] - reference["xp"][turn_idx, :]
            ref_dy = reference["yp"][turn_idx + 1, :] - reference["yp"][turn_idx, :]

        err_x = relative_error_percent(dx, ref_dx)
        err_y = relative_error_percent(dy, ref_dy)
        per_turn.append(
            {
                "turn": turn_idx,
                "dx": dx.copy(),
                "dy": dy.copy(),
                "z": reference["z"][turn_idx + 1, :].copy(),
                "ref_dx": ref_dx.copy(),
                "ref_dy": ref_dy.copy(),
                "rms_err_x_percent": float(np.std(err_x)),
                "rms_err_y_percent": float(np.std(err_y)),
            }
        )
        previous = after

    return per_turn


def _run_pyec4pyht_case(cfg, reference):
    machine = LEGACY_MACHINES.SPS(
        n_segments=cfg["n_kicks"],
        machine_configuration=cfg["machine_configuration"],
        accQ_x=cfg["pyht_accQ_x"],
        accQ_y=cfg["pyht_accQ_y"],
    )
    inj_optics = machine.transverse_map.get_injection_optics()
    sigma_x = np.sqrt(inj_optics["beta_x"] * cfg["epsn_x"] / machine.betagamma)
    sigma_y = np.sqrt(inj_optics["beta_y"] * cfg["epsn_y"] / machine.betagamma)
    x_aper = 20.0 * sigma_x
    y_aper = 20.0 * sigma_y
    dh_sc = 2.0 * x_aper / 128.0 / 2.0
    nel_mp_ref_0 = (
        cfg["init_unif_edens"] * 4.0 * x_aper * y_aper / cfg["n_mp_ele_init"]
    )
    slicer = UniformBinSlicer(n_slices=64, n_sigma_z=3.0)
    ecloud = PyEC4PyHT.Ecloud(
        L_ecloud=machine.circumference / cfg["n_kicks"],
        slicer=slicer,
        Dt_ref=25e-12,
        pyecl_input_folder=DRIFT_SIM_DIR,
        x_aper=x_aper,
        y_aper=y_aper,
        Dh_sc=dh_sc,
        init_unif_edens_flag=1,
        init_unif_edens=cfg["init_unif_edens"],
        N_mp_max=cfg["n_mp_ele_init"] * 4.0,
        nel_mp_ref_0=nel_mp_ref_0,
        B_multip=cfg["B_multip"],
    )
    machine.install_after_each_transverse_segment(ecloud)
    bunch = machine.generate_6D_Gaussian_bunch(
        n_macroparticles=cfg["n_macroparticles"],
        intensity=cfg["intensity"],
        epsn_x=cfg["epsn_x"],
        epsn_y=cfg["epsn_y"],
        sigma_z=cfg["sigma_z"],
    )

    n_ref = cfg["n_reference_particles"]
    bunch.x[:n_ref] = reference["x"][0, :]
    bunch.xp[:n_ref] = reference["xp"][0, :]
    bunch.y[:n_ref] = reference["y"][0, :]
    bunch.yp[:n_ref] = reference["yp"][0, :]
    bunch.z[:n_ref] = reference["z"][0, :]
    bunch.dp[:n_ref] = reference["dp"][0, :]

    n_turns = reference["x"].shape[0]
    initial = _snapshot_pyht_reference_particles(bunch, n_ref)
    previous = initial
    per_turn = []

    for turn_idx in range(n_turns - 1):
        machine.track(bunch)
        after = _snapshot_pyht_reference_particles(bunch, n_ref)
        if cfg["comparison_mode"] == "cumulative":
            dx = after["xp"] - initial["xp"]
            dy = after["yp"] - initial["yp"]
            ref_dx = reference["xp"][turn_idx + 1, :] - reference["xp"][0, :]
            ref_dy = reference["yp"][turn_idx + 1, :] - reference["yp"][0, :]
        else:
            dx = after["xp"] - previous["xp"]
            dy = after["yp"] - previous["yp"]
            ref_dx = reference["xp"][turn_idx + 1, :] - reference["xp"][turn_idx, :]
            ref_dy = reference["yp"][turn_idx + 1, :] - reference["yp"][turn_idx, :]

        err_x = relative_error_percent(dx, ref_dx)
        err_y = relative_error_percent(dy, ref_dy)
        per_turn.append(
            {
                "turn": turn_idx,
                "dx": dx.copy(),
                "dy": dy.copy(),
                "z": reference["z"][turn_idx + 1, :].copy(),
                "ref_dx": ref_dx.copy(),
                "ref_dy": ref_dy.copy(),
                "rms_err_x_percent": float(np.std(err_x)),
                "rms_err_y_percent": float(np.std(err_y)),
            }
        )
        previous = after

    return per_turn


def _make_case_plot(case_name, xsuite_turns, pyht_turns):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    last_xs = xsuite_turns[-1]
    last_pyht = pyht_turns[-1]

    ax = axes[0, 0]
    ax.plot(last_xs["z"], last_xs["ref_dx"], color="black", linewidth=1.2, label="HEADTAIL ref")
    ax.plot(last_pyht["z"], last_pyht["dx"], color="tab:blue", linewidth=1.2, label="PyEC4PyHT")
    ax.plot(last_xs["z"], last_xs["dx"], color="tab:orange", linewidth=1.2, label="PyEC4XS")
    ax.set_title(f"{case_name}: final-turn $\\Delta x'$")
    ax.set_xlabel("z [m]")
    ax.set_ylabel("$\\Delta x'$")
    ax.grid(True)
    ax.legend(loc="best")

    ax = axes[0, 1]
    ax.plot(last_xs["z"], last_xs["ref_dy"], color="black", linewidth=1.2, label="HEADTAIL ref")
    ax.plot(last_pyht["z"], last_pyht["dy"], color="tab:blue", linewidth=1.2, label="PyEC4PyHT")
    ax.plot(last_xs["z"], last_xs["dy"], color="tab:orange", linewidth=1.2, label="PyEC4XS")
    ax.set_title(f"{case_name}: final-turn $\\Delta y'$")
    ax.set_xlabel("z [m]")
    ax.set_ylabel("$\\Delta y'$")
    ax.grid(True)
    ax.legend(loc="best")

    ax = axes[1, 0]
    ax.plot(
        [item["turn"] for item in pyht_turns],
        [item["rms_err_x_percent"] for item in pyht_turns],
        ".-",
        color="tab:blue",
        label="PyEC4PyHT",
    )
    ax.plot(
        [item["turn"] for item in xsuite_turns],
        [item["rms_err_x_percent"] for item in xsuite_turns],
        ".-",
        color="tab:orange",
        label="PyEC4XS",
    )
    ax.set_title("RMS relative error in x'")
    ax.set_xlabel("Turn")
    ax.set_ylabel("error [%]")
    ax.grid(True)
    ax.legend(loc="best")

    ax = axes[1, 1]
    ax.plot(
        [item["turn"] for item in pyht_turns],
        [item["rms_err_y_percent"] for item in pyht_turns],
        ".-",
        color="tab:blue",
        label="PyEC4PyHT",
    )
    ax.plot(
        [item["turn"] for item in xsuite_turns],
        [item["rms_err_y_percent"] for item in xsuite_turns],
        ".-",
        color="tab:orange",
        label="PyEC4XS",
    )
    ax.set_title("RMS relative error in y'")
    ax.set_xlabel("Turn")
    ax.set_ylabel("error [%]")
    ax.grid(True)
    ax.legend(loc="best")

    fig.tight_layout()
    out_path = OUTPUT_DIR / f"{case_name}_side_by_side.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def _case_summary(xsuite_turns, pyht_turns):
    return {
        "xsuite": {
            "max_rms_err_x_percent": float(
                max(item["rms_err_x_percent"] for item in xsuite_turns)
            ),
            "max_rms_err_y_percent": float(
                max(item["rms_err_y_percent"] for item in xsuite_turns)
            ),
            "mean_rms_err_x_percent": float(
                np.mean([item["rms_err_x_percent"] for item in xsuite_turns])
            ),
            "mean_rms_err_y_percent": float(
                np.mean([item["rms_err_y_percent"] for item in xsuite_turns])
            ),
        },
        "pyec4pyht": {
            "max_rms_err_x_percent": float(
                max(item["rms_err_x_percent"] for item in pyht_turns)
            ),
            "max_rms_err_y_percent": float(
                max(item["rms_err_y_percent"] for item in pyht_turns)
            ),
            "mean_rms_err_x_percent": float(
                np.mean([item["rms_err_x_percent"] for item in pyht_turns])
            ),
            "mean_rms_err_y_percent": float(
                np.mean([item["rms_err_y_percent"] for item in pyht_turns])
            ),
        },
    }


def main():
    set_reproducible_seed(12345)

    comparison_summary = {}
    for case_name in (
        "001_compare_kick_against_headtail",
        "002_compare_kick_and_betatron_motion",
    ):
        cfg = _build_case_config(case_name)
        reference = load_legacy_reference(
            filename=cfg["benchmark_file"],
            n_part_per_turn=cfg["n_reference_particles"],
            n_kicks=cfg["n_kicks"],
        )
        xsuite_turns = _run_xsuite_case(cfg, reference)
        pyht_turns = _run_pyec4pyht_case(cfg, reference)
        plot_path = _make_case_plot(case_name, xsuite_turns, pyht_turns)
        comparison_summary[case_name] = {
            "benchmark_file": str(cfg["benchmark_file"]),
            "comparison_mode": cfg["comparison_mode"],
            "n_reference_particles": cfg["n_reference_particles"],
            "plot": str(plot_path),
            **_case_summary(xsuite_turns, pyht_turns),
        }

    write_json(OUTPUT_DIR / "comparison_summary.json", comparison_summary)
    print(json.dumps(comparison_summary, indent=2))


if __name__ == "__main__":
    main()
