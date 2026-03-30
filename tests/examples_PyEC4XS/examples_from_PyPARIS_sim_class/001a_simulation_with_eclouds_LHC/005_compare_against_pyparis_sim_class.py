import importlib
import json
import os
import shutil
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

from PyEC4XS import myfilemanager as mfm


OUTPUT_DIR = SCRIPT_DIR / "comparison_against_pyparis_sim_class"

RUN_ARTIFACTS = [
    "bunch_evolution_00.h5",
    "bunch_status_part00.h5",
    "multigrid_config_dip.pkl",
    "multigrid_config_dip.txt",
    "pyparislog.txt",
    "sim_param.pkl",
    "simulation_status.sta",
    "slice_evolution_00.h5",
]


def _prepare_run_dir(path):
    path.mkdir(parents=True, exist_ok=True)
    for artifact in RUN_ARTIFACTS:
        target = path / artifact
        if target.exists():
            target.unlink()


def _write_param_file(target_dir):
    template = (SCRIPT_DIR / "Simulation_parameters.py").read_text(encoding="utf-8")
    template = template.replace(
        '"!folder_of_this_file!/../../../../PyPARIS_example/LHC_chm_ver.mat"',
        repr(str((REPO_ROOT / "PyPARIS_example" / "LHC_chm_ver.mat").resolve())),
    )
    template = template.replace(
        '"!folder_of_this_file!/../../../../PyPARIS_example/pyecloud_config"',
        repr(str((REPO_ROOT / "PyPARIS_example" / "pyecloud_config").resolve())),
    )
    (target_dir / "Simulation_parameters.py").write_text(template, encoding="utf-8")


def _run_pyec4xs(run_dir):
    _prepare_run_dir(run_dir)
    _write_param_file(run_dir)
    module = importlib.import_module("PyEC4XS.Simulation")
    previous_cwd = Path.cwd()
    try:
        os.chdir(run_dir)
        ring = module.get_serial_CPUring(param_file=str(run_dir / "Simulation_parameters.py"))
        ring.run()
    finally:
        os.chdir(previous_cwd)


def _run_pyparis_sim_class(run_dir):
    _prepare_run_dir(run_dir)
    _write_param_file(run_dir)
    module = importlib.import_module("PyPARIS_sim_class.Simulation")
    previous_cwd = Path.cwd()
    try:
        os.chdir(run_dir)
        ring = module.get_serial_CPUring()
        ring.run()
    finally:
        os.chdir(previous_cwd)


def _load_monitors(run_dir):
    bunch = mfm.monitorh5_to_obj(str(run_dir / "bunch_evolution_00.h5"))
    slices = mfm.monitorh5_to_obj(str(run_dir / "slice_evolution_00.h5"), key="Slices")
    return bunch, slices


def _plot_monitors(pyht_bunch, xsuite_bunch, pyht_slices, xsuite_slices):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    turns = np.arange(len(pyht_bunch.mean_x))
    axes[0, 0].plot(turns, pyht_bunch.mean_x, ".-", color="tab:blue", label="PyPARIS_sim_class")
    axes[0, 0].plot(turns, xsuite_bunch.mean_x, ".-", color="tab:orange", label="PyEC4XS")
    axes[0, 0].set_title("mean_x")
    axes[0, 0].set_xlabel("Turn")
    axes[0, 0].grid(True)
    axes[0, 0].legend(loc="best")

    axes[0, 1].plot(turns, pyht_bunch.mean_y, ".-", color="tab:blue", label="PyPARIS_sim_class")
    axes[0, 1].plot(turns, xsuite_bunch.mean_y, ".-", color="tab:orange", label="PyEC4XS")
    axes[0, 1].set_title("mean_y")
    axes[0, 1].set_xlabel("Turn")
    axes[0, 1].grid(True)
    axes[0, 1].legend(loc="best")

    axes[1, 0].plot(turns, pyht_bunch.mean_z, ".-", color="tab:blue", label="PyPARIS_sim_class")
    axes[1, 0].plot(turns, xsuite_bunch.mean_z, ".-", color="tab:orange", label="PyEC4XS")
    axes[1, 0].set_title("mean_z")
    axes[1, 0].set_xlabel("Turn")
    axes[1, 0].grid(True)
    axes[1, 0].legend(loc="best")

    pyht_slice_max = np.max(pyht_slices.n_macroparticles_per_slice, axis=0)
    xsuite_slice_max = np.max(xsuite_slices.n_macroparticles_per_slice, axis=0)
    axes[1, 1].plot(turns, pyht_slice_max, ".-", color="tab:blue", label="PyPARIS_sim_class")
    axes[1, 1].plot(turns, xsuite_slice_max, ".-", color="tab:orange", label="PyEC4XS")
    axes[1, 1].set_title("max slice population")
    axes[1, 1].set_xlabel("Turn")
    axes[1, 1].grid(True)
    axes[1, 1].legend(loc="best")

    fig.tight_layout()
    out_path = OUTPUT_DIR / "pyec4xs_vs_pyparis_sim_class.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def main():
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pyht_dir = OUTPUT_DIR / "pyparis_sim_class"
    xsuite_dir = OUTPUT_DIR / "pyec4xs"

    _run_pyparis_sim_class(pyht_dir)
    _run_pyec4xs(xsuite_dir)

    pyht_bunch, pyht_slices = _load_monitors(pyht_dir)
    xsuite_bunch, xsuite_slices = _load_monitors(xsuite_dir)

    plot_path = _plot_monitors(pyht_bunch, xsuite_bunch, pyht_slices, xsuite_slices)

    summary = {
        "plot": str(plot_path),
        "mean_x_max_abs_diff": float(np.max(np.abs(pyht_bunch.mean_x - xsuite_bunch.mean_x))),
        "mean_y_max_abs_diff": float(np.max(np.abs(pyht_bunch.mean_y - xsuite_bunch.mean_y))),
        "mean_z_max_abs_diff": float(np.max(np.abs(pyht_bunch.mean_z - xsuite_bunch.mean_z))),
        "max_slice_population_abs_diff": float(
            np.max(
                np.abs(
                    np.max(pyht_slices.n_macroparticles_per_slice, axis=0)
                    - np.max(xsuite_slices.n_macroparticles_per_slice, axis=0)
                )
            )
        ),
        "n_turn_samples": int(len(pyht_bunch.mean_x)),
    }

    with open(OUTPUT_DIR / "comparison_summary.json", "w", encoding="utf-8") as fid:
        json.dump(summary, fid, indent=2, sort_keys=True)

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
