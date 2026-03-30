import json
import os
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "pyec4xs_mpl"))

from PyEC4XS import myfilemanager as mfm


def _finite_fraction(values):
    values = np.asarray(values, dtype=float)
    return float(np.mean(np.isfinite(values)))


def main():
    bunch = mfm.monitorh5_to_obj(str(SCRIPT_DIR / "bunch_evolution_00.h5"))
    slices = mfm.monitorh5_to_obj(str(SCRIPT_DIR / "slice_evolution_00.h5"), key="Slices")

    with h5py.File(SCRIPT_DIR / "bunch_status_part00.h5", "r") as fid:
        bunch_buffer_size = int(np.array(fid["bunch"]).size)

    summary = {
        "bunch_buffer_size": bunch_buffer_size,
        "finite_fraction_mean_x": _finite_fraction(bunch.mean_x),
        "finite_fraction_mean_y": _finite_fraction(bunch.mean_y),
        "finite_fraction_mean_z": _finite_fraction(bunch.mean_z),
        "initial_macroparticles": int(bunch.macroparticlenumber[0]),
        "last_mean_x": float(bunch.mean_x[-1]),
        "last_mean_y": float(bunch.mean_y[-1]),
        "last_mean_z": float(bunch.mean_z[-1]),
        "max_slice_population": float(np.max(slices.n_macroparticles_per_slice)),
        "n_turn_samples": int(len(bunch.mean_x)),
    }

    output = SCRIPT_DIR / "checks_summary.json"
    with open(output, "w", encoding="utf-8") as fid:
        json.dump(summary, fid, indent=2, sort_keys=True)

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
