import json
import os
import random
import sys
import tempfile
from pathlib import Path

import numpy as np

_mpl_dir = Path(tempfile.gettempdir()) / "pyec4xs_mpl"
_mpl_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_dir))

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import xfields as xf
import xpart as xp
import xtrack as xt
from scipy.constants import c


LEGACY_TEST_ROOT = (
    REPO_ROOT / "tests" / "legacy_tests_PyEC4PyHT" / "from_PyECLOUD" / "tests_PyEC4PyHT"
)

SPS_CONFIGS = {
    "Q26-injection": {
        "circumference": 1100.0 * 2.0 * np.pi,
        "p0c_eV": 26.0e9,
        "h_rf": 4620,
        "alpha": 0.00192,
        "accQ_x": 26.13,
        "accQ_y": 26.18,
        "beta_x": 42.0,
        "beta_y": 42.0,
        "V_RF": 2.0e6,
        "dphi_RF": 0.0,
    },
    "Q20-injection": {
        "circumference": 1100.0 * 2.0 * np.pi,
        "p0c_eV": 26.0e9,
        "h_rf": 4620,
        "alpha": 0.00308642,
        "accQ_x": 20.13,
        "accQ_y": 20.18,
        "beta_x": 54.6,
        "beta_y": 54.6,
        "V_RF": 5.75e6,
        "dphi_RF": 0.0,
    },
}


def set_reproducible_seed(seed=12345):
    np.random.seed(seed)
    random.seed(seed)


def ensure_output_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def make_pyht_compatible_slicer(n_slices, z_cut):
    template = xf.TempSlicer(n_slices=n_slices, sigma_z=1.0, mode="unibin")
    template_half_range = np.max(np.abs(template.bin_edges))
    sigma_z_equiv = float(z_cut) / float(template_half_range)
    return xf.TempSlicer(n_slices=n_slices, sigma_z=sigma_z_equiv, mode="unibin")


def build_sps_ring(machine_configuration, n_segments, accQ_x=None, accQ_y=None):
    cfg = dict(SPS_CONFIGS[machine_configuration])
    if accQ_x is not None:
        cfg["accQ_x"] = float(accQ_x)
    if accQ_y is not None:
        cfg["accQ_y"] = float(accQ_y)

    particle_ref = xp.Particles(
        p0c=cfg["p0c_eV"],
        mass0=xp.PROTON_MASS_EV,
        q0=1.0,
    )
    beta0 = float(particle_ref.beta0[0])
    gamma0 = float(particle_ref.gamma0[0])
    circumference = float(cfg["circumference"])
    frequency_rf = cfg["h_rf"] * beta0 * c / circumference

    transverse_segments = []
    for ii in range(int(n_segments)):
        seg = xt.LineSegmentMap(
            length=circumference / int(n_segments),
            qx=cfg["accQ_x"] / int(n_segments),
            qy=cfg["accQ_y"] / int(n_segments),
            betx=cfg["beta_x"],
            bety=cfg["beta_y"],
            alfx=0.0,
            alfy=0.0,
            dx=0.0,
            dy=0.0,
            dqx=0.0,
            dqy=0.0,
            longitudinal_mode="frozen",
        )
        seg.name = f"sps_segment_{ii:02d}"
        transverse_segments.append(seg)

    longitudinal_map = xt.LineSegmentMap(
        length=0.0,
        qx=0.0,
        qy=0.0,
        betx=1.0,
        bety=1.0,
        longitudinal_mode="linear_fixed_rf",
        momentum_compaction_factor=cfg["alpha"],
        slippage_length=circumference,
        voltage_rf=[cfg["V_RF"]],
        frequency_rf=[frequency_rf],
        lag_rf=[cfg["dphi_RF"]],
    )

    matching_map = xt.LineSegmentMap(
        length=circumference,
        qx=cfg["accQ_x"],
        qy=cfg["accQ_y"],
        betx=cfg["beta_x"],
        bety=cfg["beta_y"],
        alfx=0.0,
        alfy=0.0,
        dx=0.0,
        dy=0.0,
        dqx=0.0,
        dqy=0.0,
        longitudinal_mode="linear_fixed_rf",
        momentum_compaction_factor=cfg["alpha"],
        slippage_length=circumference,
        voltage_rf=[cfg["V_RF"]],
        frequency_rf=[frequency_rf],
        lag_rf=[cfg["dphi_RF"]],
    )
    matching_line = xt.Line(
        elements=[matching_map],
        particle_ref=particle_ref.copy(),
    )
    matching_line.build_tracker()

    return {
        "config": cfg,
        "particle_ref": particle_ref,
        "circumference": circumference,
        "betagamma": beta0 * gamma0,
        "injection_optics": {
            "beta_x": cfg["beta_x"],
            "beta_y": cfg["beta_y"],
        },
        "transverse_segments": transverse_segments,
        "longitudinal_map": longitudinal_map,
        "matching_line": matching_line,
    }


def compute_sigmas(injection_optics, betagamma, epsn_x, epsn_y):
    sigma_x = np.sqrt(injection_optics["beta_x"] * epsn_x / betagamma)
    sigma_y = np.sqrt(injection_optics["beta_y"] * epsn_y / betagamma)
    return float(sigma_x), float(sigma_y)


def build_tracking_line(ring_data, ecloud, aperture=None):
    elements = []
    for seg in ring_data["transverse_segments"]:
        elements.append(seg)
        elements.append(ecloud)
    elements.append(ring_data["longitudinal_map"])
    if aperture is not None:
        elements.append(aperture)

    line = xt.Line(
        elements=elements,
        particle_ref=ring_data["particle_ref"].copy(),
    )
    line.build_tracker()
    return line


def generate_matched_bunch(
    ring_data,
    n_macroparticles,
    intensity,
    epsn_x,
    epsn_y,
    sigma_z,
):
    return xp.generate_matched_gaussian_bunch(
        line=ring_data["matching_line"],
        num_particles=n_macroparticles,
        total_intensity_particles=intensity,
        nemitt_x=epsn_x,
        nemitt_y=epsn_y,
        sigma_z=sigma_z,
    )


def load_legacy_reference(filename, n_part_per_turn, n_kicks):
    raw = np.loadtxt(filename)
    shape = (-1, n_part_per_turn)
    step = slice(None, None, int(n_kicks))
    return {
        "particle_id": np.reshape(raw[:, 0], shape)[step, :],
        "x": np.reshape(raw[:, 1], shape)[step, :],
        "xp": np.reshape(raw[:, 2], shape)[step, :],
        "y": np.reshape(raw[:, 3], shape)[step, :],
        "yp": np.reshape(raw[:, 4], shape)[step, :],
        "z": np.reshape(raw[:, 5], shape)[step, :],
        "dp": np.reshape(raw[:, 6], shape)[step, :],
    }


def inject_legacy_reference_particles(particles, reference):
    n_ref = reference["x"].shape[1]
    particles.x[:n_ref] = reference["x"][0, :]
    particles.px[:n_ref] = reference["xp"][0, :]
    particles.y[:n_ref] = reference["y"][0, :]
    particles.py[:n_ref] = reference["yp"][0, :]
    particles.zeta[:n_ref] = reference["z"][0, :]
    particles.delta[:n_ref] = reference["dp"][0, :]


def snapshot_reference_particles(particles, n_reference):
    mask = (np.array(particles.state) > 0) & (np.array(particles.particle_id) < n_reference)
    particle_id = np.array(particles.particle_id[mask], dtype=int)
    order = np.argsort(particle_id)
    return {
        "particle_id": particle_id[order],
        "px": np.array(particles.px[mask], dtype=float)[order],
        "py": np.array(particles.py[mask], dtype=float)[order],
        "zeta": np.array(particles.zeta[mask], dtype=float)[order],
    }


def relative_error_percent(actual, reference):
    scale = max(float(np.std(reference)), 1e-30)
    return 100.0 * np.abs(np.asarray(actual) - np.asarray(reference)) / scale


def write_json(path, payload):
    path = Path(path)

    def _convert(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, (np.integer,)):
            return int(value)
        return value

    with open(path, "w", encoding="utf-8") as fid:
        json.dump(payload, fid, indent=2, sort_keys=True, default=_convert)
