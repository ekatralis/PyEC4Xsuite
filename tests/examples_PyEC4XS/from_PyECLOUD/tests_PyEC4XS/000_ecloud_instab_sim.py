from pathlib import Path

import numpy as np
import xtrack as xt

from common import (
    build_sps_ring,
    build_tracking_line,
    compute_sigmas,
    ensure_output_dir,
    generate_matched_bunch,
    make_pyht_compatible_slicer,
    set_reproducible_seed,
    write_json,
)


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = ensure_output_dir(SCRIPT_DIR / "output" / "000_ecloud_instab_sim")


def main():
    set_reproducible_seed(12345)

    n_segments = 10
    n_turns = 1
    epsn_x = 2.5e-6
    epsn_y = 2.5e-6
    sigma_z = 0.11
    intensity = 1.5e11
    n_macroparticles = 300000

    init_unif_edens = 1.0e11
    n_mp_ele_init = 100000
    n_mp_max = n_mp_ele_init * 4.0

    ring = build_sps_ring("Q26-injection", n_segments=n_segments)
    sigma_x, sigma_y = compute_sigmas(
        ring["injection_optics"], ring["betagamma"], epsn_x, epsn_y
    )

    x_aper = 20.0 * sigma_x
    y_aper = 20.0 * sigma_y
    dh_sc = 2.0 * x_aper / 128.0
    nel_mp_ref_0 = init_unif_edens * 4.0 * x_aper * y_aper / n_mp_ele_init

    slicer = make_pyht_compatible_slicer(n_slices=64, z_cut=2.0 * sigma_z)

    from PyEC4XS import xEcloud

    ecloud = xEcloud(
        L_ecloud=ring["circumference"] / n_segments,
        slicer=slicer,
        Dt_ref=25e-12,
        pyecl_input_folder=str((SCRIPT_DIR / "../../../../drift_sim").resolve()),
        x_aper=x_aper,
        y_aper=y_aper,
        Dh_sc=dh_sc,
        init_unif_edens_flag=1,
        init_unif_edens=init_unif_edens,
        N_mp_max=n_mp_max,
        nel_mp_ref_0=nel_mp_ref_0,
    )

    line = build_tracking_line(
        ring,
        ecloud=ecloud,
        aperture=xt.LimitEllipse(a=x_aper, b=y_aper),
    )

    bunch = generate_matched_bunch(
        ring_data=ring,
        n_macroparticles=n_macroparticles,
        intensity=intensity,
        epsn_x=epsn_x,
        epsn_y=epsn_y,
        sigma_z=sigma_z,
    )

    mean_x = []
    mean_y = []
    mean_px = []
    mean_py = []
    alive = []

    for _ in range(n_turns):
        line.track(bunch, num_turns=1)
        active = np.array(bunch.state) > 0
        mean_x.append(float(np.mean(np.array(bunch.x)[active])))
        mean_y.append(float(np.mean(np.array(bunch.y)[active])))
        mean_px.append(float(np.mean(np.array(bunch.px)[active])))
        mean_py.append(float(np.mean(np.array(bunch.py)[active])))
        alive.append(int(np.sum(active)))

    np.savez_compressed(
        OUTPUT_DIR / "turn_by_turn.npz",
        mean_x=np.array(mean_x),
        mean_y=np.array(mean_y),
        mean_px=np.array(mean_px),
        mean_py=np.array(mean_py),
        alive=np.array(alive, dtype=int),
    )

    write_json(
        OUTPUT_DIR / "summary.json",
        {
            "alive_after_last_turn": alive[-1],
            "initial_macroparticles": n_macroparticles,
            "machine_configuration": "Q26-injection",
            "mean_px_after_last_turn": mean_px[-1],
            "mean_py_after_last_turn": mean_py[-1],
            "n_segments": n_segments,
            "n_turns": n_turns,
            "sigma_z_m": sigma_z,
        },
    )

    print(f"Saved outputs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
