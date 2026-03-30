from pathlib import Path

import numpy as np

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
OUTPUT_DIR = ensure_output_dir(
    SCRIPT_DIR / "output" / "002_compare_kick_and_betatron_motion"
)


def main():
    set_reproducible_seed(12345)

    filename = (
        LEGACY_TEST_ROOT
        / "headtail_for_test"
        / "test_protons"
        / "SPS_Q20_proton_check_dipole_betatron_motion_20150212_prb.dat"
    )
    n_kicks = 3
    n_reference_particles = 5000
    epsn_x = 2.5e-6
    epsn_y = 2.5e-6
    sigma_z = 0.2
    intensity = 1.15e11
    n_macroparticles = 300000
    init_unif_edens = 2.0e11
    n_mp_ele_init = 100000
    n_mp_max = n_mp_ele_init * 4.0

    reference = load_legacy_reference(
        filename=filename,
        n_part_per_turn=n_reference_particles,
        n_kicks=n_kicks,
    )
    n_turns = reference["x"].shape[0]

    ring = build_sps_ring(
        "Q20-injection",
        n_segments=n_kicks,
        accQ_x=20.13,
        accQ_y=20.18,
    )
    sigma_x, sigma_y = compute_sigmas(
        ring["injection_optics"], ring["betagamma"], epsn_x, epsn_y
    )
    x_aper = 20.0 * sigma_x
    y_aper = 20.0 * sigma_y
    dh_sc = 2.0 * x_aper / 128.0 / 2.0
    nel_mp_ref_0 = init_unif_edens * 4.0 * x_aper * y_aper / n_mp_ele_init

    slicer = make_pyht_compatible_slicer(n_slices=64, z_cut=3.0 * sigma_z)

    from PyEC4XS import xEcloud

    ecloud = xEcloud(
        L_ecloud=ring["circumference"] / n_kicks,
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
        B_multip=[0.5],
    )
    line = build_tracking_line(ring, ecloud=ecloud)

    bunch = generate_matched_bunch(
        ring_data=ring,
        n_macroparticles=n_macroparticles,
        intensity=intensity,
        epsn_x=epsn_x,
        epsn_y=epsn_y,
        sigma_z=sigma_z,
    )
    inject_legacy_reference_particles(bunch, reference)

    before = snapshot_reference_particles(bunch, n_reference_particles)
    rms_err_x = []
    rms_err_y = []

    for turn_idx in range(n_turns - 1):
        line.track(bunch, num_turns=1)
        after = snapshot_reference_particles(bunch, n_reference_particles)

        actual_dx = after["px"] - before["px"]
        actual_dy = after["py"] - before["py"]
        ref_dx = reference["xp"][turn_idx + 1, :] - reference["xp"][turn_idx, :]
        ref_dy = reference["yp"][turn_idx + 1, :] - reference["yp"][turn_idx, :]

        err_x = relative_error_percent(actual_dx, ref_dx)
        err_y = relative_error_percent(actual_dy, ref_dy)
        rms_err_x.append(float(np.std(err_x)))
        rms_err_y.append(float(np.std(err_y)))

        before = after

    np.savez_compressed(
        OUTPUT_DIR / "errors.npz",
        rms_err_x_percent=np.array(rms_err_x),
        rms_err_y_percent=np.array(rms_err_y),
    )

    write_json(
        OUTPUT_DIR / "summary.json",
        {
            "benchmark": filename.name,
            "max_rms_err_x_percent": max(rms_err_x),
            "max_rms_err_y_percent": max(rms_err_y),
            "mean_rms_err_x_percent": float(np.mean(rms_err_x)),
            "mean_rms_err_y_percent": float(np.mean(rms_err_y)),
            "n_reference_particles": n_reference_particles,
            "turns_compared": n_turns - 1,
        },
    )

    print(f"Saved outputs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
