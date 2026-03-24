import os
import pickle
import time
from dataclasses import dataclass

import h5py
import numpy as np
import xfields as xf
import xobjects as xo
import xpart as xp
import xtrack as xt

import PyPARIS.share_segments as shs

from .LHC_custom import LHC
from .PyEC4XS import IterableSlices, xEcloud
from .Save_Load_Status import SimulationStatus
from .sim_config_manager import SimConfig


def make_pyht_compatible_slicer(n_slices, z_cut):
    template = xf.TempSlicer(n_slices=n_slices, sigma_z=1.0, mode="unibin")
    template_half_range = np.max(np.abs(template.bin_edges))
    sigma_z_equiv = float(z_cut) / float(template_half_range)
    return xf.TempSlicer(n_slices=n_slices, sigma_z=sigma_z_equiv, mode="unibin")


def _weight_array(particles):
    weight = getattr(particles, "weight", 1.0)
    if np.shape(weight) == ():
        return np.full(len(particles.x), float(weight), dtype=float)
    return np.array(weight, dtype=float, copy=False)


def _active_mask(particles):
    return np.array(particles.state > 0, dtype=bool, copy=False)


def _empty_particles_from_reference(reference):
    return xp.Particles(
        p0c=float(reference.p0c[0]),
        mass0=float(reference.mass0),
        q0=float(reference.q0),
        x=np.zeros(0),
        px=np.zeros(0),
        y=np.zeros(0),
        py=np.zeros(0),
        zeta=np.zeros(0),
        delta=np.zeros(0),
        weight=np.zeros(0),
        _context=reference._context,
    )


def _filter_particles_or_empty(particles, mask, particle_ref):
    mask = np.array(mask, dtype=bool, copy=False)
    if np.any(mask):
        return particles.filter(mask)
    return _empty_particles_from_reference(particle_ref)


def _pickle_to_float_buffer(payload):
    pickled = pickle.dumps(payload, protocol=2)
    bytes_arr = np.frombuffer(pickled, dtype="S1")
    pad = (-len(bytes_arr)) % 8
    if pad:
        bytes_arr = np.concatenate([bytes_arr, np.zeros(pad, dtype="S1")])
    float_arr = np.frombuffer(bytes_arr.tobytes(), dtype=np.float64)
    return np.concatenate([np.array([len(pickled)], dtype=np.float64), float_arr])


def _float_buffer_to_pickle(buf):
    n_bytes = int(buf[0])
    raw = np.frombuffer(buf[1:].tobytes(), dtype="S1")[:n_bytes]
    return pickle.loads(raw.tobytes())


def _intensity_of_particles(particles):
    if len(particles.x) == 0:
        return 0.0
    return float(np.sum(_weight_array(particles)))


def _covariance(coord, mom):
    if len(coord) == 0:
        return 0.0
    coord = np.asarray(coord, dtype=float)
    mom = np.asarray(mom, dtype=float)
    return float(np.mean((coord - np.mean(coord)) * (mom - np.mean(mom))))


def _geom_emittance(coord, mom, dp=None):
    if len(coord) == 0:
        return 0.0

    sigma11 = _covariance(coord, coord)
    sigma12 = _covariance(coord, mom)
    sigma22 = _covariance(mom, mom)

    if dp is not None and len(dp) == len(coord):
        cov_dp2 = _covariance(dp, dp)
        if cov_dp2 != 0.0:
            cov_u_dp = _covariance(coord, dp)
            cov_up_dp = _covariance(mom, dp)
            sigma11 -= cov_u_dp * cov_u_dp / cov_dp2
            sigma12 -= cov_u_dp * cov_up_dp / cov_dp2
            sigma22 -= cov_up_dp * cov_up_dp / cov_dp2

    value = sigma11 * sigma22 - sigma12 * sigma12
    return float(np.sqrt(max(value, 0.0)))


def _normalized_emittance(particles, plane):
    active = _active_mask(particles)
    if not np.any(active):
        return 0.0

    beta_gamma = float(particles.beta0[active][0] * particles.gamma0[active][0])
    delta = np.array(particles.delta[active], dtype=float)
    if plane == "x":
        eps_g = _geom_emittance(
            np.array(particles.x[active], dtype=float),
            np.array(particles.px[active], dtype=float),
            dp=delta,
        )
    elif plane == "y":
        eps_g = _geom_emittance(
            np.array(particles.y[active], dtype=float),
            np.array(particles.py[active], dtype=float),
            dp=delta,
        )
    else:
        raise ValueError(f"Invalid plane {plane}")

    return eps_g * beta_gamma


def _normalized_longitudinal_emittance(particles):
    active = _active_mask(particles)
    if not np.any(active):
        return 0.0

    zeta = np.array(particles.zeta[active], dtype=float)
    delta = np.array(particles.delta[active], dtype=float)
    eps_g = _geom_emittance(zeta, delta, dp=None)
    return float(4.0 * np.pi * eps_g * float(particles.p0c[active][0]) / 299792458.0)


def _safe_mean(values):
    if len(values) == 0:
        return 0.0
    return float(np.mean(values))


def _safe_std(values):
    if len(values) == 0:
        return 0.0
    return float(np.std(values))


def _bunch_stat_value(particles, stat):
    active = _active_mask(particles)
    if not np.any(active):
        return 0.0

    attr_map = {
        "mean_x": "x",
        "mean_xp": "px",
        "mean_y": "y",
        "mean_yp": "py",
        "mean_z": "zeta",
        "mean_dp": "delta",
        "sigma_x": "x",
        "sigma_y": "y",
        "sigma_z": "zeta",
        "sigma_dp": "delta",
    }

    if stat in attr_map:
        values = np.array(getattr(particles, attr_map[stat])[active], dtype=float)
        if stat.startswith("mean"):
            return _safe_mean(values)
        return _safe_std(values)

    if stat == "epsn_x":
        return _normalized_emittance(particles, "x")
    if stat == "epsn_y":
        return _normalized_emittance(particles, "y")
    if stat == "epsn_z":
        return _normalized_longitudinal_emittance(particles)
    if stat == "macroparticlenumber":
        return float(np.sum(active))

    raise ValueError(f"Unsupported bunch statistic {stat!r}")


def _slice_statistics(particles, slicer, slice_stats_to_store):
    active = _active_mask(particles)
    n_slices = int(slicer.num_slices)
    out = {stat: np.zeros(n_slices, dtype=float) for stat in slice_stats_to_store}
    if not np.any(active):
        return out

    slice_index = np.array(slicer.get_slice_indices(particles), dtype=int)
    x = np.array(particles.x, dtype=float)
    px = np.array(particles.px, dtype=float)
    y = np.array(particles.y, dtype=float)
    py = np.array(particles.py, dtype=float)
    zeta = np.array(particles.zeta, dtype=float)
    delta = np.array(particles.delta, dtype=float)

    for ii in range(n_slices):
        mask = active & (slice_index == ii)
        n_part = int(np.sum(mask))
        if "n_macroparticles_per_slice" in out:
            out["n_macroparticles_per_slice"][ii] = float(n_part)
        if n_part == 0:
            continue

        if "mean_x" in out:
            out["mean_x"][ii] = _safe_mean(x[mask])
        if "mean_xp" in out:
            out["mean_xp"][ii] = _safe_mean(px[mask])
        if "mean_y" in out:
            out["mean_y"][ii] = _safe_mean(y[mask])
        if "mean_yp" in out:
            out["mean_yp"][ii] = _safe_mean(py[mask])
        if "mean_z" in out:
            out["mean_z"][ii] = _safe_mean(zeta[mask])
        if "mean_dp" in out:
            out["mean_dp"][ii] = _safe_mean(delta[mask])

        if "sigma_x" in out:
            out["sigma_x"][ii] = _safe_std(x[mask])
        if "sigma_y" in out:
            out["sigma_y"][ii] = _safe_std(y[mask])
        if "sigma_z" in out:
            out["sigma_z"][ii] = _safe_std(zeta[mask])
        if "sigma_dp" in out:
            out["sigma_dp"][ii] = _safe_std(delta[mask])

        if n_part > 1:
            if "epsn_x" in out:
                out["epsn_x"][ii] = (
                    _geom_emittance(x[mask], px[mask], dp=delta[mask])
                    * float(particles.beta0[mask][0] * particles.gamma0[mask][0])
                )
            if "epsn_y" in out:
                out["epsn_y"][ii] = (
                    _geom_emittance(y[mask], py[mask], dp=delta[mask])
                    * float(particles.beta0[mask][0] * particles.gamma0[mask][0])
                )
            if "epsn_z" in out:
                out["epsn_z"][ii] = (
                    4.0
                    * np.pi
                    * _geom_emittance(zeta[mask], delta[mask], dp=None)
                    * float(particles.p0c[mask][0])
                    / 299792458.0
                )

    return out


@dataclass
class SlicePiece:
    particles: xp.Particles
    slice_data: dict


PYHEADTAIL_BUNCH_STATS = [
    "mean_x",
    "mean_xp",
    "mean_y",
    "mean_yp",
    "mean_z",
    "mean_dp",
    "sigma_x",
    "sigma_y",
    "sigma_z",
    "sigma_dp",
    "epsn_x",
    "epsn_y",
    "epsn_z",
    "macroparticlenumber",
]

PYHEADTAIL_SLICE_STATS = [
    "mean_x",
    "mean_xp",
    "mean_y",
    "mean_yp",
    "mean_z",
    "mean_dp",
    "sigma_x",
    "sigma_y",
    "sigma_z",
    "sigma_dp",
    "epsn_x",
    "epsn_y",
    "epsn_z",
    "n_macroparticles_per_slice",
]

PYHT_TO_XSUITE_STAT = {
    "mean_x": "mean_x",
    "mean_xp": "mean_px",
    "mean_y": "mean_y",
    "mean_yp": "mean_py",
    "mean_z": "mean_zeta",
    "mean_dp": "mean_delta",
    "sigma_x": "sigma_x",
    "sigma_y": "sigma_y",
    "sigma_z": "sigma_zeta",
    "sigma_dp": "sigma_delta",
    "epsn_x": "epsn_x",
    "epsn_y": "epsn_y",
    "epsn_z": "epsn_zeta",
    "macroparticlenumber": "num_particles",
    "n_macroparticles_per_slice": "num_particles",
}


def _slicer_zeta_range(slicer):
    if hasattr(slicer, "bin_edges"):
        edges = np.array(slicer.bin_edges, dtype=float)
        return float(np.min(edges)), float(np.max(edges))
    if hasattr(slicer, "zeta_range"):
        zeta_range = np.array(slicer.zeta_range, dtype=float)
        return float(np.min(zeta_range)), float(np.max(zeta_range))
    raise ValueError("Cannot infer zeta_range from slicer")


def _mapped_xsuite_stats(pyht_stats):
    mapped = []
    for stat in pyht_stats:
        if stat not in PYHT_TO_XSUITE_STAT:
            raise ValueError(f"Unsupported monitor statistic {stat!r}")
        xs_stat = PYHT_TO_XSUITE_STAT[stat]
        if xs_stat not in mapped:
            mapped.append(xs_stat)
    return mapped


def _monitor_bunch_key(monitor):
    if monitor.slicer is None or monitor.slicer.bunch_selection is None:
        return 0
    return int(np.atleast_1d(monitor.slicer.bunch_selection)[0])


class XsuiteBunchMonitor:
    def __init__(
        self,
        filename,
        n_turns,
        slicer,
        metadata=None,
        write_buffer_every=3,
        stats_to_store=None,
    ):
        self.filename = f"{filename}.h5"
        self.metadata = metadata or {}
        self.write_buffer_every = int(write_buffer_every)
        self.stats_to_store = list(stats_to_store or PYHEADTAIL_BUNCH_STATS)
        self._buffer = {stat: [] for stat in self.stats_to_store}
        self._written_turns = 0

        with h5py.File(self.filename, "w") as fid:
            fid.attrs["n_turns_expected"] = int(n_turns)
            for key, value in self.metadata.items():
                fid.attrs[key] = str(value)
            grp = fid.create_group("Bunch")
            for stat in sorted(self.stats_to_store):
                grp.create_dataset(
                    stat,
                    shape=(int(n_turns),),
                    compression="gzip",
                    compression_opts=9,
                )

    def dump(self, bunch):
        for stat in self.stats_to_store:
            self._buffer[stat].append(_bunch_stat_value(bunch, stat))

        if len(next(iter(self._buffer.values()))) >= self.write_buffer_every:
            self.flush()

    def flush(self):
        if not self._buffer or len(next(iter(self._buffer.values()))) == 0:
            return

        n_entries = len(next(iter(self._buffer.values())))
        start = self._written_turns
        stop = start + n_entries

        with h5py.File(self.filename, "a") as fid:
            grp = fid["Bunch"]
            for stat in self.stats_to_store:
                grp[stat][start:stop] = np.array(self._buffer[stat], dtype=float)

        self._written_turns = stop
        self._buffer = {stat: [] for stat in self.stats_to_store}

    def close(self):
        self.flush()


class XsuiteSliceMonitor:
    def __init__(
        self,
        filename,
        n_turns,
        slicer,
        metadata=None,
        write_buffer_every=3,
        bunch_stats_to_store=None,
        slice_stats_to_store=None,
    ):
        self.filename = f"{filename}.h5"
        self.metadata = metadata or {}
        self.write_buffer_every = int(write_buffer_every)
        self.bunch_stats_to_store = list(bunch_stats_to_store or PYHEADTAIL_BUNCH_STATS)
        self.slice_stats_to_store = list(slice_stats_to_store or PYHEADTAIL_SLICE_STATS)
        self._bunch_buffer = {stat: [] for stat in self.bunch_stats_to_store}
        self._slice_buffer = {stat: [] for stat in self.slice_stats_to_store}
        self._written_turns = 0
        self.n_slices = int(slicer.num_slices)
        self.slicer = slicer

        with h5py.File(self.filename, "w") as fid:
            fid.attrs["n_turns_expected"] = int(n_turns)
            for key, value in self.metadata.items():
                fid.attrs[key] = str(value)

            grp_bunch = fid.create_group("Bunch")
            for stat in self.bunch_stats_to_store:
                grp_bunch.create_dataset(
                    stat,
                    shape=(int(n_turns),),
                    compression="gzip",
                    compression_opts=9,
                )

            grp_slice = fid.create_group("Slices")
            for stat in self.slice_stats_to_store:
                grp_slice.create_dataset(
                    stat,
                    shape=(self.n_slices, int(n_turns)),
                    compression="gzip",
                    compression_opts=9,
                )

    def dump(self, bunch):
        for stat in self.bunch_stats_to_store:
            self._bunch_buffer[stat].append(_bunch_stat_value(bunch, stat))

        slice_stats = _slice_statistics(bunch, self.slicer, self.slice_stats_to_store)
        for stat in self.slice_stats_to_store:
            self._slice_buffer[stat].append(slice_stats[stat])

        if len(next(iter(self._bunch_buffer.values()))) >= self.write_buffer_every:
            self.flush()

    def flush(self):
        if not self._bunch_buffer or len(next(iter(self._bunch_buffer.values()))) == 0:
            return

        n_entries = len(next(iter(self._bunch_buffer.values())))
        start = self._written_turns
        stop = start + n_entries

        with h5py.File(self.filename, "a") as fid:
            grp_bunch = fid["Bunch"]
            grp_slice = fid["Slices"]
            for stat in self.bunch_stats_to_store:
                grp_bunch[stat][start:stop] = np.array(self._bunch_buffer[stat], dtype=float)
            for stat in self.slice_stats_to_store:
                grp_slice[stat][:, start:stop] = np.stack(
                    self._slice_buffer[stat], axis=1
                )

        self._written_turns = stop
        self._bunch_buffer = {stat: [] for stat in self.bunch_stats_to_store}
        self._slice_buffer = {stat: [] for stat in self.slice_stats_to_store}

    def close(self):
        self.flush()


class Simulation:
    def __init__(self, param_file="./Simulation_parameters.py"):
        self.pp = SimConfig(param_file)
        self.N_pieces_per_transfer = 1
        self.N_buffer_float_size = 3000000
        self.N_buffer_int_size = 100

    def init_all(self, generate_parent_eclouds=True, install_clouds=True):
        pp = self.pp

        self.N_turns = pp.N_turns
        self.n_slices = pp.n_slices

        self._build_machine()
        self._install_aperture()
        self._install_damper()
        self._install_impedance()
        self._split_machine_among_cores()

        if generate_parent_eclouds:
            self._generate_parent_eclouds()
        else:
            self.parent_eclouds = []

        if install_clouds:
            assert generate_parent_eclouds
            self._install_eclouds_in_machine_part()
        else:
            self.my_list_eclouds = []

        if pp.footprint_mode:
            self._switch_to_footprint_mode()

    def init_master(self, generate_bunch=True, prepare_monitors=True):
        pp = self.pp

        if pp.footprint_mode and pp.N_turns != pp.N_turns_target:
            raise ValueError(
                "In footprint mode you need to set N_turns_target=N_turns_per_run!"
            )

        self._setup_multijob_mode()

        self.slicer = make_pyht_compatible_slicer(pp.n_slices, pp.z_cut)

        if prepare_monitors:
            self._prepare_monitors()

        if generate_bunch:
            self._generate_bunch()
            pieces_to_be_treated = self._extract_slices(self.bunch)
        else:
            pieces_to_be_treated = []

        print("N_turns", self.N_turns)

        if pp.footprint_mode:
            self.recorded_particles = ParticleTrajectories(
                pp.n_macroparticles_for_footprint_track, self.N_turns
            )

        return pieces_to_be_treated

    def init_worker(self):
        pass

    def treat_piece(self, piece):
        if isinstance(piece, SlicePiece):
            particles = piece.particles
            for ele in self.mypart:
                if isinstance(ele, xEcloud) and ele.slice_by_slice_mode:
                    ele.track_slice(particles, piece.slice_data)
                else:
                    ele.track(particles)
            return

        for ele in self.mypart:
            ele.track(piece)

    def finalize_turn_on_master(self, pieces_treated):
        pp = self.pp

        self.bunch = self._merge_pieces(pieces_treated)

        for ele in self.non_parallel_part:
            ele.track(self.bunch)

        self.bunch = _filter_particles_or_empty(
            self.bunch, _active_mask(self.bunch), self.machine.particle_ref
        )

        self.bunch_monitor.dump(self.bunch)
        self.slice_monitor.dump(self.bunch)

        new_pieces_to_be_treated = self._extract_slices(self.bunch)
        orders_to_pass = ["reset_clouds"]

        if pp.footprint_mode:
            self.recorded_particles.dump(self.bunch)

        if self._check_stop_conditions():
            orders_to_pass.append("stop")
            self.SimSt.check_for_resubmit = False

        return orders_to_pass, new_pieces_to_be_treated

    def execute_orders_from_master(self, orders_from_master):
        if "reset_clouds" in orders_from_master:
            for ec in self.my_list_eclouds:
                ec.finalize_and_reinitialize()

    def finalize_simulation(self):
        self.bunch_monitor.close()
        self.slice_monitor.close()

        if self.pp.footprint_mode:
            raise NotImplementedError("footprint_mode is not yet implemented for PyEC4XS")

        self._finalize_multijob_mode()

    def piece_to_buffer(self, piece):
        if piece is None:
            return np.array([-1.0], dtype=np.float64)

        if isinstance(piece, SlicePiece):
            payload = {
                "kind": "slice_piece",
                "particles": piece.particles.to_dict(
                    copy_to_cpu=True,
                    remove_redundant_variables=True,
                    remove_unused_space=False,
                ),
                "slice_data": piece.slice_data,
            }
        elif isinstance(piece, xp.Particles):
            payload = {
                "kind": "particles",
                "particles": piece.to_dict(
                    copy_to_cpu=True,
                    remove_redundant_variables=True,
                    remove_unused_space=False,
                ),
            }
        else:
            raise TypeError(f"Unsupported piece type {type(piece)!r}")

        return _pickle_to_float_buffer(payload)

    def buffer_to_piece(self, buf):
        if buf[0] < 0:
            return None

        payload = _float_buffer_to_pickle(buf)
        kind = payload.get("kind", "particles")
        particles = xp.Particles.from_dict(
            payload["particles"], _context=xo.ContextCpu()
        )

        if kind == "particles":
            return particles
        if kind == "slice_piece":
            return SlicePiece(particles=particles, slice_data=payload["slice_data"])
        if kind == "slice":
            return particles

        raise ValueError(f"Unknown payload kind {kind}")

    def _build_machine(self):
        pp = self.pp

        self.optics_from_pickle = False
        mode = getattr(pp, "machine_class", "LHC_custom")

        if mode != "LHC_custom":
            raise NotImplementedError(
                f"machine_class={mode!r} is not yet implemented for PyEC4XS"
            )

        optics = None
        if pp.optics_pickle_file is not None:
            with open(pp.optics_pickle_file, "rb") as fid:
                optics = pickle.load(fid)
            self.n_kick_smooth = np.sum(["_kick_smooth_" in nn for nn in optics["name"]])
            self.optics_from_pickle = True
        else:
            self.n_kick_smooth = pp.n_segments

        self.machine = LHC(
            n_segments=pp.n_segments,
            machine_configuration=pp.machine_configuration,
            beta_x=pp.beta_x,
            beta_y=pp.beta_y,
            accQ_x=pp.Q_x,
            accQ_y=pp.Q_y,
            Qp_x=pp.Qp_x,
            Qp_y=pp.Qp_y,
            octupole_knob=pp.octupole_knob,
            optics_dict=optics,
            V_RF=pp.V_RF,
            longitudinal_mode=getattr(pp, "longitudinal_mode", "non-linear"),
            _context=xo.ContextCpu(),
        )

        self.n_segments = self.machine.transverse_map.n_segments

        inj_opt = self.machine.transverse_map.get_injection_optics()
        sigma_x_inj = np.sqrt(inj_opt["beta_x"] * pp.epsn_x / self.machine.betagamma)
        sigma_y_inj = np.sqrt(inj_opt["beta_y"] * pp.epsn_y / self.machine.betagamma)

        if not self.optics_from_pickle:
            sigma_x_smooth = sigma_x_inj
            sigma_y_smooth = sigma_y_inj
        else:
            beta_x_smooth = None
            beta_y_smooth = None
            for ele in self.machine.one_turn_map:
                if ele in self.machine.transverse_map and "_kick_smooth_" in getattr(ele, "name1", ""):
                    if beta_x_smooth is None:
                        beta_x_smooth = ele.beta_x1
                        beta_y_smooth = ele.beta_y1
                    elif beta_x_smooth != ele.beta_x1 or beta_y_smooth != ele.beta_y1:
                        raise ValueError("Smooth kicks must have all the same beta")

            if beta_x_smooth is None:
                sigma_x_smooth = None
                sigma_y_smooth = None
            else:
                sigma_x_smooth = np.sqrt(beta_x_smooth * pp.epsn_x / self.machine.betagamma)
                sigma_y_smooth = np.sqrt(beta_y_smooth * pp.epsn_y / self.machine.betagamma)

        self.sigma_x_inj = sigma_x_inj
        self.sigma_y_inj = sigma_y_inj
        self.sigma_x_smooth = sigma_x_smooth
        self.sigma_y_smooth = sigma_y_smooth

        self.n_non_parallelizable = 1

    def _generate_parent_eclouds(self):
        pp = self.pp

        sigma_x_smooth = self.sigma_x_smooth
        sigma_y_smooth = self.sigma_y_smooth

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

        self.target_grid_arcs = target_grid_arcs
        self.parent_eclouds = []

        nel_mp_ref_0 = (
            pp.init_unif_edens_dip * 4.0 * pp.x_aper * pp.y_aper / pp.N_MP_ele_init_dip
        )

        if pp.enable_arc_dip:
            ecloud_dip = xEcloud(
                L_ecloud=self.machine.circumference / self.n_kick_smooth * pp.fraction_device_dip,
                slicer=None,
                Dt_ref=pp.Dt_ref,
                pyecl_input_folder=pp.pyecl_input_folder,
                slice_by_slice_mode=True,
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
                force_interp_at_substeps_interacting_slices=pp.force_interp_at_substeps_interacting_slices,
            )
            self.parent_eclouds.append(ecloud_dip)

        if pp.enable_arc_quad:
            ecloud_quad = xEcloud(
                L_ecloud=self.machine.circumference / self.n_kick_smooth * pp.fraction_device_quad,
                slicer=None,
                Dt_ref=pp.Dt_ref,
                pyecl_input_folder=pp.pyecl_input_folder,
                slice_by_slice_mode=True,
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
                N_mp_max=pp.N_mp_max_quad,
                nel_mp_ref_0=nel_mp_ref_0,
                B_multip=pp.B_multip_quad,
                filename_init_MP_state=pp.filename_init_MP_state_quad,
                enable_kick_x=pp.enable_kick_x,
                enable_kick_y=pp.enable_kick_y,
                force_interp_at_substeps_interacting_slices=pp.force_interp_at_substeps_interacting_slices,
            )
            self.parent_eclouds.append(ecloud_quad)

        if self.ring_of_CPUs.I_am_the_master and pp.enable_arc_dip:
            with open("multigrid_config_dip.txt", "w") as fid:
                if hasattr(ecloud_dip.spacech_ele.PyPICobj, "grids"):
                    fid.write(repr(ecloud_dip.spacech_ele.PyPICobj.grids))
                else:
                    fid.write("Single grid.")

            with open("multigrid_config_dip.pkl", "wb") as fid:
                if hasattr(ecloud_dip.spacech_ele.PyPICobj, "grids"):
                    pickle.dump(ecloud_dip.spacech_ele.PyPICobj.grids, fid)
                else:
                    pickle.dump("Single grid.", fid)

        if self.ring_of_CPUs.I_am_the_master and pp.enable_arc_quad:
            with open("multigrid_config_quad.txt", "w") as fid:
                if hasattr(ecloud_quad.spacech_ele.PyPICobj, "grids"):
                    fid.write(repr(ecloud_quad.spacech_ele.PyPICobj.grids))
                else:
                    fid.write("Single grid.")

            with open("multigrid_config_quad.pkl", "wb") as fid:
                if hasattr(ecloud_quad.spacech_ele.PyPICobj, "grids"):
                    pickle.dump(ecloud_quad.spacech_ele.PyPICobj.grids, fid)
                else:
                    pickle.dump("Single grid.", fid)

    def _install_damper(self):
        pp = self.pp

        if pp.enable_transverse_damper:
            gain_x = 0.0 if not pp.dampingrate_x else 2.0 / pp.dampingrate_x
            gain_y = 0.0 if not pp.dampingrate_y else 2.0 / pp.dampingrate_y
            damper = xf.TransverseDamper(
                gain_x=gain_x,
                gain_y=gain_y,
                zeta_range=(-float(pp.z_cut), float(pp.z_cut)),
                num_slices=int(pp.n_slices),
            )
            self.machine.one_turn_map.append(damper)
            self.n_non_parallelizable += 1
            self.dampers = [damper]
        else:
            self.dampers = []

    def _install_aperture(self):
        pp = self.pp

        apt_xy = xt.LimitEllipse(
            a=pp.target_size_internal_grid_sigma * self.sigma_x_inj,
            b=pp.target_size_internal_grid_sigma * self.sigma_y_inj,
        )
        self.machine.one_turn_map.append(apt_xy)
        self.n_non_parallelizable += 1
        self.apertures = [apt_xy]

    def _split_machine_among_cores(self):
        i_end_parallel = len(self.machine.one_turn_map) - self.n_non_parallelizable

        sharing = shs.ShareSegments(i_end_parallel, self.ring_of_CPUs.N_nodes)
        myid = self.ring_of_CPUs.myid
        i_start_part, i_end_part = sharing.my_part(myid)
        self.mypart = self.machine.one_turn_map[i_start_part:i_end_part]
        self.i_start_part = i_start_part

        if self.ring_of_CPUs.I_am_a_worker:
            print(
                "I am id=%d/%d (worker) and my part is %d long"
                % (myid, self.ring_of_CPUs.N_nodes, len(self.mypart))
            )
        else:
            self.non_parallel_part = self.machine.one_turn_map[i_end_parallel:]
            print(
                "I am id=%d/%d (master) and my part is %d long"
                % (myid, self.ring_of_CPUs.N_nodes, len(self.mypart))
            )

    def _install_eclouds_in_machine_part(self):
        pp = self.pp

        my_new_part = []
        self.my_list_eclouds = []

        for ele in self.mypart:
            my_new_part.append(ele)
            if ele in self.machine.transverse_map:
                if not self.optics_from_pickle or "_kick_smooth_" in getattr(ele, "name1", ""):
                    for ee in self.parent_eclouds:
                        ecloud_new = ee.generate_twin_ecloud_with_shared_space_charge()
                        my_new_part.append(ecloud_new)
                        self.my_list_eclouds.append(ecloud_new)
                elif "_kick_element_" in getattr(ele, "name1", "") and getattr(
                    pp, "enable_eclouds_at_kick_elements", False
                ):
                    raise NotImplementedError(
                        "Kick-element ecloud installation is not yet implemented for PyEC4XS"
                    )

        self.mypart = my_new_part

    def _install_impedance(self):
        pp = self.pp
        if getattr(pp, "enable_impedance", False):
            raise NotImplementedError("Impedance support is not yet implemented for PyEC4XS")
        self.impedances = []

    def _switch_to_footprint_mode(self):
        raise NotImplementedError("footprint_mode is not yet implemented for PyEC4XS")

    def _generate_bunch(self):
        pp = self.pp

        if pp.footprint_mode:
            self.bunch = self.machine.generate_6D_Gaussian_bunch_matched(
                n_macroparticles=pp.n_macroparticles_for_footprint_track,
                intensity=pp.intensity,
                epsn_x=pp.epsn_x,
                epsn_y=pp.epsn_y,
                sigma_z=pp.sigma_z,
            )
            return

        if self.SimSt.first_run:
            if pp.bunch_from_file is not None:
                print(f"Loading bunch from file {pp.bunch_from_file} ...")
                with h5py.File(pp.bunch_from_file, "r") as fid:
                    self.bunch = self.buffer_to_piece(np.array(fid["bunch"]).copy())
                print("Bunch loaded from file.\n")
            else:
                self.bunch = self.machine.generate_6D_Gaussian_bunch_matched(
                    n_macroparticles=pp.n_macroparticles,
                    intensity=pp.intensity,
                    epsn_x=pp.epsn_x,
                    epsn_y=pp.epsn_y,
                    sigma_z=pp.sigma_z,
                )

                if getattr(pp, "recenter_all_slices", False):
                    print("Recentering all slices")
                    slices = IterableSlices(self.bunch, self.slicer)
                    for slice_data in slices:
                        idx = np.array(slice_data["particle_idx"], dtype=int)
                        if len(idx) == 0:
                            continue
                        self.bunch.x[idx] -= np.mean(self.bunch.x[idx])
                        self.bunch.px[idx] -= np.mean(self.bunch.px[idx])
                        self.bunch.y[idx] -= np.mean(self.bunch.y[idx])
                        self.bunch.py[idx] -= np.mean(self.bunch.py[idx])

                inj_opt = self.machine.transverse_map.get_injection_optics()
                sigma_x = np.sqrt(inj_opt["beta_x"] * pp.epsn_x / self.machine.betagamma)
                sigma_y = np.sqrt(inj_opt["beta_y"] * pp.epsn_y / self.machine.betagamma)

                self.bunch.x += pp.x_kick_in_sigmas * sigma_x
                self.bunch.y += pp.y_kick_in_sigmas * sigma_y
                print("Bunch initialized.")
        else:
            print("Loading bunch from file...")
            filename = "bunch_status_part%02d.h5" % (self.SimSt.present_simulation_part - 1)
            with h5py.File(filename, "r") as fid:
                self.bunch = self.buffer_to_piece(np.array(fid["bunch"]).copy())
            print("Bunch loaded from file.")

    def _prepare_monitors(self):
        pp = self.pp
        write_buffer_every = getattr(pp, "write_buffer_every", 3)
        bunch_stats_to_store = getattr(pp, "bunch_stats_to_store", PYHEADTAIL_BUNCH_STATS)
        slice_stats_to_store = getattr(pp, "slice_stats_to_store", PYHEADTAIL_SLICE_STATS)

        self.bunch_monitor = XsuiteBunchMonitor(
            "bunch_evolution_%02d" % self.SimSt.present_simulation_part,
            pp.N_turns,
            self.slicer,
            {"Comment": "PyEC4XS/Xsuite simulation"},
            write_buffer_every=write_buffer_every,
            stats_to_store=bunch_stats_to_store,
        )

        self.slice_monitor = XsuiteSliceMonitor(
            "slice_evolution_%02d" % self.SimSt.present_simulation_part,
            pp.N_turns,
            self.slicer,
            {"Comment": "PyEC4XS/Xsuite simulation"},
            write_buffer_every=write_buffer_every,
            bunch_stats_to_store=bunch_stats_to_store,
            slice_stats_to_store=slice_stats_to_store,
        )

    def _setup_multijob_mode(self):
        pp = self.pp
        check_for_resubmit = getattr(pp, "check_for_resubmit", True)
        SimSt = SimulationStatus(
            N_turns_per_run=pp.N_turns,
            check_for_resubmit=check_for_resubmit,
            N_turns_target=pp.N_turns_target,
        )
        SimSt.before_simulation()
        self.SimSt = SimSt

    def _check_stop_conditions(self):
        pp = self.pp
        stop = False

        if not pp.footprint_mode and len(self.bunch.x) < pp.sim_stop_frac * pp.n_macroparticles:
            stop = True
            print("Stop simulation due to beam losses.")

        if pp.flag_check_emittance_growth:
            epsn_x_max = pp.epsn_x * (1 + pp.epsn_x_max_growth_fraction)
            epsn_y_max = pp.epsn_y * (1 + pp.epsn_y_max_growth_fraction)
            if not pp.footprint_mode and (
                _normalized_emittance(self.bunch, "x") > epsn_x_max
                or _normalized_emittance(self.bunch, "y") > epsn_y_max
            ):
                stop = True
                print("Stop simulation due to emittance growth.")

        return stop

    def _finalize_multijob_mode(self):
        filename = "bunch_status_part%02d.h5" % self.SimSt.present_simulation_part
        with h5py.File(filename, "w") as fid:
            fid["bunch"] = self.piece_to_buffer(self.bunch)

        if not self.SimSt.first_run:
            previous = "bunch_status_part%02d.h5" % (self.SimSt.present_simulation_part - 1)
            if os.path.exists(previous):
                os.remove(previous)

        self.SimSt.after_simulation()

    def _extract_slices(self, bunch):
        iterable = list(IterableSlices(particles=bunch, slicer=self.slicer))
        sliced_pieces = []
        unsliced_piece = None
        unsliced_idx = np.where(np.array(bunch.slice) < 0)[0]

        for slice_data in iterable:
            idx = np.array(slice_data["particle_idx"], dtype=int)
            mask = np.zeros(len(bunch.x), dtype=bool)
            if len(idx) > 0:
                mask[idx] = True
            local_particles = _filter_particles_or_empty(
                bunch, mask, self.machine.particle_ref
            )

            raw_slice_index = int(str(slice_data["slice_num"]).split("/")[0]) - 1
            edge_a = float(self.slicer.bin_edges[raw_slice_index])
            edge_b = float(self.slicer.bin_edges[raw_slice_index + 1])
            z_left = min(edge_a, edge_b)
            z_right = max(edge_a, edge_b)

            local_slice_data = {
                "num_active": len(local_particles.x),
                "particle_idx": np.arange(len(local_particles.x), dtype=int),
                "beta": float(slice_data["beta"]),
                "zeta": float(slice_data["zeta"]),
                "gamma": float(slice_data["gamma"]),
                "dz": float(slice_data["dz"]),
                "dt": float(slice_data["dt"]),
                "slice_num": slice_data["slice_num"],
                "slice_info": {
                    "i_slice": raw_slice_index,
                    "z_bin_center": float(slice_data["zeta"]),
                    "z_bin_left": z_left,
                    "z_bin_right": z_right,
                    "interact_with_EC": True,
                },
            }
            sliced_pieces.append(
                SlicePiece(particles=local_particles, slice_data=local_slice_data)
            )

        if len(unsliced_idx) > 0:
            mask = np.zeros(len(bunch.x), dtype=bool)
            mask[unsliced_idx] = True
            unsliced_particles = _filter_particles_or_empty(
                bunch, mask, self.machine.particle_ref
            )
            if len(unsliced_particles.x) > 0:
                beta = float(unsliced_particles.beta0[0])
                gamma = float(unsliced_particles.gamma0[0])
            else:
                beta = 0.0
                gamma = 0.0
            unsliced_piece = SlicePiece(
                particles=unsliced_particles,
                slice_data={
                    "num_active": len(unsliced_particles.x),
                    "particle_idx": np.arange(len(unsliced_particles.x), dtype=int),
                    "beta": beta,
                    "zeta": 0.0,
                    "gamma": gamma,
                    "dz": 0.0,
                    "dt": 0.0,
                    "slice_num": "unsliced",
                    "slice_info": "unsliced",
                },
            )

        pieces = sliced_pieces[::-1]
        if unsliced_piece is not None:
            pieces.append(unsliced_piece)
        return pieces

    def _slice_bunch(self, bunch):
        return self._extract_slices(bunch)

    def _merge_pieces(self, pieces):
        particle_list = []
        for piece in pieces:
            particles = piece.particles if isinstance(piece, SlicePiece) else piece
            if len(particles.x) > 0:
                particle_list.append(particles)
        if not particle_list:
            return _empty_particles_from_reference(self.machine.particle_ref)
        return xp.Particles.merge(particle_list, _context=xo.ContextCpu())


def get_sim_instance(
    N_cores_pretend,
    id_pretend,
    init_sim_objects_auto=True,
    param_file="./Simulation_parameters.py",
):
    import PyPARIS.util as pu

    sim_instance = pu.get_sim_instance(
        Simulation(param_file=param_file),
        N_cores_pretend,
        id_pretend,
        init_sim_objects_auto,
    )

    return sim_instance


def get_serial_CPUring(
    init_sim_objects_auto=True,
    param_file="./Simulation_parameters.py",
):
    import PyPARIS.util as pu

    ring = pu.get_serial_CPUring(
        Simulation(param_file=param_file), init_sim_objects_auto
    )
    return ring


class ParticleTrajectories:
    def __init__(self, n_record, n_turns):
        self.x_i = np.empty((n_record, n_turns))
        self.px_i = np.empty((n_record, n_turns))
        self.y_i = np.empty((n_record, n_turns))
        self.py_i = np.empty((n_record, n_turns))
        self.zeta_i = np.empty((n_record, n_turns))
        self.i_turn = 0

    def dump(self, bunch):
        order = np.argsort(np.array(bunch.particle_id))
        self.x_i[:, self.i_turn] = np.array(bunch.x)[order]
        self.px_i[:, self.i_turn] = np.array(bunch.px)[order]
        self.y_i[:, self.i_turn] = np.array(bunch.y)[order]
        self.py_i[:, self.i_turn] = np.array(bunch.py)[order]
        self.zeta_i[:, self.i_turn] = np.array(bunch.zeta)[order]
        self.i_turn += 1
