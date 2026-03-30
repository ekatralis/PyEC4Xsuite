import numpy as np
import xobjects as xo
import xpart as xp
import xtrack as xt
from scipy.constants import c


class EmptyObject:
    pass


class TransverseMap(list):
    def __init__(self, elements, injection_optics):
        super().__init__(elements)
        self.n_segments = len(elements)
        self._injection_optics = dict(injection_optics)

    def get_injection_optics(self):
        return dict(self._injection_optics)


class LHC:
    def __init__(
        self,
        n_segments,
        machine_configuration,
        optics_dict=None,
        _context=None,
        **kwargs,
    ):
        pp = EmptyObject()

        pp.n_segments = n_segments
        pp.machine_configuration = machine_configuration
        pp.optics_dict = optics_dict

        pp.circumference = 26658.8832
        pp.p_increment = 0.0
        pp.alpha = 3.225e-04
        pp.h_RF = 35640
        pp.longitudinal_mode = "non-linear"
        pp.Qp_x = 0.0
        pp.Qp_y = 0.0
        pp.app_x = 0.0
        pp.app_y = 0.0
        pp.app_xy = 0.0
        pp.i_octupole_focusing = None
        pp.i_octupole_defocusing = None
        pp.octupole_knob = None

        if pp.machine_configuration == "HLLHC-injection":
            pp.alpha_x = 0.0
            pp.beta_x = 92.7
            pp.D_x = 0.0
            pp.alpha_y = 0.0
            pp.beta_y = 93.2
            pp.D_y = 0.0
            pp.accQ_x = 62.27
            pp.accQ_y = 60.295
            pp.V_RF = 8e6
            pp.dphi_RF = 0.0
            pp.p0c = 450e9
        elif pp.machine_configuration == "HLLHC-collision":
            pp.alpha_x = 0.0
            pp.beta_x = 92.7
            pp.D_x = 0.0
            pp.alpha_y = 0.0
            pp.beta_y = 93.2
            pp.D_y = 0.0
            pp.accQ_x = 62.31
            pp.accQ_y = 60.32
            pp.V_RF = 16e6
            pp.dphi_RF = 0.0
            pp.p0c = 7000e9
        elif pp.machine_configuration == "LHC-collision":
            pp.alpha_x = 0.0
            pp.beta_x = 92.7
            pp.D_x = 0.0
            pp.alpha_y = 0.0
            pp.beta_y = 93.2
            pp.D_y = 0.0
            pp.accQ_x = 62.31
            pp.accQ_y = 60.32
            pp.V_RF = 12e6
            pp.dphi_RF = 0.0
            pp.p0c = 6800e9
        else:
            raise ValueError(f"ERROR: unknown machine configuration {machine_configuration}")

        for attr, value in kwargs.items():
            if value is None:
                continue
            if not hasattr(pp, attr):
                raise NameError(f"I don't understand {attr}")
            setattr(pp, attr, value)

        if pp.i_octupole_focusing is not None or pp.i_octupole_defocusing is not None:
            if pp.octupole_knob is not None:
                raise ValueError(
                    "octupole_knobs and octupole currents cannot be used at the same time!"
                )
            pp.app_x, pp.app_y, pp.app_xy = self._anharmonicities_from_octupole_current_settings(
                pp.i_octupole_focusing, pp.i_octupole_defocusing
            )

        if pp.octupole_knob is not None:
            if pp.i_octupole_focusing is not None or pp.i_octupole_defocusing is not None:
                raise ValueError(
                    "octupole_knobs and octupole currents cannot be used at the same time!"
                )
            i_oct_f, i_oct_d = self._octupole_currents_from_octupole_knobs(
                pp.octupole_knob, pp.p0c
            )
            pp.app_x, pp.app_y, pp.app_xy = self._anharmonicities_from_octupole_current_settings(
                i_oct_f, i_oct_d
            )

        self._context = _context or xo.ContextCpu()
        self.particle_ref = xp.Particles(
            p0c=pp.p0c,
            mass0=xp.PROTON_MASS_EV,
            q0=1.0,
            _context=self._context,
        )

        self.circumference = float(pp.circumference)
        self.beta = float(self.particle_ref.beta0[0])
        self.gamma = float(self.particle_ref.gamma0[0])
        self.betagamma = self.beta * self.gamma
        self.voltage_rf = np.atleast_1d(np.array(pp.V_RF, dtype=float))
        self.frequency_rf = (
            np.atleast_1d(np.array(pp.h_RF, dtype=float))
            * self.beta
            * c
            / self.circumference
        )
        self.lag_rf = self._pyheadtail_phase_to_xsuite_lag_deg(pp.dphi_RF)
        self.requested_longitudinal_mode = pp.longitudinal_mode
        self.tracking_longitudinal_mode = self._normalize_longitudinal_mode(
            pp.longitudinal_mode
        )

        if pp.optics_dict is None:
            transverse_elements = self._build_smooth_transverse_map(pp)
            inj_optics = {
                "alpha_x": float(pp.alpha_x),
                "beta_x": float(pp.beta_x),
                "D_x": float(pp.D_x),
                "alpha_y": float(pp.alpha_y),
                "beta_y": float(pp.beta_y),
                "D_y": float(pp.D_y),
            }
        else:
            transverse_elements, inj_optics = self._build_optics_dict_transverse_map(pp)

        self.transverse_map = TransverseMap(transverse_elements, inj_optics)
        self.longitudinal_map = self._build_longitudinal_map(pp, self.tracking_longitudinal_mode)
        self.one_turn_map = list(self.transverse_map) + [self.longitudinal_map]
        self.matching_line = self._build_matching_line(pp, inj_optics)

    def generate_6D_Gaussian_bunch_matched(
        self,
        n_macroparticles,
        intensity,
        epsn_x,
        epsn_y,
        sigma_z,
    ):
        if self.matching_line.tracker is None:
            self.matching_line.build_tracker(_context=self._context)

        longitudinal_engine = None
        if self.tracking_longitudinal_mode == "nonlinear":
            # Keep the matching line linear for Twiss/closed-orbit purposes,
            # but generate the longitudinal coordinates with the RF-bucket
            # matcher once the RF phase convention has been translated.
            longitudinal_engine = "pyheadtail"

        return xp.generate_matched_gaussian_bunch(
            line=self.matching_line,
            num_particles=n_macroparticles,
            nemitt_x=epsn_x,
            nemitt_y=epsn_y,
            sigma_z=sigma_z,
            total_intensity_particles=intensity,
            engine=longitudinal_engine,
            _context=self._context,
        )

    def _build_smooth_transverse_map(self, pp):
        elements = []
        n_segments = int(pp.n_segments)
        for ii in range(n_segments):
            segment = xt.LineSegmentMap(
                length=self.circumference / n_segments,
                qx=pp.accQ_x / n_segments,
                qy=pp.accQ_y / n_segments,
                betx=pp.beta_x,
                bety=pp.beta_y,
                alfx=pp.alpha_x,
                alfy=pp.alpha_y,
                dx=pp.D_x,
                dy=pp.D_y,
                dqx=pp.Qp_x / n_segments,
                dqy=pp.Qp_y / n_segments,
                det_xx=pp.app_x / n_segments,
                det_xy=pp.app_xy / n_segments,
                det_yy=pp.app_y / n_segments,
                det_yx=pp.app_xy / n_segments,
                longitudinal_mode="frozen",
            )
            segment.name1 = f"_kick_smooth_{ii}"
            segment.beta_x1 = float(pp.beta_x)
            segment.beta_y1 = float(pp.beta_y)
            elements.append(segment)
        return elements

    def _build_optics_dict_transverse_map(self, pp):
        optics = pp.optics_dict
        names = list(optics["name"])
        beta_x = np.atleast_1d(np.array(optics["beta_x"], dtype=float))
        beta_y = np.atleast_1d(np.array(optics["beta_y"], dtype=float))
        alpha_x = np.atleast_1d(np.array(optics["alpha_x"], dtype=float))
        alpha_y = np.atleast_1d(np.array(optics["alpha_y"], dtype=float))
        D_x = np.atleast_1d(np.array(optics.get("D_x", np.zeros_like(beta_x)), dtype=float))
        D_y = np.atleast_1d(np.array(optics.get("D_y", np.zeros_like(beta_y)), dtype=float))
        s = np.atleast_1d(np.array(optics["s"], dtype=float))

        n_segments = len(names)
        lengths = np.diff(np.concatenate([s, [self.circumference]]))
        positive_lengths = lengths[lengths > 0]
        replacement_length = float(np.mean(positive_lengths)) if len(positive_lengths) > 0 else 0.0
        lengths = np.where(lengths > 0, lengths, replacement_length)

        if "accQ_x" in optics:
            pp.accQ_x = float(optics["accQ_x"])
        if "accQ_y" in optics:
            pp.accQ_y = float(optics["accQ_y"])

        elements = []
        for ii, name in enumerate(names):
            segment = xt.LineSegmentMap(
                length=float(lengths[ii]),
                qx=pp.accQ_x / n_segments,
                qy=pp.accQ_y / n_segments,
                betx=float(beta_x[ii]),
                bety=float(beta_y[ii]),
                alfx=float(alpha_x[ii]),
                alfy=float(alpha_y[ii]),
                dx=float(D_x[ii]),
                dy=float(D_y[ii]),
                dqx=pp.Qp_x / n_segments,
                dqy=pp.Qp_y / n_segments,
                det_xx=pp.app_x / n_segments,
                det_xy=pp.app_xy / n_segments,
                det_yy=pp.app_y / n_segments,
                det_yx=pp.app_xy / n_segments,
                longitudinal_mode="frozen",
            )
            segment.name1 = name
            segment.beta_x1 = float(beta_x[ii])
            segment.beta_y1 = float(beta_y[ii])
            elements.append(segment)

        inj_optics = {
            "alpha_x": float(alpha_x[0]),
            "beta_x": float(beta_x[0]),
            "D_x": float(D_x[0]),
            "alpha_y": float(alpha_y[0]),
            "beta_y": float(beta_y[0]),
            "D_y": float(D_y[0]),
        }
        return elements, inj_optics

    def _build_longitudinal_map(self, pp, mode):
        kwargs = {
            "length": 0.0,
            "qx": 0.0,
            "qy": 0.0,
            "betx": 1.0,
            "bety": 1.0,
            "longitudinal_mode": mode,
        }

        if mode in ("linear_fixed_rf", "nonlinear"):
            kwargs.update(
                {
                    "momentum_compaction_factor": pp.alpha,
                    "slippage_length": self.circumference,
                    "voltage_rf": list(self.voltage_rf),
                    "frequency_rf": list(self.frequency_rf),
                    "lag_rf": list(self.lag_rf),
                }
            )
        elif mode != "frozen":
            raise ValueError(f"Unsupported longitudinal mode {mode}")

        element = xt.LineSegmentMap(**kwargs)
        element.name1 = "_longitudinal_map_"
        return element

    def _build_matching_line(self, pp, inj_optics):
        matching_mode = "linear_fixed_rf"
        matching_map = xt.LineSegmentMap(
            length=self.circumference,
            qx=pp.accQ_x,
            qy=pp.accQ_y,
            betx=inj_optics["beta_x"],
            bety=inj_optics["beta_y"],
            alfx=inj_optics["alpha_x"],
            alfy=inj_optics["alpha_y"],
            dx=inj_optics["D_x"],
            dy=inj_optics["D_y"],
            dqx=pp.Qp_x,
            dqy=pp.Qp_y,
            det_xx=pp.app_x,
            det_xy=pp.app_xy,
            det_yy=pp.app_y,
            det_yx=pp.app_xy,
            longitudinal_mode=matching_mode,
            momentum_compaction_factor=pp.alpha,
            slippage_length=self.circumference,
            voltage_rf=list(self.voltage_rf),
            frequency_rf=list(self.frequency_rf),
            lag_rf=list(self.lag_rf),
        )
        line = xt.Line(elements=[matching_map], particle_ref=self.particle_ref.copy(_context=self._context))
        line.build_tracker(_context=self._context)
        return line

    def _pyheadtail_phase_to_xsuite_lag_deg(self, dphi_rf):
        phase_rad = np.atleast_1d(np.array(dphi_rf, dtype=float))
        # PyHEADTAIL's dphi_RF=0 is synchronous above transition, while
        # Xsuite's LineSegmentMap uses lag_rf=180 deg for the same bucket.
        return np.mod(np.degrees(np.pi - phase_rad), 360.0)

    def _normalize_longitudinal_mode(self, mode):
        if mode is None:
            return "linear_fixed_rf"

        mode_low = mode.lower().replace("_", "-")
        if mode_low in ("linear", "linear-fixed-rf", "linear-fixed_rf"):
            return "linear_fixed_rf"
        if mode_low in ("nonlinear", "non-linear"):
            return "nonlinear"
        if mode_low == "frozen":
            return "frozen"

        raise ValueError(f"Unsupported longitudinal_mode {mode}")

    def _anharmonicities_from_octupole_current_settings(
        self, i_octupole_focusing, i_octupole_defocusing
    ):
        from scipy.constants import e

        i_max = 550.0
        E_max = 7000.0

        app_x = E_max * (
            267065.0 * i_octupole_focusing / i_max
            - 7856.0 * i_octupole_defocusing / i_max
        )
        app_y = E_max * (
            9789.0 * i_octupole_focusing / i_max
            - 277203.0 * i_octupole_defocusing / i_max
        )
        app_xy = E_max * (
            -102261.0 * i_octupole_focusing / i_max
            + 93331.0 * i_octupole_defocusing / i_max
        )

        convert_to_SI = e / (1.0e-9 * c)
        app_x *= convert_to_SI
        app_y *= convert_to_SI
        app_xy *= convert_to_SI

        return app_x, app_y, app_xy

    def _octupole_currents_from_octupole_knobs(self, octupole_knob, p0c):
        i_octupole_focusing = (
            19.557 * octupole_knob / (-1.5) * p0c / 2.4049285931335872e-16
        )
        i_octupole_defocusing = -i_octupole_focusing
        return i_octupole_focusing, i_octupole_defocusing
