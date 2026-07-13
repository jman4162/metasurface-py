"""End-to-end tests for the modulated metasurface antenna."""

from __future__ import annotations

import numpy as np
import pytest

from metasurface_py.core.conventions import wavelength
from metasurface_py.core.polarization import axial_ratio_db, circular_components
from metasurface_py.core.types import AngleGrid, SubstrateInfo
from metasurface_py.em.aperture_field import pattern_directivity
from metasurface_py.em.leakywave import solve_sw_transparent
from metasurface_py.surfaces.modulated import (
    ModulatedMetasurfaceAntenna,
    SinusoidalModulation,
    TensorModulation,
    dual_band_reactance_map,
)

FREQ = 26.25e9
LAM = wavelength(FREQ)
SUBSTRATE = SubstrateInfo(name="RO3010", eps_r=10.2, thickness_mm=0.635)
X_SHEET = -1058.0


@pytest.fixture(scope="module")
def x_op() -> float:
    return solve_sw_transparent(X_SHEET, FREQ, 10.2, 0.635e-3).x_op


def _antenna(
    modulation: object, radius_lam: float = 4.0
) -> ModulatedMetasurfaceAntenna:
    return ModulatedMetasurfaceAntenna(
        radius=radius_lam * LAM,
        x_transparent=X_SHEET,
        substrate=SUBSTRATE,
        modulation=modulation,  # type: ignore[arg-type]
        n_rho=128,
        n_phi=64,
    )


def _angles() -> AngleGrid:
    return AngleGrid.from_degrees(
        theta=np.linspace(0, 90, 121), phi=np.linspace(0, 350, 36)
    )


class TestBroadsideCP:
    def test_rhcp_broadside_beam(self, x_op: float) -> None:
        mod = TensorModulation.circular_broadside(x_op, FREQ, m=0.3, hand="rhcp")
        ant = _antenna(mod)
        ff = ant.far_field(FREQ, _angles())
        d = pattern_directivity(ff)
        dv = d.values
        it, ip = np.unravel_index(int(np.argmax(dv)), dv.shape)
        # Beam at broadside
        assert float(np.rad2deg(_angles().theta[it])) == pytest.approx(0.0, abs=1.0)
        # Below the physical aperture-directivity bound
        peak_db = 10 * np.log10(dv[it, ip])
        bound_db = 10 * np.log10(4 * np.pi * ant.aperture.area / LAM**2)
        assert peak_db < bound_db
        assert peak_db > bound_db - 3.0  # reasonable aperture efficiency
        # Pure circular polarization on boresight
        ar = axial_ratio_db(ff["E_theta"], ff["E_phi"])
        assert float(ar.values[it, ip]) < 0.1
        cp = circular_components(ff["E_theta"], ff["E_phi"])
        xpd = 20 * np.log10(
            float(np.abs(cp["e_rhcp"].values[it, ip]))
            / max(float(np.abs(cp["e_lhcp"].values[it, ip])), 1e-30)
        )
        assert xpd > 30.0

    def test_lhcp_hand_flips(self, x_op: float) -> None:
        mod = TensorModulation.circular_broadside(x_op, FREQ, m=0.3, hand="lhcp")
        ant = _antenna(mod)
        ff = ant.far_field(FREQ, _angles())
        cp = circular_components(ff["E_theta"], ff["E_phi"])
        boresight_r = float(np.abs(cp["e_rhcp"].values[0]).max())
        boresight_l = float(np.abs(cp["e_lhcp"].values[0]).max())
        assert boresight_l > 10 * boresight_r


class TestTiltedBeam:
    def test_pencil_beam_at_design_angle(self, x_op: float) -> None:
        theta0 = np.deg2rad(30.0)
        mod = TensorModulation.circular_tilted(
            x_op, FREQ, m=0.3, theta0=theta0, hand="rhcp"
        )
        ant = _antenna(mod)
        ff = ant.far_field(FREQ, _angles())
        dv = pattern_directivity(ff).values
        it, ip = np.unravel_index(int(np.argmax(dv)), dv.shape)
        assert float(np.rad2deg(_angles().theta[it])) == pytest.approx(30.0, abs=1.5)
        assert float(np.rad2deg(_angles().phi[ip])) == pytest.approx(0.0, abs=10.0)

    def test_radial_period_makes_conical_beam(self, x_op: float) -> None:
        """A radial-only tilted period produces a cone: azimuthally flat power."""
        from metasurface_py.core.conventions import k0 as k0f
        from metasurface_py.em.leakywave import sw_wavenumber_tm

        big_k = sw_wavenumber_tm(x_op, FREQ) - k0f(FREQ) * np.sin(np.deg2rad(30))
        mod = SinusoidalModulation(m=0.3, period=2 * np.pi / big_k)
        ant = _antenna(mod)
        ff = ant.far_field(FREQ, _angles())
        dv = pattern_directivity(ff).values
        it, _ = np.unravel_index(int(np.argmax(dv)), dv.shape)
        cut = dv[it, :]
        assert float(np.rad2deg(_angles().theta[it])) == pytest.approx(30.0, abs=2.0)
        assert cut.min() / cut.max() > 0.5  # ring, not pencil


class TestApertureField:
    def test_dataset_structure(self, x_op: float) -> None:
        mod = SinusoidalModulation.for_broadside(x_op, FREQ, m=0.25)
        ant = _antenna(mod)
        fields = ant.aperture_field(FREQ)
        assert set(fields.data_vars) >= {"e_rho", "e_phi", "j0_rho", "alpha", "beta"}
        assert fields["e_rho"].dims == ("rho", "phi_ap")
        assert fields.attrs["beta_sw"] > 0
        # scalar modulation -> no cross component
        assert float(np.abs(fields["e_phi"]).max()) == 0.0
        # leakage decays the current amplitude radially
        j_amp = np.abs(fields["j0_rho"].values) * np.sqrt(fields["rho"].values)
        assert j_amp[-1] < j_amp[len(j_amp) // 4]

    def test_radial_taper_profile(self, x_op: float) -> None:
        mod = SinusoidalModulation.for_broadside(
            x_op, FREQ, m=lambda rho: 0.1 + 0.2 * rho / rho.max()
        )
        ant = _antenna(mod)
        fields = ant.aperture_field(FREQ)
        alpha = fields["alpha"].values
        assert alpha[-1] > alpha[0] > 0


class TestValidation:
    def test_positive_x_transparent_rejected(self, x_op: float) -> None:
        mod = SinusoidalModulation.for_broadside(x_op, FREQ, m=0.2)
        with pytest.raises(ValueError, match="capacitive"):
            ModulatedMetasurfaceAntenna(
                radius=0.05,
                x_transparent=+500.0,
                substrate=SUBSTRATE,
                modulation=mod,
            )

    def test_launch_efficiency_range(self, x_op: float) -> None:
        mod = SinusoidalModulation.for_broadside(x_op, FREQ, m=0.2)
        with pytest.raises(ValueError, match="launch_efficiency"):
            ModulatedMetasurfaceAntenna(
                radius=0.05,
                x_transparent=X_SHEET,
                substrate=SUBSTRATE,
                modulation=mod,
                launch_efficiency=1.5,
            )

    def test_bad_hand_rejected(self, x_op: float) -> None:
        with pytest.raises(ValueError, match="hand"):
            TensorModulation.circular_broadside(x_op, FREQ, m=0.2, hand="elliptical")


class TestBandwidthGain:
    def test_result_values(self, x_op: float) -> None:
        mod = SinusoidalModulation.for_broadside(x_op, FREQ, m=0.2)
        ant = _antenna(mod, radius_lam=10.0)
        bg = ant.bandwidth_gain(FREQ)
        assert 0.0 < bg.v_g_over_c < 1.0
        assert bg.radius_lambda == pytest.approx(10.0)
        assert bg.gb_uniform > bg.gb_tapered > 0
        assert 0.0 < bg.bandwidth < 1.0


class TestDualBandMap:
    def test_map_structure(self, x_op: float) -> None:
        m1 = TensorModulation.circular_broadside(x_op, FREQ, m=0.2, hand="rhcp")
        m2 = TensorModulation.circular_broadside(x_op, 32.05e9, m=0.2, hand="rhcp")
        ds = dual_band_reactance_map(
            x_avg=X_SHEET,
            modulations=[m1, m2],
            centers=[(-0.01, 0.0), (0.01, 0.0)],
            extent=0.05,
            n=64,
        )
        assert ds["x_rr"].dims == ("y", "x")
        assert ds["x_rr"].shape == (64, 64)
        # Mean over the window stays near the average reactance
        assert float(ds["x_rr"].mean()) == pytest.approx(X_SHEET, rel=0.05)
        # Peak deviation bounded by the summed modulation indices
        assert float(np.abs(ds["x_rr"] / X_SHEET - 1.0).max()) <= 0.4 + 1e-9

    def test_length_mismatch_raises(self, x_op: float) -> None:
        m1 = TensorModulation.circular_broadside(x_op, FREQ, m=0.2)
        with pytest.raises(ValueError, match="same length"):
            dual_band_reactance_map(X_SHEET, [m1], [(0, 0), (1, 1)], extent=0.05)


class TestDispersionMethod:
    def test_default_sweep(self, x_op: float) -> None:
        mod = SinusoidalModulation.for_broadside(x_op, FREQ, m=0.3)
        ant = _antenna(mod)
        ds = ant.dispersion(FREQ)
        assert "beta_over_k0" in ds and "alpha_over_k0" in ds
        assert float(ds["m"].max()) == pytest.approx(0.3)
        assert bool(ds["converged"].all())
