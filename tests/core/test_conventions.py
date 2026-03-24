"""Tests for core conventions module."""

from __future__ import annotations

import math

import pytest

from metasurface_py.core.conventions import (
    EPS_0,
    ETA_0,
    MU_0,
    PHASOR_SIGN,
    SPEED_OF_LIGHT,
    freq_to_omega,
    k0,
    wavelength,
)


class TestPhysicalConstants:
    def test_speed_of_light(self) -> None:
        assert pytest.approx(299_792_458.0) == SPEED_OF_LIGHT

    def test_impedance_of_free_space(self) -> None:
        assert pytest.approx(376.73, rel=1e-3) == ETA_0

    def test_mu0_eps0_c_relation(self) -> None:
        """mu0 * eps0 * c^2 = 1."""
        assert pytest.approx(1.0) == MU_0 * EPS_0 * SPEED_OF_LIGHT**2

    def test_eta0_from_mu0_eps0(self) -> None:
        """eta0 = sqrt(mu0/eps0)."""
        assert pytest.approx(math.sqrt(MU_0 / EPS_0), rel=1e-10) == ETA_0

    def test_phasor_sign_is_negative(self) -> None:
        assert PHASOR_SIGN == -1


class TestWavelength:
    def test_28ghz(self) -> None:
        lam = wavelength(28e9)
        assert lam == pytest.approx(SPEED_OF_LIGHT / 28e9)

    def test_roundtrip_with_k0(self) -> None:
        freq = 10e9
        lam = wavelength(freq)
        k = k0(freq)
        assert k * lam == pytest.approx(2 * math.pi)

    def test_negative_freq_raises(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            wavelength(-1e9)

    def test_zero_freq_raises(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            wavelength(0.0)


class TestK0:
    def test_value(self) -> None:
        freq = 5e9
        expected = 2.0 * math.pi * freq / SPEED_OF_LIGHT
        assert k0(freq) == pytest.approx(expected)


class TestFreqToOmega:
    def test_value(self) -> None:
        freq = 1e9
        assert freq_to_omega(freq) == pytest.approx(2 * math.pi * 1e9)

    def test_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            freq_to_omega(-1.0)
