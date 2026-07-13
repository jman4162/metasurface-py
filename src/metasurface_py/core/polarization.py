"""Polarization decompositions for far-field patterns.

Circular-polarization convention (locked to the package's exp(+j*omega*t)
time convention, see :mod:`metasurface_py.core.conventions`):

A wave propagating along +r_hat with E = (theta_hat - j*phi_hat)/sqrt(2)
has instantaneous field Re{E exp(+j*omega*t)} rotating from theta_hat
toward phi_hat. Since theta_hat x phi_hat = r_hat, that rotation follows
the right-hand rule about the propagation direction, i.e. IEEE right-hand
circular polarization (IEEE Std 145-2013). The RHCP/LHCP components are
therefore extracted by Hermitian projection onto the unit vectors
e_R = (theta_hat - j*phi_hat)/sqrt(2), e_L = (theta_hat + j*phi_hat)/sqrt(2):

    E_R = (E_theta + j*E_phi) / sqrt(2)
    E_L = (E_theta - j*E_phi) / sqrt(2)

so a pure RHCP field satisfies E_phi = -j*E_theta and |E_R|^2 = |E|^2.
"""

from __future__ import annotations

import math

import numpy as np
import xarray as xr

_SQRT2 = math.sqrt(2.0)


def circular_components(
    e_theta: xr.DataArray,
    e_phi: xr.DataArray,
) -> xr.Dataset:
    """Decompose a theta/phi far field into RHCP and LHCP components.

    Args:
        e_theta: Complex E_theta pattern (any dims, typically (theta, phi)).
        e_phi: Complex E_phi pattern with the same dims/coords.

    Returns:
        xr.Dataset with data_vars ``e_rhcp`` and ``e_lhcp`` (complex),
        same dims/coords as the inputs. Total power is preserved:
        |e_rhcp|^2 + |e_lhcp|^2 = |e_theta|^2 + |e_phi|^2.
    """
    e_rhcp = (e_theta + 1j * e_phi) / _SQRT2
    e_lhcp = (e_theta - 1j * e_phi) / _SQRT2
    ds = xr.Dataset({"e_rhcp": e_rhcp, "e_lhcp": e_lhcp})
    ds.attrs.update(
        {
            "polarization_basis": "rhcp_lhcp",
            "time_convention": "exp(+j*omega*t)",
            "definition": "E_R=(E_theta+j*E_phi)/sqrt(2), IEEE Std 145-2013",
        }
    )
    return ds


def axial_ratio_db(
    e_theta: xr.DataArray,
    e_phi: xr.DataArray,
    floor_db: float = 60.0,
) -> xr.DataArray:
    """Axial ratio [dB] of the polarization ellipse, per observation angle.

    AR = (|E_R| + |E_L|) / ||E_R| - |E_L||, expressed as 20*log10(AR).
    Pure circular polarization gives 0 dB; pure linear gives +inf,
    clipped at ``floor_db``.

    Args:
        e_theta: Complex E_theta pattern.
        e_phi: Complex E_phi pattern with the same dims/coords.
        floor_db: Clip value for (near-)linear polarization [dB].

    Returns:
        Real DataArray of axial ratio in dB, same dims as the inputs.
    """
    cp = circular_components(e_theta, e_phi)
    r_mag = abs(cp["e_rhcp"])
    l_mag = abs(cp["e_lhcp"])
    denom = abs(r_mag - l_mag)
    ratio_linear = 10.0 ** (floor_db / 20.0)
    ar = ((r_mag + l_mag) / denom.clip(min=1e-300)).clip(max=ratio_linear)
    ar_db = ar.copy(data=20.0 * np.log10(ar.values))
    ar_db.name = "axial_ratio"
    ar_db.attrs.update({"unit": "dB"})
    return ar_db
