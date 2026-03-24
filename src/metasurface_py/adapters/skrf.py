"""scikit-rf adapter (optional dependency)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import xarray as xr

from metasurface_py.elements.lookup_cell import LookupTableCell

if TYPE_CHECKING:
    from pathlib import Path


def from_touchstone(
    path: str | Path,
    state_mapping: dict[int, int] | None = None,
) -> LookupTableCell:
    """Import element response from a Touchstone S-parameter file.

    Requires scikit-rf: pip install scikit-rf

    Args:
        path: Path to Touchstone file (.s1p, .s2p, etc.).
        state_mapping: Optional mapping from port index to state index.
            If None, each port corresponds to one state.

    Returns:
        LookupTableCell with frequency-dependent response.

    Raises:
        ImportError: If scikit-rf is not installed.
    """
    try:
        import skrf
    except ImportError as e:
        raise ImportError(
            "scikit-rf is required for Touchstone import. "
            "Install with: pip install scikit-rf"
        ) from e

    ntwk = skrf.Network(str(path))
    return from_network(ntwk, state_mapping=state_mapping)


def from_network(
    network: Any,
    state_mapping: dict[int, int] | None = None,
) -> LookupTableCell:
    """Convert a scikit-rf Network to a LookupTableCell.

    For a 1-port network, creates a single-state cell with S11
    as the reflection coefficient vs frequency.

    Args:
        network: scikit-rf Network object.
        state_mapping: Optional port-to-state mapping.

    Returns:
        LookupTableCell with response data.
    """
    freqs = network.f  # frequency array in Hz
    s = network.s  # S-parameter array: (n_freq, n_ports, n_ports)

    n_ports = s.shape[1]
    if state_mapping is None:
        state_mapping = {i: i for i in range(n_ports)}

    n_states = len(state_mapping)
    # Use S11 (reflection) for each state/port
    data = np.zeros(
        (n_states, len(freqs), 1),
        dtype=np.complex128,
    )
    for port_idx, state_idx in state_mapping.items():
        data[state_idx, :, 0] = s[:, port_idx, port_idx]

    state_phases = np.linspace(
        0,
        2 * np.pi,
        n_states,
        endpoint=False,
    )
    table = xr.DataArray(
        data,
        dims=["state", "freq", "theta"],
        coords={
            "state": state_phases,
            "freq": freqs,
            "theta": np.array([0.0]),
        },
        attrs={"source": "scikit-rf", "unit": "S11"},
    )
    return LookupTableCell.from_xarray(table)
