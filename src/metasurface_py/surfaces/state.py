"""Surface state representation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Self

import numpy as np
import numpy.typing as npt

from metasurface_py.elements.states import StateSpace, quantize


@dataclass(frozen=True)
class SurfaceState:
    """A concrete realization of programmable element controls.

    Args:
        values: Phase/control values per element, shape (N,) in radians.
        space: The state space these values belong to.
        mask: Boolean mask (True = active), shape (N,), or None.
    """

    values: npt.NDArray[np.floating[Any]]
    space: StateSpace
    mask: npt.NDArray[np.bool_] | None = None

    @property
    def num_elements(self) -> int:
        """Number of elements (including masked-out ones)."""
        return int(self.values.shape[0])

    def quantize(
        self,
        codebook: (npt.NDArray[np.complexfloating[Any, Any]] | None) = None,
    ) -> Self:
        """Project to nearest discrete states.

        Args:
            codebook: Complex codebook to quantize to. If None, uses the
                      state space's codebook.

        Returns:
            New SurfaceState with quantized values.
        """
        if codebook is None:
            if self.space.codebook is None:
                raise ValueError("No codebook available for quantization")
            codebook = self.space.codebook
        quantized = quantize(self.values, codebook)
        return type(self)(values=quantized, space=self.space, mask=self.mask)

    def with_defects(self, defect_mask: npt.NDArray[np.bool_]) -> Self:
        """Zero out dead elements.

        Args:
            defect_mask: Boolean array, True = active, False = defective.

        Returns:
            New SurfaceState with defective elements set to zero phase.
        """
        new_values = self.values.copy()
        new_values[~defect_mask] = 0.0
        combined_mask = defect_mask
        if self.mask is not None:
            combined_mask = self.mask & defect_mask
        return type(self)(values=new_values, space=self.space, mask=combined_mask)

    def apply_grouping(self, group_map: npt.NDArray[np.integer[npt.NBitBase]]) -> Self:
        """Enforce bias-line grouping: all elements in a group share the same state.

        The group state is the circular mean of member phases.

        Args:
            group_map: Integer array mapping each element to a group ID, shape (N,).

        Returns:
            New SurfaceState with grouped values.
        """
        new_values = self.values.copy()
        group_ids = np.unique(group_map)
        for gid in group_ids:
            members = group_map == gid
            # Circular mean of phases
            mean_phase = np.angle(np.mean(np.exp(1j * self.values[members])))
            new_values[members] = mean_phase
        return type(self)(values=new_values, space=self.space, mask=self.mask)
