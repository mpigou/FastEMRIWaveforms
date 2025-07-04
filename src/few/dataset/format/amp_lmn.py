"""
Classes and methods to load a view of ZNAmps lmn data.
"""

from __future__ import annotations

import dataclasses
import typing as t

import h5py
import numpy as np

if t.TYPE_CHECKING:
    from ...cutils import Backend


@dataclasses.dataclass(frozen=True)
class AmpRegionLMN:
    """Class holding region data for lmn modes."""

    n_u: int
    n_w: int
    n_z: int

    coeffs: np.ndarray
    params: np.ndarray

    u_knots: np.ndarray
    w_knots: np.ndarray
    z_knots: np.ndarray

    def _check_sizes(self):
        """Perform tests on region sizes consistency"""

    def _freeze_arrays(self):
        """Freeze arrays to prevent modification"""
        self.coeffs.setflags(write=False)
        self.params.setflags(write=False)
        self.u_knots.setflags(write=False)
        self.w_knots.setflags(write=False)
        self.z_knots.setflags(write=False)

    def __post_init__(self):
        """Perform post-initialization checks and actions."""
        self._freeze_arrays()
        self._check_sizes()


@dataclasses.dataclass(frozen=True)
class AmpLMN:
    """
    Class holding amplitude data for lmn modes.
    """

    l_max: int
    m_max: int
    n_max: int

    region_a: AmpRegionLMN
    region_b: AmpRegionLMN


def build(filename: str, backend: Backend) -> AmpLMN:
    from few import get_file_manager

    file_path = get_file_manager().get_file(filename)

    with h5py.File(file_path, "r") as f:
        region_a = f["regionA"]
        region_b = f["regionB"]

        def convert(input: h5py.Dataset):
            return backend.xp.asarray(input[()])

        return AmpLMN(
            l_max=f.attrs["lmax"],
            m_max=f.attrs["mmax"],
            n_max=f.attrs["nmax"],
            region_a=AmpRegionLMN(
                n_u=region_a.attrs["NU"],
                n_w=region_a.attrs["NW"],
                n_z=region_a.attrs["NZ"],
                coeffs=convert(region_a["CoeffsRegionA"]),
                params=convert(region_a["ParamsRegionA"]),
                u_knots=convert(region_a["u_knots"]),
                w_knots=convert(region_a["w_knots"]),
                z_knots=convert(region_a["z_knots"]),
            ),
            region_b=AmpRegionLMN(
                n_u=region_b.attrs["NU"],
                n_w=region_b.attrs["NW"],
                n_z=region_b.attrs["NZ"],
                coeffs=convert(region_b["CoeffsRegionB"]),
                params=convert(region_b["ParamsRegionB"]),
                u_knots=convert(region_b["u_knots"]),
                w_knots=convert(region_b["w_knots"]),
                z_knots=convert(region_b["z_knots"]),
            ),
        )
