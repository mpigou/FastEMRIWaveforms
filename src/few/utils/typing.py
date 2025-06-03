"""Definition of type hints helpers for FEW"""

import typing as t

import numpy as np
from numpy import ndarray as np_ndarray

try:
    import cupy as cp
except ImportError:
    cp = None


class xp_ndarray(np_ndarray):
    """Class to be use as type hint for arrays that can be either np or xp ndarray"""

    def get(self) -> np_ndarray:
        """Get method to match cp.ndarray interface"""
        return self.view(np_ndarray)


ArrayLike: t.TypeAlias = float | np_ndarray | xp_ndarray

TYPE_CHECKING_GPU_GUARD: bool = True
"""
Flag turned on during type checking to disable paths where xp_ndarray is confused
with cp.ndarray during static analysis of type hints.
"""


def as_np_array(value: ArrayLike) -> np_ndarray:
    """Coerce a value into a np.ndarray."""
    if isinstance(value, xp_ndarray):
        return value.get()
    if isinstance(value, np_ndarray):
        return value
    if TYPE_CHECKING_GPU_GUARD and (cp is not None) and isinstance(value, cp.ndarray):
        return value.get()
    assert isinstance(value, float)
    return np.asarray(value)


def as_xp_array(value: ArrayLike, use_gpu: bool) -> xp_ndarray:
    """Coerce a value into a xp_ndarray."""
    assert not (use_gpu and (cp is None))  # fail if use_gpu is True and cp is None
    if use_gpu and TYPE_CHECKING_GPU_GUARD:
        if isinstance(value, cp.ndarray):
            return value
        assert isinstance(value, (float, np_ndarray, xp_ndarray))
        return cp.asarray(value)

    if TYPE_CHECKING_GPU_GUARD and (cp is not None) and isinstance(value, cp.ndarray):
        return value.get().view(xp_ndarray)

    if isinstance(value, np_ndarray):
        return value.view(xp_ndarray)
    if isinstance(value, float):
        return np.asarray(value).view(xp_ndarray)
    assert isinstance(value, xp_ndarray)
    return value
