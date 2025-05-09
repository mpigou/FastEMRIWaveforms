import typing as t

import numpy as np

from ..typing import auto_array, xp_ndarray


@auto_array
def schwarzecc_p_to_y(
    p: t.Union[float, xp_ndarray], e: t.Union[float, xp_ndarray], *, use_gpu=False
):
    """Convert from separation :math:`p` to :math:`y` coordinate

    Conversion from the semilatus rectum or separation :math:`p` to :math:`y`.

    arguments:
        p (double scalar or 1D xp.ndarray): Values of separation,
            :math:`p`, to convert.
        e (double scalar or 1D xp.ndarray): Associated eccentricity values
            of :math:`p` necessary for conversion.
        use_gpu (bool, optional): If True, use Cupy/GPUs. Default is False.

    """
    if use_gpu:
        import cupy as cp

        e_cp = cp.asarray(e)
        p_cp = cp.asarray(p)
        return cp.log(-(21 / 10) - 2 * e_cp + p_cp)

    else:
        return np.log(-(21 / 10) - 2 * e + p)
