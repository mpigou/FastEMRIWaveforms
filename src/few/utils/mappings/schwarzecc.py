import numpy as np

from ..typing import auto_array, xp_ndarray


@auto_array
def schwarzecc_p_to_y(p: xp_ndarray, e: xp_ndarray, *, use_gpu=False) -> xp_ndarray:
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

        xp = cp
    else:
        xp = np

    return xp.log(-(21 / 10) - 2 * e + p)
