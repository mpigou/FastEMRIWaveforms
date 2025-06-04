import typing as t

import numpy as np


class OdeProtocol(t.Protocol):
    @t.overload
    def min_p(self, e: float, x: float, a: float | None) -> float: ...

    @t.overload
    def min_p(
        self, e: np.ndarray, x: np.ndarray, a: np.ndarray | None
    ) -> np.ndarray: ...

    @t.overload
    def max_p(self, e: float, x: float, a: float | None) -> float: ...

    @t.overload
    def max_p(
        self, e: np.ndarray, x: np.ndarray, a: np.ndarray | None
    ) -> np.ndarray: ...
