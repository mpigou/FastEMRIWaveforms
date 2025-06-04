"""Definition of type hints helpers for FEW"""

import functools
import inspect
import typing as t

import beartype
import numpy as np
import wrapt
from numpy import ndarray as np_ndarray

try:
    import cupy as cp
except ImportError:
    cp = None

from .exceptions import FewException


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


@functools.cache
def _xp_type_check_is_disabled() -> bool:
    """
    This method is called at import time to detect whether xp type checking decorator
    should be applied or completely skipped.

    If the environment variable FEW_DISABLE_XP_TYPE_CHECKS is set to **ANY** value,
    then, the decorator is completely disabled, thus reducing its impact on
    performances to zero.
    The decorator is also disabled if python runs in optimised mode (flag -O or -OO).
    """
    import os
    import sys

    return (os.getenv("FEW_DISABLE_XP_TYPE_CHECKS", None) is not None) or (
        sys.flags.optimize > 0
    )


class FewTypingException(FewException):
    """Base class for few.utils.typing exceptions"""


class FewXpTypeCheckInvalidTarget(FewTypingException):
    """xp_type_check decorator applied to invalid object."""


class FewXpNotDeducible(FewTypingException):
    wrapped: t.Callable
    instance: object
    bound_args: inspect.BoundArguments

    def __init__(
        self,
        *args,
        wrapped: t.Callable,
        instance: object,
        bound_args: inspect.BoundArguments,
        **kwargs,
    ):
        self.wrapped = wrapped
        self.instance = instance
        self.bound_args = bound_args
        super().__init__(*args, **kwargs)

    def name(self) -> str | None:
        return getattr(self.wrapped, "__name__", None)

    def location(self) -> tuple[str, int] | None:
        source_file = inspect.getsourcefile(self.wrapped)
        if source_file is None:
            return None
        lineno = inspect.getsourcelines(self.wrapped)[1]

        return (source_file, lineno)

    def formatted_location(self) -> str:
        location = self.location()
        return "" if location is None else "(from {0}:{1}) ".format(*location)

    def __str__(self) -> str:
        name: str | t.Callable = self.name() or self.wrapped
        return f"Could not deduce meaning of 'xp' for {name} {self.formatted_location()} for instance={self.instance} and arguments: {self.bound_args}"


if cp is not None:
    _xp2cp_beartype = beartype.beartype(
        conf=beartype.BeartypeConf(
            hint_overrides=beartype.FrozenDict({xp_ndarray: cp.ndarray})
        )
    )


class _XpTypeCheckDispatcher:
    """
    Object built for any callable decorated with xp_type_check and handling
    the dispatch to either a CPU or GPU runtime type checker.
    """

    bare_wrapped: t.Callable
    cpu_wrapped: t.Callable
    gpu_wrapped: t.Callable | None

    _signature: inspect.Signature

    def __init__(self, wrapped: t.Callable):
        if (not inspect.isfunction(wrapped)) and (not inspect.ismethod(wrapped)):
            raise FewXpTypeCheckInvalidTarget(
                "@xp_type_check decorator should only be applied on "
                "free-functions or bound methods"
            )

        self.bare_wrapped = wrapped
        self.cpu_wrapped = beartype.beartype(wrapped)

        if cp is not None:
            self.gpu_wrapped = _xp2cp_beartype(wrapped)
        else:
            self.gpu_wrapped = None

        self._signature = inspect.signature(self.bare_wrapped, eval_str=True)

    def should_skip(self) -> bool:
        """Determines if the wrappers are relevant"""
        return (self.cpu_wrapped is self.bare_wrapped) and (
            self.gpu_wrapped is None or self.gpu_wrapped is self.bare_wrapped
        )

    def should_dispatch(self) -> bool:
        """Determine if dispatching call to either cpu or gpu wrapper is relevant."""
        return not self.should_skip() and (self.gpu_wrapped is not None)

    def is_gpu_context(self, instance, args, kwargs) -> bool:
        """Determine whether wrapped is called in a GPU context"""
        from .baseclasses import ParallelModuleBase

        if isinstance(instance, ParallelModuleBase) and hasattr(instance, "backend"):
            return instance.backend.uses_cupy

        bound_args = (
            self._signature.bind(*args, **kwargs)
            if (instance is None)
            else self._signature.bind(instance, *args, **kwargs)
        )

        if (
            isinstance(instance, ParallelModuleBase)
            and "force_backend" in bound_args.arguments
        ):
            return instance.select_backend(
                bound_args.arguments["force_backend"]
            ).uses_cupy

        if "use_gpu" in bound_args.arguments and isinstance(
            use_gpu := bound_args.arguments["use_gpu"], bool
        ):
            return use_gpu

        raise FewXpNotDeducible(
            wrapped=self.bare_wrapped, instance=instance, bound_args=bound_args
        )

    @wrapt.decorator
    def __call__(self, _, instance, args, kwargs):
        wrapper = (
            self.gpu_wrapped
            if self.is_gpu_context(instance, args, kwargs)
            else self.cpu_wrapped
        )
        return (
            wrapper(*args, **kwargs)
            if instance is None
            else wrapper(instance, *args, **kwargs)
        )


def xp_type_check(wrapped: t.Callable) -> t.Callable:
    """
    Decorator applied on functions whose type hints must be analyzed at runtime.

    This decorator expects the decorated method to be used in a context where
    one can interpret a xp_ndarray type hint as either an actual instance of
    xp_ndarray (data on CPU memory with a .get() method defined in the class
    xp_ndarray), or actually a cp.ndarray (data on GPU memory with a .get() method
    to trigger data transfer to CPU memory).
    This currently applies to the following cases:
      - Free functions with a boolean `use_gpu` argument (if use_gpu is True, then
        xp_ndarray is reinterpreted as cp.ndarray)
      - __init__ method of a ParallelModuleBase derivate class with an explicitely
        defined `force_backend`
      - Any instance method of a ParallelModuleBase derivate class

    """
    if _xp_type_check_is_disabled():
        # shortcut that disables applying the decorator through global configuration
        return wrapped

    dispatcher = _XpTypeCheckDispatcher(wrapped)

    if dispatcher.should_skip():
        return wrapped

    if not dispatcher.should_dispatch():
        return dispatcher.cpu_wrapped

    return dispatcher(wrapped)


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


@xp_type_check
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


__all__ = ["xp_type_check", "as_np_array", "as_xp_array", "ArrayLike", "xp_ndarray"]
