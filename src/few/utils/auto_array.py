"""Define the auto_array decorators to handle np and cp array conversions."""

import enum
import inspect
import logging
import types
from typing import Optional, Sequence, Union

import numpy as np
import wrapt

from .exceptions import FewException
from .parallel_base import ParallelModuleBase


class AutoArrayException(FewException):
    """Base-class for exceptions related to the automatic array conversion feature."""


class AutoArrayMode(enum.Enum):
    """Enumeration of automatic array conversion modes"""

    DEFAULT = "default"
    """Default mode is no-op: no conversion take place"""

    STRICT = "strict"
    """Strict mode always convert from/into np and xp arrays and raise warnings"""

    NUMPY = "numpy"
    """This mode always return np arrays and convert back into xp array in method inputs"""

    DEMO = "demo"
    """Methods called by user use NUMPY mode and nested methods use strict mode"""

    def __str__(self) -> str:
        return str(self.value)

    @classmethod
    def __contains__(cls, item):
        return isinstance(item, cls) or item in [
            v.value for v in cls.__members__.values()
        ]


class _BaseBoundFunctionWrapper(wrapt.BoundFunctionWrapper):
    pass


class _BaseFunctionWrapper(wrapt.FunctionWrapper):
    __bound_function_wrapper__ = _BaseBoundFunctionWrapper


class _InputBoundFunctionWrapper(_BaseBoundFunctionWrapper):
    pass


class _InputFunctionWrapper(_BaseFunctionWrapper):
    __bound_function_wrapper__ = _InputBoundFunctionWrapper


class _OutputBoundFunctionWrapper(_BaseBoundFunctionWrapper):
    pass


class _OutputFunctionWrapper(_BaseFunctionWrapper):
    __bound_function_wrapper__ = _OutputBoundFunctionWrapper


class _Base:
    """Base class for all auto_array decorators"""

    _active_mode: Optional[AutoArrayMode]
    _logger: Optional[logging.Logger]

    def __init__(self):
        self._active_mode = None
        self._logger = None

    def activate_mode(self, mode: AutoArrayMode):
        from .globals import get_logger

        self._active_mode = mode
        self._logger = get_logger()

    def deactivate_mode(self):
        self._active_mode = None
        self._logger = None

    @property
    def active_mode(self) -> Optional[AutoArrayMode]:
        return self._active_mode

    @property
    def logger(self) -> logging.Logger:
        if self._logger is None:
            raise AutoArrayException(
                "Logger cannot be accessed outside of decorator call."
            )
        return self._logger


class _AutoArrayContext:
    """Context manager to set/unset decorator mode"""

    def __init__(self, decorator: _Base, mode: AutoArrayMode):
        self.decorator = decorator
        self.mode = mode

    def __enter__(self) -> AutoArrayMode:
        self.decorator.activate_mode(self.mode)
        return self.mode

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.decorator.deactivate_mode()


class _BaseContext(_Base):
    """Base class for decorators with mode context management"""

    def _detect_mode(self, wrapped, instance) -> AutoArrayMode:
        """Detect the mode to activate for this decorator."""
        from .globals import get_config, get_logger

        get_logger().debug(
            f"Detecting mode for {wrapped=}, {instance=}, {isinstance(wrapped, wrapt.ObjectProxy)=}, {hasattr(wrapped, "active_mode")=}",
            stack_info=True,
        )
        raise Exception()
        return get_config().auto_array_mode

    def mode_context(self, wrapped, instance) -> _AutoArrayContext:
        """Return a context manager for this decorator."""
        return _AutoArrayContext(self, self._detect_mode(wrapped, instance))


class _InputBase(_BaseContext):
    """Base class for auto_input_array decorators"""

    def __init__(self):
        super().__init__()


class _OutputBase(_BaseContext):
    """Base class for auto_output_array decorators"""

    def __init__(self):
        super().__init__()


class _Input(_InputBase):
    def mode_context(self, wrapped, instance) -> _AutoArrayContext:
        if instance is not None and isinstance(instance, _InputBase):
            raise AutoArrayException(
                "%s is decorated multiple times ",
                "with auto_input_array decorators",
                wrapped.__name__,
            )
        return super().mode_context(wrapped, instance)


class _Output(_OutputBase):
    def __init__(self):
        super().__init__()

    def mode_context(self, wrapped, instance) -> _AutoArrayContext:
        if instance is not None and isinstance(instance, _InputBase):
            raise AutoArrayException(
                "%s is decorated multiple times ",
                "with auto_output_array decorators",
                wrapped.__name__,
            )
        return super().mode_context(wrapped, instance)


class AutoArrayXpDetectionException(AutoArrayException):
    """Base-class for exceptions related to the automatic array conversion feature."""

    def __init__(self, wrap_wrapped, wrap_instance, wrap_kwargs, *args, **kwargs):
        self.wrapped = wrap_wrapped
        self.instance = wrap_instance
        self.kwargs = wrap_kwargs
        super().__init__(*args, **kwargs)


def _universal_xp_extractor(wrapped, instance, kwargs) -> types.ModuleType:
    """
    Method detecting the xp module associated to a decorator instance.

    It detects the following situations:
      - instance methods of classes deriving from ParallelModuleBase
      - function or static method taking a use_gpu argument where
          use_gpu==False indicates a np.ndarray return value
          use_gpu==True indicated a cp.ndarray return value
    """
    if instance is None:
        if inspect.isclass(wrapped):
            raise AutoArrayXpDetectionException(
                wrapped, instance, kwargs, "Cannot use auto array decorators on a class"
            )

        if "use_gpu" in kwargs:
            if kwargs["use_gpu"]:
                import cupy

                return cupy
            return np

        raise AutoArrayXpDetectionException(
            wrapped,
            instance,
            kwargs,
            "Cannot use auto array decorators on a function without a use_gpu argument",
        )

    if inspect.isclass(instance):
        raise AutoArrayXpDetectionException(
            wrapped,
            instance,
            kwargs,
            "Cannot use auto array decorators on a classmethod",
        )

    if isinstance(instance, ParallelModuleBase):
        return instance.xp

    raise AutoArrayXpDetectionException(
        wrapped,
        instance,
        kwargs,
        "Cannot use auto array decorators on a class %s which does not derive from ParallelModuleBase",
    )


OutLocation = Sequence[int]
InLocation = Sequence[Union[int, str]]


class auto_output_array_np(_Output):
    @wrapt.decorator
    def __call__(self, wrapped, instance, args, kwargs):
        """Decorator applied on method/functions expected to return a np.ndarray."""
        with self.mode_context(wrapped, instance) as mode:
            logger = self.logger
            logger.warning(
                "Executing auto_output_array_np decorator with mode %s", mode
            )

            if mode == AutoArrayMode.STRICT:
                out = wrapped(*args, **kwargs)

                if isinstance(out, np.ndarray):
                    return out

                logger.warning(
                    "%s should have returned a np.ndarray but returned %s instead.",
                    wrapped.__name__,
                    type(out),
                    stack_info=True,
                )
                logger.debug("Converting %s output to np.ndarray", wrapped.__name__)
                return np.asarray(out)

            # Call directly wrapped since we are in either default, numpy or demo mode
            return wrapped(*args, **kwargs)


class auto_output_array_xp(_Output):
    """
    Decorator applied on method/functions expected to return a xp.ndarray

    This decorator can only be applied on:
    - instance methods of classes deriving from ParallelModuleBase
    - function or static method taking a use_gpu argument where
        use_gpu==False indicates a np.ndarray return value
        use_gpu==True indicated a cp.ndarray return value
    """

    def __init__(self):
        super().__init__()

    @wrapt.decorator
    def __call__(self, wrapped, instance, args, kwargs):
        with self.mode_context(wrapped, instance) as mode:
            logger = self.logger
            logger.warning(
                "Executing auto_output_array_xp decorator with mode %s", mode
            )

            if mode == AutoArrayMode.DEFAULT:
                return wrapped(*args, **kwargs)

            # Detect xp
            xp = _universal_xp_extractor(wrapped, instance, kwargs)

            # Apply Numpy strategy
            if mode == AutoArrayMode.NUMPY:
                out = wrapped(*args, **kwargs)
                if isinstance(out, np.ndarray):
                    return out
                return np.asarray(out)

            if mode == AutoArrayMode.STRICT:
                out = wrapped(*args, **kwargs)
                if isinstance(out, xp.ndarray):
                    return out
                logger.warning(
                    "%s: expected to return %s but return %s instead",
                    wrapped.__name__,
                    xp.ndarray,
                    type(out),
                    stack_info=True,
                )
                logger.debug("Converting %s output to %s", wrapped.__name__, xp.ndarray)
                return xp.asarray(out)

            if mode == AutoArrayMode.DEMO:
                raise NotImplementedError(
                    "DEMO mode not yet implemented for decorator auto_output_array_xp"
                )


class auto_output_array_mix(_Output):
    """
    Decorator applied on method/function returning multiple values, some of
    which are np.ndarray and/or xp.ndarray.
    """

    xp_at: OutLocation
    np_at: OutLocation

    def __init__(
        self,
        *,
        xp_at: Optional[OutLocation] = None,
        np_at: Optional[OutLocation] = None,
    ):
        self.xp_at = [] if xp_at is None else xp_at
        self.np_at = [] if np_at is None else np_at

    @wrapt.decorator
    def __call__(self, wrapped, instance, args, kwargs):
        with self.mode_context(wrapped, instance) as mode:
            logger = self.logger
            logger.warning(
                "Executing auto_output_array_mix decorator with mode %s", mode
            )
            logger.warning("NOT YET IMPLEMENTED")
            return wrapped(*args, **kwargs)


class auto_input_array(_Input):
    """
    Decorator applied on method/function taking inputs some of which are
    np.ndarray and/or xp.ndarray.
    """

    xp_at: InLocation
    np_at: InLocation

    def __init__(
        self, *, xp_at: Optional[InLocation] = None, np_at: Optional[InLocation] = None
    ):
        self.xp_at = [] if xp_at is None else xp_at
        self.np_at = [] if np_at is None else np_at

    @wrapt.decorator
    def __call__(self, wrapped, instance, args, kwargs):
        with self.mode_context(wrapped, instance) as mode:
            logger = self.logger
            logger.warning("Executing auto_input_array decorator with mode %s", mode)
            logger.warning("NOT YET IMPLEMENTED")
            return wrapped(*args, **kwargs)
