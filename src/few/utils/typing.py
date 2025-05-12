"""Definition of type hints helpers for FEW"""

from __future__ import annotations

import enum
import inspect
import logging
import textwrap
import typing as t

import wrapt
from numpy import ndarray as np_ndarray

from .exceptions import FewException


class FewTypingException(FewException):
    """Base class for few.utils.typing exceptions"""


class FewMissingTypeHintsExceptions(FewTypingException):
    """Function is decorated with auto_array but does not have type hints for the input and output ndarrays."""

    wrapped: t.Callable

    def __init__(self, wrapped: t.Callable, signature: inspect.Signature):
        self.wrapped = wrapped
        self.signature = signature

        super().__init__(
            textwrap.dedent(f"""
    From "{inspect.getsourcefile(wrapped)}:{inspect.getsourcelines(wrapped)[1]}"):
        Function {wrapped.__name__} is decorated with @auto_array but does not
        have type hints for the input and output ndarrays.
        It's signature is {self.signature}.""")
        )


class FewAutoArrayInvalidTarget(FewTypingException):
    """Auto-array applied to invalid object."""


class FewAutoArrayInvalidConversion(FewTypingException):
    """Value cannot be converted into target type."""


class AutoArrayMode(enum.Enum):
    """Enumeration of modes defining the behaviour or @auto_array decorator."""

    NOOP = "noop"
    """Does nothing"""

    WARN = "warn"
    """Raise warning when unexpected array types are encountered, do not convert"""

    RAISE = "raise"
    """Raise an exception when unexpected array types are encountered"""

    ENFORCE = "enforce"
    """Convert any input/output type to the expected type"""

    NUMPY = "numpy"
    """Convert all function outputs to np.ndarray and all inputs to their expected type"""

    DEFAULT = NOOP


class xp_ndarray(np_ndarray):
    """Class to be use as type hint for arrays that can be either np or xp ndarray"""

    pass


class gettable_ndarray(np_ndarray):
    """Class adding a get() method to np.ndarray to match cp.ndarray interface"""

    def get(self):
        return self

    @staticmethod
    def from_np_ndarray(other: np_ndarray) -> gettable_ndarray:
        """Convert np.ndarray into gettable ndarray"""
        return other.view(gettable_ndarray)

    @staticmethod
    def from_xp_ndarray(other: xp_ndarray) -> gettable_ndarray:
        """Convert np.ndarray into gettable ndarray or return cp.ndarray directly"""
        return other.view(gettable_ndarray) if isinstance(other, np_ndarray) else other


class HintKind(enum.Flag):
    NP = enum.auto()
    """Hint contains np.ndarray"""

    XP = enum.auto()
    """Hint contains xp_ndarray"""

    PLAIN = enum.auto()
    """Hint is for a plain value"""

    UNION = enum.auto()
    """Hint is for a Union between multiple type including ndarray"""

    LIST = enum.auto()
    """Hint is for a list container of only ndarray values"""

    DICT = enum.auto()
    """Hint is for dict container of only ndarray values"""

    NONE = 0
    """Hint is neither of previous options"""

    PLAIN_NP = PLAIN | NP  # np.ndarray
    PLAIN_XP = PLAIN | XP  # xp_ndarray
    UNION_NP_ONLY = UNION | NP  # t.Union[np.ndarray, ...]; t.Optional[np.ndarray]
    UNION_XP_ONLY = UNION | XP  # t.Union[xp_ndarray, ...]; t.Optional[xp_ndarray]
    UNION_MIX = UNION | NP | XP  # t.Union[np.ndarray, xp_ndarray, ...]

    LIST_PURE_NP = LIST | PLAIN_NP  # list[np.ndarray]
    LIST_PURE_XP = LIST | PLAIN_XP  # list[xp_ndarray]

    DICT_PURE_NP = DICT | PLAIN_NP  # dict[t.Any, np.ndarray]
    DICT_PURE_XP = DICT | PLAIN_XP  # dict[t.Any, xp_ndarray]

    def as_np_variant(self) -> HintKind:
        """Convert a xp or mix hint kind to np variant."""
        return self ^ HintKind.XP | HintKind.NP

    def isa(self, value: t.Any, xp) -> bool:
        """Check if a value matched the hint"""
        if self & HintKind.DICT:
            return isinstance(value, dict) and all(
                isinstance(v, np_ndarray if (self & HintKind.NP) else xp.ndarray)
                for v in value.values()
            )

        if self & HintKind.LIST:
            return isinstance(value, list) and all(
                isinstance(v, np_ndarray if (self & HintKind.NP) else xp.ndarray)
                for v in value
            )

        if isinstance(value, np_ndarray):
            return (self & (HintKind.PLAIN | HintKind.UNION)) and (self & HintKind.NP)

        if isinstance(value, xp.ndarray):
            return (self & (HintKind.PLAIN | HintKind.UNION)) and (self & HintKind.XP)

        return False

    def convert_into(self, value: t.Any, xp):
        """Convert arrays from a value into self type"""
        raise NotImplementedError
        # Convert from any np/xp ndarray into self type
        # Warning: when converting from xp to np, remember that .get() calls
        # might be necessary (if xp is not np)


InputKinds = t.Dict[str, HintKind]
"""
Type for storing mapping of parameter name with their HintKind for auto_array.
"""

OutputKind = t.Union[None, HintKind, t.Sequence[HintKind]]
"""
Output is:
- None: a value not interpretable by auto array
- HintKind: a single value whose content can be interpreted by auto_array
- Sequence[HintKind]: a tuple whose n-th entry has a kind given by the sequence
                       n-th value (at least one element in the sequence must
                       be a np or xp array)
"""


def _are_auto_array_enabled() -> bool:
    """
    This method is called at import time to detect whether auto_array decorator
    should be applied or completely skipped.

    If the environment variable FEW_DISABLE_AUTO_ARRAY is set to **ANY** value,
    then, the auto_array decorators are completely disabled, thus reducing their
    impact on performances to zero.
    """
    import os

    return os.getenv("FEW_DISABLE_AUTO_ARRAY", None) is None


def annotation_to_hint_kind(annotation: t.Any) -> HintKind:
    """Generic method to detect what HintKind corresponds to a given annotation"""
    if annotation is inspect.Signature.empty:
        return HintKind.NONE

    if annotation is np_ndarray:
        return HintKind.PLAIN_NP

    if annotation is xp_ndarray:
        return HintKind.PLAIN_XP

    if t.get_origin(annotation) is t.Union:
        args = t.get_args(annotation)
        hint = HintKind.UNION
        if np_ndarray in args:
            hint |= HintKind.NP
        if xp_ndarray in args:
            hint |= HintKind.XP
        if hint == HintKind.UNION:
            """The Union does not contain np or xp array"""
            return HintKind.NONE
        return hint

    if t.get_origin(annotation) is list:
        args = t.get_args(annotation)
        if args[0] is np_ndarray:
            return HintKind.LIST_PURE_NP
        if args[0] is xp_ndarray:
            return HintKind.LIST_PURE_XP
        return HintKind.NONE

    if t.get_origin(annotation) is dict:
        args = t.get_args(annotation)
        if args[1] is np_ndarray:
            return HintKind.DICT_PURE_NP
        if args[1] is xp_ndarray:
            return HintKind.DICT_PURE_XP
        return HintKind.NONE

    return HintKind.NONE


def _detect_input_kinds_from_(signature: inspect.Signature) -> InputKinds:
    """Build a mapping of parameter names to hint kind"""
    return {
        param.name: annotation_to_hint_kind(param.annotation)
        for param in signature.parameters.values()
    }


def _detect_output_kind_from_(signature: inspect.Signature) -> OutputKind:
    """Detect the hint kind of return type hint"""
    return_annotation = signature.return_annotation

    if t.get_origin(return_annotation) is tuple:
        output_kinds = [
            annotation_to_hint_kind(hint) for hint in t.get_args(return_annotation)
        ]
        if any((kind != HintKind.NONE for kind in output_kinds)):
            return output_kinds
        return None

    output_kind = annotation_to_hint_kind(return_annotation)

    return None if output_kind == HintKind.NONE else output_kind


def _detect_wrapped_in_out(
    signature: inspect.Signature,
) -> t.Tuple[InputKinds, OutputKind]:
    """Detect the in and output ndarray (np and xp) from a function type hints."""
    return _detect_input_kinds_from_(signature), _detect_output_kind_from_(signature)


def __process_value_convert(xp, value, kind: HintKind) -> t.Any:
    pass


def __process_value_log_convert(
    xp, value, kind: HintKind, level, convert: bool
) -> t.Any:
    from few import get_logger

    matches: bool = kind.isa(value, xp)
    if matches:
        return value

    get_logger().log(
        level,
        "  parameter '%s' is not of expected type.",
    )

    pass


def __process_value(
    xp, value, kind: HintKind, mode: AutoArrayMode, input: bool
) -> t.Any:
    """
    Process a single input or output value according to its hint kind and current mode

    This method will compare a value type with its expected type and act based
    on the result and current mode.
    It will return a value that must be passed as input to the wrapped function,
    or returned as output of that function.
    """
    if mode == AutoArrayMode.WARN:
        return __process_value_log_convert(
            xp, value, kind, level=logging.WARNING, convert=False
        )
    if mode == AutoArrayMode.RAISE:
        return __process_value_log_convert(
            xp, value, kind, level=logging.FATAL, convert=False
        )
    if mode == AutoArrayMode.ENFORCE:
        return __process_value_log_convert(
            xp, value, kind, level=logging.DEBUG, convert=True
        )
    if mode == AutoArrayMode.NUMPY:
        if input:
            return __process_value_convert(xp, value, kind)
        return __process_value_convert(xp, value, kind ^ HintKind.XP | HintKind.NP)

    return value


class _auto_array_decorator:
    _signature: inspect.Signature
    _input_kinds: InputKinds
    _output_kind: OutputKind

    def __init__(self, wrapped: t.Callable):
        if not _are_auto_array_enabled():
            return

        if (not inspect.isfunction(wrapped)) and (not inspect.ismethod(wrapped)):
            raise FewAutoArrayInvalidTarget(
                "@auto_array decorator should only "
                "be applied on free-functions or bound methods"
            )

        self._signature = inspect.signature(wrapped, eval_str=True)

        input_kinds, output_kind = _detect_wrapped_in_out(self._signature)

        if (
            not any(
                (input_kind != HintKind.NONE) for input_kind in input_kinds.values()
            )
        ) and (output_kind is None):
            raise FewMissingTypeHintsExceptions(
                wrapped=wrapped, signature=self._signature
            )

        self._input_kinds = input_kinds
        self._output_kind = output_kind

    @wrapt.decorator
    def __call__(self, wrapped, instance, args, kwargs):
        from few import get_config, get_logger

        logger = get_logger()
        logger.warning(
            textwrap.dedent(f"""
            Calling {wrapped.__name__} (from {inspect.getsourcefile(wrapped)}:{inspect.getsourcelines(wrapped)[1]}) through auto_array decorator with
                - input_kinds: {self._input_kinds}
                - output_kind: {self._output_kind}""")
        )

        mode = get_config().auto_array_mode

        if mode == AutoArrayMode.NOOP:
            # Shortcut in no-op mode
            return wrapped(*args, **kwargs)

        xp = self.__detect_xp(instance, kwargs)
        logger.warning(f"xp is detected to be '{xp.__name__}'")

        p_args, p_kwargs = self.__process_inputs(
            xp, args, kwargs, fake_self=instance is not None
        )

        out = wrapped(*p_args, **p_kwargs)

        p_out = self.__process_output(xp, out)

        return p_out

    def __detect_xp(self, instance, kwargs):
        """
        Detect the xp module associated to a decorated function call
        """
        if hasattr(instance, "xp"):
            return getattr(instance, "xp")

        if "use_gpu" in kwargs:
            if kwargs["use_gpu"]:
                import cupy

                return cupy
            import numpy

            return numpy

    def __process_inputs(self, xp, args, kwargs, fake_self: bool, mode: AutoArrayMode):
        """Process given postional and keyword arguments for a specific xp module."""
        # 1 - Bind input arguments to function signature
        bound_args = (
            self._signature.bind(None, *args, **kwargs)
            if fake_self
            else self._signature.bind(*args, **kwargs)
        )
        bound_args.apply_defaults()

        # 2 - Iterate over arguments to process and update them if necessary
        for param_name, param_kind in self._input_kinds.items():
            bound_args.arguments[param_name] = self.__process_value(
                xp=xp,
                value=bound_args.arguments[param_name],
                kind=param_kind,
                mode=mode,
                input=True,
            )

        # 3 - Return resulting arguments
        return bound_args.args[1:] if fake_self else bound_args.args, bound_args.kwargs

    def __process_output(self, xp, out, output_kind: OutputKind, mode: AutoArrayMode):
        """Process result output for a specific xp module"""
        if output_kind is None:
            return out

        if output_kind is HintKind:
            return __process_value(xp=xp, value=out, kind=output_kind, mode=mode)

        return tuple(
            __process_value(xp=xp, value=o, kind=k, mode=mode, input=False)
            for o, k in zip(out, output_kind)
        )


def auto_array(wrapped):
    if not _are_auto_array_enabled():
        # shortcut that disables applying the decorator through global configuration
        return wrapped

    # The following "(wrapped)(wrapped)" call is no typo.
    # The first call builds a _auto_array_decorator instance which analyses
    #  "wrapped" signature to determine how its input parameter and output should
    #  be typed.
    # The second call decorates "wrapped" using a wrapt.decorator which will
    #  intercept args, kwargs and out at runtime and apply the requested mode.
    return _auto_array_decorator(wrapped)(wrapped)


__all__ = [xp_ndarray, auto_array]
