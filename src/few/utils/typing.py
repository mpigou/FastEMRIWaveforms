"""Definition of type hints helpers for FEW"""

from __future__ import annotations

import enum
import inspect
import textwrap
import typing as t

import wrapt
from numpy import ndarray as np_ndarray

from .exceptions import FewException


class FewTypingException(FewException):
    """Base class for few.utils.typing exceptions"""


class FewInvalidTypeHints(FewTypingException):
    wrapped: t.Callable
    signature: inspect.Signature

    def __init__(self, wrapped: t.Callable, signature: inspect.Signature):
        self.wrapped = wrapped
        self.signature = signature
        super().__init__(
            f"Function {wrapped.__name__} has invalid type hints "
            "for the input and output ndarrays. "
            f"Signature is {self.signature}."
        )


class FewMissingCompatibleTypeHints(FewInvalidTypeHints):
    """Function is decorated with auto_array but does not have type hints for the input and output ndarrays."""


class FewUnsupportedTypeHint(FewInvalidTypeHints):
    """Function has not-supported type hints in its signature"""


class FewAutoArrayInvalidTarget(FewTypingException):
    """Auto-array applied to invalid object."""


class FewAutoArrayInvalidConversion(FewTypingException):
    """Value cannot be converted into target type."""


class AutoArrayAction(enum.Enum):
    """Enumeration of actions to apply through the auto_array decorator"""

    OFF = "off"
    """Do not apply any action"""

    MINIMAL = "minimal"
    """Apply only np.ndarray to xp_ndarray conversion"""

    CONVERT = "convert"
    """Same as minimal + in GPU mode, convert and log np.ndarray to/from cp.ndarray"""

    FAIL = "fail"
    """Same as minimal + in GPU mode, fail if np to/from cp conversion is needed"""

    DEFAULT = CONVERT


class xp_ndarray(np_ndarray):
    """Class to be use as type hint for arrays that can be either np or xp ndarray"""

    def get(self):
        """No-op get method to match cp.ndarray interface"""
        return self

    @staticmethod
    def from_np_ndarray(other: np_ndarray) -> xp_ndarray:
        """Convert np.ndarray into gettable ndarray"""
        return other.view(xp_ndarray)

    @staticmethod
    def from_any_ndarray(other: xp_ndarray) -> xp_ndarray:
        """Convert np.ndarray into gettable ndarray or return cp.ndarray directly"""
        return other.view(xp_ndarray) if isinstance(other, np_ndarray) else other


class _Conversions(enum.Enum):
    """Type of conversions that can be applied"""

    NONE = enum.auto()
    """No-op conversion"""

    NP2XP = enum.auto()
    """Convert np.ndarray to actual xp_ndarray instance"""

    NP2CP = enum.auto()
    """Convert np.ndarray or xp_ndarray to cp.ndarray"""

    CP2XP = enum.auto()
    """Convert xp.ndarray to xp_ndarray instance"""

    INVALID = enum.auto()
    """Value cannot possibly be converted into hinted type"""


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
    UNION_NP = UNION | NP  # t.Union[np.ndarray, ...]; t.Optional[np.ndarray]
    UNION_XP = UNION | XP  # t.Union[xp_ndarray, ...]; t.Optional[xp_ndarray]

    LIST_PURE_NP = LIST | PLAIN_NP  # list[np.ndarray]
    LIST_PURE_XP = LIST | PLAIN_XP  # list[xp_ndarray]

    DICT_PURE_NP = DICT | PLAIN_NP  # dict[t.Any, np.ndarray]
    DICT_PURE_XP = DICT | PLAIN_XP  # dict[t.Any, xp_ndarray]

    # To add later
    # LIST_MIX_NP = LIST | UNION_NP  # list[np.ndarray | other | ...]
    # LIST_MIX_XP = LIST | UNION_XP  # list[xp_ndarray | other | ...]
    # DICT_MIX_NP = DICT | UNION_NP  # dict[np.ndarray | other | ...]
    # DICT_MIX_XP = DICT | UNION_XP  # dict[xp_ndarray | other | ...]
    # TUPLE = enum.auto()
    # """Hint for tuple containers of homogenous types"
    #
    # TUPLE_NP = TUPLE | NP  # tuple[np.ndarray, ...] or fixed length
    # TUPLE_XP = TUPLE | XP  # tuple[xp.ndarray, ...] or fixed length

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

    def from_value(self, value: t.Any, xp):
        """Convert arrays from a value into self type"""
        if isinstance(value, xp_ndarray):
            return self._from_xp_value(value, xp)

        if isinstance(value, np_ndarray):
            return self._from_np_value(value, xp)

        return self._from_cp_value(value, xp)

    def _from_np_value(self, value: np_ndarray, xp):
        """Convert arrays from a np.ndarray value into self type"""
        raise NotImplementedError

    def _from_xp_value(self, value: xp_ndarray, xp):
        """Convert arrays from an actual xp_ndarray value into self type"""
        raise NotImplementedError

    def _from_cp_value(self, value: xp_ndarray, xp):
        """Convert arrays from a cp.ndarray (passing as xp_ndarray) value into self type"""
        raise NotImplementedError


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

    Note however that this will break the code whenever the assumption is made
    that an array is either a xp_ndarray instance (np.ndarray + phony get()
    method) or a cp.ndarray since, in CPU mode, without this decorator, only bare
    np.ndarray will be passed -> ONLY DISABLE THE DECORATOR IF IN GPU MODE WITH
    ONLY GPU-ENABLED OBJECTS.
    """
    import os

    return os.getenv("FEW_DISABLE_AUTO_ARRAY", None) is None


class FewUnsupportedAnnotation(FewTypingException):
    """Raised when an annotation is not supported by FEW auto_array decorator"""

    annotation: t.Any

    def __init__(self, /, *args, annotation: t.Any, **kwargs):
        self.annotation = annotation
        super().__init__(*args, **kwargs)


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
        if (hint & HintKind.NP) and (hint & HintKind.XP):
            raise FewUnsupportedAnnotation(
                "auto_array cannot be used with Union[np.ndarray, xp_ndarray] annotations",
                annotation=annotation,
            )
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


class FewXpNotDeducible(FewTypingException):
    """Raised when xp cannot be deduced from execution context"""


class _auto_array_decorator:
    _wrapped_name: str
    """Name of the wrapped callable (for logging purposes)"""

    _wrapped_file: str
    """Definition file of the wraped callable (for logging purposes)"""

    _wrapped_lineno: int
    """Definition line of the wraped callable (for logging purposes)"""

    _signature: inspect.Signature
    """Signature of the wrapped callable (for parameter binding)"""

    _input_kinds: InputKinds
    """Precomputed kind of callable input parameters"""

    _output_kind: OutputKind
    """Precomputed kind of callable output value"""

    def __init__(self, wrapped: t.Callable):
        if (not inspect.isfunction(wrapped)) and (not inspect.ismethod(wrapped)):
            raise FewAutoArrayInvalidTarget(
                "@auto_array decorator should only "
                "be applied on free-functions or bound methods"
            )

        self._wrapped_name = wrapped.__name__
        self._wrapped_file = inspect.getsourcefile(wrapped)
        self._wrapped_lineno = inspect.getsourcelines(wrapped)[1]

        self._signature = inspect.signature(wrapped, eval_str=True)

        try:
            input_kinds = _detect_input_kinds_from_(self._signature)
            output_kind = _detect_output_kind_from_(self._signature)
        except FewUnsupportedAnnotation as e:
            raise FewInvalidTypeHints(wrapped=wrapped, signature=self._signature) from e

        if (
            not any(
                (input_kind != HintKind.NONE) for input_kind in input_kinds.values()
            )
        ) and (output_kind is None):
            raise FewMissingCompatibleTypeHints(
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
            Calling {self._wrapped_name} (from {self._wrapped_file}:{self._wrapped_lineno}) through auto_array decorator with
                - input_kinds: {self._input_kinds}
                - output_kind: {self._output_kind}""")
        )

        cfg = get_config()
        action = cfg.auto_array_action
        log_level = cfg.auto_array_log_level

        if action == AutoArrayAction.OFF:
            # Shortcut in OFF mode
            return wrapped(*args, **kwargs)

        xp = self.__detect_xp(instance, kwargs)
        logger.warning(f"xp is detected to be '{xp.__name__}'")

        p_args, p_kwargs = self.__process_inputs(
            xp,
            args,
            kwargs,
            fake_self=instance is not None,
            action=action,
            log_level=log_level,
        )

        out = wrapped(*p_args, **p_kwargs)

        p_out = self.__process_output(xp, out, action=action, log_level=log_level)

        return p_out

    def __detect_xp(self, instance, kwargs):
        """
        Detect the xp module associated to a decorated function call
        """
        from .baseclasses import ParallelModuleBase

        if isinstance(instance, ParallelModuleBase):
            if hasattr(instance, "xp"):
                return getattr(instance, "xp")

            # We are calling __init__ if xp is not yet set
            return instance.select_backend(
                kwargs["force_backend"] if "force_backend" in kwargs else None
            ).xp

        if "use_gpu" in kwargs:
            if kwargs["use_gpu"]:
                import cupy

                return cupy
            import numpy

            return numpy

        raise FewXpNotDeducible(
            "Could not detect xp module for call to "
            f"{self._wrapped_name} ({self._wrapped_file}:{self._wrapped_lineno})."
        )

    def __process_inputs(
        self, xp, args, kwargs, fake_self: bool, action: AutoArrayAction, log_level: int
    ):
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
            pass
            # bound_args.arguments[param_name] = bound_args.arguments[param_name]

        # 3 - Return resulting arguments
        return bound_args.args[1:] if fake_self else bound_args.args, bound_args.kwargs

    def __process_output(self, xp, out, action: AutoArrayAction, log_level: int):
        """Process result output for a specific xp module"""
        if self._output_kind is None:
            return out

        if isinstance(self._output_kind, HintKind):
            return out  # __process_value(xp=xp, value=out, kind=output_kind, mode=mode)

        return tuple(
            o  # __process_value(xp=xp, value=o, kind=k, mode=mode, input=False)
            for o, k in zip(out, self._output_kind)
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
