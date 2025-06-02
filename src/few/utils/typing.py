"""Definition of type hints helpers for FEW"""

from __future__ import annotations

import enum
import inspect
import textwrap
import types
import typing as t

import beartype.door
import numpy as np
import wrapt
from numpy import ndarray as np_ndarray

try:
    import cupy as cp
except ImportError:
    cp = None

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
            f"Function {wrapped.__name__} has invalid type hints for the input "
            f"and output ndarrays. Signature is {self.signature}."
        )


class FewMissingCompatibleTypeHints(FewInvalidTypeHints):
    """
    Function is decorated with auto_array but does not have type hints for the
    input and output ndarrays.
    """


class FewInvalidUnionTypeHint(FewTypingException):
    """Incompatible type hints are in the same Union"""

    def __init__(self, incompatible_hint, all_hints):
        super().__init__(
            f"Hint {incompatible_hint} is not compatible with one of {all_hints}."
        )


class FewAutoArrayInvalidTarget(FewTypingException):
    """Auto-array applied to invalid object."""


class FewAutoArrayInvalidConversion(FewTypingException):
    """Value cannot be converted into target type."""


class FewUnsupportedTypeHint(FewTypingException):
    """Type hint is not supported"""


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

    def get(self) -> np_ndarray:
        """Get method to match cp.ndarray interface"""
        return self.view(np_ndarray)

    @staticmethod
    def from_np_ndarray(other: np_ndarray) -> xp_ndarray:
        """Convert np.ndarray into gettable ndarray"""
        return other.view(xp_ndarray)


xp_ndarray_like = xp_ndarray
"""
When used as type hint, indicates that the value is either an actual xp_ndarray,
or a cp.ndarray
"""


class ConversionLevel(enum.IntEnum):
    """Type of conversions that can be applied"""

    NOOP = enum.auto()
    """No-op conversion"""

    MINIMAL = enum.auto()
    """Conversion required between np.ndarray and xp_ndarray"""

    COSTLY = enum.auto()
    """Conversion required between CPU and GPU type"""

    INVALID = enum.auto()
    """Value cannot possibly be converted into hinted type"""

    @property
    def is_invalid(self) -> bool:
        """Whether this conversion is invalid"""
        return self is ConversionLevel.INVALID

    @property
    def is_noop(self) -> bool:
        """Whether this conversion is a no-op"""
        return self is ConversionLevel.NOOP

    @property
    def is_minimal(self) -> bool:
        """Whether this conversion is minimal"""
        return self is ConversionLevel.MINIMAL

    @property
    def is_costly(self) -> bool:
        """Whether conversion has high cost"""
        return self is ConversionLevel.COSTLY


class HintBuilder(t.Protocol):
    def __call__(self, value, xp) -> HintKind:
        """Build a hint kind instance"""
        ...

    def is_union_compatible(self, other: HintBuilder) -> bool:
        """Check compatibility of two hint builders in a Union context"""
        ...

    def __hash__(self) -> int: ...


class HintKind:
    """Base class for hint kinds"""

    _conversion_level: ConversionLevel

    def __init__(self, conversion_level: ConversionLevel, value, xp):
        self._conversion_level = conversion_level
        self._value = value
        self._xp = xp
        super().__init__()

    def convert(self) -> t.Any:
        """Convert a value into the hinted type"""
        return self.value

    @property
    def conversion_level(self) -> ConversionLevel:
        return self._conversion_level

    @property
    def value(self):
        return self._value

    @property
    def xp(self):
        return self._xp


class HintNone(HintKind):
    """Hint to be used in the processing of function outputs only."""

    def __init__(self, value):
        super().__init__(ConversionLevel.NOOP, value, None)

    def convert(self):
        return self.value


class HintNP(HintKind):
    """For type hints of the form 'np.ndarray'"""

    def __init__(self, value, xp):
        super().__init__(HintNP.detect_conversion(value, xp), value, xp)

    @staticmethod
    def detect_conversion(value: t.Any, xp) -> ConversionLevel:
        if isinstance(value, (xp_ndarray, float)):
            return ConversionLevel.MINIMAL
        if isinstance(value, np_ndarray):
            return ConversionLevel.NOOP
        if (cp is not None) and (xp is cp) and isinstance(value, cp.ndarray):
            return ConversionLevel.COSTLY
        return ConversionLevel.INVALID

    def convert(self) -> np_ndarray:
        conversion = self.conversion_level
        value = self.value

        if conversion.is_noop:
            return self.value
        if conversion.is_minimal:
            # Assuming value is xp_ndarray
            if isinstance(self.value, xp_ndarray):
                return value.get()
            return np.asarray(value)
        if conversion.is_costly:
            assert isinstance(value, cp.ndarray)
            # Assuming value is cp.ndarray
            return value.get()

        raise FewAutoArrayInvalidConversion(
            f"Invalid conversion {conversion} for value {value} and "
            "type hint 'np.ndarray'."
        )

    @classmethod
    def is_union_compatible(cls, other: HintBuilder) -> bool:
        return other is not HintXP

    @classmethod
    def __hash__(cls) -> int:
        return hash(HintNP)


class HintXP(HintKind):
    """For type hints of the form 'xp_ndarray'"""

    def __init__(self, value, xp):
        super().__init__(HintXP.detect_conversion(value, xp), value, xp)

    @staticmethod
    def detect_conversion(value: t.Any, xp) -> ConversionLevel:
        if xp is np:
            # Output type must be xp_ndarray
            if isinstance(value, xp_ndarray):
                return ConversionLevel.NOOP
            if isinstance(value, (float, np_ndarray)):
                return ConversionLevel.MINIMAL
            if cp is not None and isinstance(value, cp.ndarray):
                return ConversionLevel.COSTLY
        elif cp is not None and xp is cp:
            # Output type must be cp_ndarray
            if isinstance(value, float):
                return ConversionLevel.MINIMAL
            if isinstance(value, np_ndarray):
                return ConversionLevel.COSTLY
            if isinstance(value, xp_ndarray):
                return ConversionLevel.COSTLY
            if isinstance(value, cp.ndarray):
                return ConversionLevel.NOOP
        return ConversionLevel.INVALID

    def convert(self) -> xp_ndarray_like:
        conversion = self.conversion_level
        value = self.value
        xp = self.xp

        if conversion.is_noop:
            return value

        if conversion.is_minimal:
            if isinstance(value, np_ndarray):
                return xp_ndarray.from_np_ndarray(value)
            assert isinstance(value, float)
            if xp is np:
                return xp_ndarray.from_np_ndarray(np.asarray(value))
            assert cp is not None and xp is cp
            return cp.asarray(value)
        if conversion.is_costly:
            if xp is np:
                assert cp is not None and isinstance(value, cp.ndarray)
                return xp_ndarray.from_np_ndarray(value.get())

            assert cp is not None
            return cp.asarray(value)

        raise FewAutoArrayInvalidConversion(
            f"Invalid conversion {conversion} for value {value} and "
            "type hint 'xp_ndarray'."
        )

    @classmethod
    def is_union_compatible(cls, other: HintBuilder) -> bool:
        return other is not HintNP

    @classmethod
    def __hash__(cls) -> int:
        return hash(HintXP)


class HintList(HintKind):
    """For type hints of the form 'list[X]'"""

    _itemHints: list[HintKind]

    def __init__(self, value, xp, item_hint_builder: HintBuilder):
        try:
            self._itemHints = [item_hint_builder(v, xp) for v in value]
        except TypeError:
            super().__init__(ConversionLevel.INVALID, value, xp)
            return

        conversion_level = max(hint.conversion_level for hint in self._itemHints)

        super().__init__(conversion_level, value, xp)

    def convert(self):
        return [hint.convert() for hint in self._itemHints]


class HintDict(HintKind):
    """For type hints of the form 'dict[X, Y]'."""

    _itemHints: dict[t.Any, HintKind]

    def __init__(self, value, xp, item_hint_builder: HintBuilder):
        try:
            assert isinstance(value, dict)
            self._itemHints = {k: item_hint_builder(v, xp) for k, v in value.items()}
        except (AssertionError, AttributeError):
            super().__init__(ConversionLevel.INVALID, value, xp)
            return

        conversion_level = max(
            hint.conversion_level for hint in self._itemHints.values()
        )

        super().__init__(conversion_level, value, xp)

    def convert(self):
        return {k: hint.convert() for k, hint in self._itemHints.items()}


class HintPlainBuilder:
    """HintKind builder for plain values"""

    @staticmethod
    def from_annotation(annotation: t.Any) -> HintBuilder | None:
        if annotation is inspect.Signature.empty:
            return None
        if annotation is np_ndarray:
            return HintNP
        if annotation is xp_ndarray:
            return HintXP
        return None


class HintListBuilder:
    """For type hints of the form 'list[X]' where X is a valid handled type hint"""

    _subhint: HintBuilder

    def __init__(self, subhint: HintBuilder):
        self._subhint = subhint

    def __eq__(self, other: t.Any) -> bool:
        return isinstance(other, HintListBuilder) and other.subhint == self.subhint

    def __repr__(self):
        return f"HintListBuilder[{self.subhint}]"

    def __hash__(self) -> int:
        return hash(HintListBuilder) ^ hash(self.subhint)

    @property
    def subhint(self) -> HintBuilder:
        return self._subhint

    def __call__(self, value, xp) -> HintKind:
        try:
            self._itemHints = [self.subhint(v, xp) for v in value]
        except TypeError:
            return HintKind(ConversionLevel.INVALID, value, xp)

        conversion_level = max(hint.conversion_level for hint in self._itemHints)

        return HintKind(conversion_level, value, xp)

    def is_union_compatible(self, other: HintBuilder) -> bool:
        return (
            self.subhint.is_union_compatible(other.subhint)
            if isinstance(other, HintListBuilder)
            else True
        )


class HintDictBuilder:
    """For type hints of the form 'dict[Y, X]' where X is a valid handled type hint"""

    _key_type: type
    _item_hint: HintBuilder

    def __init__(self, key_type: type, item_hint: HintBuilder):
        self._key_type = key_type
        self._item_hint = item_hint

    def __eq__(self, other: t.Any):
        return (
            isinstance(other, HintDictBuilder)
            and (self._key_type == other._key_type)
            and (other.item_hint == self.item_hint)
        )

    def __hash__(self) -> int:
        return (
            hash(HintDictBuilder)
            ^ (hash(self._key_type) << 1)
            ^ (hash(self._item_hint) << 2)
        )

    @property
    def key_type(self) -> type:
        return self._key_type

    @property
    def item_hint(self) -> HintBuilder:
        return self._item_hint

    def __repr__(self):
        return f"HintDictBuilder({self._key_type}, {self._item_hint})"

    def __call__(self, value, xp) -> HintKind:
        try:
            assert isinstance(value, dict)
            for k in value.keys():
                if not beartype.door.is_bearable(k, self.key_type):
                    return HintKind(ConversionLevel.INVALID, value, xp)
            self._itemHints = {k: HintNP(v, xp) for k, v in value.items()}
        except (AssertionError, AttributeError):
            return HintKind(ConversionLevel.INVALID, value, xp)

        conversion_level = max(
            hint.conversion_level for hint in self._itemHints.values()
        )

        return HintKind(conversion_level, value, xp)

    def is_union_compatible(self, other: HintBuilder) -> bool:
        return (
            self.item_hint.is_union_compatible(other.item_hint)
            if isinstance(other, HintDictBuilder)
            else True
        )


class HintUnionBuilder:
    """For type hints of the form 'X | Y | Z'."""

    _sub_builders: set[HintBuilder]
    _unmanaged_types: set[type]
    _allow_any: bool

    def __init__(
        self,
        sub_builders: list[HintBuilder],
        other_types: list[type],
        allow_any: bool = False,
    ):
        self._sub_builders = set(sub_builders)
        self._unmanaged_types = set(other_types)

        for type in self._unmanaged_types:
            try:
                _ = beartype.door.is_bearable(None, type)
            except TypeError as e:
                raise FewUnsupportedTypeHint(
                    f"Hint {type} not compatible with isinstance"
                ) from e

        self._allow_any = allow_any

        for sub_builder in self._sub_builders:
            for other_builder in self._sub_builders:
                if not other_builder.is_union_compatible(sub_builder):
                    raise FewInvalidUnionTypeHint(sub_builder, self._sub_builders)

    def __hash__(self):
        from functools import reduce

        hash_sub = reduce(
            lambda x, y: x ^ y, (hash(builder) for builder in self._sub_builders), 0
        )
        hahs_other_types = reduce(
            lambda x, y: x ^ y,
            (hash(other_type) for other_type in self._unmanaged_types),
            0,
        )

        return (
            hash(HintUnionBuilder)
            ^ (hash(self._allow_any) << 1)
            ^ (hash_sub << 2)
            ^ (hahs_other_types << 3)
        )

    def __repr__(self):
        return f"HintUnionBuilder(subs={self._sub_builders}, other_types={self._unmanaged_types}, allow_any={self._allow_any})"

    def __call__(self, value, xp) -> HintKind:
        if not self._sub_builders:
            return HintKind(ConversionLevel.NOOP, value, xp)
        for builder in self._sub_builders:
            if not (hint := builder(value, xp)).conversion_level.is_invalid:
                return hint
        if self._allow_any:
            return HintKind(ConversionLevel.NOOP, value, xp)

        return HintKind(
            ConversionLevel.NOOP
            if any(
                beartype.door.is_bearable(value, subtype)
                for subtype in self._unmanaged_types
            )
            else ConversionLevel.INVALID,
            value,
            xp,
        )

    def __eq__(self, other: t.Any) -> bool:
        if not isinstance(other, HintUnionBuilder):
            return False
        return (
            (self._sub_builders == other._sub_builders)
            and (self._unmanaged_types == other._unmanaged_types)
            and (self._allow_any is other._allow_any)
        )

    def is_union_compatible(self, other: HintBuilder) -> bool:
        """Check compatibility of two hint builders in a Union context"""
        return True


# """Hint for tuple containers of homogenous types"
# TUPLE_NP = TUPLE | NP  # tuple[np.ndarray, ...] or fixed length
# TUPLE_XP = TUPLE | XP  # tuple[xp.ndarray, ...] or fixed length


InputKinds = t.Dict[str, HintBuilder | None]
"""
Type for storing mapping of parameter name with their HintKind for auto_array.
"""

OutputKind = t.Union[None, HintBuilder, list[HintBuilder | None]]
"""
Output is:
- None: a value not interpretable by auto array
- HintKind: a single value whose content can be interpreted by auto_array
- Sequence[HintKind | None]: a tuple whose n-th entry has a kind given by the
                       sequence n-th value (at least one element in the sequence
                       must be a np or xp array)
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


def annotation_to_hint_kind(annotation: t.Any) -> HintBuilder | None:
    """Generic method to detect what HintKind corresponds to a given annotation"""
    if annotation is inspect.Signature.empty:
        return None

    if annotation is np_ndarray:
        return HintNP

    if annotation is xp_ndarray:
        return HintXP

    if t.get_origin(annotation) in (t.Union, types.UnionType):
        args = t.get_args(annotation)
        sub_builders = []
        other_types = []
        allow_any = False
        for arg in args:
            if (arg_builder := annotation_to_hint_kind(arg)) is None:
                if arg is t.Any:
                    allow_any = True
                else:
                    other_types.append(arg)
            else:
                sub_builders.append(arg_builder)
        if sub_builders:
            try:
                return HintUnionBuilder(
                    sub_builders=sub_builders,
                    other_types=other_types,
                    allow_any=allow_any,
                )
            except TypeError as e:
                raise FewUnsupportedAnnotation(annotation=annotation) from e
        else:
            return None

    if t.get_origin(annotation) is list:
        args = t.get_args(annotation)
        if args[0] is np_ndarray:
            return HintListBuilder(HintNP)
        if args[0] is xp_ndarray:
            return HintListBuilder(HintXP)
        if (hint_list_arg := annotation_to_hint_kind(args[0])) is not None:
            return HintListBuilder(hint_list_arg)
        return None

    if t.get_origin(annotation) is dict:
        args = t.get_args(annotation)
        if args[1] is np_ndarray:
            return HintDictBuilder(args[0], HintNP)
        if args[1] is xp_ndarray:
            return HintDictBuilder(args[0], HintXP)
        if (hint_items_arg := annotation_to_hint_kind(args[1])) is not None:
            return HintDictBuilder(args[0], hint_items_arg)
        return None

    return None


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
        if any((kind is not None for kind in output_kinds)):
            return output_kinds
        return None

    output_kind = annotation_to_hint_kind(return_annotation)

    return output_kind


class FewXpNotDeducible(FewTypingException):
    """Raised when xp cannot be deduced from execution context"""


class FewAutoArayForbiddenConversion(FewTypingException):
    """Raised when costly conversion is needed but forbidden"""


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
        source_file = inspect.getsourcefile(wrapped)

        self._wrapped_file = "unknown" if source_file is None else source_file
        self._wrapped_lineno = inspect.getsourcelines(wrapped)[1]

        self._signature = inspect.signature(wrapped, eval_str=True)

        try:
            input_kinds = _detect_input_kinds_from_(self._signature)
            output_kind = _detect_output_kind_from_(self._signature)
        except FewUnsupportedAnnotation as e:
            raise FewInvalidTypeHints(wrapped=wrapped, signature=self._signature) from e

        if (
            not any((input_kind is not None) for input_kind in input_kinds.values())
        ) and (output_kind is None):
            raise FewMissingCompatibleTypeHints(
                wrapped=wrapped, signature=self._signature
            )

        self._input_kinds = input_kinds
        self._output_kind = output_kind

    @wrapt.decorator
    def __call__(self, wrapped, instance, args, kwargs):
        """Apply type detection and conversions on call to wrapped function."""

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

        if action is AutoArrayAction.OFF:
            return

        log_level = cfg.auto_array_log_level

        if action == AutoArrayAction.OFF:
            # Shortcut in OFF mode
            return wrapped(*args, **kwargs)

        needs_self = instance is not None
        """Whether we must supply a fake "self" argument on argument binding"""

        bound_args = (
            self._signature.bind(instance, *args, **kwargs)
            if needs_self
            else self._signature.bind(*args, **kwargs)
        )
        bound_args.apply_defaults()

        xp = self.__detect_xp(instance, bound_args)
        logger.warning(f"xp is detected to be '{xp.__name__}'")

        p_bound_args = self.__process_inputs(
            xp,
            bound_args=bound_args,
            action=action,
            log_level=log_level,
        )

        p_args = p_bound_args.args[1:] if needs_self else p_bound_args.args
        p_kwargs = p_bound_args.kwargs

        out = wrapped(*p_args, **p_kwargs)

        p_out = self.__process_output(xp, out, action=action, log_level=log_level)

        return p_out

    def __detect_xp(self, instance, bound_args: inspect.BoundArguments):
        """
        Detect the xp module associated to a decorated function call
        """
        from .baseclasses import ParallelModuleBase

        if isinstance(instance, ParallelModuleBase):
            if hasattr(instance, "xp"):
                return getattr(instance, "xp")

            # We are calling __init__ if xp is not yet set
            if "force_backend" in bound_args.arguments:
                return instance.select_backend(bound_args.arguments["force_backend"]).xp

        if "use_gpu" in bound_args.arguments and isinstance(
            bound_args.arguments["use_gpu"], bool
        ):
            if bound_args.arguments["use_gpu"]:
                import cupy

                return cupy
            import numpy

            return numpy

        raise FewXpNotDeducible(
            "Could not detect xp module for call to "
            f"{self._wrapped_name} ({self._wrapped_file}:{self._wrapped_lineno}) from {instance=} and {bound_args=}."
        )

    def __process_inputs(
        self,
        xp,
        bound_args: inspect.BoundArguments,
        action: AutoArrayAction,
        log_level: int,
    ) -> inspect.BoundArguments:
        """Process given postional and keyword arguments for a specific xp module."""

        # 1. Go through each parameter in _input_kinds and detect what conversion
        #    it requires
        input_converters = {
            param_name: param_kind(bound_args.arguments[param_name], xp)
            for param_name, param_kind in self._input_kinds.items()
            if param_kind is not None
        }

        # 2. Depending on mode and needed conversions, send log message or fail
        #    program
        max_conversion_level = max(
            converter.conversion_level for converter in input_converters.values()
        )

        if max_conversion_level.is_invalid:
            raise FewAutoArrayInvalidConversion(
                "In function %s (%s:%u), parameters %s need invalid "
                "conversion to match their hinted types.",
                self._wrapped_name,
                self._wrapped_file,
                self._wrapped_lineno,
                {
                    param_name: converter
                    for param_name, converter in input_converters.items()
                    if converter.conversion_level.is_invalid
                },
                bound_args,
            )

        if max_conversion_level.is_costly and action is AutoArrayAction.FAIL:
            self.__action_input_log(input_converters, log_level=None, fail=True)

        if max_conversion_level.is_costly and action is AutoArrayAction.CONVERT:
            self.__action_input_log(input_converters, log_level=log_level, fail=False)

        # 3. Perform the needed conversions
        for param_name, converter in input_converters.items():
            bound_args.arguments[param_name] = converter.convert()

        return bound_args

    def __action_input_log(
        self, input_converters: dict[str, HintKind], log_level: int | None, fail: bool
    ):
        """If converters requires loggable operation, log them"""
        from few import get_logger

        logger = get_logger()

        parameters = [
            key
            for key, converter in input_converters.items()
            if converter.conversion_level.is_costly
        ]

        if log_level is not None:
            logger.log(
                log_level,
                "In function %s (%s:%u), parameters %s need costly "
                "conversion between np and cp arrays",
                self._wrapped_name,
                self._wrapped_file,
                self._wrapped_lineno,
                parameters,
            )

        if fail:
            raise FewAutoArayForbiddenConversion(
                "In function %s (%s:%u), parameters %s need costly "
                "conversion between np and cp arrays",
                self._wrapped_name,
                self._wrapped_file,
                self._wrapped_lineno,
                parameters,
            )

    def __process_output(self, xp, out, action: AutoArrayAction, log_level: int):
        """Process result output for a specific xp module"""
        if self._output_kind is None:
            return out

        if isinstance(self._output_kind, list):
            return self.__process_sequence_output(
                xp=xp,
                out=out,
                action=action,
                log_level=log_level,
                output_kinds=self._output_kind,
            )

        return self.__process_pure_output(
            xp=xp, out=out, action=action, log_level=log_level
        )

    def __process_pure_output(self, xp, out, action, log_level):
        """Process function output when it has a single value"""
        converter = self._output_kind(out, xp)

        if converter.conversion_level.is_noop:
            return out

        if converter.conversion_level.is_invalid:
            FewAutoArrayInvalidConversion(
                "In function %s (%s:%u), output has invalid "
                "type for conversion into its hinted types.",
                self._wrapped_name,
                self._wrapped_file,
                self._wrapped_lineno,
                out,
            )

        if converter.conversion_level.is_costly:
            if action is AutoArrayAction.FAIL:
                raise FewAutoArayForbiddenConversion(
                    "In function %s (%s:%u), output %s value needs costly "
                    "conversion between np and cp arrays",
                    self._wrapped_name,
                    self._wrapped_file,
                    self._wrapped_lineno,
                    out,
                )
            if action is AutoArrayAction.CONVERT:
                from few import get_logger

                logger = get_logger()
                logger.log(
                    log_level,
                    "In function %s (%s:%u), output %s needs costly "
                    "conversion between np and cp arrays",
                    self._wrapped_name,
                    self._wrapped_file,
                    self._wrapped_lineno,
                    out,
                )

        # At this point, we are in the context of a minimal conversion, or costly
        # conversion is authorized

        return converter.convert()

    def __process_sequence_output(
        self, xp, out, action, log_level, output_kinds: list[HintBuilder | None]
    ):
        """Process function outputs when it contains multiple values in a tuple"""
        output_converters = [
            output_kind(value, xp) if output_kind is not None else HintNone(value)
            for value, output_kind in zip(out, output_kinds)
        ]

        max_conversion_level = max(
            converter.conversion_level for converter in output_converters
        )

        if max_conversion_level.is_invalid:
            raise FewAutoArrayInvalidConversion(
                "In function %s (%s:%u), outputs at indices %s need invalid "
                "conversion to match their hinted types.",
                self._wrapped_name,
                self._wrapped_file,
                self._wrapped_lineno,
                [
                    i
                    for i, c in enumerate(output_converters)
                    if c.conversion_level.is_invalid
                ],
            )

        if max_conversion_level.is_costly and action is AutoArrayAction.FAIL:
            raise FewAutoArayForbiddenConversion(
                "In function %s (%s:%u), outputs at indices %s need costly "
                "conversion between np and cp arrays",
                self._wrapped_name,
                self._wrapped_file,
                self._wrapped_lineno,
            )

        if max_conversion_level.is_costly and action is AutoArrayAction.CONVERT:
            from few import get_logger

            logger = get_logger()
            logger.log(
                log_level,
                "In function %s (%s:%u), outputs at indices %s need costly "
                "conversion between np and cp arrays",
                self._wrapped_name,
                self._wrapped_file,
                self._wrapped_lineno,
                [
                    i
                    for i, c in enumerate(output_converters)
                    if c.conversion_level.is_costly
                ],
            )

        return tuple(k.convert() for k in output_converters)


def auto_array(wrapped):
    if not _are_auto_array_enabled():
        # shortcut that disables applying the decorator through global configuration
        return wrapped

    # The following "(wrapped)(wrapped)" call is no typo.
    # The first call builds a _auto_array_decorator instance which analyses
    #  "wrapped" signature to determine only once how its input parameter and
    #   output should be typed.
    # The second call decorates "wrapped" using a wrapt.decorator which will
    #  intercept args, kwargs and out at runtime and execute the requested
    #  action in case of type mismatch.
    return _auto_array_decorator(wrapped)(wrapped)


__all__ = ["xp_ndarray", "auto_array"]
