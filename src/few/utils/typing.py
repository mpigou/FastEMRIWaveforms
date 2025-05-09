"""Definition of type hints helpers for FEW"""

import inspect
import textwrap
import typing as t

import wrapt
from numpy import ndarray as np_ndarray

from .exceptions import FewException


class xp_ndarray(np_ndarray):
    pass


class xp:
    ndarray = xp_ndarray


input_at = t.Optional[t.Sequence[t.Union[int, str]]]
"""
Input np or xp array are either not set (None), set by args locations (int)
or set by kwargs keys (str).
"""

output_is = t.Union[None, bool, t.Sequence[t.Union[None, bool]]]
"""
Output is:
- None: not a np nor xp ndarray
- False: a np.ndarray
- True: a xp.ndarray
- Sequence[None|bool]: a tuple whose n-th entry has type given by the sequence
                       n-th value following previous rules
"""


def _are_auto_array_enabled() -> bool:
    import os

    return os.getenv("FEW_DISABLE_AUTO_ARRAY", None) is None


class _auto_array_decorator:
    _input_np: input_at
    _input_xp: input_at
    _output: output_is

    def __init__(
        self,
        input_np: input_at = None,
        input_xp: input_at = None,
        output: output_is = None,
    ):
        self._input_np = input_np
        self._input_xp = input_xp
        self._output = output

    @wrapt.decorator(enabled=_are_auto_array_enabled)
    def __call__(self, wrapped, instance, args, kwargs):
        from few import get_logger

        logger = get_logger()
        logger.warning(
            textwrap.dedent(f"""
            Calling {wrapped.__name__} through auto_array decorator with
                - input_np: {self._input_np}
                - input_xp: {self._input_xp}
                - output: {self._output}""")
        )

        return wrapped(*args, **kwargs)


def _is_annotation_np(annotation: t.Any) -> bool:
    return annotation is np_ndarray


def _is_annotation_xp(annotation: t.Any) -> bool:
    return annotation is xp_ndarray


def _build_locations(index: int, param: inspect.Parameter) -> input_at:
    if param.kind == param.POSITIONAL_ONLY:
        return [index]
    if param.kind == param.POSITIONAL_OR_KEYWORD:
        return [index, param.name]
    if param.kind == param.KEYWORD_ONLY:
        return [param.name]
    # We do not (yet?) support arrays transmitted via variaditc positionals or
    # keyword parameters (*args or **kwargs)
    return []


def _detect_in_from_(sig: inspect.Signature) -> t.Tuple[input_at, input_at]:
    np_at: input_at = []
    xp_at: input_at = []

    for idx, param in enumerate(sig.parameters.values()):
        annotation = param.annotation
        if annotation is inspect.Signature.empty:
            continue
        if _is_annotation_np(annotation):
            np_at.extend(_build_locations(idx, param))
        if _is_annotation_xp(annotation):
            xp_at.extend(_build_locations(idx, param))

    return np_at if np_at else None, xp_at if xp_at else None


def _detect_out_from_(sig: inspect.Signature) -> output_is:
    return_annotation = sig.return_annotation
    if return_annotation is inspect.Signature.empty:
        return None

    if _is_annotation_np(return_annotation):
        return False

    if _is_annotation_xp(return_annotation):
        return True

    return None


def _detect_wrapped_in_out(
    wrapped: t.Callable,
) -> t.Tuple[input_at, input_at, output_is]:
    """Detect the in and output ndarray (np and xp) from a function type hints."""
    signature = inspect.signature(wrapped, eval_str=True)

    return *_detect_in_from_(signature), _detect_out_from_(signature)


class FewTypingException(FewException):
    """Base class for few.utils.typing exceptions"""


class FewMissingTypeHintsExceptions(FewTypingException):
    """Function is decorated with auto_array but does not have type hints for the input and output ndarrays."""

    wrapped: t.Callable

    def __init__(self, wrapped: t.Callable):
        self.wrapped = wrapped

        self.signature = inspect.signature(wrapped, eval_str=True)
        super().__init__(
            textwrap.dedent(f"""
    From "{inspect.getsourcefile(wrapped)}:{inspect.getsourcelines(wrapped)[1]}"):
        Function {wrapped.__name__} is decorated with @auto_array but does not
        have type hints for the input and output ndarrays.
        It's signature is {self.signature}.""")
        )


def auto_array(wrapped):
    if not _are_auto_array_enabled():
        return wrapped

    input_np, input_xp, output = _detect_wrapped_in_out(wrapped)

    if (input_np is None) and (input_xp is None) and (output is None):
        raise FewMissingTypeHintsExceptions(wrapped=wrapped)

    return _auto_array_decorator(input_np=input_np, input_xp=input_xp, output=output)(
        wrapped
    )


__all__ = [xp, xp_ndarray, auto_array]
