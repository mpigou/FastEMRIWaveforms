"""
Implementation of a Holder class that loads data by itself on demand.
"""

from __future__ import annotations

import dataclasses
import typing as t

from ..utils.exceptions import FewException
from . import format
from .holder import Holder, T

if t.TYPE_CHECKING:
    from ..cutils import Backend


@dataclasses.dataclass
class FileDefinition:
    name: str
    builder: t.Callable[[str, Backend], t.Any]


FILE_DEFINITIONS: dict[str, FileDefinition[t.Any]] = {
    "ZNAmps_l10_m10_n55_DS2Outer": FileDefinition(
        "ZNAmps_l10_m10_n55_DS2Outer.h5", format.amp_lmn.build
    )
}


class ReferenceNotOnDemand(FewException):
    """Exception raised when requested file does not support on-demand loading."""


class OnDemandHolder(Holder):
    """
    Implementation of a Holder class that loads data by itself on demand.
    """

    _backend: Backend
    """Backend of the OnDemandHolder."""

    def __init__(self, backend: Backend):
        self._backend = backend
        super().__init__()

    @t.overload
    def view(self, reference: str, expected_type: None) -> t.Any: ...

    @t.overload
    def view(self, reference: str, expected_type: type[T]) -> T: ...

    def view(self, reference, expected_type=None):
        if reference not in FILE_DEFINITIONS:
            raise ReferenceNotOnDemand(f"File {reference} cannot be loaded on-demand.")
        if reference not in self._data:
            definition = FILE_DEFINITIONS[reference]
            self.hold(reference, definition.builder(definition.name, self._backend))
        return super().view(reference, expected_type)

    def has(self, reference: str) -> bool:
        """Whether the holder has data for the given reference."""
        if super().has(reference=reference):
            return True
        return reference in FILE_DEFINITIONS
