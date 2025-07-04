"""
Implementation of a Holder class to hold large datasets.
"""

import typing as t

from ..utils.exceptions import FewException

T = t.TypeVar("T")


class HoldReferenceAlreadyDefined(FewException):
    """Exception raised when holding data with a reference already used."""


class HoldReferenceNotFound(FewException):
    """Exception raised when viewing data with a reference not found."""


class HolderReferenceTypeMismatch(FewException):
    """Exception raised when viewing data with a reference type mismatch."""


class Holder:
    _data: dict[str, t.Any]

    def __init__(self):
        """Initialize an empty holder."""
        self._data = {}

    def has(self, reference: str) -> bool:
        """Whether the holder contains data for the given reference."""
        return reference in self._data

    def hold(self, reference: str, data: T) -> None:
        """Hold a reference to the data."""
        if reference in self._data:
            raise HoldReferenceAlreadyDefined(f"Reference {reference} already defined.")

        self._data[reference] = data

    @t.overload
    def view(self, reference: str, expected_type: None) -> t.Any: ...

    @t.overload
    def view(self, reference: str, expected_type: type[T]) -> T: ...

    def view(self, reference, expected_type=None):
        """Return the data associated with the reference."""
        if reference not in self._data:
            raise HoldReferenceNotFound(f"Data with reference '{reference}' not found.")

        data = self._data[reference]

        if expected_type is not None and not isinstance(data, expected_type):
            raise HolderReferenceTypeMismatch(
                f"Data with reference '{reference}' has type {type(data)}, expected {expected_type}."
            )

        return data

    @t.overload
    def pop(self, reference: str, expected_type: None) -> t.Any: ...

    @t.overload
    def pop(self, reference: str, expected_type: type[T]) -> T: ...

    def pop(self, reference, expected_type=None):
        """Remove the data associated with the reference."""
        if reference not in self._data:
            raise HoldReferenceNotFound(f"Data with reference '{reference}' not found.")

        if expected_type is not None and not isinstance(
            self._data[reference], expected_type
        ):
            raise HolderReferenceTypeMismatch(
                f"Data with reference '{reference}' has type {type(self._data[reference])}, expected {expected_type}."
            )

        return self._data.pop(reference)
