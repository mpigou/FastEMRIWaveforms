import typing as t

try:
    import cupy as cp
except ImportError:
    cp = None

import numpy as np
import wrapt

from few import get_config_setter
from few.amplitude import ModeIndex
from few.tests.base import FewTest
from few.utils.typing import (
    AutoArrayAction,
    ConversionLevel,
    FewAutoArrayInvalidConversion,
    FewInvalidUnionTypeHint,
    FewTypingException,
    HintDictBuilder,
    HintListBuilder,
    HintNP,
    HintUnionBuilder,
    HintXP,
    xp_ndarray,
)
from few.utils.typing import annotation_to_hint_kind as a2h


class UtilsTyping(FewTest):
    @classmethod
    def name(self) -> str:
        return "few.utils.typing annotation parsing"

    def test_parse_annotation(self):
        a2h_expected_pairs = {
            # Base unhandled types
            None: None,
            float: None,
            str: None,
            int: None,
            list: None,
            # np.ndarray type
            np.ndarray: HintNP,
            # xp_ndarray type
            xp_ndarray: HintXP,
            # Unions with |
            np.ndarray | int: HintUnionBuilder(
                sub_builders=[HintNP], other_types=[int]
            ),
            int | np.ndarray | t.Any: HintUnionBuilder(
                sub_builders=[HintNP], other_types=[int], allow_any=True
            ),
            int | np.ndarray: HintUnionBuilder(
                sub_builders=[HintNP], other_types=[int]
            ),
            xp_ndarray | float | str: HintUnionBuilder(
                sub_builders=[HintXP], other_types=[str, float]
            ),
            float | xp_ndarray | str: HintUnionBuilder(
                sub_builders=[HintXP], other_types=[str, float]
            ),
            float | str: None,
            # Unions with t.Union
            t.Union[np.ndarray, int]: HintUnionBuilder(
                sub_builders=[HintNP], other_types=[int]
            ),
            t.Union[int, np.ndarray]: HintUnionBuilder(
                sub_builders=[HintNP], other_types=[int]
            ),
            t.Union[xp_ndarray, float, str]: HintUnionBuilder(
                sub_builders=[HintXP], other_types=[str, float]
            ),
            t.Union[float, xp_ndarray, str]: HintUnionBuilder(
                sub_builders=[HintXP], other_types=[str, float]
            ),
            t.Union[float, str]: None,
            # Lists
            t.List[np.ndarray]: HintListBuilder(HintNP),
            list[np.ndarray]: HintListBuilder(HintNP),
            t.List[xp_ndarray]: HintListBuilder(HintXP),
            list[xp_ndarray]: HintListBuilder(HintXP),
            list[float | np.ndarray]: HintListBuilder(
                HintUnionBuilder(sub_builders=[HintNP], other_types=[float])
            ),
            # Dict
            t.Dict[str, np.ndarray]: HintDictBuilder(str, HintNP),
            t.Dict[float, np.ndarray]: HintDictBuilder(float, HintNP),
            dict[tuple[int, int, int], np.ndarray]: HintDictBuilder(ModeIndex, HintNP),
            t.Dict[str, xp_ndarray]: HintDictBuilder(str, HintXP),
            t.Dict[float, xp_ndarray]: HintDictBuilder(float, HintXP),
            dict[tuple[int, int, int], xp_ndarray]: HintDictBuilder(ModeIndex, HintXP),
            # Composite hints
            xp_ndarray | list[xp_ndarray]: HintUnionBuilder(
                sub_builders=[HintXP, HintListBuilder(HintXP)], other_types=[]
            ),
            xp_ndarray | dict[str, xp_ndarray | float]: HintUnionBuilder(
                sub_builders=[
                    HintXP,
                    HintDictBuilder(
                        str,
                        HintUnionBuilder(sub_builders=[HintXP], other_types=[float]),
                    ),
                ],
                other_types=[],
            ),
        }

        a2h_unssuported_annotations = [
            np.ndarray | float | xp_ndarray,
        ]

        for annotation, expected_kind in a2h_expected_pairs.items():
            self.assertEqual(
                expected_kind,
                res := a2h(annotation),
                f"annotation_to_hint_kind({annotation=})={res} [hash: {hex(hash(res))}], expected={expected_kind} [hash: {hex(hash(expected_kind))}]",
            )

        for annotation in a2h_unssuported_annotations:
            with self.assertRaises(
                FewInvalidUnionTypeHint,
                msg=f"a2h({annotation=}) should not be supported",
            ):
                _ = a2h(annotation)


class expected_conversion:
    def __init__(
        self,
        annotation,
        xp,
        value_type: type,
        expected_level: ConversionLevel,
        output_type,
    ):
        self.annotation = annotation
        self.xp = xp
        self.value_type = value_type
        self.expected_level = expected_level
        self.output_type = output_type

    @wrapt.decorator
    def __call__(self, wrapped, instance: FewTest, args, kwargs):
        hint = a2h(self.annotation)
        converter = hint(instance.values[self.value_type], self.xp)
        assert converter.conversion_level == self.expected_level
        if issubclass(self.output_type, FewTypingException):
            with instance.assertRaises(self.output_type):
                _ = converter.convert()
        else:
            assert isinstance(converter.convert(), self.output_type)
        wrapped(*args, **kwargs)


class AutoArrayCpuModeMinimal(FewTest):
    """Test auto_array conversions when in CPU mode and minimal mode."""

    float_val: float
    npndarray_val: np.ndarray

    @classmethod
    def name(self) -> str:
        return "few.utils.typing: CPU mode conversions"

    def setUp(self):
        self.values = {
            float: 2.0,
            np.ndarray: np.ones((2,)),
            xp_ndarray: xp_ndarray.from_np_ndarray(
                np.ones(
                    3,
                )
            ),
            str: "invalid",
        }

        get_config_setter(reset=True).set_auto_array_action(AutoArrayAction.MINIMAL)
        return super().setUp()

    @expected_conversion(
        annotation=np.ndarray,
        xp=np,
        value_type=float,
        expected_level=ConversionLevel.MINIMAL,
        output_type=np.ndarray,
    )
    def test_annotation_purenp_with_float_should_return_npndarray(self):
        pass

    @expected_conversion(
        annotation=np.ndarray,
        xp=np,
        value_type=np.ndarray,
        expected_level=ConversionLevel.NOOP,
        output_type=np.ndarray,
    )
    def test_annotation_purenp_with_np_should_be_noop(self):
        pass

    @expected_conversion(
        annotation=np.ndarray,
        xp=np,
        value_type=xp_ndarray,
        expected_level=ConversionLevel.MINIMAL,
        output_type=np.ndarray,
    )
    def test_annotation_purenp_with_xp_instance_should_be_minimal(self):
        pass

    @expected_conversion(
        annotation=np.ndarray,
        xp=np,
        value_type=str,
        expected_level=ConversionLevel.INVALID,
        output_type=FewAutoArrayInvalidConversion,
    )
    def test_annotation_purenp_with_str_should_be_invalid(self):
        pass

    @expected_conversion(
        annotation=xp_ndarray,
        xp=np,
        value_type=float,
        expected_level=ConversionLevel.MINIMAL,
        output_type=xp_ndarray,
    )
    def test_annotation_purexp_with_float_should_return_xpndarray(self):
        pass

    @expected_conversion(
        annotation=xp_ndarray,
        xp=np,
        value_type=np.ndarray,
        expected_level=ConversionLevel.MINIMAL,
        output_type=xp_ndarray,
    )
    def test_annotation_purexp_with_np_should_be_minimal(self):
        pass
