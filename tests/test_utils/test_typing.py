import typing as t

try:
    import cupy as cp
except ImportError:
    cp = None

import numpy as np

from few.tests.base import FewTest
from few.utils.typing import FewUnsupportedAnnotation, Hint, xp_ndarray
from few.utils.typing import annotation_to_hint_kind as a2h


class UtilsTyping(FewTest):
    @classmethod
    def name(self) -> str:
        return "few.utils.typing tests"

    def test_parse_annotation(self):
        a2h_expected_pairs = {
            # Base unhandled types
            None: None,
            float: None,
            str: None,
            int: None,
            list: None,
            # np.ndarray type
            np.ndarray: Hint.NP,
            # xp_ndarray type
            xp_ndarray: Hint.XP,
            # Unions with |
            np.ndarray | int: Hint.UNION_NP,
            int | np.ndarray: Hint.UNION_NP,
            xp_ndarray | float | str: Hint.UNION_XP,
            float | xp_ndarray | str: Hint.UNION_XP,
            float | str: None,
            # Unions with t.Union
            t.Union[np.ndarray, int]: Hint.UNION_NP,
            t.Union[int, np.ndarray]: Hint.UNION_NP,
            t.Union[xp_ndarray, float, str]: Hint.UNION_XP,
            t.Union[float, xp_ndarray, str]: Hint.UNION_XP,
            t.Union[float, str]: None,
            # Lists
            t.List[np.ndarray]: Hint.LIST_NP,
            list[np.ndarray]: Hint.LIST_NP,
            t.List[xp_ndarray]: Hint.LIST_XP,
            list[xp_ndarray]: Hint.LIST_XP,
            # Dict
            t.Dict[str, np.ndarray]: Hint.DICT_NP,
            t.Dict[t.Any, np.ndarray]: Hint.DICT_NP,
            dict[tuple[int, int, int], np.ndarray]: Hint.DICT_NP,
            t.Dict[str, xp_ndarray]: Hint.DICT_XP,
            t.Dict[t.Any, xp_ndarray]: Hint.DICT_XP,
            dict[tuple[int, int, int], xp_ndarray]: Hint.DICT_XP,
        }

        a2h_unssuported_annotations = [
            np.ndarray | float | xp_ndarray,
        ]

        for annotation, expected_kind in a2h_expected_pairs.items():
            self.assertIs(
                a2h(annotation),
                expected_kind,
                f"annotation_to_hint_kind({annotation=})={a2h(annotation)}, expected={expected_kind}",
            )

        for annotation in a2h_unssuported_annotations:
            with self.assertRaises(
                FewUnsupportedAnnotation,
                msg=f"a2h({annotation=}) should not be supported",
            ):
                _ = a2h(annotation)

        self.assertIs(a2h(list[str]), None)
        self.assertIs(a2h(list[np.ndarray]), Hint.LIST_NP)
