"""Classes for representing comparisons such as x < y, x >= y, semi-comparisons etc."""
from typing import Type

from galois import GF, FieldArray

from oraqle.compiler.arithmetic.subtraction import Subtraction
from oraqle.compiler.boolean.bool_neg import Neg
from oraqle.compiler.circuit import Circuit
from oraqle.compiler.comparison.in_upper_half import IliashenkoZuccaInUpperHalf, InUpperHalf
from oraqle.compiler.nodes.abstract import CostParetoFront, Node, iterate_increasing_depth
from oraqle.compiler.nodes.leafs import Constant, Input
from oraqle.compiler.nodes.non_commutative import NonCommutativeBinaryNode


class AbstractComparison(NonCommutativeBinaryNode):
    """An abstract class for comparisons, representing that they can be flipped: i.e. x > y <=> y < x."""

    def __init__(self, left, right, less_than: bool, gf: Type[FieldArray]):
        """Initialize an abstract comparison, indicating the direction of the comparison by specifying `less_than`."""
        self._less_than = less_than
        super().__init__(left, right, gf)

    def __hash__(self) -> int:
        if self._hash is None:
            left_hash = hash(self._left)
            right_hash = hash(self._right)

            if self._less_than:
                self._hash = hash((self._hash_name, (left_hash, right_hash)))
            else:
                self._hash = hash((self._hash_name, (right_hash, left_hash)))

        return self._hash

    def is_equivalent(self, other: Node) -> bool:  # noqa: D102
        if not isinstance(other, self.__class__):
            return False

        if hash(self) != hash(other):
            return False

        if self._less_than ^ other._less_than:
            return self._left.is_equivalent(other._right) and self._right.is_equivalent(other._left)
        else:
            return self._left.is_equivalent(other._left) and self._right.is_equivalent(other._right)


class SemiStrictComparison(AbstractComparison):
    """A node representing a comparison x < y or x > y that only works when x and y are at most p // 2 elements apart.
    
    Semi-comparisons are only valid if the absolute difference between the inputs does not exceed half of the field size.
    """

    @property
    def _hash_name(self) -> str:
        return "semi_strict_comparison"

    @property
    def _node_label(self) -> str:
        return "~<" if self._less_than else ">~"

    def _operation_inner(self, x, y) -> FieldArray:
        assert abs(int(x) - int(y)) <= self._gf.characteristic // 2

        if self._less_than:
            return self._gf(int(int(x) < int(y)))
        else:
            return self._gf(int(int(x) > int(y)))

    def _arithmetize_inner(self, strategy: str) -> Node:
        if self._less_than:
            left = self._left
            right = self._right
        else:
            left = self._right
            right = self._left

        return InUpperHalf(
            Subtraction(left.arithmetize(strategy), right.arithmetize(strategy), self._gf),
            self._gf,
        ).arithmetize(strategy)

    def _arithmetize_depth_aware_inner(self, cost_of_squaring: float) -> CostParetoFront:
        front = CostParetoFront(cost_of_squaring)

        if self._less_than:
            left = self._left
            right = self._right
        else:
            left = self._right
            right = self._left

        left_front = left.arithmetize_depth_aware(cost_of_squaring)
        right_front = right.arithmetize_depth_aware(cost_of_squaring)

        for left, right in iterate_increasing_depth(left_front, right_front):
            _, _, left_node = left
            _, _, right_node = right

            sub_front = InUpperHalf(
                Subtraction(left_node, right_node, self._gf),
                self._gf,
            ).arithmetize_depth_aware(cost_of_squaring)

            front.add_front(sub_front)

        assert not front.is_empty()
        return front


class StrictComparison(AbstractComparison):
    """A node representing a comparison x < y or x > y."""

    @property
    def _hash_name(self) -> str:
        return "strict_comparison"

    @property
    def _node_label(self) -> str:
        return "<" if self._less_than else ">"

    def _operation_inner(self, x, y) -> FieldArray:
        if self._less_than:
            return self._gf(int(int(x) < int(y)))
        else:
            return self._gf(int(int(x) > int(y)))

    def _arithmetize_inner(self, strategy: str) -> Node:
        p = self._gf.characteristic

        if self._less_than:
            left = self._left
            right = self._right
        else:
            left = self._right
            right = self._left

        left = left.arithmetize(strategy)
        right = right.arithmetize(strategy)

        left_is_small = SemiStrictComparison(
            left, Constant(self._gf(p // 2)), less_than=True, gf=self._gf
        )
        right_is_small = SemiStrictComparison(
            right, Constant(self._gf(p // 2)), less_than=True, gf=self._gf
        )

        # Test whether left and right are in the same range
        same_range = (left_is_small & right_is_small) + (
            Neg(left_is_small, self._gf) & Neg(right_is_small, self._gf)
        )

        # Performs left < right on the reduced inputs, note that if both are in the upper half the difference is still small enough for a semi-comparison
        comparison = SemiStrictComparison(left, right, less_than=True, gf=self._gf)
        result = same_range * comparison

        # Performs left < right when one if small and the other is large
        right_is_larger = left_is_small & Neg(right_is_small, self._gf)
        result += right_is_larger

        return result.arithmetize(strategy)

    def _arithmetize_depth_aware_inner(self, cost_of_squaring: float) -> CostParetoFront:
        p = self._gf.characteristic

        if self._less_than:
            left = self._left
            right = self._right
        else:
            left = self._right
            right = self._left

        left_front = left.arithmetize_depth_aware(cost_of_squaring)
        right_front = right.arithmetize_depth_aware(cost_of_squaring)

        # TODO: This is just exhaustive. We can instead add a method decompose so that we do not have to copy this from arithmetize.
        front = CostParetoFront(cost_of_squaring)

        for _, _, left_node in left_front:
            for _, _, right_node in right_front:
                left_is_small = SemiStrictComparison(
                    left_node, Constant(self._gf(p // 2)), less_than=True, gf=self._gf
                )
                right_is_small = SemiStrictComparison(
                    right_node, Constant(self._gf(p // 2)), less_than=True, gf=self._gf
                )

                # Test whether left and right are in the same range
                same_range = (left_is_small & right_is_small) + (
                    Neg(left_is_small, self._gf) & Neg(right_is_small, self._gf)
                )

                # Performs left < right on the reduced inputs, note that if both are in the upper half the difference is still small enough for a semi-comparison
                comparison = SemiStrictComparison(
                    left_node, right_node, less_than=True, gf=self._gf
                )
                result = same_range * comparison

                # Performs left < right when one if small and the other is large
                right_is_larger = left_is_small & Neg(right_is_small, self._gf)
                result += right_is_larger

                front.add_front(result.arithmetize_depth_aware(cost_of_squaring))

        return front


class SemiComparison(AbstractComparison):
    """A node representing a comparison x <= y or x >= y that only works when x and y are at most p // 2 elements apart."""

    @property
    def _hash_name(self) -> str:
        return "semi_comparison"

    @property
    def _node_label(self) -> str:
        return "~<=" if self._less_than else ">=~"

    def _operation_inner(self, x, y) -> FieldArray:
        assert abs(int(x) - int(y)) <= self._gf.characteristic // 2

        if self._less_than:
            return self._gf(int(int(x) <= int(y)))
        else:
            return self._gf(int(int(x) >= int(y)))

    def _arithmetize_inner(self, strategy: str) -> Node:
        return Neg(
            SemiStrictComparison(
                self._left.arithmetize(strategy),
                self._right.arithmetize(strategy),
                less_than=not self._less_than,
                gf=self._gf,
            ),
            self._gf,
        ).arithmetize(strategy)

    def _arithmetize_depth_aware_inner(self, cost_of_squaring: float) -> CostParetoFront:
        return Neg(
            SemiStrictComparison(
                self._left, self._right, less_than=not self._less_than, gf=self._gf
            ),
            self._gf,
        ).arithmetize_depth_aware(cost_of_squaring)


class Comparison(AbstractComparison):
    """A node representing a comparison x <= y or x >= y."""

    @property
    def _hash_name(self) -> str:
        return "comparison"

    @property
    def _node_label(self) -> str:
        return "<=" if self._less_than else ">="

    def _operation_inner(self, x, y) -> FieldArray:
        if self._less_than:
            return self._gf(int(int(x) <= int(y)))
        else:
            return self._gf(int(int(x) >= int(y)))

    def _arithmetize_inner(self, strategy: str) -> Node:
        return Neg(
            StrictComparison(
                self._left.arithmetize(strategy),
                self._right.arithmetize(strategy),
                less_than=not self._less_than,
                gf=self._gf,
            ),
            self._gf,
        ).arithmetize(strategy)

    def _arithmetize_depth_aware_inner(self, cost_of_squaring: float) -> CostParetoFront:
        return Neg(
            StrictComparison(self._left, self._right, less_than=not self._less_than, gf=self._gf),
            self._gf,
        ).arithmetize_depth_aware(cost_of_squaring)


class T2SemiLessThan(NonCommutativeBinaryNode):
    """Implementation of [the algorithm from the T2 framework](https://petsymposium.org/popets/2023/popets-2023-0075.pdf) for performing x < y."""

    @property
    def _hash_name(self) -> str:
        return "less_than_t2"

    @property
    def _node_label(self) -> str:
        return "< [t2]"

    def _operation_inner(self, x, y) -> FieldArray:
        return self._gf(int(int(x) < int(y)))

    def _arithmetize_inner(self, strategy: str) -> Node:
        out = Constant(self._gf(0))

        p = self._gf.characteristic
        for a in range((p + 1) // 2, p):
            out += Constant(self._gf(1)) - (self._left - self._right - Constant(self._gf(a))) ** (
                p - 1
            )

        return out.arithmetize(strategy)

    def _arithmetize_depth_aware_inner(self, cost_of_squaring: float) -> CostParetoFront:
        raise NotImplementedError()


class IliashenkoZuccaSemiLessThan(NonCommutativeBinaryNode):
    """Implementation of the [Illiashenko-Zucca algorithm](https://eprint.iacr.org/2021/315) for performing x < y."""

    @property
    def _hash_name(self) -> str:
        return "less_than_t2"

    @property
    def _node_label(self) -> str:
        return "< [t2]"

    def _operation_inner(self, x, y) -> FieldArray:
        return self._gf(int(int(x) < int(y)))

    def _arithmetize_inner(self, strategy: str) -> Node:
        return IliashenkoZuccaInUpperHalf(
            Subtraction(
                self._left.arithmetize(strategy), self._right.arithmetize(strategy), self._gf
            ),
            self._gf,
        ).arithmetize(strategy)

    def _arithmetize_depth_aware_inner(self, cost_of_squaring: float) -> CostParetoFront:
        raise NotImplementedError()


def test_evaluate_semi_mod5_lt():  # noqa: D103
    gf = GF(5)

    a = Input("a", gf)
    b = Input("b", gf)
    node = SemiStrictComparison(a, b, less_than=True, gf=gf)

    for x in range(3):
        for y in range(3):
            assert node.evaluate({"a": gf(x), "b": gf(y)}) == gf(int(x < y))
            node.clear_cache(set())


def test_evaluate_semi_arithmetized_mod5_lt():  # noqa: D103
    gf = GF(5)

    a = Input("a", gf)
    b = Input("b", gf)
    node = SemiStrictComparison(a, b, less_than=True, gf=gf).arithmetize("best-effort")
    node.clear_cache(set())

    for x in range(3):
        for y in range(3):
            assert node.evaluate({"a": gf(x), "b": gf(y)}) == gf(int(x < y))
            node.clear_cache(set())


def test_evaluate_mod5_lt():  # noqa: D103
    gf = GF(5)

    a = Input("a", gf)
    b = Input("b", gf)
    node = StrictComparison(a, b, less_than=True, gf=gf)

    for x in range(5):
        for y in range(5):
            assert node.evaluate({"a": gf(x), "b": gf(y)}) == gf(int(x < y))
            node.clear_cache(set())


def test_evaluate_arithmetized_mod5_lt():  # noqa: D103
    gf = GF(5)

    a = Input("a", gf)
    b = Input("b", gf)
    node = StrictComparison(a, b, less_than=True, gf=gf).arithmetize("best-effort")
    node.clear_cache(set())

    for x in range(5):
        for y in range(5):
            assert node.evaluate({"a": gf(x), "b": gf(y)}) == gf(int(x < y))
            node.clear_cache(set())


def test_evaluate_arithmetized_mod11_lt():  # noqa: D103
    gf = GF(11)

    a = Input("a", gf)
    b = Input("b", gf)
    node = StrictComparison(a, b, less_than=True, gf=gf).arithmetize("best-effort")
    node.clear_cache(set())

    for x in range(11):
        for y in range(11):
            assert node.evaluate({"a": gf(x), "b": gf(y)}) == gf(int(x < y))
            node.clear_cache(set())


def test_evaluate_arithmetized_depth_aware_semi_mod11_lt():  # noqa: D103
    gf = GF(11)

    a = Input("a", gf)
    b = Input("b", gf)
    front = SemiStrictComparison(a, b, less_than=True, gf=gf).arithmetize_depth_aware(1.0)

    for _, _, node in front:
        for x in range(6):
            for y in range(6):
                assert node.evaluate({"a": gf(x), "b": gf(y)}) == gf(int(x < y))
                node.clear_cache(set())


def test_evaluate_arithmetized_depth_aware_mod11_lt():  # noqa: D103
    gf = GF(11)

    a = Input("a", gf)
    b = Input("b", gf)
    front = StrictComparison(a, b, less_than=True, gf=gf).arithmetize_depth_aware(1.0)

    for _, _, node in front:
        for x in range(11):
            for y in range(11):
                assert node.evaluate({"a": gf(x), "b": gf(y)}) == gf(int(x < y))
                node.clear_cache(set())


def test_evaluate_semi_arithmetized_mod5_t2():  # noqa: D103
    gf = GF(5)

    a = Input("a", gf)
    b = Input("b", gf)
    node = T2SemiLessThan(a, b, gf).arithmetize("best-effort")
    node.clear_cache(set())

    for x in range(3):
        for y in range(3):
            assert node.evaluate({"a": gf(x), "b": gf(y)}) == gf(int(x < y))
            node.clear_cache(set())


def test_evaluate_semi_arithmetized_mod11_t2():  # noqa: D103
    gf = GF(11)

    a = Input("a", gf)
    b = Input("b", gf)
    node = T2SemiLessThan(a, b, gf).arithmetize("best-effort")
    node.clear_cache(set())

    for x in range(6):
        for y in range(6):
            assert node.evaluate({"a": gf(x), "b": gf(y)}) == gf(int(x < y))
            node.clear_cache(set())


def test_evaluate_semi_mod5_gt():  # noqa: D103
    gf = GF(5)

    a = Input("a", gf)
    b = Input("b", gf)
    node = SemiStrictComparison(a, b, less_than=False, gf=gf)

    for x in range(3):
        for y in range(3):
            assert node.evaluate({"a": gf(x), "b": gf(y)}) == gf(int(x > y))
            node.clear_cache(set())


def test_evaluate_semi_arithmetized_mod5_gt():  # noqa: D103
    gf = GF(5)

    a = Input("a", gf)
    b = Input("b", gf)
    node = SemiStrictComparison(a, b, less_than=False, gf=gf).arithmetize("best-effort")
    node.clear_cache(set())

    for x in range(3):
        for y in range(3):
            assert node.evaluate({"a": gf(x), "b": gf(y)}) == gf(int(x > y))
            node.clear_cache(set())


def test_evaluate_mod5_gt():  # noqa: D103
    gf = GF(5)

    a = Input("a", gf)
    b = Input("b", gf)
    node = StrictComparison(a, b, less_than=False, gf=gf)

    for x in range(5):
        for y in range(5):
            assert node.evaluate({"a": gf(x), "b": gf(y)}) == gf(int(x > y))
            node.clear_cache(set())


def test_evaluate_arithmetized_mod5_gt():  # noqa: D103
    gf = GF(5)

    a = Input("a", gf)
    b = Input("b", gf)
    node = StrictComparison(a, b, less_than=False, gf=gf).arithmetize("best-effort")
    node.clear_cache(set())

    for x in range(5):
        for y in range(5):
            assert node.evaluate({"a": gf(x), "b": gf(y)}) == gf(int(x > y))
            node.clear_cache(set())


def test_evaluate_semi_mod5_ge():  # noqa: D103
    gf = GF(5)

    a = Input("a", gf)
    b = Input("b", gf)
    node = SemiComparison(a, b, less_than=False, gf=gf)

    for x in range(3):
        for y in range(3):
            assert node.evaluate({"a": gf(x), "b": gf(y)}) == gf(int(x >= y))
            node.clear_cache(set())


def test_evaluate_semi_arithmetized_mod5_ge():  # noqa: D103
    gf = GF(5)

    a = Input("a", gf)
    b = Input("b", gf)
    node = SemiComparison(a, b, less_than=False, gf=gf).arithmetize("best-effort")
    node.clear_cache(set())

    for x in range(3):
        for y in range(3):
            assert node.evaluate({"a": gf(x), "b": gf(y)}) == gf(int(x >= y))
            node.clear_cache(set())


def test_evaluate_mod5_ge():  # noqa: D103
    gf = GF(5)

    a = Input("a", gf)
    b = Input("b", gf)
    node = Comparison(a, b, less_than=False, gf=gf)

    for x in range(5):
        for y in range(5):
            assert node.evaluate({"a": gf(x), "b": gf(y)}) == gf(int(x >= y))
            node.clear_cache(set())


def test_evaluate_arithmetized_mod5_ge():  # noqa: D103
    gf = GF(5)

    a = Input("a", gf)
    b = Input("b", gf)
    node = Comparison(a, b, less_than=False, gf=gf).arithmetize("best-effort")
    node.clear_cache(set())

    for x in range(5):
        for y in range(5):
            assert node.evaluate({"a": gf(x), "b": gf(y)}) == gf(int(x >= y))
            node.clear_cache(set())


def test_evaluate_arithmetized_depth_aware_mod5_ge():  # noqa: D103
    gf = GF(5)

    a = Input("a", gf)
    b = Input("b", gf)
    node = Comparison(a, b, less_than=False, gf=gf)
    front = node.arithmetize_depth_aware(0.75)

    for _, _, n in front:
        for x in range(5):
            for y in range(5):
                assert n.evaluate({"a": gf(x), "b": gf(y)}) == gf(int(x >= y))
                n.clear_cache(set())


def test_evaluate_semi_mod5_le():  # noqa: D103
    gf = GF(5)

    a = Input("a", gf)
    b = Input("b", gf)
    node = SemiComparison(a, b, less_than=True, gf=gf)

    for x in range(3):
        for y in range(3):
            assert node.evaluate({"a": gf(x), "b": gf(y)}) == gf(int(x <= y))
            node.clear_cache(set())


def test_evaluate_semi_arithmetized_mod5_le():  # noqa: D103
    gf = GF(5)

    a = Input("a", gf)
    b = Input("b", gf)
    node = SemiComparison(a, b, less_than=True, gf=gf).arithmetize("best-effort")
    node.clear_cache(set())

    for x in range(3):
        for y in range(3):
            assert node.evaluate({"a": gf(x), "b": gf(y)}) == gf(int(x <= y))
            node.clear_cache(set())


def test_evaluate_mod5_le():  # noqa: D103
    gf = GF(5)

    a = Input("a", gf)
    b = Input("b", gf)
    node = Comparison(a, b, less_than=True, gf=gf)

    for x in range(5):
        for y in range(5):
            assert node.evaluate({"a": gf(x), "b": gf(y)}) == gf(int(x <= y))
            node.clear_cache(set())


def test_evaluate_arithmetized_mod5_le():  # noqa: D103
    gf = GF(5)

    a = Input("a", gf)
    b = Input("b", gf)
    node = Comparison(a, b, less_than=True, gf=gf).arithmetize("best-effort")
    node.clear_cache(set())

    for x in range(5):
        for y in range(5):
            assert node.evaluate({"a": gf(x), "b": gf(y)}) == gf(int(x <= y))
            node.clear_cache(set())


def test_evaluate_semi_arithmetized_mod101_lt():  # noqa: D103
    gf = GF(101)

    a = Input("a", gf)
    b = Input("b", gf)
    node = SemiStrictComparison(a, b, less_than=True, gf=gf).arithmetize("best-effort")
    node.clear_cache(set())

    for x in range(51):
        for y in range(51):
            assert node.evaluate({"a": gf(x), "b": gf(y)}) == gf(int(x < y))
            node.clear_cache(set())


def test_evaluate_semi_depth_aware_arithmetized_mod61_lt():  # noqa: D103
    gf = GF(61)

    a = Input("a", gf)
    b = Input("b", gf)
    front = SemiStrictComparison(a, b, less_than=True, gf=gf).arithmetize_depth_aware(cost_of_squaring=1.0)

    for _, _, node in front:
        node.clear_cache(set())

        for x in range(31):
            for y in range(31):
                assert node.evaluate({"a": gf(x), "b": gf(y)}) == gf(int(x < y))
                node.clear_cache(set())


def test_evaluate_semi_depth_aware_arithmetized_mod61_lt_05sq():  # noqa: D103
    gf = GF(61)

    a = Input("a", gf)
    b = Input("b", gf)
    front = SemiStrictComparison(a, b, less_than=True, gf=gf).arithmetize_depth_aware(cost_of_squaring=0.5)

    for _, _, node in front:
        node.clear_cache(set())

        for x in range(31):
            for y in range(31):
                assert node.evaluate({"a": gf(x), "b": gf(y)}) == gf(int(x < y))
                node.clear_cache(set())


def test_lessthan_mod101():  # noqa: D103
    gf = GF(101)

    x = Input("x", gf)
    circuit = Circuit([x < 30])

    for _, _, arithmetization in circuit.arithmetize_depth_aware():
        assert arithmetization.evaluate({
                "x": gf(90),
            })[0] == 0
