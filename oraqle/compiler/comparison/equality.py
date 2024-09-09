"""This module contains classes for representing equality checks."""
from galois import GF, FieldArray

from oraqle.compiler.arithmetic.exponentiation import Power
from oraqle.compiler.arithmetic.subtraction import Subtraction
from oraqle.compiler.boolean.bool_neg import Neg
from oraqle.compiler.nodes.abstract import CostParetoFront, Node
from oraqle.compiler.nodes.binary_arithmetic import CommutativeBinaryNode
from oraqle.compiler.nodes.leafs import Input
from oraqle.compiler.nodes.univariate import UnivariateNode


class IsNonZero(UnivariateNode):
    """This node represents a zero check: x == 0."""

    @property
    def _node_shape(self) -> str:
        return "box"

    @property
    def _hash_name(self) -> str:
        return "is_nonzero"

    @property
    def _node_label(self) -> str:
        return "!= 0"

    def _operation_inner(self, input: FieldArray) -> FieldArray:
        return input != 0

    def _arithmetize_inner(self, strategy: str) -> Node:
        return Power(self._node, self._gf.order - 1, self._gf).arithmetize(strategy)

    def _arithmetize_depth_aware_inner(self, cost_of_squaring: float) -> CostParetoFront:
        return Power(self._node, self._gf.order - 1, self._gf).arithmetize_depth_aware(
            cost_of_squaring
        )


class Equals(CommutativeBinaryNode):
    """This node represents an equality operation: x == y."""

    @property
    def _hash_name(self) -> str:
        return "equals"

    @property
    def _node_label(self) -> str:
        return "=="

    def _operation_inner(self, x, y) -> FieldArray:
        return self._gf(int(x == y))

    def _arithmetize_inner(self, strategy: str) -> Node:
        return Neg(
            IsNonZero(Subtraction(self._left, self._right, self._gf), self._gf),
            self._gf,
        ).arithmetize(strategy)

    def _arithmetize_depth_aware_inner(self, cost_of_squaring: float) -> CostParetoFront:
        return Neg(
            IsNonZero(Subtraction(self._left, self._right, self._gf), self._gf),
            self._gf,
        ).arithmetize_depth_aware(cost_of_squaring)


def test_evaluate_mod5():  # noqa: D103
    gf = GF(5)

    a = Input("a", gf)
    b = Input("b", gf)
    node = Equals(a, b, gf)

    assert node.evaluate({"a": gf(3), "b": gf(2)}) == gf(0)
    node.clear_cache(set())
    assert node.evaluate({"a": gf(4), "b": gf(4)}) == gf(1)
    node.clear_cache(set())
    assert node.evaluate({"a": gf(1), "b": gf(2)}) == gf(0)
    node.clear_cache(set())
    assert node.evaluate({"a": gf(0), "b": gf(0)}) == gf(1)


def test_evaluate_arithmetized_mod5():  # noqa: D103
    gf = GF(5)

    a = Input("a", gf)
    b = Input("b", gf)
    node = Equals(a, b, gf).arithmetize("best-effort")
    node.clear_cache(set())

    assert node.evaluate({"a": gf(3), "b": gf(2)}) == gf(0)
    node.clear_cache(set())
    assert node.evaluate({"a": gf(4), "b": gf(4)}) == gf(1)
    node.clear_cache(set())
    assert node.evaluate({"a": gf(1), "b": gf(2)}) == gf(0)
    node.clear_cache(set())
    assert node.evaluate({"a": gf(0), "b": gf(0)}) == gf(1)


def test_equality_equivalence_commutative():  # noqa: D103
    gf = GF(5)

    a = Input("a", gf)
    b = Input("b", gf)

    assert (a == b).is_equivalent(b == a)
