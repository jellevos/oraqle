"""This module contains classes for representing subtraction: x - y."""
from galois import GF, FieldArray

from oraqle.compiler.nodes.abstract import CostParetoFront, Node
from oraqle.compiler.nodes.leafs import Constant, Input
from oraqle.compiler.nodes.non_commutative import NonCommutativeBinaryNode


class Subtraction(NonCommutativeBinaryNode):
    """Represents a subtraction, which can be arithmetized using addition and constant-multiplication."""

    @property
    def _overriden_graphviz_attributes(self) -> dict:
        return {"shape": "square", "style": "rounded,filled", "fillcolor": "cornsilk"}

    @property
    def _hash_name(self) -> str:
        return "sub"

    @property
    def _node_label(self) -> str:
        return "-"

    def _operation_inner(self, x, y) -> FieldArray:
        return x - y

    def _arithmetize_inner(self, strategy: str) -> Node:
        # TODO: Reorganize the files: let the arithmetic folder only contain pure arithmetic (including add and mul) and move exponentiation elsewhere.
        # TODO: For schemes that support subtraction we do not need to do this. We should only do this transformation during the compiler stage.
        return (self._left.arithmetize(strategy) + (Constant(-self._gf(1)) * self._right.arithmetize(strategy))).arithmetize(strategy)  # type: ignore  # TODO: Should we always perform a final arithmetization in every node for constant folding? E.g. in Node?

    def _arithmetize_depth_aware_inner(self, cost_of_squaring: float) -> CostParetoFront:
        result = self._left + (Constant(-self._gf(1)) * self._right)
        front = result.arithmetize_depth_aware(cost_of_squaring)
        return front


def test_evaluate_mod5():  # noqa: D103
    gf = GF(5)

    a = Input("a", gf)
    b = Input("b", gf)
    node = Subtraction(a, b, gf)

    assert node.evaluate({"a": gf(3), "b": gf(2)}) == gf(1)
    node.clear_cache(set())
    assert node.evaluate({"a": gf(4), "b": gf(1)}) == gf(3)
    node.clear_cache(set())
    assert node.evaluate({"a": gf(1), "b": gf(3)}) == gf(3)
    node.clear_cache(set())
    assert node.evaluate({"a": gf(0), "b": gf(4)}) == gf(1)


def test_evaluate_arithmetized_mod5():  # noqa: D103
    gf = GF(5)

    a = Input("a", gf)
    b = Input("b", gf)
    node = Subtraction(a, b, gf).arithmetize("best-effort")
    node.clear_cache(set())

    assert node.evaluate({"a": gf(3), "b": gf(2)}) == gf(1)
    node.clear_cache(set())
    assert node.evaluate({"a": gf(4), "b": gf(1)}) == gf(3)
    node.clear_cache(set())
    assert node.evaluate({"a": gf(1), "b": gf(3)}) == gf(3)
    node.clear_cache(set())
    assert node.evaluate({"a": gf(0), "b": gf(4)}) == gf(1)
