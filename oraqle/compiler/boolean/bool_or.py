"""This module contains tools for evaluating OR operations between many inputs."""
import itertools
from typing import Set

from galois import GF, FieldArray

from oraqle.compiler.boolean.bool_and import And, _find_depth_cost_front
from oraqle.compiler.boolean.bool_neg import Neg
from oraqle.compiler.nodes.abstract import CostParetoFront, Node, UnoverloadedWrapper
from oraqle.compiler.nodes.flexible import CommutativeUniqueReducibleNode
from oraqle.compiler.nodes.leafs import Constant, Input

# TODO: Reduce code duplication between OR and AND


class Or(CommutativeUniqueReducibleNode):
    """Performs an OR operation over several operands. The user must ensure that the operands are Booleans."""

    @property
    def _hash_name(self) -> str:
        return "or"

    @property
    def _node_label(self) -> str:
        return "OR"

    def _inner_operation(self, a: FieldArray, b: FieldArray) -> FieldArray:
        return self._gf(bool(a) | bool(b))

    def _arithmetize_inner(self, strategy: str) -> Node:
        # FIXME: Handle what happens when arithmetize outputs a constant!
        # TODO: Also consider the arithmetization using randomness
        return Neg(
            And(
                {
                    UnoverloadedWrapper(Neg(operand.node.arithmetize(strategy), self._gf))
                    for operand in self._operands
                },
                self._gf,
            ),
            self._gf,
        ).arithmetize(strategy)

    def _arithmetize_depth_aware_inner(self, cost_of_squaring: float) -> CostParetoFront:
        # TODO: This is mostly copied from AND
        new_operands: Set[CostParetoFront] = set()
        for operand in self._operands:
            new_operand = operand.node.arithmetize_depth_aware(cost_of_squaring)
            new_operands.add(new_operand)

        if len(new_operands) == 0:
            return CostParetoFront.from_leaf(Constant(self._gf(1)), cost_of_squaring)
        elif len(new_operands) == 1:
            return next(iter(new_operands))

        # TODO: We can check if any of the element in new_operands are constants and return early

        front = CostParetoFront(cost_of_squaring)

        # TODO: This is brute force composition
        for operands in itertools.product(*(iter(new_operand) for new_operand in new_operands)):
            checked_operands = []
            for depth, cost, node in operands:
                if isinstance(node, Constant):
                    assert node._value in {0, 1}
                    if node._value == 0:
                        return CostParetoFront.from_leaf(Constant(self._gf(0)), cost_of_squaring)
                else:
                    checked_operands.append((depth, cost, node))

            if len(checked_operands) == 0:
                return CostParetoFront.from_leaf(Constant(self._gf(1)), cost_of_squaring)

            if len(checked_operands) == 1:
                depth, cost, node = checked_operands[0]
                front.add(node, depth, cost)
                continue

            this_front = _find_depth_cost_front(
                checked_operands,
                self._gf,
                float("inf"),
                squaring_cost=cost_of_squaring,
                is_and=False,
            )
            front.add_front(this_front)

        return front

    def or_flatten(self, other: Node) -> Node:
        """Performs an OR operation with `other`, flattening the `Or` node if either of the two is also an `Or` and absorbing `Constant`s.
        
        Returns:
            An `Or` node containing the flattened OR operation, or a `Constant` node.
        """
        if isinstance(other, Constant):
            if bool(other._value):
                return Constant(self._gf(1))
            else:
                return self

        if isinstance(other, Or):
            return Or(self._operands | other._operands, self._gf)

        new_operands = self._operands.copy()
        new_operands.add(UnoverloadedWrapper(other))
        return Or(new_operands, self._gf)


def any_(*operands: Node) -> Or:
    """Returns an `Or` node that evaluates to true if any of the given `operands` evaluates to true."""
    assert len(operands) > 0
    return Or(set(UnoverloadedWrapper(operand) for operand in operands), operands[0]._gf)


def test_evaluate_mod3():  # noqa: D103
    gf = GF(3)

    a = Input("a", gf)
    b = Input("b", gf)
    node = a | b

    assert node.evaluate({"a": gf(0), "b": gf(0)}) == gf(0)
    node.clear_cache(set())
    assert node.evaluate({"a": gf(0), "b": gf(1)}) == gf(1)
    node.clear_cache(set())
    assert node.evaluate({"a": gf(1), "b": gf(0)}) == gf(1)
    node.clear_cache(set())
    assert node.evaluate({"a": gf(1), "b": gf(1)}) == gf(1)


def test_evaluate_arithmetized_depth_aware_mod2():  # noqa: D103
    gf = GF(2)

    a = Input("a", gf)
    b = Input("b", gf)
    node = a | b
    front = node.arithmetize_depth_aware(cost_of_squaring=1.0)

    for _, _, n in front:
        n.clear_cache(set())
        assert n.evaluate({"a": gf(0), "b": gf(0)}) == gf(0)
        n.clear_cache(set())
        assert n.evaluate({"a": gf(0), "b": gf(1)}) == gf(1)
        n.clear_cache(set())
        assert n.evaluate({"a": gf(1), "b": gf(0)}) == gf(1)
        n.clear_cache(set())
        assert n.evaluate({"a": gf(1), "b": gf(1)}) == gf(1)


def test_evaluate_arithmetized_mod3():  # noqa: D103
    gf = GF(3)

    a = Input("a", gf)
    b = Input("b", gf)
    node = (a | b).arithmetize("best-effort")

    node.clear_cache(set())
    assert node.evaluate({"a": gf(0), "b": gf(0)}) == gf(0)
    node.clear_cache(set())
    assert node.evaluate({"a": gf(0), "b": gf(1)}) == gf(1)
    node.clear_cache(set())
    assert node.evaluate({"a": gf(1), "b": gf(0)}) == gf(1)
    node.clear_cache(set())
    assert node.evaluate({"a": gf(1), "b": gf(1)}) == gf(1)


def test_evaluate_arithmetized_depth_aware_50_mod31():  # noqa: D103
    gf = GF(31)

    xs = {Input(f"x{i}", gf) for i in range(50)}
    node = Or({UnoverloadedWrapper(x) for x in xs}, gf)
    front = node.arithmetize_depth_aware(cost_of_squaring=1.0)

    for _, _, n in front:
        n.clear_cache(set())
        assert n.evaluate({f"x{i}": gf(0) for i in range(50)}) == gf(0)
        n.clear_cache(set())
        assert n.evaluate({f"x{i}": gf(i % 2) for i in range(50)}) == gf(1)
        n.clear_cache(set())
        assert n.evaluate({f"x{i}": gf(1) for i in range(50)}) == gf(1)
