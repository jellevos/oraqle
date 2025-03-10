"""This module contains tools for evaluating conditional statements."""
from typing import List, Type

from galois import GF, FieldArray

from oraqle.compiler.circuit import Circuit
from oraqle.compiler.nodes.fp.abstract import CostParetoFront
from oraqle.compiler.nodes.fp.fixed import FixedFpNode
from oraqle.compiler.nodes.fp.leafs import FpConstant, FpInput
from oraqle.compiler.nodes.fpd.abstract import FpNode


class IfElse(FixedFpNode):
    """A node representing an if-else clause."""

    @property
    def _node_label(self):
        return "If"

    @property
    def _hash_name(self):
        return "if_else"

    def __init__(self, condition: FpNode, positive: FpNode, negative: FpNode, gf: Type[FieldArray]):
        """Initialize an if-else node: If condition evaluates to true, then it outputs positive, otherwise it outputs negative."""
        self._condition = condition
        self._positive = positive
        self._negative = negative
        super().__init__(gf)

    def __hash__(self) -> int:
        return hash((self._hash_name, self._condition, self._positive, self._negative))

    def is_equivalent(self, other: FpNode) -> bool:  # noqa: D102
        if not isinstance(other, self.__class__):
            return False

        return (
            self._condition.is_equivalent(other._condition)
            and self._positive.is_equivalent(other._positive)
            and self._negative.is_equivalent(other._negative)
        )

    def operands(self) -> List[FpNode]:  # noqa: D102
        return [self._condition, self._positive, self._negative]

    def set_operands(self, operands: List[FpNode]):  # noqa: D102
        self._condition = operands[0]
        self._positive = operands[1]
        self._negative = operands[2]

    def operation(self, operands: List[FieldArray]) -> FieldArray:  # noqa: D102
        assert operands[0] == 0 or operands[0] == 1
        return operands[1] if operands[0] == 1 else operands[2]

    def _arithmetize_inner(self, strategy: str) -> FpNode:
        return (self._condition * (self._positive - self._negative) + self._negative).arithmetize_fpd(
            strategy
        )

    def _arithmetize_depth_aware_inner(self, cost_of_squaring: float) -> CostParetoFront:
        return (
            self._condition * (self._positive - self._negative) + self._negative
        ).arithmetize_depth_aware(cost_of_squaring)


def if_else(condition: FpNode, positive: FpNode, negative: FpNode) -> IfElse:
    """Sugar expression for creating an if-else clause.
    
    Returns:
        An `IfElse` node that equals `positive` if `condition` is true, and `negative` otherwise.
    """
    assert condition._gf == positive._gf
    assert condition._gf == negative._gf
    return IfElse(condition, positive, negative, condition._gf)


def test_if_else():  # noqa: D103
    gf = GF(11)

    a = FpInput("a", gf)
    b = FpInput("b", gf)

    output = if_else(a == b, FpConstant(gf(3)), FpConstant(gf(5)))

    circuit = Circuit([output])

    for val_a in range(11):
        for val_b in range(11):
            expected = gf(3) if val_a == val_b else gf(5)

            values = {"a": gf(val_a), "b": gf(val_b)}
            assert circuit.evaluate(values) == expected


def test_if_else_arithmetized():  # noqa: D103
    gf = GF(11)

    a = FpInput("a", gf)
    b = FpInput("b", gf)

    output = if_else(a == b, FpConstant(gf(3)), FpConstant(gf(5)))

    arithmetic_circuit = Circuit([output]).arithmetize()

    for val_a in range(11):
        for val_b in range(11):
            expected = gf(3) if val_a == val_b else gf(5)

            values = {"a": gf(val_a), "b": gf(val_b)}
            assert arithmetic_circuit.evaluate(values) == expected
