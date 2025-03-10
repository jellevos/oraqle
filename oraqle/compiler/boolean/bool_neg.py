"""Classes for describing Boolean negation."""
from typing import Type
from galois import FieldArray

from oraqle.compiler.arithmetic.subtraction import Subtraction
from oraqle.compiler.nodes.fp.abstract import CostParetoFront
from oraqle.compiler.nodes.fp.fixed import ArithmeticNode
from oraqle.compiler.nodes.fp.leafs import FpConstant
from oraqle.compiler.nodes.fp.univariate import UnivariateFpNode
from oraqle.compiler.nodes.fpd.abstract import FpNode


class Neg(UnivariateFpNode):
    """A node that negates a Boolean input."""

    @property
    def _node_shape(self) -> str:
        return "box"

    @property
    def _hash_name(self) -> str:
        return "neg"

    @property
    def _node_label(self) -> str:
        return "NEG"

    def _operation_inner(self, input: FieldArray) -> FieldArray:
        assert input in {0, 1}
        return self._gf(not bool(input))

    def _arithmetize_inner(self, strategy: str, circuit_gf: Type[FieldArray]) -> ArithmeticNode:
        return Subtraction(
            FpConstant(self._gf(1)), self._node.arithmetize_fpd(strategy, circuit_gf), self._gf
        ).arithmetize_fpd(strategy, circuit_gf)

    def _arithmetize_depth_aware_inner(self, cost_of_squaring: float) -> CostParetoFront:
        return Subtraction(FpConstant(self._gf(1)), self._node, self._gf).arithmetize_depth_aware(
            cost_of_squaring
        )
