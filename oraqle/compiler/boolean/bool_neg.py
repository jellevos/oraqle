"""Classes for describing Boolean negation."""
from galois import FieldArray

from oraqle.compiler.arithmetic.subtraction import Subtraction
from oraqle.compiler.nodes.abstract import CostParetoFront, Node
from oraqle.compiler.nodes.leafs import Constant
from oraqle.compiler.nodes.univariate import UnivariateNode


class Neg(UnivariateNode):
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

    def _arithmetize_inner(self, strategy: str) -> Node:
        return Subtraction(
            Constant(self._gf(1)), self._node.arithmetize(strategy), self._gf
        ).arithmetize(strategy)

    def _arithmetize_depth_aware_inner(self, cost_of_squaring: float) -> CostParetoFront:
        return Subtraction(Constant(self._gf(1)), self._node, self._gf).arithmetize_depth_aware(
            cost_of_squaring
        )
