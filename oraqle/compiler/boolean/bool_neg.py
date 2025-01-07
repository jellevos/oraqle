"""Classes for describing Boolean negation."""
from galois import FieldArray

from oraqle.compiler.arithmetic.subtraction import Subtraction
from oraqle.compiler.boolean.bool import Boolean, InvUnreducedBoolean, ReducedBoolean, UnreducedBoolean
from oraqle.compiler.nodes.abstract import CostParetoFront, Node
from oraqle.compiler.nodes.leafs import Constant
from oraqle.compiler.nodes.univariate import UnivariateNode


class Neg(UnivariateNode[Boolean], Boolean):
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
        raise NotImplementedError("TODO: Not sure if it makes sense to implement this")
    
    # TODO: We can create a high-level implementation that tries all three transformations and chooses the lowest size one
    def _arithmetize_inner(self, strategy: str) -> Node:
        return self.arithmetize_all_representations(strategy)
    
    def _arithmetize_depth_aware_inner(self, cost_of_squaring: float) -> CostParetoFront:
        raise NotImplementedError("TODO!")
    
    def transform_to_reduced_boolean(self) -> ReducedBoolean:
        return ReducedNeg(self._node.transform_to_reduced_boolean(), self._gf)
    
    def transform_to_unreduced_boolean(self) -> UnreducedBoolean:
        raise NotImplementedError("TODO!")
    
    def transform_to_inv_unreduced_boolean(self) -> InvUnreducedBoolean:
        raise NotImplementedError("TODO!")


class ReducedNeg(UnivariateNode[ReducedBoolean], ReducedBoolean):
    """A node that negates a ReducedBoolean input."""

    @property
    def _node_shape(self) -> str:
        return "box"

    @property
    def _hash_name(self) -> str:
        return "reduced_neg"

    @property
    def _node_label(self) -> str:
        return "NEG"

    def _operation_inner(self, input: FieldArray) -> FieldArray:
        assert input in {0, 1}
        return self._gf(not bool(input))

    def _arithmetize_inner(self, strategy: str) -> Boolean:
        return Subtraction(
            Constant(self._gf(1)), self._node.arithmetize(strategy), self._gf
        ).arithmetize(strategy)  # type: ignore

    # FIXME: CostParetoFront should be generic
    def _arithmetize_depth_aware_inner(self, cost_of_squaring: float) -> CostParetoFront:
        return Subtraction(Constant(self._gf(1)), self._node, self._gf).arithmetize_depth_aware(
            cost_of_squaring
        )
