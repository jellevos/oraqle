"""This module contains classes for representing equality checks."""
from galois import GF, FieldArray

from oraqle.compiler.arithmetic.exponentiation import Power
from oraqle.compiler.arithmetic.subtraction import Subtraction
from oraqle.compiler.boolean.bool import Boolean, InvUnreducedBoolean, ReducedBoolean, UnreducedBoolean, cast_to_reduced_boolean
from oraqle.compiler.boolean.bool_neg import Neg, ReducedNeg
from oraqle.compiler.nodes.abstract import CostParetoFront, ExtendedArithmeticNode, Node
from oraqle.compiler.nodes.binary_arithmetic import CommutativeBinaryNode
from oraqle.compiler.nodes.extended import UnknownRandom, Reveal
from oraqle.compiler.nodes.leafs import Input
from oraqle.compiler.nodes.univariate import UnivariateNode


class IsNonZero(UnivariateNode, Boolean):
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
    
    def _expansion(self) -> Node:
        raise NotImplementedError()

    def _arithmetize_inner(self, strategy: str) -> Node:
        return self.arithmetize_all_representations(strategy)
    
    def _arithmetize_extended_inner(self) -> ExtendedArithmeticNode:
        arithmetic = self.arithmetize_all_representations("best-effort")
        # TODO: Reveal is not okay unless the output is allowed to be known
        #extended_arithmetic = (Reveal(UnknownRandom(self._gf) * self._node.arithmetize()) == 0).arithmetize("best-effort")

        # TODO: In the future, we can use a metric to decide if we want the extended arithmetic solution
        # if metric(extended_arithmetic) < metric(arithmetic):
        #     return extended_arithmetic
        
        return arithmetic  # type: ignore
    
    def _arithmetize_depth_aware_inner(self, cost_of_squaring: float) -> CostParetoFront:
        raise NotImplementedError("TODO!")
    
    def transform_to_reduced_boolean(self) -> ReducedBoolean:
        return ReducedIsNonZero(self._node)
    
    def transform_to_unreduced_boolean(self) -> UnreducedBoolean:
        raise NotImplementedError("TODO!")
    
    def transform_to_inv_unreduced_boolean(self) -> InvUnreducedBoolean:
        raise NotImplementedError("TODO!")
    

class ReducedIsNonZero(UnivariateNode, ReducedBoolean):
    """This node represents a zero check: x == 0."""

    @property
    def _node_shape(self) -> str:
        return "box"

    @property
    def _hash_name(self) -> str:
        return "reduced_is_nonzero"

    @property
    def _node_label(self) -> str:
        return "!= 0"
    
    def _operation_inner(self, input: FieldArray) -> FieldArray:
        return input != 0
    
    def _expansion(self) -> Node:
        return cast_to_reduced_boolean(Power(self._node, self._gf.order - 1, self._gf))


# TODO: UnreducedEquals MUST multiply with randomness
class Equals(CommutativeBinaryNode, Boolean):
    """This node represents an equality operation: x == y."""

    @property
    def _hash_name(self) -> str:
        return "equals"

    @property
    def _node_label(self) -> str:
        return "=="

    def _operation_inner(self, x, y) -> FieldArray:
        return self._gf(int(x == y))
    
    def _expansion(self) -> Node:
        raise NotImplementedError()

    def _arithmetize_inner(self, strategy: str) -> Node:
        return self.arithmetize_all_representations(strategy)
    
    def _arithmetize_depth_aware_inner(self, cost_of_squaring: float) -> CostParetoFront:
        raise NotImplementedError("TODO!")
    
    def transform_to_reduced_boolean(self) -> ReducedBoolean:
        return ReducedEquals(self._left, self._right, self._gf)
    
    def transform_to_unreduced_boolean(self) -> UnreducedBoolean:
        raise NotImplementedError("TODO!")
    
    def transform_to_inv_unreduced_boolean(self) -> InvUnreducedBoolean:
        raise NotImplementedError("TODO!")
    

class ReducedEquals(CommutativeBinaryNode, ReducedBoolean):
    """This node represents an equality operation: x == y."""

    @property
    def _hash_name(self) -> str:
        return "reduced_equals"

    @property
    def _node_label(self) -> str:
        return "=="

    def _operation_inner(self, x, y) -> FieldArray:
        return self._gf(int(x == y))
    
    def _expansion(self) -> Node:
        return Neg(
            IsNonZero(Subtraction(self._left, self._right, self._gf))
        )


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
