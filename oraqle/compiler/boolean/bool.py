from __future__ import annotations

from abc import abstractmethod
from typing import Dict, Optional, Type
from galois import GF, FieldArray
from oraqle.compiler.circuit import Circuit
from oraqle.compiler.nodes.abstract import Node, UnoverloadedWrapper
from oraqle.compiler.nodes.leafs import Constant
from oraqle.compiler.nodes.leafs import Input


# TODO: Make outgoing edges a specific color
class Boolean(Node):
    """A Boolean node indicates that this Node outputs a Boolean."""
    
    def __invert__(self) -> "Node":
        from oraqle.compiler.boolean.bool_neg import Neg

        return Neg(self, self._gf)
    
    def bool_or(self, other: "Boolean", flatten=True) -> "Boolean":
        """Performs an OR operation between `self` and `other`, possibly flattening the result into an OR operation between many operands.

        It is possible to disable flattening by setting `flatten=False`.
        
        Returns:
            A possibly flattened `Or` node or a `Constant` representing self & other.
        """
        from oraqle.compiler.boolean.bool_or import Or
        from oraqle.compiler.nodes.leafs import Constant

        if flatten and isinstance(other, Or):
            return other.or_flatten(self)

        if isinstance(other, Constant):
            if bool(other._value):
                return BooleanConstant(self._gf(1))
            else:
                return self

        if self.is_equivalent(other):
            return self
        else:
            return Or({UnoverloadedWrapper(self), UnoverloadedWrapper(other)}, self._gf)

    def __or__(self, other: "Boolean") -> "Boolean":
        if not isinstance(other, Node):
            raise Exception(f"The RHS of this OR is not a Node: {self} | {other}")

        return self.bool_or(other)

    def bool_and(self, other: "Boolean", flatten=True) -> "Boolean":
        """Performs an AND operation between `self` and `other`, possibly flattening the result into an AND operation between many operands.

        It is possible to disable flattening by setting `flatten=False`.
        
        Returns:
            A possibly flattened `And` node or a `Constant` representing self & other.
        """
        from oraqle.compiler.boolean.bool_and import And
        from oraqle.compiler.nodes.leafs import Constant

        if flatten and isinstance(other, And):
            return other.and_flatten(self)

        if isinstance(other, Constant):
            if bool(other._value):
                return self
            else:
                return BooleanConstant(self._gf(0))

        if self.is_equivalent(other):
            return self
        else:
            return And({UnoverloadedWrapper(self), UnoverloadedWrapper(other)}, self._gf)

    def __and__(self, other: "Boolean") -> "Boolean":
        if not isinstance(other, Node):
            raise Exception(f"The RHS of this AND is not a Node: {self} & {other}")

        return self.bool_and(other)

    # TODO: Should we override the signature of arithmetize and depth_aware_arithmetize? Perhaps it is best done using generics

    def arithmetize_all_representations(self, strategy: str) -> "Boolean":
        # TODO: Should this be cached?

        # TODO: Consider a more efficient strategy and do not catch exceptions
        best_arithmetization: Optional[Node] = None
        lowest_size: Optional[int] = None
        errors = []

        try:
            reduced = self.transform_to_reduced_boolean().arithmetize(strategy)
            size = reduced.to_arithmetic().multiplicative_size()
            if lowest_size is None or size < lowest_size:
                lowest_size = size
                best_arithmetization = reduced
        except NotImplementedError as err:
            errors.append(err)

        try:
            unreduced = self.transform_to_unreduced_boolean().arithmetize(strategy)
            size = unreduced.to_arithmetic().multiplicative_size()
            if lowest_size is None or size < lowest_size:
                lowest_size = size
                best_arithmetization = unreduced
        except NotImplementedError as err:
            errors.append(err)

        try:
            inv_unreduced = self.transform_to_inv_unreduced_boolean().arithmetize(strategy)
            size = inv_unreduced.to_arithmetic().multiplicative_size()
            if lowest_size is None or size < lowest_size:
                lowest_size = size
                best_arithmetization = inv_unreduced
        except NotImplementedError as err:
            errors.append(err)

        if best_arithmetization is None:
            raise Exception(f"Arithmetization failed, these were the caught errors: {errors}")
        
        Circuit([best_arithmetization]).to_pdf(f"test_{best_arithmetization}.pdf")
        return best_arithmetization  # type: ignore

    
    # TODO: Also make the output of depth-aware arithmetization a front of Booleans

    @abstractmethod
    def transform_to_reduced_boolean(self) -> ReducedBoolean:
        pass

    @abstractmethod
    def transform_to_unreduced_boolean(self) -> UnreducedBoolean:
        pass

    @abstractmethod
    def transform_to_inv_unreduced_boolean(self) -> InvUnreducedBoolean:
        pass


# FIXME: This is actually a ReducedBooleanConstant
class BooleanConstant(Constant, Boolean):
    
    def bool_or(self, other: Boolean, flatten=True) -> Node:  # noqa: D102
        if isinstance(other, Constant):
            return Constant(self._gf(bool(self._value) | bool(other._value)))

        return other.bool_or(self, flatten)
    
    def bool_and(self, other: Boolean, flatten=True) -> Node:  # noqa: D102
        if isinstance(other, Constant):
            return Constant(self._gf(bool(self._value) & bool(other._value)))

        return other.bool_and(self, flatten)
    
    def transform_to_reduced_boolean(self) -> ReducedBoolean:
        return ReducedBooleanConstant(self._value)
    
    def transform_to_unreduced_boolean(self) -> UnreducedBoolean:
        return UnreducedBooleanConstant(self._value)
    
    def transform_to_inv_unreduced_boolean(self) -> InvUnreducedBoolean:
        return InvUnreducedBooleanConstant(gf(1) - self._value)


_class_cache = {}

def _get_dynamic_class(name, bases, attrs):
    """Tracks dynamic classes so that cast_to_reduced_boolean on a specific class always returns the same dynamic Boolean class."""
    key = (name, bases)
    if key not in _class_cache:
        _class_cache[key] = type(name, bases, attrs)
    return _class_cache[key]


def _cast_to[N: Node](node: Node, to: Type[N]) -> N:
    CastedNode = _get_dynamic_class(f'{node.__class__.__name__}_{N.__name__}', (node.__class__, N), dict(node.__class__.__dict__))  # type: ignore
    node.__class__ = CastedNode
    return node  # type: ignore
    

def cast_to_reduced_boolean(node: Node) -> ReducedBoolean:
    """
    Casts this Node to a ReducedBoolean. This results in a new class called <node's class name>_ReducedBool.

    !!! warning
        This modifies the node *in place*, so the node now has a different (extended) class.
    """
    return _cast_to(node, ReducedBoolean)


def cast_to_unreduced_boolean(node: Node) -> UnreducedBoolean:
    """
    Casts this Node to a UnreducedBoolean. This results in a new class called <node's class name>_UnreducedBoolean.

    !!! warning
        This modifies the node *in place*, so the node now has a different (extended) class.
    """
    return _cast_to(node, UnreducedBoolean)


def cast_to_inv_unreduced_boolean(node: Node) -> InvUnreducedBoolean:
    """
    Casts this Node to a InvUnreducedBoolean. This results in a new class called <node's class name>_InvUnreducedBoolean.

    !!! warning
        This modifies the node *in place*, so the node now has a different (extended) class.
    """
    return _cast_to(node, InvUnreducedBoolean)


class UnreducedBoolean(Boolean):

    def transform_to_reduced_boolean(self) -> ReducedBoolean:
        return cast_to_reduced_boolean(self == 1)
    
    def transform_to_unreduced_boolean(self) -> UnreducedBoolean:
        return self
    
    def transform_to_inv_unreduced_boolean(self) -> InvUnreducedBoolean:
        return cast_to_inv_unreduced_boolean(~self)


class InvUnreducedBoolean(Boolean):

    def transform_to_reduced_boolean(self) -> ReducedBoolean:
        return cast_to_reduced_boolean(self == 0)
    
    def transform_to_unreduced_boolean(self) -> UnreducedBoolean:
        return cast_to_unreduced_boolean(~self)
    
    def transform_to_inv_unreduced_boolean(self) -> InvUnreducedBoolean:
        return self


# TODO: Make the old implementations part of ReducedBoolean
class ReducedBoolean(UnreducedBoolean):
    
    def transform_to_reduced_boolean(self) -> ReducedBoolean:
        return self
    
    def transform_to_unreduced_boolean(self) -> UnreducedBoolean:
        return self
    
    def transform_to_inv_unreduced_boolean(self) -> InvUnreducedBoolean:
        return cast_to_inv_unreduced_boolean(~self)
    

class ReducedBooleanConstant(BooleanConstant, ReducedBoolean):
    pass


class UnreducedBooleanConstant(BooleanConstant, UnreducedBoolean):
    pass


class InvUnreducedBooleanConstant(BooleanConstant, InvUnreducedBoolean):
    pass


# TODO: Define mappings from bool inputs True/False to Boolean representations
class BooleanInput(Input, Boolean):

    # TODO:
    def _arithmetize_inner(self, strategy: str) -> Node:
        return super()._arithmetize_inner(strategy)

    def transform_to_reduced_boolean(self) -> ReducedBoolean:
        return ReducedBooleanInput(self._name, self._gf)
    
    def transform_to_unreduced_boolean(self) -> UnreducedBoolean:
        return UnreducedBooleanInput(self._name, self._gf)
    
    def transform_to_inv_unreduced_boolean(self) -> InvUnreducedBoolean:
        return InvUnreducedBooleanInput(self._name, self._gf)


class ReducedBooleanInput(Input, ReducedBoolean):
    
    def evaluate(self, actual_inputs: Dict[str, FieldArray]) -> FieldArray:
        output = super().evaluate(actual_inputs)
        if not (output == 0 or output == 1):
            raise ValueError(f"Not a Boolean: {output}")
        return output
    

class UnreducedBooleanInput(Input, UnreducedBoolean):
    
    # TODO: Consider changing these arguments to allow bool inputs that are then mapped
    def evaluate(self, actual_inputs: Dict[str, FieldArray]) -> FieldArray:
        output = super().evaluate(actual_inputs)
        if not (output == 0 or output == 1):
            raise ValueError(f"Not a Boolean: {output}")
        return output
    

class InvUnreducedBooleanInput(Input, InvUnreducedBoolean):
    
    def evaluate(self, actual_inputs: Dict[str, FieldArray]) -> FieldArray:
        output = super().evaluate(actual_inputs)
        if not (output == 0 or output == 1):
            raise ValueError(f"Not a Boolean: {output}")
        return output
    

def test_isinstance_cast_reduced_boolean():
    from oraqle.compiler.nodes.leafs import Input
    from oraqle.compiler.nodes.arbitrary_arithmetic import Sum, sum_

    gf = GF(7)
    a = Input("a", gf)
    b = Input("b", gf)
    c = Input("c", gf)

    s = sum_(a, b, c)
    assert isinstance(s, Sum)
    assert isinstance(cast_to_reduced_boolean(s), Sum)


if __name__ == "__main__":
    gf = GF(7)
    a = Input("a", gf)
    b = Input("b", gf)
    c = Input("c", gf)

    res1 = a == b
    res2 = b == c

    res = res1 & res2
