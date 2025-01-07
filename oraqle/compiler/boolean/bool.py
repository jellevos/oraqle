from abc import ABCMeta
from typing import Callable, Dict, Type
from galois import GF, FieldArray
from oraqle.compiler.nodes.abstract import CostParetoFront, Node, UnoverloadedWrapper
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
    
    def arithmetize(self, strategy: str) -> "Boolean":
        return super().arithmetize(strategy)  # type: ignore
    
    # TODO: Also make the output of depth-aware arithmetization a front of Booleans


class BooleanInput(Input, Boolean):
    
    def evaluate(self, actual_inputs: Dict[str, FieldArray]) -> FieldArray:
        output = super().evaluate(actual_inputs)
        if not (output == 0 or output == 1):
            raise ValueError(f"Not a Boolean: {output}")
        return output
    

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


# TODO: Make the old implementations part of ReducedBoolean
class ReducedBoolean(Boolean):
    pass


_class_cache = {}

def _get_dynamic_class(name, bases, attrs):
    """Tracks dynamic classes so that cast_to_reduced_boolean on a specific class always returns the same dynamic Boolean class."""
    key = (name, bases)
    if key not in _class_cache:
        _class_cache[key] = type(name, bases, attrs)
    return _class_cache[key]
    

def cast_to_reduced_boolean(node: Node) -> ReducedBoolean:
    """
    Casts this Node to a Boolean. This results in a new class called <node's class name>_ReducedBool.

    !!! warning
        This modifies the node *in place*, so the node is now a Boolean node.
    """
    BooleanNode = _get_dynamic_class(f'{node.__class__.__name__}_ReducedBool', (node.__class__, ReducedBoolean), dict(node.__class__.__dict__))  # type: ignore
    node.__class__ = BooleanNode
    return node  # type: ignore
    

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
