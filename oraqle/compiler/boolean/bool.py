from galois import GF
from oraqle.compiler.nodes.abstract import Node, UnoverloadedWrapper
from oraqle.compiler.nodes.leafs import Input


class Boolean(Node):
    """A Boolean node indicates that the wrapped Node is a Boolean."""
    
    def __invert__(self) -> "Node":
        from oraqle.compiler.boolean.bool_neg import Neg

        return Neg(self, self._gf)
    
    def bool_or(self, other: "Node", flatten=True) -> "Node":
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
                return Constant(self._gf(1))
            else:
                return self

        if self.is_equivalent(other):
            return self
        else:
            return Or({UnoverloadedWrapper(self), UnoverloadedWrapper(other)}, self._gf)

    def __or__(self, other) -> "Node":
        if not isinstance(other, Node):
            raise Exception(f"The RHS of this OR is not a Node: {self} | {other}")

        return self.bool_or(other)

    def bool_and(self, other: "Node", flatten=True) -> "Node":
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
                return Constant(self._gf(0))

        if self.is_equivalent(other):
            return self
        else:
            return And({UnoverloadedWrapper(self), UnoverloadedWrapper(other)}, self._gf)

    def __and__(self, other) -> "Node":
        if not isinstance(other, Node):
            raise Exception(f"The RHS of this AND is not a Node: {self} & {other}")

        return self.bool_and(other)


if __name__ == "__main__":
    gf = GF(7)
    a = Input("a", gf)
    b = Input("b", gf)
    c = Input("c", gf)

    res1 = a == b
    res2 = b == c

    res = res1 & res2
