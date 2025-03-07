from abc import abstractmethod
from collections import Counter
from typing import TYPE_CHECKING, Optional, Set, Type

from galois import FieldArray
from oraqle.compiler.nodes.abstract import Node, UnoverloadedWrapper

if TYPE_CHECKING:
    from oraqle.compiler.nodes.fp.abstract import CostParetoFront
    from oraqle.compiler.nodes.fp.fixed import ArithmeticNode, GaloisArithmeticNode


class FpdNode(Node):
    """
    An element of F_{p^d}.
    """

    @abstractmethod
    def galois_arithmetize(self) -> "GaloisArithmeticNode":
        pass


class FpNode(FpdNode):  # noqa: PLR0904
    """Abstract node representing an element in Fp in a circuit."""

    def __init__(self, gf: Type[FieldArray]):
        """Creates a new node, of which the result is known by the parties identified by `known_by`, as well as those who know all input operands."""
        self._gf = gf
        Node.__init__(self)

        self._evaluate_cache: Optional[FieldArray] = None
        self._arithmetize_cache: Optional[ArithmeticNode] = None
        self._arithmetize_depth_cache: Optional[CostParetoFront] = None
        self._arithmetic_cache: Optional[ArithmeticNode] = None

    def clear_cache(self, already_cleared: Set[int]):
        # FIXME: The cache should not be cleared twice for the same node, but there is no way to check this.
        self._evaluate_cache: Optional[FieldArray] = None
        self._arithmetize_cache: Optional["ArithmeticNode"] = None
        self._arithmetize_depth_cache: Optional[CostParetoFront] = None
        self._arithmetic_cache: Optional[ArithmeticNode] = None

        Node.clear_cache(self, already_cleared)

    @abstractmethod
    def arithmetize(self, strategy: str) -> "ArithmeticNode":
        """Arithmetizes this node, replacing it with only arithmetic operations (constants, additions, and multiplications).

        The current implementation only aims at reducing the total number of multiplications.
        """

    def galois_arithmetize(self) -> "GaloisArithmeticNode":
        """
        The default implementation simply calls arithmetize("best-effort").
        """
        return self.arithmetize("best-effort")

    @abstractmethod
    def arithmetize_depth_aware(
        self, cost_of_squaring: float
    ) -> "CostParetoFront":
        """Arithmetizes this node in a depth-aware fashion, replacing high-level nodes with only arithmetic operations (constants, additions, and multiplications).
        
        Returns:
            `CostParetoFront` containing a front that trades off multiplicative depth and multiplicative cost.
        """

    def to_arithmetic(self) -> "ArithmeticNode":
        """Outputs this node's equivalent ArithmeticNode. Errors if this node does not have a direct arithmetic equivalent.

        Raises:
            Exception: If there is no direct arithmetic equivalent.
        """
        # TODO: Make this a non-generic exception
        raise Exception(
            f"This node does not have a direct arithmetic equivalent: {self}. Consider first calling `arithmetize`."
        )

    def add(self, other: "FpNode", flatten=True) -> "FpNode":
        """Performs a summation between `self` and `other`, possibly flattening any sums.

        It is possible to disable flattening by setting `flatten=False`.
        
        Returns:
            A possibly flattened `Sum` node or a `Constant` representing self & other.
        """
        from oraqle.compiler.nodes.fp.arbitrary_arithmetic import Sum
        from oraqle.compiler.nodes.fp.leafs import Constant

        if flatten and isinstance(self, Sum):
            return self.add_flatten(other)

        if flatten and isinstance(other, Sum):
            return other.add_flatten(self)

        if isinstance(other, Constant):
            if int(other._value) == 0:
                return self
            return Sum(Counter({UnoverloadedWrapper(self): 1}), self._gf, constant=other._value)

        if id(self) == id(other):
            return Sum(Counter({UnoverloadedWrapper(self): 2}), self._gf)
        else:
            return Sum(
                Counter({UnoverloadedWrapper(self): 1, UnoverloadedWrapper(other): 1}), self._gf
            )

    def __add__(self, other) -> "FpNode":
        from oraqle.compiler.nodes.fp.fixed import try_to_node
        other_node = try_to_node(other, self._gf)
        if other_node is None:
            raise Exception(f"The RHS of this + cannot be made into a Node: {self} - {other}")

        return self.add(other_node)

    def __radd__(self, other) -> "FpNode":
        from oraqle.compiler.nodes.fp.fixed import try_to_node
        other_node = try_to_node(other, self._gf)
        if other_node is None:
            raise Exception(f"The LHS of this + cannot be made into a Node: {other} - {self}")

        return self.add(other_node)

    # TODO: Make separate method for flatten=False, which creates an ArithmeticNode (same for additions)
    def mul(self, other: "FpNode", flatten=True) -> "FpNode":  # noqa: PLR0911
        """Performs a multiplication between `self` and `other`, possibly flattening any products.

        It is possible to disable flattening by setting `flatten=False`.
        
        Returns:
            A possibly flattened `Product` node or a `Constant` representing self & other.
        """
        from oraqle.compiler.nodes.fp.arbitrary_arithmetic import Product
        from oraqle.compiler.nodes.fp.leafs import Constant

        if flatten and isinstance(self, Product):
            return self.mul_flatten(other)

        if flatten and isinstance(other, Product):
            return other.mul_flatten(self)

        if isinstance(other, Constant):
            if int(other._value) == 0:
                return other
            if int(other._value) == 1:
                return self
            return Product(Counter({UnoverloadedWrapper(self): 1}), self._gf, constant=other._value)

        if id(self) == id(other):
            return Product(Counter({UnoverloadedWrapper(self): 2}), self._gf)
        else:
            return Product(
                Counter({UnoverloadedWrapper(self): 1, UnoverloadedWrapper(other): 1}), self._gf
            )

    def __mul__(self, other) -> "FpNode":
        if not isinstance(other, FpNode):
            raise Exception(f"The RHS of this multiplication is not a Node: {self} * {other}")

        return self.mul(other)

    def bool_or(self, other: "FpNode", flatten=True) -> "FpNode":
        """Performs an OR operation between `self` and `other`, possibly flattening the result into an OR operation between many operands.

        It is possible to disable flattening by setting `flatten=False`.
        
        Returns:
            A possibly flattened `Or` node or a `Constant` representing self & other.
        """
        from oraqle.compiler.boolean.bool_or import Or
        from oraqle.compiler.nodes.fp.leafs import Constant

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

    def __or__(self, other) -> "FpNode":
        if not isinstance(other, FpNode):
            raise Exception(f"The RHS of this OR is not a Node: {self} | {other}")

        return self.bool_or(other)

    def bool_and(self, other: "FpNode", flatten=True) -> "FpNode":
        """Performs an AND operation between `self` and `other`, possibly flattening the result into an AND operation between many operands.

        It is possible to disable flattening by setting `flatten=False`.
        
        Returns:
            A possibly flattened `And` node or a `Constant` representing self & other.
        """
        from oraqle.compiler.boolean.bool_and import And
        from oraqle.compiler.nodes.fp.leafs import Constant

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

    def __and__(self, other) -> "FpNode":
        if not isinstance(other, FpNode):
            raise Exception(f"The RHS of this AND is not a Node: {self} & {other}")

        return self.bool_and(other)

    def __lt__(self, other) -> "FpNode":
        from oraqle.compiler.nodes.fp.fixed import try_to_node
        other_node = try_to_node(other, self._gf)
        if other_node is None:
            raise Exception(f"The RHS of this < cannot be made into a Node: {self} < {other}")

        from oraqle.compiler.comparison.comparison import StrictComparison

        return StrictComparison(self, other_node, less_than=True, gf=self._gf)

    def __gt__(self, other) -> "FpNode":
        from oraqle.compiler.nodes.fp.fixed import try_to_node
        other_node = try_to_node(other, self._gf)
        if other_node is None:
            raise Exception(f"The RHS of this > cannot be made into a Node: {self} > {other}")

        from oraqle.compiler.comparison.comparison import StrictComparison

        return StrictComparison(self, other_node, less_than=False, gf=self._gf)

    def __le__(self, other) -> "FpNode":
        from oraqle.compiler.nodes.fp.fixed import try_to_node
        other_node = try_to_node(other, self._gf)
        if other_node is None:
            raise Exception(f"The RHS of this <= cannot be made into a Node: {self} <= {other}")

        from oraqle.compiler.comparison.comparison import Comparison

        return Comparison(self, other_node, less_than=True, gf=self._gf)

    def __ge__(self, other) -> "FpNode":
        from oraqle.compiler.nodes.fp.fixed import try_to_node
        other_node = try_to_node(other, self._gf)
        if other_node is None:
            raise Exception(f"The RHS of this >= cannot be made into a Node: {self} >= {other}")

        from oraqle.compiler.comparison.comparison import Comparison

        return Comparison(self, other_node, less_than=False, gf=self._gf)

    def __neg__(self) -> "FpNode":
        from oraqle.compiler.nodes.fp.leafs import Constant

        return Constant(-self._gf(1)) * self

    def __invert__(self) -> "FpNode":
        from oraqle.compiler.boolean.bool_neg import Neg

        return Neg(self, self._gf)

    def __pow__(self, other) -> "FpNode":
        if not isinstance(other, int):
            raise Exception(f"The exponent must be an integer: {self}**{other}")

        from oraqle.compiler.arithmetic.exponentiation import Power

        return Power(self, other, self._gf)

    def __sub__(self, other) -> "FpNode":
        from oraqle.compiler.nodes.fp.fixed import try_to_node
        other_node = try_to_node(other, self._gf)
        if other_node is None:
            raise Exception(f"The RHS of this - cannot be made into a Node: {self} - {other}")

        from oraqle.compiler.arithmetic.subtraction import Subtraction

        return Subtraction(self, other_node, self._gf)

    def __rsub__(self, other) -> "FpNode":
        from oraqle.compiler.nodes.fp.fixed import try_to_node
        other_node = try_to_node(other, self._gf)
        if other_node is None:
            raise Exception(f"The LHS of this - cannot be made into a Node: {other} - {self}")

        from oraqle.compiler.arithmetic.subtraction import Subtraction

        return Subtraction(other_node, self, self._gf)

    def __eq__(self, other) -> "FpNode":
        from oraqle.compiler.nodes.fp.fixed import try_to_node
        other_node = try_to_node(other, self._gf)
        if other_node is None:
            raise Exception(f"The RHS of this == cannot be made into a Node: {self} == {other}")

        from oraqle.compiler.comparison.equality import Equals

        return Equals(self, other_node, self._gf)
