"""Module containing the most fundamental classes in the compiler."""
from abc import ABC, abstractmethod
from collections import Counter
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterator, List, Optional, Self, Set, Tuple, Type, Union

from galois import FieldArray

from oraqle.compiler.graphviz import DotFile
from oraqle.compiler.instructions import ArithmeticInstruction
from oraqle.compiler.nodes.abstract import Node, UnoverloadedWrapper


if TYPE_CHECKING:
    from oraqle.compiler.nodes.fp.fixed import ArithmeticNode


def _to_node(obj: Union["FpNode", int, bool], gf: Type[FieldArray]) -> "FpNode":
    if isinstance(obj, FpNode):
        return obj

    if isinstance(obj, int):
        from oraqle.compiler.nodes.fp.leafs import Constant

        return Constant(gf(obj))


# TODO: This should output higher level types (not necessarily elements of Fp, Fpd, ..., we do not yet know which it should be)
def try_to_node(obj: Any, gf: Type[FieldArray]) -> Optional["FpNode"]:
    """Tries to cast this object into a valid `Node`.
    
    This can be used to transform e.g. an `int` or `bool` into a `Constant`.
    If it is applied to a `Node`, it does nothing.
    
    Returns:
    A `Node` or `None` depending on whether the object is castable.
    """
    return _to_node(obj, gf)


# TODO: It would be great if we can move out this ParetoFront class, but it's hard to do without circular imports
class ParetoFront(ABC):
    """Abstract base class for ParetoFronts.
    
    One objective is to minimize the multiplicative depth, while the other objective is minimizing some value, such as the multiplicative size or cost.
    """

    def __init__(self) -> None:
        """Initialize an empty ParetoFront."""
        self._nodes_by_depth: Dict[int, Tuple[Union[int, float], ArithmeticNode]] = {}
        self._highest_depth: int = -1

    @abstractmethod
    def _get_value(self, node: "ArithmeticNode") -> Union[int, float]:
        pass

    @abstractmethod
    def _default_value(self) -> Union[int, float]:
        pass

    @classmethod
    def from_node(
        cls,
        node: "ArithmeticNode",
        depth: Optional[int] = None,
        value: Optional[Union[int, float]] = None,
    ) -> "ParetoFront":
        """Initialize a `ParetoFront` with one node in it.
        
        Returns:
            New `ParetoFront`.
        """
        self = cls()
        self.add(node, depth, value)
        return self

    @classmethod
    def from_leaf(cls, leaf) -> "ParetoFront":
        """Initialize a `ParetoFront` with one leaf node in it.
        
        Returns:
            New `ParetoFront`.
        """
        self = cls()
        self.add_leaf(leaf)
        return self

    def add(
        self,
        node: "ArithmeticNode",
        depth: Optional[int] = None,
        value: Optional[Union[int, float]] = None,
    ) -> bool:
        """Adds the given `Node` to the `ParetoFront` by computing its multiplicative depth and value.
        
        Alternatively, the user can supply an unchecked `depth` and `value` so that these values do not have to be (re)computed.
        
        Returns:
        `True` if and only if the node was inserted into the ParetoFront (so it was in some way better than the current `Nodes`).
        """
        if depth is None:
            depth = node.multiplicative_depth()

        if value is None:
            value = self._get_value(node)

        return self._add(depth, value, node)

    def _add(self, depth: int, value: Union[int, float], node: "ArithmeticNode") -> bool:
        """Returns True if and only if the node was inserted into the ParetoFront."""
        for d in range(depth + 1):
            if d in self._nodes_by_depth and self._nodes_by_depth[d][0] <= value:
                return False

        self._nodes_by_depth[depth] = (value, node)
        self._highest_depth = max(depth, self._highest_depth)

        for d in range(depth + 1, self._highest_depth + 1):
            if d in self._nodes_by_depth and self._nodes_by_depth[d][0] >= value:
                del self._nodes_by_depth[d]

        return True

    def add_leaf(self, leaf):
        """Add a leaf node to this `ParetoFront`."""
        self._add(0, 0, leaf)  # type: ignore

    def add_front(self, front: "ParetoFront"):
        """Add all elements from `front` to `self`."""
        # TODO: This can be optimized
        for d, s, n in front:
            self.add(n, d, s)

    def __iter__(self) -> Iterator[Tuple[int, Union[int, float], "ArithmeticNode"]]:
        for depth in range(self._highest_depth + 1):
            if depth in self._nodes_by_depth:
                yield depth, self._nodes_by_depth[depth][0], self._nodes_by_depth[depth][1]

    def get_smallest_at_depth(
        self, max_depth: int
    ) -> Optional[Tuple[int, Union[int, float], "ArithmeticNode"]]:
        """Returns the circuit with the smallest value that has at most depth `max_depth`."""
        for depth in reversed(range(max_depth + 1)):
            if depth in self._nodes_by_depth:
                return depth, self._nodes_by_depth[depth][0], self._nodes_by_depth[depth][1]

    def is_empty(self) -> bool:
        """Returns whether the front is empty."""
        return len(self._nodes_by_depth) == 0

    def get_lowest_value(self) -> Optional["ArithmeticNode"]:
        """Returns the value (size or cost) of the Node with the highest depth, and therefore the lowest value."""
        if self._highest_depth == -1:
            return None

        return self._nodes_by_depth[self._highest_depth][1]


def iterate_increasing_depth(front1: ParetoFront, front2: ParetoFront) -> Iterator[
    Tuple[
        Tuple[int, Union[int, float], "ArithmeticNode"],
        Tuple[int, Union[int, float], "ArithmeticNode"],
    ]
]:
    """Iterates over two ParetoFronts, returning pairs of ArithmeticNodes such that the multiplicative depth grows monotonically.

    Yields:
        Pairs of tuples, containing the multiplicative depth, the multiplicative size/cost, and the arithmetization, in that order.
    """
    highest_depth = max(front1._highest_depth, front2._highest_depth)
    last_depth: Optional[int] = None

    # TODO: This is quite inefficient because we constantly loop over the same parts of the fronts, we could instead iterate over both fronts in sequence
    for depth in range(highest_depth + 1):
        res1 = front1.get_smallest_at_depth(depth)
        res2 = front2.get_smallest_at_depth(depth)

        if res1 is None or res2 is None:
            continue

        d1, _, _ = res1
        d2, _, _ = res2

        if last_depth is None or d1 > last_depth or d2 > last_depth:
            yield res1, res2


class SizeParetoFront(ParetoFront):
    """A `ParetoFront` that trades off multiplicative depth with multiplicative size."""

    def _get_value(self, node: "ArithmeticNode") -> int:
        return node.multiplicative_size()

    def _default_value(self) -> int:
        return 0

    def add(self, node: "ArithmeticNode", depth: Optional[int] = None, size: Optional[int] = None):
        """Adds the given `Node` to the `SizeParetoFront` by computing its multiplicative depth and size.
        
        Alternatively, the user can supply an unchecked `depth` and `size` so that these values do not have to be (re)computed.
        
        Returns:
        `True` if and only if the node was inserted into the ParetoFront (so it was in some way better than the current `Nodes`).
        """
        return super().add(node, depth, value=size)


class CostParetoFront(ParetoFront):
    """A `ParetoFront` that trades off multiplicative depth with multiplicative cost."""

    def __init__(self, cost_of_squaring: float) -> None:
        """Initialize an empty `CostParetoFront` with the given `cost_of_squaring`."""
        self._cost_of_squaring = cost_of_squaring
        super().__init__()

    @classmethod
    def from_node(
        cls,
        node: "ArithmeticNode",
        cost_of_squaring: float,
        depth: Optional[int] = None,
        cost: Optional[float] = None,
    ) -> "CostParetoFront":
        """Initialize a `CostParetoFront` with one node in it.
        
        Returns:
            New `CostParetoFront`.
        """
        self = cls(cost_of_squaring)
        self.add(node, depth, cost)
        return self

    @classmethod
    def from_leaf(cls, leaf, cost_of_squaring: float) -> "CostParetoFront":
        """Initialize a `CostParetoFront` with one leaf node in it.
        
        Returns:
            New `CostParetoFront`.
        """
        self = cls(cost_of_squaring)
        self.add_leaf(leaf)
        return self

    def _get_value(self, node: "ArithmeticNode") -> float:
        return node.multiplicative_cost(self._cost_of_squaring)

    def _default_value(self) -> float:
        return 0.0

    def add(
        self, node: "ArithmeticNode", depth: Optional[int] = None, cost: Optional[float] = None
    ) -> bool:
        """Adds the given `Node` to the `CostParetoFront` by computing its multiplicative depth and cost.
        
        Alternatively, the user can supply an unchecked `depth` and `cost` so that these values do not have to be (re)computed.
        
        Returns:
        `True` if and only if the node was inserted into the ParetoFront (so it was in some way better than the current `Nodes`).
        """
        return super().add(node, depth, value=cost)


class FpNode(Node):  # noqa: PLR0904
    """Abstract node representing an element in Fp in a circuit."""

    def __init__(self, gf: Type[FieldArray]):
        """Creates a new node, of which the result is known by the parties identified by `known_by`, as well as those who know all input operands."""
        self._gf = gf
        Node.__init__(self)

        self._evaluate_cache: Optional[FieldArray] = None
        self._arithmetize_cache: Optional["ArithmeticNode"] = None
        self._arithmetize_depth_cache: Optional[CostParetoFront] = None
        self._arithmetic_cache: Optional[ArithmeticNode] = None


    @abstractmethod
    def evaluate(self, actual_inputs: Dict[str, FieldArray]) -> FieldArray:
        """Evaluates the node in the arithmetic circuit. The output should always be reduced modulo the modulus."""

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
        other_node = try_to_node(other, self._gf)
        if other_node is None:
            raise Exception(f"The RHS of this + cannot be made into a Node: {self} - {other}")

        return self.add(other_node)

    def __radd__(self, other) -> "FpNode":
        other_node = try_to_node(other, self._gf)
        if other_node is None:
            raise Exception(f"The LHS of this + cannot be made into a Node: {other} - {self}")

        return self.add(other_node)

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
        other_node = try_to_node(other, self._gf)
        if other_node is None:
            raise Exception(f"The RHS of this < cannot be made into a Node: {self} < {other}")

        from oraqle.compiler.comparison.comparison import StrictComparison

        return StrictComparison(self, other_node, less_than=True, gf=self._gf)

    def __gt__(self, other) -> "FpNode":
        other_node = try_to_node(other, self._gf)
        if other_node is None:
            raise Exception(f"The RHS of this > cannot be made into a Node: {self} > {other}")

        from oraqle.compiler.comparison.comparison import StrictComparison

        return StrictComparison(self, other_node, less_than=False, gf=self._gf)

    def __le__(self, other) -> "FpNode":
        other_node = try_to_node(other, self._gf)
        if other_node is None:
            raise Exception(f"The RHS of this <= cannot be made into a Node: {self} <= {other}")

        from oraqle.compiler.comparison.comparison import Comparison

        return Comparison(self, other_node, less_than=True, gf=self._gf)

    def __ge__(self, other) -> "FpNode":
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
        other_node = try_to_node(other, self._gf)
        if other_node is None:
            raise Exception(f"The RHS of this - cannot be made into a Node: {self} - {other}")

        from oraqle.compiler.arithmetic.subtraction import Subtraction

        return Subtraction(self, other_node, self._gf)

    def __rsub__(self, other) -> "FpNode":
        other_node = try_to_node(other, self._gf)
        if other_node is None:
            raise Exception(f"The LHS of this - cannot be made into a Node: {other} - {self}")

        from oraqle.compiler.arithmetic.subtraction import Subtraction

        return Subtraction(other_node, self, self._gf)

    def __eq__(self, other) -> "FpNode":
        other_node = try_to_node(other, self._gf)
        if other_node is None:
            raise Exception(f"The RHS of this == cannot be made into a Node: {self} == {other}")

        from oraqle.compiler.comparison.equality import Equals

        return Equals(self, other_node, self._gf)


# class ArithmeticNode(FixedFpNode["ArithmeticNode"]):
#     """Extension of Node to indicate that this is a node permitted in a purely arithmetic circuit (with binary additions and multiplications).
    
#     The ArithmeticNode 'mixin' must always come before the base class in the class declaration.
#     """

#     # ArithmeticNode should be like an interface; it should not have an __init__ method.

#     def clear_cache(self, already_cleared: Set[int]):
#         """Clears any cached values of the node and any of its operands."""
#         # FIXME: The cache should not be cleared twice for the same node, but there is no way to check this.
#         if id(self) not in already_cleared:
#             for node in self.operands():
#                 node.clear_cache(already_cleared)

#         self._evaluate_cache: Optional[FieldArray] = None
#         self._to_graph_cache: Optional[int] = None
#         self._arithmetize_cache: Optional[ArithmeticNode] = None
#         self._arithmetize_depth_cache: Optional[ParetoFront] = None
#         self._instruction_cache: Optional[int] = None
#         self._arithmetic_cache: Optional[ArithmeticNode] = None
#         self._parent_count_cache: Optional[int] = None

#         self._hash = None

#         already_cleared.add(id(self))

#     @abstractmethod
#     def multiplicative_depth(self) -> int:
#         """Computes the multiplicative depth of this node and its children recursively.
        
#         Returns:
#         The largest number of multiplications from the output of this node to the leafs of this subcircuit.
#         """

#     def multiplicative_size(self) -> int:
#         """Computes the multiplicative size (number of multiplications) by counting the size of the set returned by self.multiplications().
        
#         Returns:
#         The number of multiplications in this subcircuit.
#         """
#         return len(self.multiplications())

#     def multiplicative_cost(self, cost_of_squaring: float) -> float:
#         """Computes the multiplicative cost (number of general multiplications + cost_of_squaring * squarings).
        
#         It does so by counting the size of the sets returned by self.multiplications() and self.squarings().

#         Returns:
#             The number of proper multiplications + the cost of squaring * the number of squarings.
#         """
#         return (
#             len(self.multiplications())
#             - len(self.squarings())
#             + cost_of_squaring * len(self.squarings())
#         )

#     @abstractmethod
#     def multiplications(self) -> Set[int]:
#         """Returns a set of all the multiplications in this tree of descendants, including itself.
        
#         This includes any squarings.
#         """

#     @abstractmethod
#     def squarings(self) -> Set[int]:
#         """Returns a set of all the squarings in this tree of descendants, including itself."""

#     @abstractmethod
#     def create_instructions(
#         self,
#         instructions: List[ArithmeticInstruction],
#         stack_counter: int,
#         stack_occupied: List[bool],
#     ) -> Tuple[int, int]:
#         """Creates a set of instructions of this node to the given file. Returns the index in the stack of the output and the stack_counter.
        
#         !!! note
#             This method assumes that the _parent_count of each node is up to date.
#         """

#     def to_arithmetic(self) -> "ArithmeticNode":  # noqa: D102
#         return self
