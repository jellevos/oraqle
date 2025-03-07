"""Module containing the most fundamental classes in the compiler."""
from abc import ABC, abstractmethod
from collections import Counter
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterator, List, Optional, Self, Set, Tuple, Type, Union


if TYPE_CHECKING:
    from oraqle.compiler.nodes.fp.fixed import ArithmeticNode


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
