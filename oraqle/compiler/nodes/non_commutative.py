"""A collection of abstract nodes representing operations that are non-commutative."""
from abc import abstractmethod
from typing import List, Type

from galois import FieldArray

from oraqle.compiler.graphviz import DotFile
from oraqle.compiler.nodes.abstract import Node
from oraqle.compiler.nodes.fixed import BinaryNode


class NonCommutativeBinaryNode(BinaryNode):
    """Represents a non-cummutative binary operation such as `x < y` or `x - y`."""

    def __init__(self, left, right, gf: Type[FieldArray]):
        """Initialize a Node that performs an operation between two operands that is not commutative."""
        self._left = left
        self._right = right
        super().__init__(gf)

    @abstractmethod
    def _operation_inner(self, x, y) -> FieldArray:
        """Applies the binary operation on x and y."""

    def operation(self, operands: List[FieldArray]) -> FieldArray:  # noqa: D102
        return self._operation_inner(operands[0], operands[1])

    def operands(self) -> List[Node]:  # noqa: D102
        return [self._left, self._right]

    def set_operands(self, operands: List["Node"]):  # noqa: D102
        self._left = operands[0]
        self._right = operands[1]

    def __hash__(self) -> int:
        if self._hash is None:
            left_hash = hash(self._left)
            right_hash = hash(self._right)

            self._hash = hash((self._hash_name, (left_hash, right_hash)))

        return self._hash
    
    def is_equivalent(self, other: Node) -> bool:  # noqa: D102
        if not isinstance(other, self.__class__):
            return False

        if hash(self) != hash(other):
            return False

        return self._left.is_equivalent(other._left) and self._right.is_equivalent(other._right)

    def to_graph(self, graph_builder: DotFile) -> int:  # noqa: D102
        if self._to_graph_cache is None:
            attributes = {"shape": "box"}
            attributes.update(self._overriden_graphviz_attributes)

            self._to_graph_cache = graph_builder.add_node(
                label=self._node_label,
                **attributes,
            )

            left = self._left.to_graph(graph_builder)
            right = self._right.to_graph(graph_builder)

            graph_builder.add_link(left, self._to_graph_cache, headport="nw")
            graph_builder.add_link(right, self._to_graph_cache, headport="ne")

        return self._to_graph_cache
