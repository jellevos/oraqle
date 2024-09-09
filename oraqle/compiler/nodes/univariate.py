"""Abstract nodes for univariate operations."""

from abc import abstractmethod
from typing import List, Type

from galois import FieldArray

from oraqle.compiler.graphviz import DotFile
from oraqle.compiler.nodes.abstract import Node
from oraqle.compiler.nodes.fixed import FixedNode
from oraqle.compiler.nodes.leafs import Constant


class UnivariateNode(FixedNode):
    """An abstract node with a single input."""

    @property
    @abstractmethod
    def _node_shape(self) -> str:
        """Graphviz node shape."""

    def __init__(self, node: Node, gf: Type[FieldArray]):
        """Initialize a univariate node."""
        self._node = node
        assert not isinstance(node, Constant)
        super().__init__(gf)

    
    def operands(self) -> List["Node"]:  # noqa: D102
        return [self._node]

    
    def set_operands(self, operands: List["Node"]):  # noqa: D102
        self._node = operands[0]

    @abstractmethod
    def _operation_inner(self, input: FieldArray) -> FieldArray:
        """Evaluate the operation on the input. This method does not have to cache."""

    
    def operation(self, operands: List[FieldArray]) -> FieldArray:  # noqa: D102
        return self._operation_inner(operands[0])

    
    def to_graph(self, graph_builder: DotFile) -> int:  # noqa: D102
        if self._to_graph_cache is None:
            attributes = {}

            attributes.update(self._overriden_graphviz_attributes)

            self._to_graph_cache = graph_builder.add_node(
                label=self._node_label, shape=self._node_shape, **attributes
            )

            graph_builder.add_link(self._node.to_graph(graph_builder), self._to_graph_cache)

        return self._to_graph_cache

    def __hash__(self) -> int:
        if self._hash is None:
            self._hash = hash((self._hash_name, self._node))

        return self._hash

    def is_equivalent(self, other: Node) -> bool:
        """Check whether `self` is semantically equivalent to `other`.

        This function may have false negatives but it should never return false positives.

        Returns:
        -------
            `True` if `self` is semantically equivalent to `other`, `False` if they are not or that they cannot be shown to be equivalent.

        """
        if not isinstance(other, self.__class__):
            return False

        if hash(self) != hash(other):
            return False

        return self._node.is_equivalent(other._node)
