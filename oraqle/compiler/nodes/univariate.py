from abc import abstractmethod
from typing import List, Type, override
from oraqle.compiler.graphviz import DotFile
from oraqle.compiler.nodes.abstract import Node
from oraqle.compiler.nodes.fixed import FixedNode
from oraqle.compiler.nodes.fp.leafs import Constant


class UnivariateNode[Operand: Node](FixedNode[Operand]):
    """An abstract node with a single input."""

    @property
    @abstractmethod
    def _node_shape(self) -> str:
        """Graphviz node shape."""

    def __init__(self, node: Operand):
        """Initialize a univariate node."""
        self._node = node
        assert not isinstance(node, Constant)
        FixedNode.__init__(self)

    def operands(self) -> List[Operand]:  # noqa: D102
        return [self._node]

    def set_operands(self, operands: List[Operand]):  # noqa: D102
        self._node = operands[0]

    @override
    def to_graph(self, graph_builder: DotFile) -> int:  # noqa: D102
        if self._to_graph_cache is None:
            attributes = {}

            attributes.update(self._overriden_graphviz_attributes)

            self._to_graph_cache = graph_builder.add_node(
                label=self._node_label, shape=self._node_shape, **attributes
            )

            graph_builder.add_link(self._node.to_graph(graph_builder), self._to_graph_cache)

        return self._to_graph_cache

    @override
    def __hash__(self) -> int:
        if self._hash is None:
            self._hash = hash((self._hash_name, self._node))

        return self._hash

    @override
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
