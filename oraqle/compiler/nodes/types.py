"""TypeNodes are no-op nodes that wrap another node or multiple of them."""

from oraqle.compiler.nodes.abstract import Node


class TypeNode(Node):
    pass


InvisibleTypeNode
- single Node
- invisible: forwards arrow

BundleTypeNode
- ordered list
- draws group around nodes
