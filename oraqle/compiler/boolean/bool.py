from typing import Type
from galois import FieldArray
from oraqle.compiler.nodes.abstract import Node
from oraqle.compiler.nodes.types import TypeNode


# TODO: We should move __and__, __or__, etc. to this class instead of Node
class Boolean(TypeNode):
    """A Boolean node indicates that the wrapped Node is a Boolean."""

    # TODO: Consider implementing node name etc. in TypeNode (waar mogelijk). Iig voor rendering.
    
    def __init__(self, node: Node):
        super().__init__(node._gf)

    # FIXME: arithmetize should return Boolean, consider doing so by adding it as generic of TypeNode
