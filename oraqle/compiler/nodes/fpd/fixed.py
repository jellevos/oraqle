# from typing import Type

# from galois import FieldArray
# from oraqle.compiler.nodes.fp.fixed import FixedFpNode
# from oraqle.compiler.nodes.fpd.abstract import FpdNode
# from oraqle.compiler.nodes.univariate import UnivariateNode


# class UnivariateFpdNode[Operand: FpdNode](UnivariateNode[Operand], FixedFpdNode[Operand]):
#     """An abstract node with a single FpNode input."""

#     def __init__(self, node: FpdNode, gf: Type[FieldArray]):
#         """Initialize a univariate node."""
#         self._node = node
#         # TODO: assert not isinstance(node, FpdConstant)
#         FixedFpNode.__init__(self, gf)
#         UnivariateNode.__init__(self, node)
