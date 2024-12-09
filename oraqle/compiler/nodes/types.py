"""TypeNodes are no-op nodes that wrap another node or multiple of them."""

from typing import Callable, Dict, List, Sequence, Type
from galois import GF, FieldArray
from oraqle.compiler.circuit import Circuit
from oraqle.compiler.graphviz import DotFile
from oraqle.compiler.nodes.abstract import CostParetoFront, Node
from oraqle.compiler.nodes.leafs import Input


# class TypeNode(Node):

    # def evaluate(self, actual_inputs: Dict[str, FieldArray]) -> FieldArray:
    #     raise NotImplementedError("Should not be called on a type")
    
    # def arithmetize(self, strategy: str) -> Node:
    #     raise NotImplementedError("Should not be called on a type")
    
    # def arithmetize_depth_aware(self, cost_of_squaring: float) -> CostParetoFront:
    #     raise NotImplementedError("Should not be called on a type")


# class InvisibleTypeNode(TypeNode):
#     pass


# class BundleTypeNode[N: Node](TypeNode):
    
#     def __init__(self, operands: Sequence[N], gf: type[FieldArray]):
#         super().__init__(gf)
#         self._operands = list(operands)

#     def apply_function_to_operands(self, function: Callable[[N], None]):
#         for operand in self._operands:
#             function(operand)

#     def replace_operands_using_function(self, function: Callable[[N], N]):
#         self._operands = [function(operand) for operand in self._operands]
    
#     def __hash__(self) -> int:
#         if self._hash is None:
#             self._hash = hash(tuple(self._operands))

#         return self._hash
    
#     def is_equivalent(self, other: Node) -> bool:
#         if not isinstance(other, self.__class__):
#             return False
        
#         return all(a.is_equivalent(b) for a, b in zip(self._operands, other._operands))
    
#     def to_graph(self, graph_builder: DotFile) -> int:
#         if self._to_graph_cache is None:
#             operand_ids = [operand.to_graph(graph_builder) for operand in self._operands]
#             graph_builder.add_cluster(operand_ids, label=self._node_label)
#             self._to_graph_cache = -1

#         return self._to_graph_cache


# TODO:
# InvisibleTypeNode
# - single Node
# - invisible: forwards arrow
