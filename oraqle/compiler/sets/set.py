from abc import abstractmethod
from typing import Set, Type
from galois import FieldArray
from oraqle.compiler.nodes.abstract import Node, UnoverloadedWrapper
from oraqle.compiler.nodes.flexible import CommutativeUniqueReducibleNode
from oraqle.compiler.nodes.leafs import Input


# class AbstractSet(CommutativeUniqueReducibleNode[Node]):

#     # # TODO: This should have a certain (runtime-available) type for the elements
#     def __init__(self, operands: Set[UnoverloadedWrapper[Node]], gf: Type[FieldArray], universe_size: int):
#         super().__init__(operands, gf)
#         self._universe_size = universe_size
    
#     # @abstractmethod
#     # def query_constant(self, element: FieldArray):
#     #     pass


# # TODO: In the future, this could be an object describing all the possible (i.e. a product of) set representations that one wants to consider
# class InputSet(AbstractSet):

#     @property
#     def _hash_name(self) -> str:
#         return "set"

#     @property
#     def _node_label(self) -> str:
#         return f"Set {self._name}"

#     def __init__(self, name: str, gf: Type[FieldArray], universe_size: int):
#         super().__init__(set(), gf, universe_size)
#         self._name = name

# # if __name__ == "__main__":
# #     Set()
