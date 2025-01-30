import random
from typing import List, Optional, Sequence, Set, Type
from galois import FieldArray
from oraqle.compiler.nodes.abstract import CostParetoFront, ExtendedArithmeticNode, Node, ExtendedArithmeticCosts
from oraqle.compiler.nodes.leafs import Constant, ExtendedArithmeticLeafNode, LeafNode
from oraqle.compiler.nodes.univariate import UnivariateNode
from oraqle.mpc.parties import PartyId

from pysat.formula import WCNF, IDPool


class Reveal(UnivariateNode, ExtendedArithmeticNode):

    @property
    def _node_shape(self) -> str:
        return "circle"
    
    @property
    def _hash_name(self) -> str:
        return "reveal"

    @property
    def _node_label(self) -> str:
        return "Reveal"
    
    def _expansion(self) -> Node:
        raise NotImplementedError("Reveal cannot be arithmetized: arithmetic circuits only contain arithmetic operations.")
    
    def _arithmetize_inner(self, strategy: str) -> Node:
        raise NotImplementedError("Reveal cannot be arithmetized: arithmetic circuits only contain arithmetic operations.")
    
    def _arithmetize_depth_aware_inner(self, cost_of_squaring: float) -> CostParetoFront:
        raise NotImplementedError("Reveal cannot be arithmetized: arithmetic circuits only contain arithmetic operations.")
    
    def _operation_inner(self, input: FieldArray) -> FieldArray:
        return input
    
    # FIXME: Overload operators to create *plaintext* operations


# TODO: Reduce code duplication
# TODO: We can consider creating NonZero randomness (we can still generate it as usual but with a small probability of incorrectness)
class UnknownRandom(ExtendedArithmeticLeafNode):  # UnknownRandom(LeafNode):

    # @property
    # def _overriden_graphviz_attributes(self) -> dict:
    #     return {"shape": "circle", "style": "filled", "fillcolor": "thistle"}

    # @property
    # def _node_shape(self) -> str:
    #     return "circle"
    
    @property
    def _hash_name(self) -> str:
        return "unknown_random"

    @property
    def _node_label(self) -> str:
        return "Rand"
    
    def __init__(self, gf: type[FieldArray]):
        super().__init__(gf)
        self._generated_hash = random.randint(-2**63, 2**63 - 1)
    
    def __hash__(self) -> int:
        return self._generated_hash

    def _arithmetize_inner(self, strategy: str) -> Node:
        raise NotImplementedError("UnknownRandom cannot be arithmetized: arithmetic circuits only contain arithmetic operations.")
    
    def _arithmetize_depth_aware_inner(self, cost_of_squaring: float) -> CostParetoFront:
        raise NotImplementedError("UnknownRandom cannot be arithmetized: arithmetic circuits only contain arithmetic operations.")
    
    # def _arithmetize_extended_inner(self) -> ExtendedArithmeticNode:
    #     raise NotImplementedError("TODO: Sum of KnownRandomness")
    
    def is_equivalent(self, other: Node) -> bool:
        if not isinstance(other, self.__class__):
            return False
        
        return self._hash == other._hash
    
    def operation(self, operands: List[FieldArray]) -> FieldArray:
        return self._gf.Random()
    
    def _computational_cost(self, costs: Sequence[ExtendedArithmeticCosts], party_id: PartyId) -> float:
        raise NotImplementedError("This is not an operation")

    def _replace_randomness_inner(self, party_count: int) -> ExtendedArithmeticNode:
        raise NotImplementedError("TODO!")
    
#     def __add__(self, other) -> Node:
#         if isinstance(other, Constant):
#             # TODO: Should output itself if this randomness is not used in other places
#             raise NotImplementedError("TODO")
        
#         return RandomAddition(other)
    
#     def __mul__(self, other) -> Node:
#         if isinstance(other, Constant):
#             # TODO: Should output itself if this randomness is not used in other places
#             raise NotImplementedError("TODO")
        
#         return RandomMultiplication(other)


class KnownRandom(ExtendedArithmeticLeafNode):

    @property
    def _overriden_graphviz_attributes(self) -> dict:
        return {"shape": "circle", "style": "filled", "fillcolor": "thistle"}

    @property
    def _node_shape(self) -> str:
        return "circle"
    
    @property
    def _hash_name(self) -> str:
        return "known_random"

    @property
    def _node_label(self) -> str:
        return f"Uniform\\nrandom"
    
    def __init__(self, gf: type[FieldArray], known_by: Optional[Set[PartyId]]):
        super().__init__(gf, known_by)
        self._generated_hash = random.randint(-2**63, 2**63 - 1)
    
    def __hash__(self) -> int:
        return self._generated_hash

    def _arithmetize_inner(self, strategy: str) -> Node:
        raise NotImplementedError("KnownRandom cannot be arithmetized: arithmetic circuits only contain arithmetic operations.")
    
    def _arithmetize_depth_aware_inner(self, cost_of_squaring: float) -> CostParetoFront:
        raise NotImplementedError("KnownRandom cannot be arithmetized: arithmetic circuits only contain arithmetic operations.")
    
    def is_equivalent(self, other: Node) -> bool:
        if not isinstance(other, self.__class__):
            return False
        
        return self._hash == other._hash
    
    def operation(self, operands: List[FieldArray]) -> FieldArray:
        return self._gf.Random()
    
    def _computational_cost(self, costs: Sequence[ExtendedArithmeticCosts], party_id: PartyId) -> float:
        raise NotImplementedError("This is not an operation")

    def _replace_randomness_inner(self, party_count: int) -> ExtendedArithmeticNode:
        return self


# class RandomAddition(CommutativeBinaryNode):
    
#     @property
#     def _overriden_graphviz_attributes(self) -> dict:
#         return {"style": "rounded,filled", "fillcolor": "grey80"}

#     @property
#     def _node_shape(self) -> str:
#         return "square"

#     @property
#     def _hash_name(self) -> str:
#         return f"constant_mul_{self._constant}"

#     @property
#     def _node_label(self) -> str:
#         return "×"  # noqa: RUF001
        

# class RandomMultiplication(CommutativeBinaryNode):
    
#     @property
#     def _overriden_graphviz_attributes(self) -> dict:
#         return {"style": "rounded,filled", "fillcolor": "grey80"}

#     @property
#     def _node_shape(self) -> str:
#         return "square"

#     @property
#     def _hash_name(self) -> str:
#         return f"constant_mul_{self._constant}"

#     @property
#     def _node_label(self) -> str:
#         return "×"  # noqa: RUF001


# class PublicRandom(LeafNode, ExtendedArithmeticNode):

#     @property
#     def _node_shape(self) -> str:
#         return "circle"
    
#     @property
#     def _hash_name(self) -> str:
#         return "public_random"

#     @property
#     def _node_label(self) -> str:
#         return "Random [pub]"
    
#     def __init__(self, gf: type[FieldArray]):
#         super().__init__(gf)
#         self._hash = random.randint(-2**63, 2**63 - 1)
    
#     def __hash__(self) -> int:
#         return self._hash

#     def _arithmetize_inner(self, strategy: str) -> Node:
#         raise NotImplementedError("PublicRandom cannot be arithmetized: arithmetic circuits only contain arithmetic operations.")
    
#     def _arithmetize_depth_aware_inner(self, cost_of_squaring: float) -> CostParetoFront:
#         raise NotImplementedError("PublicRandom cannot be arithmetized: arithmetic circuits only contain arithmetic operations.")
    
#     def is_equivalent(self, other: Node) -> bool:
#         if not isinstance(other, self.__class__):
#             return False
        
#         return self._hash == other._hash
    
#     def operation(self, operands: List[FieldArray]) -> FieldArray:
#         return self._gf.Random()
