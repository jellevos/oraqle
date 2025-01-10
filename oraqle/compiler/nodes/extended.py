import random
from typing import List, Type
from galois import FieldArray
from oraqle.compiler.nodes.abstract import CostParetoFront, ExtendedArithmeticNode, Node
from oraqle.compiler.nodes.leafs import LeafNode
from oraqle.compiler.nodes.univariate import UnivariateNode


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
    
    def _arithmetize_inner(self, strategy: str) -> Node:
        raise NotImplementedError("Reveal cannot be arithmetized: arithmetic circuits only contain arithmetic operations.")
    
    def _arithmetize_depth_aware_inner(self, cost_of_squaring: float) -> CostParetoFront:
        raise NotImplementedError("Reveal cannot be arithmetized: arithmetic circuits only contain arithmetic operations.")
    
    def _operation_inner(self, input: FieldArray) -> FieldArray:
        return input
    
    # FIXME: Overload operators to create *plaintext* operations


class Random(LeafNode, ExtendedArithmeticNode):

    @property
    def _node_shape(self) -> str:
        return "circle"
    
    @property
    def _hash_name(self) -> str:
        return "random"

    @property
    def _node_label(self) -> str:
        return "Random"
    
    def __init__(self, gf: type[FieldArray]):
        self._hash = hash(random.randbytes(16))  # TODO: Not neat
        super().__init__(gf)
    
    def __hash__(self) -> int:
        return self._hash

    def _arithmetize_inner(self, strategy: str) -> Node:
        raise NotImplementedError("Reveal cannot be arithmetized: arithmetic circuits only contain arithmetic operations.")
    
    def _arithmetize_depth_aware_inner(self, cost_of_squaring: float) -> CostParetoFront:
        raise NotImplementedError("Reveal cannot be arithmetized: arithmetic circuits only contain arithmetic operations.")
    
    def is_equivalent(self, other: Node) -> bool:
        if not isinstance(other, self.__class__):
            return False
        
        return self._hash == other._hash
    
    def operation(self, operands: List[FieldArray]) -> FieldArray:
        return self._gf.Random()
