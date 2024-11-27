from typing import List, Type, Union
from galois import FieldArray
from oraqle.compiler.boolean.bool_and import all_
from oraqle.compiler.nodes.abstract import CostParetoFront, Node
from oraqle.compiler.nodes.flexible import CommutativeUniqueReducibleNode
from oraqle.compiler.nodes.leafs import Input
from oraqle.compiler.sets.set import AbstractSet, Set


# FIXME: gf should be moved to arithmetization and Input should be abstract. Instead we should have integer Inputs. ShortInt should be what is currently Input.
class BitSet(AbstractSet):

    def __init__(self, name: str, gf: type[FieldArray], bits: List[Input]) -> None:  # TODO: Input should become Boolean
        # TODO: Change constructor
        super().__init__(name, gf)
        self._bits = bits

    @classmethod
    def from_universe(cls, name: str, gf: type[FieldArray], universe_size: int) -> "BitSet":
        return cls(name, gf, [Input(f"{name}_{i}", gf) for i in range(universe_size)])

    @classmethod
    def coerce_from(cls, set: Union[Set, "BitSet"]) -> "BitSet":
        if isinstance(set, Set):
            return cls.from_universe(set._name, set._gf, set._universe_size)
        elif isinstance(set, BitSet):
            return set
        else:
            raise TypeError(f"A set of type {type(set)} cannot be coerced to BitSet.")

    def _arithmetize_inner(self, strategy: str) -> Node:
        # TODO: In the future, this might arithmetize the Booleans
        return self

class BitSetIntersection(CommutativeUniqueReducibleNode[BitSet]):

    @property
    def _hash_name(self) -> str:
        return "bitset_intersection"

    @property
    def _node_label(self) -> str:
        return "∩"

    def _arithmetize_inner(self, strategy: str) -> BitSet:
        # TODO: Turn all inputs into BitSets and TypeError if not possible
        # TODO: Do not arithmetize each bit separately
        bit_count = len(next(iter(self._operands)).node._bits)
        return BitSet(" ∩ ".join(operand.node._name for operand in self._operands), self._gf, [all_(*(operand.node._bits[i] for operand in self._operands)) for i in range(bit_count)])
    
    def _arithmetize_depth_aware_inner(self, cost_of_squaring: float) -> CostParetoFront:
        raise NotImplementedError()
    
    def _inner_operation(self, a: FieldArray, b: FieldArray) -> FieldArray:
        # TODO: This should take sets as input
        raise NotImplementedError()
