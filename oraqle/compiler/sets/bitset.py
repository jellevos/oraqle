from abc import abstractmethod
from typing import Callable, Dict, List, Sequence, Set, Type, Union
from galois import GF, FieldArray
from oraqle.compiler.boolean.bool_and import all_
from oraqle.compiler.circuit import Circuit
from oraqle.compiler.nodes.abstract import CostParetoFront, Node, UnoverloadedWrapper
from oraqle.compiler.nodes.fixed import FixedNode
from oraqle.compiler.nodes.flexible import CommutativeUniqueReducibleNode
from oraqle.compiler.nodes.leafs import Input
#from oraqle.compiler.nodes.types import BundleTypeNode, TypeNode
#from oraqle.compiler.sets.set import AbstractSet, InputSet


# TODO: At some point we can implement __and__ for intersections
class BitSet(Node):  # TODO: Node should become Boolean
    
    def __getitem__(self, index) -> Node:  # TODO: Make Boolean
        assert 0 <= index < len(self)
        return BitSetIndex(self, index, self._gf)
    
    @abstractmethod
    def __len__(self) -> int:
        pass
    
    def contains_element(self, element: int) -> Node:  # TODO: Node should become Boolean
        return self[element - 1]
    
    @staticmethod
    def intersection(*bitsets: "BitSet") -> "BitSet":
        intersection = BitSetIntersection({UnoverloadedWrapper(bitset) for bitset in bitsets}, bitsets[0]._gf)
        return intersection


class BitSetContainer(FixedNode[Node], BitSet):  # TODO: Boolean

    @property
    def _hash_name(self) -> str:
        return "bitset"

    @property
    def _node_label(self) -> str:
        return "Bitset"
    
    def __init__(self, bits: Sequence[Node], gf: Type[FieldArray]):
        super().__init__(gf)
        self._bits = list(bits)

    # def apply_function_to_operands(self, function: Callable[[Node], None]):
    #     for operand in self._bits:
    #         function(operand)

    # def replace_operands_using_function(self, function: Callable[[Node], Node]):
    #     self._bits = [function(operand) for operand in self._bits]
    
    def __hash__(self) -> int:
        if self._hash is None:
            self._hash = hash((self._hash_name, tuple(self._bits)))

        return self._hash
    
    def is_equivalent(self, other: Node) -> bool:
        if not isinstance(other, BitSetContainer):
            return False
        
        return all(a.is_equivalent(b) for a, b in zip(self._bits, other._bits))
    
    def operands(self) -> List[Node]:
        return self._bits
    
    def set_operands(self, operands: List[Node]):
        self._bits = operands

    def operation(self, operands: List[FieldArray]) -> FieldArray:
        raise NotImplementedError("Incompatible: must return all operands")
    
    def _arithmetize_inner(self, strategy: str) -> Node:
        # TODO: Consider changing the arithmetize type
        return BitSetContainer([bit.arithmetize(strategy) for bit in self._bits], self._gf)
    
    def _arithmetize_depth_aware_inner(self, cost_of_squaring: float) -> CostParetoFront:
        raise NotImplementedError("TODO")
    
    def __len__(self) -> int:
        return len(self._bits)


class BitSetIndex(Node):

    @property
    def _hash_name(self) -> str:
        return "bitset_index"

    @property
    def _node_label(self) -> str:
        return f"Bitset index #{self._index}"

    def __init__(self, bitset: BitSet, index: int, gf: Type[FieldArray]):
        super().__init__(gf)
        self._bitset = bitset
        self._index = index

    def arithmetize(self, strategy: str) -> Node:
        if self._arithmetize_cache is None:
            arithmetized_bitset = self._bitset.arithmetize(strategy)
            assert isinstance(arithmetized_bitset, BitSetContainer)
            self._arithmetize_cache = arithmetized_bitset._bits[self._index]

        return self._arithmetize_cache
    
    def arithmetize_depth_aware(self, cost_of_squaring: float) -> CostParetoFront:
        raise NotImplementedError("TODO")
    
    def apply_function_to_operands(self, function: Callable[[Node], None]):
        function(self._bitset)

    def replace_operands_using_function(self, function: Callable[[Node], Node]):
        self._bitset = function(self._bitset)

    def evaluate(self, actual_inputs: Dict[str, FieldArray]) -> FieldArray:
        raise NotImplementedError("TODO: Requires refactors")
    
    def __hash__(self) -> int:
        if self._hash is None:
            self._hash = hash((self._hash_name, self._index, self._bitset))

        return self._hash
    
    def is_equivalent(self, other: Node) -> bool:
        if not isinstance(other, BitSetIndex):
            return False
        
        return self._index == other._index and self._bitset.is_equivalent(other._bitset)


# FIXME: gf should be moved to arithmetization and Input should be abstract. Instead we should have integer Inputs. ShortInt should be what is currently Input.
# class EncodeBitSet(AbstractSet):

#     @property
#     def _hash_name(self) -> str:
#         return "encode_bitset"

#     @property
#     def _node_label(self) -> str:
#         return "Encode bitset"
    
#     def __init__(self, operands: Set[UnoverloadedWrapper[Node]], gf: Type[FieldArray], universe_size: int):
#         super().__init__(operands, gf, universe_size)

#     # def __init__(self, gf: type[FieldArray], universe_size: int) -> None:  # TODO: Input should become Boolean
#     #     # TODO: Change constructor
#     #     super().__init__(gf, universe_size)
#     #     self._bits = bits

#     # @classmethod
#     # def from_inputs(cls, name: str, gf: type[FieldArray], universe_size: int) -> "BitSet":
#     #     # TODO: Consider making this a separate class
#     #     return cls(gf, [Input(f"{name}_{i}", gf) for i in range(universe_size)])

#     @classmethod
#     def coerce_from(cls, set: Union[InputSet, "BitSet"]) -> "BitSet":
#         if isinstance(set, InputSet):
#             return cls.from_inputs(set._name, set._gf, set._universe_size)
#         elif isinstance(set, BitSet):
#             return set
#         else:
#             raise TypeError(f"A set of type {type(set)} cannot be coerced to BitSet.")

#     def _arithmetize_inner(self, strategy: str) -> Node:
#         # TODO: In the future, this might arithmetize the Booleans
#         return self


class BitSetIntersection(CommutativeUniqueReducibleNode[BitSet], BitSet):

    @property
    def _hash_name(self) -> str:
        return "bitset_intersection"

    @property
    def _node_label(self) -> str:
        return "∩"
    
    def __len__(self) -> int:
        # TODO: Compute once
        return len(next(iter(self._operands)).node)

    def _arithmetize_inner(self, strategy: str) -> BitSet:
        # TODO: Assert all lengths are equal? Or that they map the same universe?
        bit_count = len(self)
        # " ∩ ".join(operand.node._name for operand in self._operands)
        return BitSetContainer([all_(*(operand.node[i] for operand in self._operands)).arithmetize(strategy) for i in range(bit_count)], self._gf)
    
    def _arithmetize_depth_aware_inner(self, cost_of_squaring: float) -> CostParetoFront:
        raise NotImplementedError()
    
    def _inner_operation(self, a: FieldArray, b: FieldArray) -> FieldArray:
        # TODO: This should take sets as input
        raise NotImplementedError()


if __name__ == "__main__":
    gf = GF(11)

    bits1 = [Input(f"b1_{i}", gf) for i in range(10)]
    bitset1 = BitSetContainer(bits1, gf)

    bits2 = [Input(f"b2_{i}", gf) for i in range(10)]
    bitset2 = BitSetContainer(bits2, gf)

    final_bitset = BitSet.intersection(bitset1, bitset2)

    # TODO: Only check contains for elements in server's set
    circuit = Circuit([final_bitset.contains_element(3)])
    circuit.to_pdf("debug.pdf")
    arithmetic_circuit = circuit.arithmetize()
    arithmetic_circuit.to_pdf("debug2.pdf")
