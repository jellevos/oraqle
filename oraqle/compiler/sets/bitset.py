from abc import abstractmethod
from typing import Callable, Dict, List, Sequence, Set, Type, Union
from galois import GF, FieldArray
from oraqle.compiler.boolean.bool import Boolean, BooleanInput, InvUnreducedBoolean, ReducedBoolean, ReducedBooleanInput, UnreducedBoolean
from oraqle.compiler.boolean.bool_and import all_
from oraqle.compiler.circuit import Circuit
from oraqle.compiler.nodes.abstract import CostParetoFront, ExtendedArithmeticNode, Node, UnoverloadedWrapper
from oraqle.compiler.nodes.fixed import FixedNode
from oraqle.compiler.nodes.flexible import CommutativeUniqueReducibleNode
from oraqle.compiler.nodes.leafs import Input
#from oraqle.compiler.nodes.types import BundleTypeNode, TypeNode
#from oraqle.compiler.sets.set import AbstractSet, InputSet


# TODO: At some point we can implement __and__ for intersections
class BitSet(Node):
    
    def __getitem__(self, index) -> Boolean:
        assert 0 <= index < len(self)
        return BitSetIndex(self, index, self._gf)
    
    @abstractmethod
    def __len__(self) -> int:
        pass
    
    def contains_element(self, element: int) -> Boolean:
        return self[element - 1]
    
    @staticmethod
    def intersection(*bitsets: "BitSet") -> "BitSet":
        intersection = BitSetIntersection({UnoverloadedWrapper(bitset) for bitset in bitsets}, bitsets[0]._gf)
        return intersection


# TODO: Reduce duplication
class BitSetContainer(FixedNode[Boolean], BitSet):

    @property
    def _hash_name(self) -> str:
        return "bitset"

    @property
    def _node_label(self) -> str:
        return "Bitset"
    
    def __init__(self, bits: Sequence[Boolean]):
        super().__init__(bits[0]._gf)
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
    
    def operands(self) -> List[Boolean]:
        return self._bits
    
    def set_operands(self, operands: List[Boolean]):
        self._bits = operands

    def operation(self, operands: List[FieldArray]) -> FieldArray:
        raise NotImplementedError("Incompatible: must return all operands")
    
    def _expansion(self) -> Node:
        raise NotImplementedError()
    
    def _arithmetize_inner(self, strategy: str) -> Node:
        # TODO: Consider changing the arithmetize type
        return BitSetContainer([bit.arithmetize(strategy) for bit in self._bits])  # type: ignore
    
    def _arithmetize_depth_aware_inner(self, cost_of_squaring: float) -> CostParetoFront:
        raise NotImplementedError("TODO")
    
    def _arithmetize_extended_inner(self) -> Node:
        # TODO: Consider changing the arithmetize type
        return BitSetContainer([bit.arithmetize_extended() for bit in self._bits])  # type: ignore
    
    def __len__(self) -> int:
        return len(self._bits)
    

# TODO: Should this be a FixedNode?
class ReducedBitSet(FixedNode[BitSet], BitSet):
    
    @property
    def _hash_name(self) -> str:
        return "reduced_bitset"

    @property
    def _node_label(self) -> str:
        return "Bitset"
    
    def __init__(self, bitset: BitSetContainer):
        super().__init__(bitset._gf)
        self._bitset = bitset

    # def apply_function_to_operands(self, function: Callable[[Node], None]):
    #     for operand in self._bits:
    #         function(operand)

    # def replace_operands_using_function(self, function: Callable[[Node], Node]):
    #     self._bits = [function(operand) for operand in self._bits]
    
    def __hash__(self) -> int:
        if self._hash is None:
            self._hash = hash((self._hash_name, self._bitset))

        return self._hash
    
    def is_equivalent(self, other: Node) -> bool:
        if not isinstance(other, ReducedBitSet):
            return False
        
        return self._bitset.is_equivalent(other._bitset)
    
    def operands(self) -> List[BitSet]:
        return [self._bitset]
    
    def set_operands(self, operands: List[BitSet]):
        self._bitset = operands[0]

    def operation(self, operands: List[FieldArray]) -> FieldArray:
        raise NotImplementedError("Incompatible: must return all operands")
    
    def _expansion(self) -> Node:
        raise NotImplementedError()
    
    def _arithmetize_inner(self, strategy: str) -> Node:
        # TODO: Consider changing the arithmetize type
        return BitSetContainer([bit.transform_to_reduced_boolean().arithmetize(strategy) for bit in self._bitset._bits])  # type: ignore
    
    def _arithmetize_depth_aware_inner(self, cost_of_squaring: float) -> CostParetoFront:
        raise NotImplementedError("TODO")
    
    def _arithmetize_extended_inner(self) -> ExtendedArithmeticNode:
        return BitSetContainer([bit.transform_to_reduced_boolean().arithmetize_extended() for bit in self._bitset._bits])  # type: ignore
    
    def __len__(self) -> int:
        return len(self._bitset)
    

# TODO: Should this be a FixedNode?
class InvUnreducedBitSet(FixedNode[BitSet], BitSet):
    
    @property
    def _hash_name(self) -> str:
        return "inv_unreduced_bitset"

    @property
    def _node_label(self) -> str:
        return "Bitset"
    
    def __init__(self, bitset: BitSetContainer):
        super().__init__(bitset._gf)
        self._bitset = bitset

    # def apply_function_to_operands(self, function: Callable[[Node], None]):
    #     for operand in self._bits:
    #         function(operand)

    # def replace_operands_using_function(self, function: Callable[[Node], Node]):
    #     self._bits = [function(operand) for operand in self._bits]
    
    def __hash__(self) -> int:
        if self._hash is None:
            self._hash = hash((self._hash_name, self._bitset))

        return self._hash
    
    def is_equivalent(self, other: Node) -> bool:
        if not isinstance(other, InvUnreducedBitSet):
            return False
        
        return self._bitset.is_equivalent(other._bitset)
    
    def operands(self) -> List[BitSet]:
        return [self._bitset]
    
    def set_operands(self, operands: List[BitSet]):
        self._bitset = operands[0]

    def operation(self, operands: List[FieldArray]) -> FieldArray:
        raise NotImplementedError("Incompatible: must return all operands")
    
    def _expansion(self) -> Node:
        raise NotImplementedError()
    
    def _arithmetize_inner(self, strategy: str) -> Node:
        # TODO: Consider changing the arithmetize type
        return BitSetContainer([bit.transform_to_inv_unreduced_boolean().arithmetize(strategy) for bit in self._bitset._bits])  # type: ignore
    
    def _arithmetize_depth_aware_inner(self, cost_of_squaring: float) -> CostParetoFront:
        raise NotImplementedError("TODO")
    
    def _arithmetize_extended_inner(self) -> Node:
        # TODO: Consider changing the arithmetize type
        return BitSetContainer([bit.transform_to_inv_unreduced_boolean().arithmetize_extended() for bit in self._bitset._bits])  # type: ignore
    
    def __len__(self) -> int:
        return len(self._bitset)


class BitSetIndex(Boolean):

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

    def _expansion(self) -> Node:
        raise NotImplementedError()

    def arithmetize(self, strategy: str) -> Node:
        if self._arithmetize_cache is None:
            arithmetized_bitset = self._bitset.arithmetize(strategy)
            assert isinstance(arithmetized_bitset, BitSetContainer)
            self._arithmetize_cache = arithmetized_bitset._bits[self._index]

        return self._arithmetize_cache
    
    def arithmetize_depth_aware(self, cost_of_squaring: float) -> CostParetoFront:
        raise NotImplementedError("TODO")
    
    def arithmetize_extended(self) -> ExtendedArithmeticNode:
        if self._arithmetize_extended_cache is None:
            arithmetized_bitset = self._bitset.arithmetize_extended()
            assert isinstance(arithmetized_bitset, BitSetContainer)
            self._arithmetize_extended_cache = arithmetized_bitset._bits[self._index]  # type: ignore

        return self._arithmetize_extended_cache  # type: ignore
    
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
    
    def transform_to_reduced_boolean(self) -> ReducedBoolean:
        return ReducedBitSetIndex(ReducedBitSet(self._bitset), self._index, self._gf)  # type: ignore
    
    def transform_to_unreduced_boolean(self) -> UnreducedBoolean:
        raise NotImplementedError("TODO!")
    
    def transform_to_inv_unreduced_boolean(self) -> InvUnreducedBoolean:
        return InvUnreducedBitSetIndex(InvUnreducedBitSet(self._bitset), self._index, self._gf)  # type: ignore


# TODO: Reduce duplication
class ReducedBitSetIndex(ReducedBoolean):
    
    @property
    def _hash_name(self) -> str:
        return "reduced_bitset_index"

    @property
    def _node_label(self) -> str:
        return f"Bitset index #{self._index}"

    def __init__(self, bitset: BitSet, index: int, gf: Type[FieldArray]):
        super().__init__(gf)
        self._bitset = bitset
        self._index = index

    def _expansion(self) -> Node:
        raise NotImplementedError()

    def arithmetize(self, strategy: str) -> Node:
        if self._arithmetize_cache is None:
            arithmetized_bitset = self._bitset.arithmetize(strategy)
            assert isinstance(arithmetized_bitset, BitSetContainer)
            self._arithmetize_cache = arithmetized_bitset._bits[self._index]

        return self._arithmetize_cache
    
    def arithmetize_depth_aware(self, cost_of_squaring: float) -> CostParetoFront:
        raise NotImplementedError("TODO")
    
    def arithmetize_extended(self) -> ExtendedArithmeticNode:
        if self._arithmetize_extended_cache is None:
            arithmetized_bitset = self._bitset.arithmetize_extended()
            assert isinstance(arithmetized_bitset, BitSetContainer)
            self._arithmetize_extended_cache = arithmetized_bitset._bits[self._index]  # type: ignore

        return self._arithmetize_extended_cache  # type: ignore
    
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


class InvUnreducedBitSetIndex(InvUnreducedBoolean):
    
    @property
    def _hash_name(self) -> str:
        return "inv_unreduced_bitset_index"

    @property
    def _node_label(self) -> str:
        return f"Bitset index #{self._index}"

    def __init__(self, bitset: BitSet, index: int, gf: Type[FieldArray]):
        super().__init__(gf)
        self._bitset = bitset
        self._index = index

    def _expansion(self) -> Node:
        raise NotImplementedError()

    def arithmetize(self, strategy: str) -> Node:
        if self._arithmetize_cache is None:
            arithmetized_bitset = self._bitset.arithmetize(strategy)
            assert isinstance(arithmetized_bitset, BitSetContainer)
            self._arithmetize_cache = arithmetized_bitset._bits[self._index]

        return self._arithmetize_cache
    
    def arithmetize_depth_aware(self, cost_of_squaring: float) -> CostParetoFront:
        raise NotImplementedError("TODO")
    
    def arithmetize_extended(self) -> ExtendedArithmeticNode:
        if self._arithmetize_extended_cache is None:
            arithmetized_bitset = self._bitset.arithmetize_extended()
            assert isinstance(arithmetized_bitset, BitSetContainer)
            self._arithmetize_extended_cache = arithmetized_bitset._bits[self._index]  # type: ignore

        return self._arithmetize_extended_cache  # type: ignore
    
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
    
    def _expansion(self) -> Node:
        bit_count = len(self)
        return BitSetContainer([all_(*(operand.node[i] for operand in self._operands)) for i in range(bit_count)])

    # def _arithmetize_inner(self, strategy: str) -> BitSet:
    #     # TODO: Assert all lengths are equal? Or that they map the same universe?
    #     bit_count = len(self)
    #     # TODO: After arithmetizing one of the bitsets, we can consider reusing that arithmetization for the rest (so not to run in O(n))
    #     return BitSetContainer([all_(*(operand.node[i] for operand in self._operands)).arithmetize(strategy) for i in range(bit_count)])  # type: ignore
    
    # def _arithmetize_depth_aware_inner(self, cost_of_squaring: float) -> CostParetoFront:
    #     raise NotImplementedError()
    
    def _inner_operation(self, a: FieldArray, b: FieldArray) -> FieldArray:
        # TODO: This should take sets as input
        raise NotImplementedError()


if __name__ == "__main__":
    gf = GF(11)

    bits1 = [BooleanInput(f"b1_{i}", gf) for i in range(10)]
    bitset1 = BitSetContainer(bits1)

    bits2 = [BooleanInput(f"b2_{i}", gf) for i in range(10)]
    bitset2 = BitSetContainer(bits2)

    bits3 = [BooleanInput(f"b3_{i}", gf) for i in range(10)]
    bitset3 = BitSetContainer(bits3)

    final_bitset = BitSet.intersection(bitset1, bitset2, bitset3)

    # TODO: Only check contains for elements in server's set
    circuit = Circuit([final_bitset.contains_element(3)])
    circuit.to_pdf("debug.pdf")
    arithmetic_circuit = circuit.arithmetize()
    arithmetic_circuit.to_pdf("debug2.pdf")


# TODO: Write tests
# FIXME: Make it so that arithmetizing a single bit in the bitset does not arithmetize all the other bits
