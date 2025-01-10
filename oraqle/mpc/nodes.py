from typing import Set

from galois import GF

from oraqle.compiler.boolean.bool import ReducedBooleanInput
from oraqle.compiler.circuit import Circuit
from oraqle.compiler.nodes.abstract import Node
from oraqle.compiler.sets.bitset import BitSet, BitSetContainer
from oraqle.mpc.parties import PartyId


# FIXME: all the inputs must also be mpc nodes...
class MpcNode:
    
    def __init__(self, node: Node, known_by: Set[PartyId], leakable_to: Set[PartyId], computed_by: Set[PartyId]):
        self._node = node
        self._known_by = known_by
        self._leakable_to = leakable_to
        self._computed_by = computed_by


if __name__ == "__main__":
    # TODO: Add proper set intersection interface
    gf = GF(11)
    
    # TODO: Consider immediately creating a bitset (container) using bitset params/set params
    party_bitsets = []
    for party_id in range(5):
        bits = [ReducedBooleanInput(f"b{party_id}_{i}", gf) for i in range(10)]
        bitset = BitSetContainer(bits)
        party_bitsets.append(bitset)

    intersection = BitSet.intersection(*party_bitsets)

    circuit = Circuit([intersection.contains_element(element) for element in [1, 4, 5, 9]])
    circuit.to_pdf("debug.pdf")

    arithmetic_circuit = circuit.arithmetize()
    arithmetic_circuit.to_pdf("debug2.pdf")
