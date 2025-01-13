from typing import Set

from galois import GF

from oraqle.compiler.boolean.bool import BooleanInput, ReducedBooleanInput, _cast_to
from oraqle.compiler.circuit import Circuit
from oraqle.compiler.nodes.abstract import Node
from oraqle.compiler.sets.bitset import BitSet, BitSetContainer
from oraqle.mpc.parties import PartyId


# FIXME: all the inputs must also be mpc nodes...
class MpcNode(Node):
    
    _known_by: Set[PartyId]
    _leakable_to: Set[PartyId]
    _computed_by: Set[PartyId]
    
    # def __init__(self, node: Node, known_by: Set[PartyId], leakable_to: Set[PartyId], computed_by: Set[PartyId]):
    #     self._node = node
    #     self._known_by = known_by
    #     self._leakable_to = leakable_to  # TODO: Leakable to should always be a superset of known_by
    #     self._computed_by = computed_by  # TODO: This is an inconvient interface


def to_mpc(node: Node, known_by: Set[PartyId], leakable_to: Set[PartyId], computed_by: Set[PartyId]) -> MpcNode:
    result = _cast_to(node, MpcNode)

    result._known_by = known_by
    result._leakable_to = leakable_to
    result._computed_by = computed_by

    return result


if __name__ == "__main__":
    # TODO: Add proper set intersection interface
    gf = GF(11)
    
    # TODO: Consider immediately creating a bitset (container) using bitset params/set params
    party_bitsets = []
    for party_id in range(5):
        #bits = [to_mpc(ReducedBooleanInput(f"b{party_id}_{i}", gf), {PartyId(party_id)}, {PartyId(party_id)}, {PartyId(i) for i in range(5)}) for i in range(10)]
        bits = [BooleanInput(f"b{party_id}_{i}", gf) for i in range(10)]
        bitset = BitSetContainer(bits)
        party_bitsets.append(bitset)

    intersection = BitSet.intersection(*party_bitsets)

    circuit = Circuit([intersection.contains_element(element) for element in [1, 4, 5, 9]])  # TODO: Currently we output to party 1
    circuit.to_pdf("debug.pdf")

    arithmetic_circuit = circuit.arithmetize()
    arithmetic_circuit.to_pdf("debug2.pdf")

    extended_arithmetic_circuit = circuit.arithmetize_extended()
    extended_arithmetic_circuit.to_pdf("debug3.pdf")

    # TODO: After generating extended arithmetic circuits, schedule & assign
