from typing import Set

from oraqle.compiler.nodes.abstract import Node
from oraqle.mpc.parties import PartyId


# FIXME: all the inputs must also be mpc nodes...
class MpcNode:
    
    def __init__(self, node: Node, known_by: Set[PartyId], leakable_to: Set[PartyId], computed_by: Set[PartyId]):
        self._node = node
        self._known_by = known_by
        self._leakable_to = leakable_to
        self._computed_by = computed_by



if __name__ == "__main__":
    # gf = GF(11)
    # bits = [Input(f"b_{i}", gf) for i in range(10)]
    # circuit = Circuit([BitSet(bits, gf).contains_element(3)]).to_pdf("debug.pdf")

    # TODO: Encode bitset, whose inputs are known by party 1
    # TODO: Encode bitset, whose inputs are known by party 2
    # TODO: Intersect bitsets, computed by both 1 and 2
    # TODO: Query bitset on inputs known by party 1
    pass
