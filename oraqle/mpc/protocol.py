from typing import List
from oraqle.compiler.nodes.abstract import Node
from oraqle.mpc.parties import PartyId


class Protocol:

    def __init__(self, party_count) -> None:
        self._operations: List[List[Node]] = [[]] * party_count

    def assign_operation(self, to_party: PartyId, node: Node):
        self._operations[to_party - 1].append(node)
