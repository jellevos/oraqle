import subprocess
from typing import Any, List, Optional, Sequence, Set

from galois import GF

from oraqle.add_chains.solving import solve
from oraqle.compiler.boolean.bool import BooleanInput, ReducedBooleanInput, _cast_to
from oraqle.compiler.circuit import Circuit, ExtendedArithmeticCircuit
from oraqle.compiler.graphviz import DotFile
from oraqle.compiler.nodes.abstract import ArithmeticCosts, ExtendedArithmeticNode, Node, ExtendedArithmeticCosts
from oraqle.compiler.nodes.binary_arithmetic import Addition, Multiplication
from oraqle.compiler.nodes.extended import KnownRandom
from oraqle.compiler.nodes.leafs import Constant, Input
from oraqle.compiler.nodes.unary_arithmetic import ConstantAddition, ConstantMultiplication
from oraqle.compiler.sets.bitset import BitSet, BitSetContainer
from oraqle.mpc.parties import PartyId

from pysat.formula import WCNF, IDPool
from pysat.card import EncType
from pysat.examples.fm import FM

import time


class LeaderCosts(ExtendedArithmeticCosts):

    def __init__(self, arithmetic_costs: ArithmeticCosts, receive: float, send: float) -> None:
        super().__init__(arithmetic_costs)
        self._receive = receive
        self._send = send
    
    def receive(self, from_party: PartyId) -> float:
        assert from_party != 0
        return self._receive
    
    def send(self, to_party: PartyId) -> float:
        assert to_party != 0
        return self._send


class NonLeaderCosts(ExtendedArithmeticCosts):

    def __init__(self, arithmetic_costs: ArithmeticCosts, receive: float, send: float) -> None:
        super().__init__(arithmetic_costs)
        self._receive = receive
        self._send = send
    
    def receive(self, from_party: PartyId) -> float:
        if from_party == 0:
            return self._receive
        
        return float('inf')
    
    def send(self, to_party: PartyId) -> float:
        if to_party == 0:
            return self._send
        
        return float('inf')


def create_star_topology_costs(leader_arithmetic_costs: ArithmeticCosts, other_arithmetic_costs: ArithmeticCosts, leader_send: float, leader_receive: float, other_send: float, other_receive: float, party_count: int) -> List[ExtendedArithmeticCosts]:
    parties: List[ExtendedArithmeticCosts] = [LeaderCosts(leader_arithmetic_costs, leader_send, leader_receive)]

    for _ in range(party_count - 1):
        parties.append(NonLeaderCosts(other_arithmetic_costs, other_receive, other_send))

    return parties


def minimize_total_protocol_cost(circuit: Circuit, supported_multiplications: int, precomputed_randomness: bool, max_colluders: int, costs: Sequence[ExtendedArithmeticCosts], at_most_1_enc: Optional[int]):
    # FIXME: Add collusion threshold to signature

    extended_arithmetic_circuit = circuit.arithmetize_extended()

    assert supported_multiplications == 0
    assert not precomputed_randomness
    party_count = len(costs)
    assert max_colluders + 1 == party_count

    # Replace unknown randomness with known randomness
    processed_circuit = extended_arithmetic_circuit.replace_randomness(party_count)
    processed_circuit.to_pdf("processed.pdf")

    # Let each node add their own clauses to the formulation
    wcnf = WCNF()
    id_pool = IDPool()
    processed_circuit._add_constraints_minimize_cost_formulation(wcnf, id_pool, costs, party_count, at_most_1_enc)
    # TODO: We now assume that party 0 must learn the final results
    for output in processed_circuit._outputs:
        party_zero = 0
        h = id_pool.id(("h", id(output), party_zero))
        wcnf.append([h])

    # solver = FM(wcnf, solver='glucose42')
    # assert solver.compute()
    # result = solver.model

    result = solve(wcnf, "glucose42", None, minz=True)
    #result = solve(wcnf, "cadical195", None)
    assert result is not None

    # TODO: Assert that all inputs are known by at least one person

    # TODO: For now, let's just immediately encrypt for who knows it. Assert there is only one who knows it. So known_by => held_by

    processed_circuit.to_clustered_graph("test.dot", party_count, result, id_pool)
    subprocess.run(["dot", "-Tpdf", "test.dot", "-o", "test.pdf"], check=True)
    processed_circuit._clear_cache()

    # TODO: The backend should be threshold EC-ElGamal


def to_subscript(n):
    subscript_map = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
    return str(n).translate(subscript_map)
    

# FIXME: For visualizing these circuits, we want to make sure that the ranks in a cluster match globally.
# ... we can do so by adding invisible nodes and edges in a cluster, so that arrows do not go up.
# ... The easiest way is to duplicate all nodes across all parties, and make them invisible (and the edges) if they are not computed.
# ... We should then also draw an edge from the right source if the element was sent.
# TODO: Visualize how messages can be routed through other parties, e.g. using a point node
