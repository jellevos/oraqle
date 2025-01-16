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

    result = solve(wcnf, "glucose42", None)
    #result = solve(wcnf, "cadical195", None)
    print(result)
    assert result is not None

    # TODO: Assert that all inputs are known by at least one person

    # TODO: For now, let's just immediately encrypt for who knows it. Assert there is only one who knows it. So known_by => held_by

    processed_circuit.to_clustered_graph("test.dot", party_count, result, id_pool)
    subprocess.run(["dot", "-Tpdf", "test.dot", "-o", "test.pdf"], check=True)
    processed_circuit._clear_cache()

    # TODO: The backend should be threshold EC-ElGamal


if __name__ == "__main__":
    # party_count = 3
    # gf = GF(11)

    # # a = Input("a", gf, {PartyId(0)})
    # # a_neg = (a * Constant(gf(10))) + 1

    # b = Input("b", gf, {PartyId(1)})
    # b_neg = (b * Constant(gf(10))) + 1

    # c = Input("c", gf, {PartyId(2)})
    # c_neg = (c * KnownRandom(gf, {PartyId(0)})) + 1

    # circuit = Circuit([(b_neg + c_neg) * KnownRandom(gf, {PartyId(1)})])
    # circuit.to_pdf("simple.pdf")
    # circuit = circuit.arithmetize_extended()
    # circuit.to_pdf("simple-arith.pdf")
    
    # leader_arithmetic_costs = ArithmeticCosts(1., float('inf'), 1., 100.)
    # other_arithmetic_costs = leader_arithmetic_costs * 10.
    # print(leader_arithmetic_costs)
    # print(other_arithmetic_costs)
    # all_costs = create_star_topology_costs(leader_arithmetic_costs, other_arithmetic_costs, 1000., 1000., 2000., 2000., party_count)
    # minimize_total_protocol_cost(circuit, 0, False, party_count - 1, all_costs)



    # exit(0)
    # TODO: Add proper set intersection interface
    gf = GF(11)
    
    # TODO: Consider immediately creating a bitset (container) using bitset params/set params
    party_count = 3
    party_bitsets = []
    for party_id in range(party_count):
        #bits = [to_mpc(ReducedBooleanInput(f"b{party_id}_{i}", gf), {PartyId(party_id)}, {PartyId(party_id)}, {PartyId(i) for i in range(5)}) for i in range(10)]
        bits = [BooleanInput(f"b{party_id}_{i}", gf, {PartyId(party_id)}) for i in range(10)]
        bitset = BitSetContainer(bits)
        party_bitsets.append(bitset)

    intersection = BitSet.intersection(*party_bitsets)

    circuit = Circuit([intersection.contains_element(element) for element in [1]])#, 4, 5, 9]])  # TODO: Currently we output to party 1
    circuit.to_pdf("debug.pdf")

    arithmetic_circuit = circuit.arithmetize()
    arithmetic_circuit.to_pdf("debug2.pdf")

    extended_arithmetic_circuit = circuit.arithmetize_extended()
    extended_arithmetic_circuit.to_pdf("debug3.pdf")

    addition = 1.
    other_computation_factor = 0.25 #0.25 #0.1
    all_communication_factor = 100.

    # FIXME: What to do with the cost of a scalar mul!
    leader_arithmetic_costs = ArithmeticCosts(addition, float('inf'), addition, 100.)
    other_arithmetic_costs = leader_arithmetic_costs * other_computation_factor
    print(leader_arithmetic_costs)
    print(other_arithmetic_costs)

    communication_cost = addition * all_communication_factor
    all_costs = create_star_topology_costs(leader_arithmetic_costs, other_arithmetic_costs, communication_cost, communication_cost, communication_cost, communication_cost, party_count)

    t = time.monotonic()
    minimize_total_protocol_cost(circuit, 0, False, party_count - 1, all_costs, at_most_1_enc=EncType.ladder)
    print(time.monotonic() - t)

# We kunnen denk ik 2^k doen voor k=-2,-1,0,1,2,3,4 ofzo voor hoeveel efficienter de leader is (kan ook vanaf -2 omdat we soms juist dingen door de user willen laten doen)
# En dan 2^k k=-2,-1,-0,1,2 voor hoeveel efficienter de leader communiceert [DEZE ZOU IK NIET DOEN]
# Ik zou ipv dat doen dat communicatie 10^k keer duurder is dan een addition, dan heb je k=-1,0,1,2,3
# En dan meten we de tijd dat het duurt om te solven en welk protocol eruit komt
