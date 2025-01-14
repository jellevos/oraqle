import subprocess
from typing import List, Set

from galois import GF

from oraqle.add_chains.solving import solve
from oraqle.compiler.boolean.bool import BooleanInput, ReducedBooleanInput, _cast_to
from oraqle.compiler.circuit import Circuit, ExtendedArithmeticCircuit
from oraqle.compiler.graphviz import DotFile
from oraqle.compiler.nodes.abstract import ExtendedArithmeticNode, Node, SecureComputationCosts
from oraqle.compiler.nodes.binary_arithmetic import Addition, Multiplication
from oraqle.compiler.nodes.unary_arithmetic import ConstantAddition, ConstantMultiplication
from oraqle.compiler.sets.bitset import BitSet, BitSetContainer
from oraqle.mpc.parties import PartyId

from pysat.formula import WCNF, IDPool


class StarTopologyCosts(SecureComputationCosts):

    def __init__(self, addition: float, multiplication: float, receive_leader: float, receive_other: float, send_leader: float, send_other: float) -> None:
        super().__init__(addition, multiplication, addition / 10, multiplication / 10)  # FIXME: This is incorrect!! Also input constant_add and constant_mul
        self._receive_leader = receive_leader
        self._receive_other = receive_other
        self._send_leader = send_leader
        self._send_other = send_other
    
    def receive(self, from_party: PartyId) -> float:
        if from_party == 0:
            return self._receive_leader
        
        return self._receive_other
    
    def send(self, to_party: PartyId) -> float:
        if to_party == 0:
            return self._send_leader
        
        return self._send_other


def create_star_topology_costs(leader_add: float, leader_mul: float, leader_send: float, leader_receive: float, other_factor: float, other_com_factor: float, mesh_cost: float, party_count: int) -> List[StarTopologyCosts]:
    parties = [StarTopologyCosts(leader_add, leader_mul, leader_send, leader_receive, leader_send, leader_receive)]

    for _ in range(party_count - 1):
        parties.append(StarTopologyCosts(leader_add * other_factor, leader_mul * other_factor, leader_receive * other_com_factor, mesh_cost, leader_send * other_com_factor, mesh_cost))

    return parties


def minimize_total_protocol_cost(circuit: Circuit, supported_multiplications: int, precomputed_randomness: bool, max_colluders: int, costs: List[SecureComputationCosts]):
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
    processed_circuit._add_constraints_minimize_cost_formulation(wcnf, id_pool, costs, party_count)
    # TODO: We now assume that party 0 must learn the final results
    for output in processed_circuit._outputs:
        party_zero = 0
        h = id_pool.id(("h", id(output), party_zero))
        wcnf.append([h])
    result = solve(wcnf, "glucose42", None)
    print(result)
    assert result is not None

    # TODO: Assert that all inputs are known by at least one person

    # TODO: For now, let's just immediately encrypt for who knows it. Assert there is only one who knows it. So known_by => held_by

    graph_builder = DotFile()
    for output in processed_circuit._outputs:
        for party_id in range(party_count):
            def assign_to_cluster(node: ExtendedArithmeticNode):
                # TODO: Do we need a cache?
                node_id = node.to_graph(graph_builder)
                
                if isinstance(node, (Addition, Multiplication, ConstantAddition, ConstantMultiplication)):
                    c = id_pool.id(("c", id(node), party_id))
                    if result[c] > 0:
                        graph_builder.add_node_to_cluster(node_id, party_id)
                    # h = id_pool.id(("h", id(node), party_id))
                    # if result[h] > 0:
                    #     graph_builder.add_node_to_cluster(node_id, party_id)
                else:
                    if PartyId(party_id) in node._known_by:
                        graph_builder.add_node_to_cluster(node_id, party_id)

                node.apply_function_to_operands(assign_to_cluster)  # type: ignore

            output.apply_function_to_operands(assign_to_cluster)  # type: ignore

        graph_builder.add_link(
            output.to_graph(graph_builder),
            graph_builder.add_node(label="Output", shape="plain"),
        )
    
    graph_builder.to_file("test.dot")
    subprocess.run(["dot", "-Tpdf", "test.dot", "-o", "test.pdf"], check=True)
    processed_circuit._clear_cache()


if __name__ == "__main__":
    # TODO: Add proper set intersection interface
    gf = GF(11)
    
    # TODO: Consider immediately creating a bitset (container) using bitset params/set params
    party_bitsets = []
    for party_id in range(5):
        #bits = [to_mpc(ReducedBooleanInput(f"b{party_id}_{i}", gf), {PartyId(party_id)}, {PartyId(party_id)}, {PartyId(i) for i in range(5)}) for i in range(10)]
        bits = [BooleanInput(f"b{party_id}_{i}", gf, {PartyId(i)}) for i in range(10)]
        bitset = BitSetContainer(bits)
        party_bitsets.append(bitset)

    intersection = BitSet.intersection(*party_bitsets)

    circuit = Circuit([intersection.contains_element(element) for element in [1, 4, 5, 9]])  # TODO: Currently we output to party 1
    circuit.to_pdf("debug.pdf")

    arithmetic_circuit = circuit.arithmetize()
    arithmetic_circuit.to_pdf("debug2.pdf")

    extended_arithmetic_circuit = circuit.arithmetize_extended()
    extended_arithmetic_circuit.to_pdf("debug3.pdf")

    minimize_total_protocol_cost(circuit, 0, False, 4, create_star_topology_costs(0.01, 1.00, 10., 10., 100., 1., 100000., 5))
