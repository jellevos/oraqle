import time

from galois import GF
from oraqle.compiler.boolean.bool import BooleanInput, NegReducedBooleanInput
from oraqle.compiler.circuit import Circuit
from oraqle.compiler.nodes.abstract import ArithmeticCosts
from oraqle.compiler.nodes.arbitrary_arithmetic import Product
from oraqle.compiler.nodes.binary_arithmetic import Addition, Multiplication
from oraqle.compiler.nodes.extended import UnknownRandom
from oraqle.experiments.thesis.extended_arith import Zp
from oraqle.mpc.compilation import create_star_topology_costs, minimize_total_protocol_cost, to_subscript
from pysat.card import EncType

from oraqle.mpc.parties import PartyId


if __name__ == "__main__":
    zp = Zp(2**252 + 27742317777372353535851937790883648493)
    party_count = 3

    b1 = NegReducedBooleanInput(f"b{to_subscript(1)},{to_subscript(1)}", zp, {PartyId(0)})  # type: ignore
    b2 = NegReducedBooleanInput(f"b{to_subscript(2)},{to_subscript(1)}", zp, {PartyId(1)})  # type: ignore
    b3 = NegReducedBooleanInput(f"b{to_subscript(3)},{to_subscript(1)}", zp, {PartyId(2)})  # type: ignore

    add1 = b1.add(b2, flatten=False)
    add2 = add1.add(b3, flatten=False)

    rand = UnknownRandom(zp)  # type: ignore

    mul = rand.mul(add2)

    circuit = Circuit([mul])

    addition = 1.
    other_computation_factor = 0.1
    all_communication_factor = 1000.

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

# TODO: Maak nog een file waarin we meerdere params testen en de cost, compile time en type reporten
