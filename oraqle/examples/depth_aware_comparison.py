"""Depth-aware arithmetization of a comparison modulo 101."""

from galois import GF

from oraqle.compiler.circuit import Circuit
from oraqle.compiler.nodes.leafs import Input

gf = GF(101)
cost_of_squaring = 1.0

a = Input("a", gf)
b = Input("b", gf)

output = a < b

circuit = Circuit(outputs=[output])
circuit.to_graph("high_level_circuit.dot")

arithmetic_circuits = circuit.arithmetize_depth_aware(cost_of_squaring)

for depth, cost, arithmetic_circuit in arithmetic_circuits:
    assert arithmetic_circuit.multiplicative_depth() == depth
    assert arithmetic_circuit.multiplicative_cost(cost_of_squaring) == cost

    print("pre CSE", depth, cost)

    arithmetic_circuit.eliminate_subexpressions()

    print(
        "post CSE",
        arithmetic_circuit.multiplicative_depth(),
        arithmetic_circuit.multiplicative_cost(cost_of_squaring),
    )
