"""Depth-aware arithmetization of a comparison modulo 101."""

import sys
from galois import GF

from oraqle.compiler.circuit import Circuit
from oraqle.compiler.nodes.leafs import Input

import mpmath


if __name__ == "__main__":
    mpmath.mp.dps = 50
    mpmath.mp.pretty = True
    mpmath.mp.maxsteps = 1000

    sys.setrecursionlimit(10000)

    gf = GF(786433) #GF(12289)  #GF(65537) 786433
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

    _, _, ac = arithmetic_circuits[0]
    params = ac.generate_code("test_code2.cpp")
    print(params)


# pre CSE 24 2872.0
# post CSE 24 2770.0
