"""Depth-aware arithmetization of a comparison modulo 101."""

import math
import sys
from galois import GF

from oraqle.compiler.circuit import Circuit
from oraqle.compiler.nodes.leafs import Input
from oraqle.compiler.polynomials.univariate import UnivariatePoly


if __name__ == "__main__":
    sys.setrecursionlimit(10000)

    gf = GF(31) #GF(786433) #GF(12289)  #GF(65537) 786433
    cost_of_squaring = 1.0

    x = Input("x", gf)


    # print("ceil")
    # for d in [2, 3, 4, 5, 6, 7, 8, 9]:
    #     output = UnivariatePoly.from_function(x, gf, lambda xx: math.ceil(xx / d))
    #     print(d, [int(coeff) for coeff in output._coefficients])
    
    # print("round")
    # for d in [2, 3, 4, 5, 6, 7, 8, 9]:
    #     output = UnivariatePoly.from_function(x, gf, lambda xx: round(xx / d))
    #     print(d, [int(coeff) for coeff in output._coefficients])
    
    # print("floor")
    # for d in [2, 3, 4, 5, 6, 7, 8, 9]:
    #     output = UnivariatePoly.from_function(x, gf, lambda xx: math.floor(xx / d))
    #     print(d, [int(coeff) for coeff in output._coefficients])


    from sklearn.datasets import load_breast_cancer
    import numpy as np

    data = load_breast_cancer()
    counts = np.bincount(data.target)


    n_benign = counts[1]
    n_malignant = counts[0]
    n = n_benign + n_malignant

    p = 786433

    print("All")
    for d in reversed(range(n + 1)):
        if (p % d) == (d - 1):
            print(d)
            break

    print("Benign")
    for d in reversed(range(n_benign + 1)):
        if (p % d) == (d - 1):
            print(d)
            break

    print("Mal")
    for d in reversed(range(n_malignant + 1)):
        if (p % d) == (d - 1):
            print(d)
            break

    print("High")
    for d in reversed(range(100000 + 1)):
        if (p % d) == (d - 1):
            print(d)
            #break


    # circuit = Circuit(outputs=[output])
    # circuit.to_graph("high_level_circuit.dot")

    # arithmetic_circuits = circuit.arithmetize_depth_aware(cost_of_squaring)

    # for depth, cost, arithmetic_circuit in arithmetic_circuits:
    #     assert arithmetic_circuit.multiplicative_depth() == depth
    #     assert arithmetic_circuit.multiplicative_cost(cost_of_squaring) == cost

    #     print("pre CSE", depth, cost)

    #     arithmetic_circuit.eliminate_subexpressions()

    #     print(
    #         "post CSE",
    #         arithmetic_circuit.multiplicative_depth(),
    #         arithmetic_circuit.multiplicative_cost(cost_of_squaring),
    #     )

    # _, _, ac = arithmetic_circuits[0]
    # params = ac.generate_code("test_code2.cpp")
    # print(params)
