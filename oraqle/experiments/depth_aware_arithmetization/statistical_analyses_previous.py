"""Depth-aware arithmetization of a comparison modulo 101."""

import math
import random
import sys
import time
from galois import GF

from oraqle.compiler.circuit import Circuit
from oraqle.compiler.division.const_divisor import DivideBy, IliashenkoZuccaDivideBy
from oraqle.compiler.nodes.arbitrary_arithmetic import sum_
from oraqle.compiler.nodes.leafs import Input
from oraqle.compiler.polynomials.univariate import UnivariatePoly


if __name__ == "__main__":
    sys.setrecursionlimit(100000)

    cost_of_squaring = 1.0

    p = 786433
    gf = GF(p) #GF(786433) #GF(12289)  #GF(65537) 786433

    distr_mean = 6
    distr_variance = 2

    entry_count = 35747 #71494
    entries = [random.randint(0, 10) for _ in range(entry_count)]
    inputs = [Input(f"x{i}", gf) for i in range(entry_count)]
    total = sum_(*inputs)
    mean = IliashenkoZuccaDivideBy(total, entry_count)

    const_mean = distr_mean #round(sum(entries) / entry_count)
    print(const_mean)
    sum_of_squares = sum_(*((inputs[i] - const_mean)**2 for i in range(entry_count))) 
    variance = IliashenkoZuccaDivideBy(sum_of_squares, entry_count)

    print("--- mean ---")
    start = time.monotonic()
    circuit_mean = Circuit(outputs=[mean])
    arithmetic_circuit = circuit_mean.arithmetize("best-effort")
    print("Arith:", time.monotonic() - start)
    start2 = time.monotonic()
    print("Pre CSE", arithmetic_circuit.multiplicative_depth(), arithmetic_circuit.multiplicative_cost(cost_of_squaring))
    # arithmetic_circuit.eliminate_subexpressions()
    # print("CSE:", time.monotonic() - start2)
    # print(
    #         "post CSE",
    #         arithmetic_circuit.multiplicative_depth(),
    #         arithmetic_circuit.multiplicative_cost(cost_of_squaring),
    #     )
    params = arithmetic_circuit.generate_code("mean.cpp", measure_time=True, decrypt_outputs=True)
    params = arithmetic_circuit.generate_code_openfhe("mean_bfv.cpp", measure_time=True, decrypt_outputs=True)
    print(params)

    print("--- variance ---")
    start = time.monotonic()
    circuit_var = Circuit(outputs=[variance])
    arithmetic_circuit = circuit_var.arithmetize("best-effort")
    print("Arith:", time.monotonic() - start)
    start2 = time.monotonic()
    print("Pre CSE", arithmetic_circuit.multiplicative_depth(), arithmetic_circuit.multiplicative_cost(cost_of_squaring))
    # arithmetic_circuit.eliminate_subexpressions()
    # print("CSE:", time.monotonic() - start2)
    # print(
    #         "post CSE",
    #         arithmetic_circuit.multiplicative_depth(),
    #         arithmetic_circuit.multiplicative_cost(cost_of_squaring),
    #     )
    params = arithmetic_circuit.generate_code("variance.cpp", measure_time=True, decrypt_outputs=True)
    params = arithmetic_circuit.generate_code_openfhe("variance_bfv.cpp", measure_time=True, decrypt_outputs=True)
    print(params)

# 619.152179458004
