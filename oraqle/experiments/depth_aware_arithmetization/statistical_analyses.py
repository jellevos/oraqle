"""Depth-aware arithmetization of a comparison modulo 101."""

import math
import os
import random
import sys
import time
from galois import GF

from oraqle.compiler.circuit import Circuit
from oraqle.compiler.division.const_divisor import DivideBy
from oraqle.compiler.nodes.arbitrary_arithmetic import sum_
from oraqle.compiler.nodes.leafs import Input
from oraqle.compiler.polynomials.univariate import UnivariatePoly


if __name__ == "__main__":
    sys.setrecursionlimit(1000000)

    cost_of_squaring = 1.0

    p = 786433
    gf = GF(p) #GF(786433) #GF(12289)  #GF(65537) 786433

    distr_mean = 6
    distr_variance = 2

    entry_count = 35747 #71494
    entries = [random.randint(0, 10) for _ in range(entry_count)]
    inputs = [Input(f"x{i}", gf) for i in range(entry_count)]
    total = sum_(*inputs)
    mean = DivideBy(total, entry_count)

    const_mean = distr_mean #round(sum(entries) / entry_count)
    print(const_mean)
    sum_of_squares = sum_(*((inputs[i] - const_mean)**2 for i in range(entry_count))) 
    variance = DivideBy(sum_of_squares, entry_count)

    print("--- mean ---")
    start = time.monotonic()
    circuit_mean = Circuit(outputs=[mean])
    arithmetic_circuits = circuit_mean.arithmetize_depth_aware(cost_of_squaring)
    print("Arith:", time.monotonic() - start)
    start2 = time.monotonic()
    d, c, ac = arithmetic_circuits[0]
    print("Pre CSE", d, c)
    # ac.eliminate_subexpressions()
    # print("CSE:", time.monotonic() - start2)
    # print(
    #         "post CSE",
    #         ac.multiplicative_depth(),
    #         ac.multiplicative_cost(cost_of_squaring),
    #     )

    cd = os.getcwd()

    folder = "mean_ours_helib"
    os.makedirs(folder, exist_ok=True)
    os.chdir(folder)
    params = ac.generate_code_chunked("main.cpp", "split", measure_time=True, decrypt_outputs=True)

    os.chdir(cd)

    folder = "mean_ours_openfhe"
    os.makedirs(folder, exist_ok=True)
    os.chdir(folder)
    params = ac.generate_code_chunked_openfhe("main.cpp", "split", measure_time=True, decrypt_outputs=True)
    print(params)
    os.chdir(cd)

    print("--- variance ---")
    start = time.monotonic()
    circuit_var = Circuit(outputs=[variance])
    arithmetic_circuits = circuit_var.arithmetize_depth_aware(cost_of_squaring)
    print("Arith:", time.monotonic() - start)
    start2 = time.monotonic()
    d, c, ac = arithmetic_circuits[0]
    print("Pre CSE", d, c)
    # ac.eliminate_subexpressions()
    # print("CSE:", time.monotonic() - start2)
    # print(
    #         "post CSE",
    #         ac.multiplicative_depth(),
    #         ac.multiplicative_cost(cost_of_squaring),
    #     )

    cd = os.getcwd()

    folder = "var_ours_helib"
    os.makedirs(folder, exist_ok=True)
    os.chdir(folder)
    params = ac.generate_code_chunked("main.cpp", "split", measure_time=True, decrypt_outputs=True)

    os.chdir(cd)

    folder = "var_ours_openfhe"
    os.makedirs(folder, exist_ok=True)
    os.chdir(folder)
    params = ac.generate_code_chunked_openfhe("main.cpp", "split", measure_time=True, decrypt_outputs=True)
    print(params)
    os.chdir(cd)
