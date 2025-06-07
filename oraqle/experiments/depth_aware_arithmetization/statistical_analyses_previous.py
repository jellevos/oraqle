"""Depth-aware arithmetization of a comparison modulo 101."""

import math
import os
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

    p = 6143
    gf = GF(p)

    distr_mean = 6
    distr_variance = 2

    entry_count = 512
    inputs = [Input(f"x{i}", gf) for i in range(entry_count)]
    total = sum_(*inputs)
    mean = IliashenkoZuccaDivideBy(total, entry_count)

    const_mean = distr_mean
    print(const_mean)
    sum_of_squares = sum_(*((inputs[i] - const_mean)**2 for i in range(entry_count))) 
    variance = IliashenkoZuccaDivideBy(sum_of_squares, entry_count)

    print("--- mean ---")
    start = time.monotonic()
    circuit_mean = Circuit(outputs=[mean])
    arithmetic_circuit = circuit_mean.arithmetize("best-effort")
    print("Arith:", time.monotonic() - start)
    start2 = time.monotonic()
    print("Properties:", arithmetic_circuit.multiplicative_depth(), arithmetic_circuit.multiplicative_cost(cost_of_squaring))

    cd = os.getcwd()

    folder = "mean_IZ_helib"
    os.makedirs(folder, exist_ok=True)
    os.chdir(folder)
    params = arithmetic_circuit.generate_code_chunked("main.cpp", "split", measure_time=True, decrypt_outputs=True, iterations=10)

    os.chdir(cd)

    print("--- variance ---")
    start = time.monotonic()
    circuit_var = Circuit(outputs=[variance])
    arithmetic_circuit = circuit_var.arithmetize("best-effort")
    print("Arith:", time.monotonic() - start)
    start2 = time.monotonic()
    print("Properties:", arithmetic_circuit.multiplicative_depth(), arithmetic_circuit.multiplicative_cost(cost_of_squaring))

    cd = os.getcwd()

    folder = "var_IZ_helib"
    os.makedirs(folder, exist_ok=True)
    os.chdir(folder)
    params = arithmetic_circuit.generate_code_chunked("main.cpp", "split", measure_time=True, decrypt_outputs=True, iterations=10)

    os.chdir(cd)
