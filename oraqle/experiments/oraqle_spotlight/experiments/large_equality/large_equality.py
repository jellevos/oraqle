import math
import random
import subprocess
import time
from typing import List, Tuple

from galois import GF
from sympy import sieve

from oraqle.compiler.boolean.bool_and import all_
from oraqle.compiler.circuit import ArithmeticCircuit, Circuit
from oraqle.compiler.nodes.leafs import Input


def generate_circuits(bits: int) -> List[Tuple[int, ArithmeticCircuit, int, float]]:
    circuits = []

    primes = list(sieve.primerange(300))[:10]  # [:55]  # p <= 257
    start = time.monotonic()
    times = []
    for p in primes:
        # (6, 63.0): p=2
        # (7, 58.0): p=5
        # (8, 51.0): p=17

        limbs = math.ceil(bits / math.log2(p))

        gf = GF(p)

        xs = [Input(f"x{i}", gf) for i in range(limbs)]
        ys = [Input(f"y{i}", gf) for i in range(limbs)]
        circuit = Circuit([all_(*(xs[i] == ys[i] for i in range(limbs)))])

        inbetween = time.monotonic()
        front = circuit.arithmetize_depth_aware(0.75)

        print(f"{p}.", end=" ")

        for f in front:
            circuits.append((p, f[2], f[0], f[1]))
            print(f[0], f[1], end="   ")

        inbetween_time = time.monotonic() - inbetween
        print(inbetween_time)
        times.append((p, inbetween_time))

    print(times)
    print("Total time", time.monotonic() - start)

    return circuits


if __name__ == "__main__":
    bits = 64
    benchmark_circuits = False
    generate_table = True

    # Run a benchmark for all circuits in the front
    if benchmark_circuits:
        # Generate all circuits per p
        circuits = generate_circuits(bits)

        results = []
        for p, arithmetic_circuit, d, c in circuits:
            # Prepare the benchmark
            params = arithmetic_circuit.generate_code("main.cpp", iterations=10, measure_time=True)
            subprocess.run("make", check=True)

            # Run the benchmark
            command = ["./main"]
            limbs = math.ceil(bits / math.log2(p))
            for i in range(limbs):
                command.append(f"x{i}={random.randint(0, p - 1)}")
                command.append(f"y{i}={random.randint(0, p - 1)}")
            print("Running:", " ".join(command))
            result = subprocess.run(command, capture_output=True, text=True, check=False)

            if result.returncode != 0:
                print("stderr:")
                print(result.stderr)
                print()
                print("stdout:")
                print(result.stdout)

            # Check if the noise was not too large
            print(result.stdout)
            lines = result.stdout.splitlines()
            for line in lines[:-1]:
                assert line.endswith("1")

            run_time = float(lines[-1]) / 10
            print(p, run_time, d, c, params)
            results.append((p, d, c, params, run_time))

        print(results)

    if generate_table:
        gen_times = [(2, 0.007554411888122559), (3, 0.06264467351138592), (5, 8.457202550023794), (7, 0.05447225831449032), (11, 0.0478445328772068), (13, 0.052152080461382866), (17, 0.04349260404706001), (19, 0.04553743451833725), (23, 0.05198719538748264), (29, 0.046183058992028236)]
        results = [(2, 6, 63.0, (16383, 1, 142, 3), 3.27577), (3, 7, 60.75, (32768, 1, 170, 3), 1.51993), (5, 7, 58.0, (32768, 1, 178, 3), 1.7679099999999999), (5, 8, 55.5, (32768, 1, 197, 3), 1.93994), (7, 8, 74.0, (32768, 1, 206, 3), 2.90913), (7, 9, 70.0, (32768, 1, 226, 3), 2.6624600000000003), (7, 10, 69.5, (32768, 1, 246, 3), 3.00814), (11, 9, 69.25, (32768, 1, 228, 3), 2.50603), (11, 12, 68.25, (32768, 1, 300, 3), 3.25469), (13, 9, 68.75, (32768, 1, 237, 3), 2.67845), (13, 10, 67.75, (32768, 1, 237, 3), 2.7718), (13, 11, 66.0, (32768, 1, 237, 3), 2.56386), (13, 12, 65.0, (32768, 1, 301, 3), 3.10959), (17, 8, 51.0, (32768, 1, 217, 3), 1.8792300000000002), (19, 9, 79.0, (32768, 1, 238, 3), 2.85011), (19, 10, 68.0, (32768, 1, 259, 3), 2.8636500000000003), (23, 9, 89.0, (32768, 1, 248, 3), 4.135730000000001), (23, 10, 80.0, (32768, 1, 270, 3), 3.75128), (29, 9, 83.0, (32768, 1, 249, 3), 3.7119), (29, 10, 75.0, (32768, 1, 271, 3), 3.46666)]

        gen_times = {p: t for p, t in gen_times}

        for p, d, c, params, run_time in results:
            print(f"{p} & {d} & {c} & {params[0]} & {params[1]} & {params[2]} & {params[3]} & {round(gen_times[p], 2)} & {round(run_time, 2)} \\\\")
