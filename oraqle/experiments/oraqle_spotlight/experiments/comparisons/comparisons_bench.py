import random
import subprocess

from galois import GF
from matplotlib import pyplot as plt
from sympy import sieve

from oraqle.compiler.circuit import ArithmeticCircuit, Circuit
from oraqle.compiler.comparison.comparison import SemiStrictComparison, T2SemiLessThan
from oraqle.compiler.nodes.leafs import Input


def run_benchmark(arithmetic_circuit: ArithmeticCircuit) -> float:
    # Prepare the benchmark
    arithmetic_circuit.generate_code("main.cpp", iterations=10, measure_time=True)
    subprocess.run("make", capture_output=True, check=True)

    # Run the benchmark
    command = ["./main"]
    p = arithmetic_circuit._gf.characteristic
    command.append(f"x={random.randint(0, p - 1)}")
    command.append(f"y={random.randint(0, p - 1)}")
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
    print(p, run_time)

    return run_time


if __name__ == "__main__":
    slides = True
    run_benchmarks = False
    gen_plots = True

    if run_benchmarks:
        primes = list(sieve.primerange(300))[2:20]

        our_times = []
        t2_times = []

        for p in primes:
            gf = GF(p)

            x = Input("x", gf)
            y = Input("y", gf)

            print(f"-------- p = {p}: ---------")
            our_circuit = Circuit([SemiStrictComparison(x, y, less_than=True, gf=gf)])
            our_front = our_circuit.arithmetize_depth_aware()
            print("Our circuits:", our_front)

            ts = []
            for _, _, arithmetic_circuit in our_front:
                ts.append(run_benchmark(arithmetic_circuit))
            our_times.append(tuple(ts))

            t2_circuit = Circuit([T2SemiLessThan(x, y, gf)])
            t2_arithmetization = t2_circuit.arithmetize()
            print(
                "T2 circuit:",
                t2_arithmetization.multiplicative_depth(),
                t2_arithmetization.multiplicative_size(),
            )

            t2_times.append(run_benchmark(t2_arithmetization))

        print(primes)
        print(our_times)
        print(t2_times)

    if gen_plots:
        primes = [5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71]
        our_times = [(0.0156603,), (0.0523416,), (0.0954489,), (0.0936497,), (0.111959,), (0.128402,), (0.288951,), (0.42076, 0.368583), (0.416362,), (0.40343,), (0.385652,), (0.437486,), (0.481356,), (0.522607, 0.504944), (0.526451,), (0.5904119999999999, 0.5146740000000001), (0.592896,), (0.621265, 0.598357)]
        t2_times = [0.0156379, 0.0938689, 0.23473899999999998, 0.319668, 0.366707, 0.6632450000000001, 1.8380299999999998, 1.14859, 2.9022200000000002, 3.2060299999999997, 3.5419899999999997, 4.53918, 5.02624, 5.4439, 8.64118, 6.6267499999999995, 6.99609, 9.21295]

        if slides:
            plt.figure(figsize=(7, 4))
        else:
            plt.figure(figsize=(4, 2))
        plt.grid(axis="y", zorder=-1000, alpha=0.5)

        plt.scatter(
            range(len(primes)), t2_times, marker="_", label="T2's Circuit", color="tab:orange", s=100 if slides else None
        )

        for x, ts in enumerate(our_times):
            for t in ts:
                plt.scatter(
                    x,
                    t,
                    marker="_",
                    label="Oraqle's circuits" if x == 0 else None,
                    color="tab:cyan",
                    s=100 if slides else None
                )

        plt.xticks(range(len(primes)), primes, fontsize=8)  # type: ignore

        plt.xlabel("Modulus")
        plt.ylabel("Run time (s)")

        plt.legend()

        plt.savefig(f"t2_comparison{'_slides' if slides else ''}.pdf", bbox_inches="tight")
        plt.show()
