import random
import subprocess
import time

from galois import GF

from oraqle.circuits.cardio import (
    construct_cardio_elevated_risk_circuit,
    construct_cardio_risk_circuit,
)
from oraqle.compiler.circuit import ArithmeticCircuit, Circuit


def run_benchmark(arithmetic_circuit: ArithmeticCircuit) -> float:
    # Prepare the benchmark
    arithmetic_circuit.generate_code("main.cpp", iterations=10, measure_time=True)
    subprocess.run("make", capture_output=True, check=True)

    # Run the benchmark
    command = ["./main"]
    command.append(f"man={random.randint(0, 1)}")
    command.append(f"smoking={random.randint(0, 1)}")
    command.append(f"diabetic={random.randint(0, 1)}")
    command.append(f"hbp={random.randint(0, 1)}")
    command.append(f"cholesterol={random.randint(0, 1)}")

    command.append(f"age={random.randint(0, 100)}")
    command.append(f"cholesterol={random.randint(0, 60)}")
    command.append(f"weight={random.randint(40, 150)}")
    command.append(f"height={random.randint(80, 210)}")
    command.append(f"activity={random.randint(0, 250)}")
    command.append(f"alcohol={random.randint(0, 5)}")
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

    return run_time


if __name__ == "__main__":
    gf = GF(257)

    for cost_of_squaring in [0.75]:
        print(f"--- Cardio risk assessment ({cost_of_squaring}) ---")
        circuit = Circuit([construct_cardio_risk_circuit(gf)])

        start = time.monotonic()
        front = circuit.arithmetize_depth_aware(cost_of_squaring=cost_of_squaring)
        print("Compile time:", time.monotonic() - start, "s")

        for depth, cost, arithmetic_circuit in front:
            print(depth, cost)
            arithmetic_circuit.to_graph(f"cardio_arith_d{depth}_c{cost}.dot")
            print("Run time:", run_benchmark(arithmetic_circuit))

        print(f"--- Cardio elevated risk assessment ({cost_of_squaring}) ---")
        circuit = Circuit([construct_cardio_elevated_risk_circuit(gf)])

        start = time.monotonic()
        front = circuit.arithmetize_depth_aware(cost_of_squaring=cost_of_squaring)
        print("Compile time:", time.monotonic() - start, "s")

        for depth, cost, arithmetic_circuit in front:
            print(depth, cost)
            arithmetic_circuit.to_graph(f"cardio_elevated_arith_d{depth}_c{cost}.dot")
            print("Run time:", run_benchmark(arithmetic_circuit))
