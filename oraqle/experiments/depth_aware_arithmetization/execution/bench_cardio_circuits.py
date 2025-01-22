import random
import subprocess
import time
from typing import Dict

from galois import GF

from oraqle.circuits.cardio import (
    construct_cardio_elevated_risk_circuit,
    construct_cardio_risk_circuit,
)
from oraqle.compiler.circuit import ArithmeticCircuit, Circuit


def gen_params() -> Dict[str, int]:
    params = {}

    params["man"] = random.randint(0, 1)
    params["smoking"] = random.randint(0, 1)
    params["diabetic"] = random.randint(0, 1)
    params["hbp"] = random.randint(0, 1)

    params["age"] = random.randint(0, 100)
    params["cholesterol"] = random.randint(0, 60)
    params["weight"] = random.randint(40, 150)
    params["height"] = random.randint(80, 210)
    params["activity"] = random.randint(0, 250)
    params["alcohol"] = random.randint(0, 5)

    return params


if __name__ == "__main__":
    gf = GF(257)
    iterations = 10

    for cost_of_squaring in [0.75]:
        print(f"--- Cardio risk assessment ({cost_of_squaring}) ---")
        circuit = Circuit([construct_cardio_risk_circuit(gf)])

        start = time.monotonic()
        front = circuit.arithmetize_depth_aware(cost_of_squaring=cost_of_squaring)
        print("Compile time:", time.monotonic() - start, "s")

        for depth, cost, arithmetic_circuit in front:
            print(depth, cost)
            run_time = arithmetic_circuit.run_using_helib(iterations, True, False, *gen_params())
            print("Run time:", run_time)

        print(f"--- Cardio elevated risk assessment ({cost_of_squaring}) ---")
        circuit = Circuit([construct_cardio_elevated_risk_circuit(gf)])

        start = time.monotonic()
        front = circuit.arithmetize_depth_aware(cost_of_squaring=cost_of_squaring)
        print("Compile time:", time.monotonic() - start, "s")

        for depth, cost, arithmetic_circuit in front:
            print(depth, cost)
            run_time = arithmetic_circuit.run_using_helib(iterations, True, False, *gen_params())
            print("Run time:", run_time)
