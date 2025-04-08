import random
import time
from typing import Dict

from galois import GF

from oraqle.circuits.cardio import (
    construct_cardio_elevated_risk_circuit,
    construct_cardio_risk_circuit,
)
from oraqle.compiler.circuit import Circuit


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

    for cost_of_squaring in [0.5, 0.75, 1.0]:
        print(f"--- Cardio risk assessment ({cost_of_squaring}) ---")
        circuit = Circuit([construct_cardio_risk_circuit(gf)]).to_naive()

        start = time.monotonic()
        arithmetic_circuit = circuit.arithmetize()
        print("Compile time:", time.monotonic() - start, "s")
        depth = arithmetic_circuit.multiplicative_depth()
        cost = arithmetic_circuit.multiplicative_cost(cost_of_squaring)
        print(depth, cost)
        arithmetic_circuit.to_graph(f"cardio_arith_d{depth}_c{cost}.dot")
        run_time = arithmetic_circuit.run_using_helib(iterations, True, False, **gen_params())
        print("Run time:", run_time)

        print(f"--- Cardio elevated risk assessment ({cost_of_squaring}) ---")
        circuit = Circuit([construct_cardio_elevated_risk_circuit(gf)]).to_naive()

        start = time.monotonic()
        arithmetic_circuit = circuit.arithmetize()
        print("Compile time:", time.monotonic() - start, "s")
        depth = arithmetic_circuit.multiplicative_depth()
        cost = arithmetic_circuit.multiplicative_cost(cost_of_squaring)
        print(depth, cost)
        arithmetic_circuit.to_graph(f"cardio_elevated_arith_d{depth}_c{cost}.dot")
        run_time = arithmetic_circuit.run_using_helib(iterations, True, False, **gen_params())
        print("Run time:", run_time)
