import time

from galois import GF

from oraqle.circuits.cardio import (
    construct_cardio_elevated_risk_circuit,
    construct_cardio_risk_circuit,
)
from oraqle.compiler.circuit import Circuit

if __name__ == "__main__":
    gf = GF(257)

    for cost_of_squaring in [0.5, 0.75, 1.0]:
        print(f"--- Cardio risk assessment ({cost_of_squaring}) ---")
        circuit = Circuit([construct_cardio_risk_circuit(gf)])

        start = time.monotonic()
        front = circuit.arithmetize_depth_aware(cost_of_squaring=cost_of_squaring)
        print("Run time:", time.monotonic() - start, "s")

        for depth, cost, arithmetic_circuit in front:
            print(depth, cost)
            arithmetic_circuit.to_graph(f"cardio_arith_d{depth}_c{cost}.dot")

        print(f"--- Cardio elevated risk assessment ({cost_of_squaring}) ---")
        circuit = Circuit([construct_cardio_elevated_risk_circuit(gf)])

        start = time.monotonic()
        front = circuit.arithmetize_depth_aware(cost_of_squaring=cost_of_squaring)
        print("Run time:", time.monotonic() - start, "s")

        for depth, cost, arithmetic_circuit in front:
            print(depth, cost)
            arithmetic_circuit.to_graph(f"cardio_elevated_arith_d{depth}_c{cost}.dot")
