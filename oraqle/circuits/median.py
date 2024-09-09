"""This module implements circuits for computing the median."""
from typing import Sequence, Type

from galois import GF, FieldArray

from oraqle.circuits.sorting import cswp
from oraqle.compiler.circuit import Circuit
from oraqle.compiler.nodes import Input

gf = GF(1037347783)


def gen_median_circuit(inputs: Sequence[int], gf: Type[FieldArray]):
    """Returns a naive circuit for finding the median value of `inputs`."""
    input_nodes = [Input(f"Input {v}", gf) for v in inputs]

    outputs = [n for n in input_nodes]

    for i in range(len(outputs) - 1, -1, -1):
        for j in range(i):
            outputs[j], outputs[j + 1] = cswp(outputs[j], outputs[j + 1])  # type: ignore

    if len(outputs) % 2 == 1:
        return Circuit([outputs[len(outputs) // 2]])
    return Circuit([outputs[len(outputs) // 2 + 1]])


if __name__ == "__main__":
    circuit = gen_median_circuit(range(10), gf)
    circuit.to_graph("median.dot")
