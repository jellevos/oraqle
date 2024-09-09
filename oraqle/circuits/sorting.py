"""This module contains sorting circuits and comparators."""
from typing import Sequence, Tuple, Type

from galois import GF, FieldArray

from oraqle.compiler.circuit import ArithmeticCircuit, Circuit
from oraqle.compiler.nodes import Input
from oraqle.compiler.nodes.abstract import Node

gf = GF(13)


def cswp(lhs: Node, rhs: Node) -> Tuple[Node, Node]:
    """Conditionally swap inputs `lhs` and `rhs` such that `lhs <= rhs`.

    Returns:
        A tuple representing (lower, higher)
    """
    teq = lhs < rhs

    first = teq * (lhs - rhs) + rhs
    second = lhs + rhs - first

    return (
        first,
        second,
    )


def gen_naive_sort_circuit(inputs: Sequence[int], gf: Type[FieldArray]) -> ArithmeticCircuit:
    """Returns a naive sorting circuit for the given sequence of `inputs`."""
    input_nodes = [Input(f"Input {v}", gf) for v in inputs]

    outputs = [n for n in input_nodes]

    for i in range(len(outputs) - 1, -1, -1):
        for j in range(i):
            outputs[j], outputs[j + 1] = cswp(outputs[j], outputs[j + 1])  # type: ignore

    return Circuit(outputs).arithmetize()  # type: ignore


if __name__ == "__main__":
    circuit = gen_naive_sort_circuit(range(2), gf)
    circuit.to_graph("sorting.dot")
