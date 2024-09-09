"""The veto voting circuit is the inverse of a consensus vote between a number of participants.

The circuit is essentially a large OR operation, returning 1 if any participant vetoes (by submitting a 1).
This represents a vote that anyone can veto.
"""
from typing import Type

from galois import GF, FieldArray

from oraqle.compiler.boolean.bool_or import any_
from oraqle.compiler.circuit import Circuit
from oraqle.compiler.nodes import Input

gf = GF(103)


def gen_veto_voting_circuit(participants: int, gf: Type[FieldArray]):
    """Returns a veto voting circuit between the number of `participants`."""
    input_nodes = {Input(f"Input {i}", gf) for i in range(participants)}
    return Circuit([any_(*input_nodes)])


if __name__ == "__main__":
    circuit = gen_veto_voting_circuit(10, gf).arithmetize()

    circuit.eliminate_subexpressions()
    circuit.to_graph("veto-voting.dot")
