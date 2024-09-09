"""Test file for sugar expressions."""

from galois import GF

from oraqle.compiler.circuit import Circuit
from oraqle.compiler.nodes.arbitrary_arithmetic import sum_
from oraqle.compiler.nodes.leafs import Input


def test_sum():
    """Tests the sum_ function."""
    gf = GF(127)

    a = Input("a", gf)
    b = Input("b", gf)

    arithmetic_circuit = Circuit([sum_(a, 4, b, 3)]).arithmetize()

    for val_a in range(127):
        for val_b in range(127):
            expected = gf(val_a) + gf(val_b) + gf(7)
            assert arithmetic_circuit.evaluate({"a": gf(val_a), "b": gf(val_b)}) == expected
