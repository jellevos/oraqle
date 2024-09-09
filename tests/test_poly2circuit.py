"""Test file for generating circuits using polynomial interpolation."""

import itertools

from oraqle.compiler.func2poly import interpolate_polynomial
from oraqle.compiler.poly2circuit import construct_circuit


def _construct_and_test_circuit_from_bivariate_lambda(function, modulus: int, cse=False):
    poly = interpolate_polynomial(function, modulus, ["x", "y"])
    circuit, gf = construct_circuit([poly], modulus)
    circuit = circuit.arithmetize()

    if cse:
        circuit.eliminate_subexpressions()

    for x, y in itertools.product(range(modulus), repeat=2):
        print(function, x, y)
        assert circuit.evaluate({"x": gf(x), "y": gf(y)}) == [function(x, y)]


def test_inequality_mod7():
    """Tests x != y (mod 7)."""
    _construct_and_test_circuit_from_bivariate_lambda(lambda x, y: int(x != y), modulus=7)


def test_inequality_mod13():
    """Tests x != y (mod 13)."""
    _construct_and_test_circuit_from_bivariate_lambda(lambda x, y: int(x != y), modulus=13)


def test_max_mod7():
    """Tests max(x, y) (mod 7)."""
    _construct_and_test_circuit_from_bivariate_lambda(max, modulus=7)


def test_max_mod13():
    """Tests max(x, y) (mod 13)."""
    _construct_and_test_circuit_from_bivariate_lambda(max, modulus=13)


def test_xor_mod11():
    """Tests x ^ y (mod 11)."""
    _construct_and_test_circuit_from_bivariate_lambda(lambda x, y: (x ^ y) % 11, modulus=11)


def test_inequality_mod11_cse():
    """Tests x ^ y (mod 11) with CSE."""
    _construct_and_test_circuit_from_bivariate_lambda(
        lambda x, y: int(x != y), modulus=11, cse=True
    )


def test_max_mod7_cse():
    """Tests max(x, y) (mod 7) with CSE."""
    _construct_and_test_circuit_from_bivariate_lambda(max, modulus=7, cse=True)


def test_xor_mod13_cse():
    """Tests x ^ y (mod 13) with CSE."""
    _construct_and_test_circuit_from_bivariate_lambda(
        lambda x, y: (x ^ y) % 13, modulus=13, cse=True
    )
