"""Module for automatic circuit generation for any functions with any number of inputs.

Warning: These circuits can be very large!
"""

from collections import Counter
from typing import Dict, List, Tuple, Type

from galois import GF, FieldArray
from sympy import Add, Integer, Mul, Poly, Pow, Symbol
from sympy.core.numbers import NegativeOne

from oraqle.compiler.circuit import Circuit
from oraqle.compiler.func2poly import interpolate_polynomial
from oraqle.compiler.nodes import Constant, Input, Node
from oraqle.compiler.nodes.abstract import UnoverloadedWrapper
from oraqle.compiler.nodes.arbitrary_arithmetic import Product


def construct_subcircuit(expression, gf, modulus: int, inputs: Dict[str, Input]) -> Node:  # noqa: PLR0912
    """Build a circuit with a single output given an expression of simple arithmetic operations in Sympy.
    
    Raises:
    ------
    Exception: Exponents must be integers, or an exception will be raised.

    Returns:
    -------
    A subcircuit (Node) computing the given sympy expression.

    """
    if expression.func == Add:
        arg_iter = iter(expression.args)

        # The first argument can be a scalar.
        first = next(arg_iter)
        if first.func in {Integer, NegativeOne}:
            if first.func == Integer:
                scalar = Constant(gf(int(first) % modulus))
            else:
                scalar = Constant(-gf(1))
            result = scalar + construct_subcircuit(next(arg_iter), gf, modulus, inputs)
        else:
            # TODO: Replace this entire part with a sum
            result = construct_subcircuit(first, gf, modulus, inputs) + construct_subcircuit(
                next(arg_iter), gf, modulus, inputs
            )

        for arg in arg_iter:
            result = construct_subcircuit(arg, gf, modulus, inputs) + result

        return result
    elif expression.func == Mul:
        arg_iter = iter(expression.args)

        # The first argument can be a scalar.
        first = next(arg_iter)
        if first.func in {Integer, NegativeOne}:
            if first.func == Integer:
                scalar = Constant(gf(int(first) % modulus))
            else:
                scalar = Constant(-gf(1))
            result = scalar * construct_subcircuit(next(arg_iter), gf, modulus, inputs)
        else:
            # TODO: Replace this entire part with a product
            result = construct_subcircuit(first, gf, modulus, inputs) * construct_subcircuit(
                next(arg_iter), gf, modulus, inputs
            )

        for arg in arg_iter:
            result = construct_subcircuit(arg, gf, modulus, inputs) * result

        return result
    elif expression.func == Pow:
        if expression.args[1].func != Integer:
            raise Exception("There was an exponent with a non-integer exponent")
        # Change powers to series of multiplications
        subcircuit = construct_subcircuit(expression.args[0], gf, modulus, inputs)
        # TODO: This is not the most efficient way; we can use re-balancing.
        return Product(
            Counter({UnoverloadedWrapper(subcircuit): int(expression.args[1])}), gf
        )  # FIXME: This could be flattened
    elif expression.func == Symbol:
        assert len(expression.args) == 0
        var = str(expression)
        if var in inputs:
            return inputs[var]
        new_input = Input(var, gf)
        inputs[var] = new_input
        return new_input
    else:
        raise Exception(
            f"The expression contained an invalid operation (not one implemented in arithmetic circuits): {expression.func}."
        )


def construct_circuit(polynomials: List[Poly], modulus: int) -> Tuple[Circuit, Type[FieldArray]]:
    """Construct an arithmetic circuit from a list of polynomials and the fixed modulus.
    
    Returns:
    -------
    A circuit outputting the evaluation of each polynomial.

    """
    inputs = {}
    gf = GF(modulus)
    return (
        Circuit(
            [construct_subcircuit(poly.expr, gf, modulus, inputs) for poly in polynomials],
        ),
        gf,
    )


if __name__ == "__main__":
    # Use function max(x, y)
    function = max
    modulus = 7

    # Create a polynomial and then a circuit that evalutes this expression
    poly = interpolate_polynomial(function, modulus, ["x", "y"])
    circuit, gf = construct_circuit([poly], modulus)

    # Output a DOT file for this high-level circuit (you can visualize it using https://dreampuf.github.io/GraphvizOnline/)
    circuit.to_graph("max_7_hl.dot")

    # Arithmetize the high-level circuit, afterwards it will only contain arithmetic operations
    circuit = circuit.arithmetize()
    circuit.to_graph("max_7_hl.dot")

    # Print the initial metrics of the circuit
    print("depth", circuit.multiplicative_depth())
    print("size", circuit.multiplicative_size())

    # Apply common subexpression elimination (CSE) to remove duplicate operations from the circuit
    circuit.eliminate_subexpressions()

    # Output a DOT file for this arithmetic circuit (you can visualize it using https://dreampuf.github.io/GraphvizOnline/)
    circuit.to_graph("max_7.dot")

    # Print the resulting metrics of the circuit
    print("depth", circuit.multiplicative_depth())
    print("size", circuit.multiplicative_size())

    # Test that given x=4 and y=2 indeed max(x, y) = 4
    assert circuit.evaluate({"x": gf(4), "y": gf(2)}) == [4]

    # Output a DOT file for this arithmetic circuit (you can visualize it using https://dreampuf.github.io/GraphvizOnline/)
    circuit.to_graph("max_7.dot")
