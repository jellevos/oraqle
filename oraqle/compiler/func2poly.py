"""Tools for interpolating polynomials from arbitrary functions."""
import itertools
from typing import Callable, List

from sympy import Poly, symbols


def principal_character(x, prime_modulus):
    """Computes the principal character. This expression always returns 1 when x = 0 and 0 otherwise. Only works for prime moduli.

    Returns:
        The principal character x**(p-1).
    """
    return x ** (prime_modulus - 1)


def interpolate_polynomial(
    function: Callable[..., int], prime_modulus: int, input_names: List[str]
) -> Poly:
    """Interpolates a polynomial for the given function. This is currently only implemented for prime moduli. This function interpolates the polynomial on all possible inputs.

    Returns:
        A sympy `Poly` object representing the unique polynomial that evaluates to the same outputs for all inputs as `function`.
    """
    variables = symbols(input_names)
    poly = 0

    for inputs in itertools.product(range(prime_modulus), repeat=len(input_names)):
        output = function(*inputs)
        assert 0 <= output < prime_modulus

        product = output
        for input, variable in zip(inputs, variables):
            product *= Poly(
                1 - principal_character(variable - input, prime_modulus),
                variable,
                modulus=prime_modulus,
            )
            product = Poly(product, variables, modulus=prime_modulus)

        poly += product
        poly = Poly(poly, variables, modulus=prime_modulus)

    return Poly(poly, variables, modulus=prime_modulus)
