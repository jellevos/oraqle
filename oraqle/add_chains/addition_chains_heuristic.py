"""This module contains functions for finding addition chains, while sometimes resorting to heuristics to prevent long computations."""

from functools import lru_cache
import math
from typing import List, Optional, Tuple

from oraqle.add_chains.addition_chains import add_chain
from oraqle.add_chains.addition_chains_mod import add_chain_modp, hw
from oraqle.add_chains.solving import extract_indices


def _mul(current_chain: List[Tuple[int, int]], other_chain: List[Tuple[int, int]]):
    length = len(current_chain)
    for a, b in other_chain:
        current_chain.append((a + length, b + length))


def _chain(n, k) -> List[Tuple[int, int]]:
    q = n // k
    r = n % k
    if r in {0, 1}:
        chain_k = _minchain(k)
        _mul(chain_k, _minchain(q))
        if r == 1:
            chain_k.append((0, len(chain_k)))
        return chain_k
    else:
        chain_k = _chain(k, r)
        index_r = len(chain_k)
        _mul(chain_k, _minchain(q))
        chain_k.append((index_r, len(chain_k)))
        return chain_k


def _minchain(n: int) -> List[Tuple[int, int]]:
    log_n = n.bit_length() - 1
    if n == 1 << log_n:
        return [(i, i) for i in range(log_n)]
    elif n == 3:
        return [(0, 0), (0, 1)]
    else:
        k = n // (1 << (log_n // 2))
        return _chain(n, k)


@lru_cache
def add_chain_guaranteed(  # noqa: PLR0913, PLR0917
    target: int,
    modulus: Optional[int],
    squaring_cost: float,
    solver: str = "glucose421",
    encoding: int = 1,
    thurber: bool = True,
    precomputed_values: Optional[Tuple[Tuple[int, int], ...]] = None,
) -> List[Tuple[int, int]]:
    """Always generates an addition chain for a given target, which is suboptimal if the inputs are too large.
    
    In some cases, the result is not necessarily optimal. These are the cases where we resort to a heuristic.
    This currently happens if:
    - The target exceeds 1000.
    - The modulus (if provided) exceeds 200.
    - MAXSAT_TIMEOUT is not None and a MaxSAT instance timed out

    !!! note
        This function is useful for preventing long computation, but the result is not guaranteed to be (close to) optimal.
        Unlike `add_chain`, this function will always return an addition chain.

    Parameters:
        target: The target integer.
        modulus: Modulus to take into account. In an exponentiation chain, this is the modulus in the exponent, i.e. x^target mod p corresponds to `modulus = p - 1`.
        squaring_cost: The cost of doubling (squaring), compared to other additions (multiplications), which cost 1.0.
        solver: Name of the SAT solver, e.g. "glucose421" for glucose 4.2.1. See: https://pysathq.github.io/docs/html/api/solvers.html.
        encoding: The encoding to use for cardinality constraints. See: https://pysathq.github.io/docs/html/api/card.html#pysat.card.EncType.
        thurber: Whether to use the Thurber bounds, which provide lower bounds for the elements in the chain. The bounds are ignored when `precomputed_values = True`.
        precomputed_values: If there are any precomputed values that can be used for free, they can be specified as a tuple of pairs (value, chain_depth).
    
    Raises:  # noqa: DOC502
        TimeoutError: If the global MAXSAT_TIMEOUT is not None, and it is reached before a maxsat instance could be solved.

    Returns:
        An addition chain.
    """
    # We want to do better than square and multiply, so we find an upper bound
    sam_cost = math.ceil(math.log2(target)) * squaring_cost + hw(target) - 1

    # Apply CSE to the square & mutliply chain
    if precomputed_values is not None:
        for exp, depth in precomputed_values:
            if exp > 0 and (exp & (exp - 1)) == 0 and depth == math.log2(exp):
                sam_cost -= squaring_cost

    try:
        addition_chain = None
        if modulus is not None and modulus <= 200:
            addition_chain = add_chain_modp(
                target,
                modulus,
                None,
                sam_cost,
                squaring_cost,
                solver,
                encoding,
                thurber,
                min_size=math.ceil(math.log2(target)) if precomputed_values is None else 1,
                precomputed_values=precomputed_values,
            )
        elif target <= 1000:
            addition_chain = add_chain(
                target,
                None,
                sam_cost,
                squaring_cost,
                solver,
                encoding,
                thurber,
                min_size=math.ceil(math.log2(target)) if precomputed_values is None else 1,
                precomputed_values=precomputed_values,
            )

        if addition_chain is not None:
            addition_chain = extract_indices(
                addition_chain, precomputed_values=None if precomputed_values is None else list(k for k, _ in precomputed_values), modulus=modulus
            )
    except TimeoutError:
        # The MaxSAT solver timed out, so we resort to a heuristic
        pass

    if addition_chain is None:
        # If no other addition chain algorithm has been called or if we could not do better than square and multiply

        # Uses the minchain algorithm from ["Addition chains using continued fractions."][BBBD1989]
        # The implementation was adapted from the `addchain` Rust crate (https://github.com/str4d/addchain).
        # This algorithm is not optimal: Below 1000 it requires one too many multiplication in 29 cases.
        addition_chain = _minchain(target)

        if precomputed_values is not None:
            # We must shift the indices in the addition chain
            shift = len(precomputed_values)
            addition_chain = [(0 if x == 0 else x + shift, 0 if y == 0 else y + shift) for (x, y) in addition_chain]

    assert addition_chain is not None

    return addition_chain
