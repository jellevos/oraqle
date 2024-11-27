"""Tools for computing addition chains, taking into account the modular nature of the algebra."""
import math
from typing import List, Optional, Tuple

from oraqle.add_chains.addition_chains import add_chain


def hw(n: int) -> int:
    """Returns the Hamming weight of n."""
    c = 0
    while n:
        c += 1
        n &= n - 1

    return c


def size_lower_bound(target: int) -> int:
    """Returns a lower bound on the size of the addition chain for this target."""
    return math.ceil(
        max(
            math.log2(target) + math.log2(hw(target)) - 2.13,
            math.log2(target),
            math.log2(target) + math.log(hw(target), 3) - 1,
        )
    )


def cost_lower_bound_monotonic(target: int, squaring_cost: float) -> float:
    """Returns a lower bound on the cost of the addition chain for this target. The bound is guaranteed to grow monotonically with the target."""
    return math.ceil(math.log2(target)) * squaring_cost


def chain_cost(chain: List[Tuple[int, int]], squaring_cost: float) -> float:
    """Returns the cost of the addition chain, considering doubling (squaring) to be cheaper than other additions (multiplications)."""
    return sum(squaring_cost if x == y else 1.0 for x, y in chain)


def add_chain_modp(  # noqa: PLR0913, PLR0917
    target: int,
    modulus: int,
    max_depth: Optional[int],
    strict_cost_max: float,
    squaring_cost: float,
    solver,
    encoding,
    thurber,
    min_size: int,
    precomputed_values: Optional[Tuple[Tuple[int, int], ...]] = None,
) -> Optional[List[Tuple[int, int]]]:
    """Computes an addition chain for target modulo p with the given constraints and optimization parameters.

    The precomputed_powers are an optional set of powers that have previously been computed along with their depth.
    This means that those powers can be reused for free.

    Returns:
        If it exists, a minimal addition chain meeting the given constraints and optimization parameters.
    """
    if precomputed_values is not None:
        # The shortest chain in (t + (k-1)p, t + kp] will have length at least k
        # The cheapest chain in (t + (k-1)p, t + kp] will have cost at least k / sqr_cost
        best_chain = None

        k = 0
        while (k / squaring_cost) < strict_cost_max:
            # Add multiples of the precomputed_values
            new_precomputed_values = []
            for precomputed_value, depth in precomputed_values:
                for i in range(k + 1):
                    new_precomputed_values.append((precomputed_value + i * modulus, depth))

            chain = add_chain(
                target + k * modulus,
                max_depth,
                strict_cost_max,
                squaring_cost,
                solver,
                encoding,
                thurber,
                min_size=max(min_size, k),
                precomputed_values=tuple(new_precomputed_values),
            )

            if chain is not None:
                cost = chain_cost(chain, squaring_cost)
                strict_cost_max = min(strict_cost_max, cost)
                best_chain = chain

            k += 1

        return best_chain

    best_chain = None
    best_cost = None

    current_target = target

    i = 0

    while cost_lower_bound_monotonic(current_target, squaring_cost) < strict_cost_max and (
        max_depth is None or math.ceil(math.log2(current_target)) <= max_depth
    ):
        tightest_min_size = max(size_lower_bound(current_target), min_size)
        if (tightest_min_size * squaring_cost) >= (
            strict_cost_max if best_cost is None else min(strict_cost_max, best_cost)
        ):
            current_target += modulus
            continue

        chain = add_chain(
            current_target,
            max_depth,
            strict_cost_max,
            squaring_cost,
            solver,
            encoding,
            thurber,
            tightest_min_size,
            precomputed_values,
        )

        if chain is not None:
            cost = chain_cost(chain, squaring_cost)
            if best_cost is None or cost < best_cost:
                best_cost = cost
                best_chain = chain
                strict_cost_max = min(best_cost, strict_cost_max)

        current_target += modulus

    i += 1
    return best_chain


def test_add_chain_modp_over_modulus():  # noqa: D103
    chain = add_chain_modp(
        62,
        66,
        None,
        8.0,
        0.75,
        solver="glucose42",
        encoding=1,
        thurber=True,
        min_size=1,
        precomputed_values=None,
    )
    assert chain == [(1, 1), (2, 2), (4, 4), (8, 8), (16, 16), (32, 32), (64, 64)]


def test_add_chain_modp_precomputations():  # noqa: D103
    chain = add_chain_modp(
        64,  # 64+66 = 65+65
        66,
        None,
        2.0,
        0.75,
        solver="glucose42",
        encoding=1,
        thurber=True,
        min_size=1,
        precomputed_values=((65, 5),),
    )
    assert chain == [(65, 65)]


if __name__ == "__main__":
    print(add_chain_modp(
        254,
        255,
        None,
        8.0,
        0.5,
        solver="glucose42",
        encoding=1,
        thurber=True,
        min_size=11,
        precomputed_values=None,
    ))
