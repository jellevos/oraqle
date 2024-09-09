"""Tools for generating addition chains that trade off depth and cost."""
import math
from typing import List, Optional, Tuple

from oraqle.add_chains.addition_chains import add_chain
from oraqle.add_chains.addition_chains_mod import add_chain_modp, hw, size_lower_bound


def chain_depth(
    chain: List[Tuple[int, int]],
    precomputed_values: Optional[Tuple[Tuple[int, int], ...]] = None,
    modulus: Optional[int] = None,
) -> int:
    """Return the depth of the addition chain."""
    depths = {1: 0}
    if precomputed_values is not None:
        depths.update(precomputed_values)

    if modulus is None:
        for x, y in chain:
            depths[x + y] = max(depths[x], depths[y]) + 1
    else:
        for x, y in chain:
            depths[(x + y) % modulus] = max(depths[x % modulus], depths[y % modulus]) + 1

    return max(depths.values())


def gen_pareto_front(  # noqa: PLR0912, PLR0913, PLR0917
    target: int,
    modulus: Optional[int],
    squaring_cost: float,
    solver="glucose42",
    encoding=1,
    thurber=True,
    precomputed_values: Optional[Tuple[Tuple[int, int], ...]] = None,
) -> List[Tuple[int, List[Tuple[int, int]]]]:
    """Returns a Pareto front of addition chains, trading of cost and depth."""
    if target == 1:
        return [(0, [])]

    if modulus is not None:
        assert target <= modulus

    # Find the lowest depth chain using square & multiply (SaM)
    sam_depth = math.ceil(math.log2(target))
    sam_cost = math.ceil(math.log2(target)) * squaring_cost + hw(target) - 1
    sam_target = target

    # If there is a modulus, we should also consider it to find an upper bound on the cost of a minimum-depth chain
    if modulus is not None:
        current_target = target + modulus - 1
        while math.log2(current_target) <= sam_depth:
            current_cost = (
                math.ceil(math.log2(current_target)) * squaring_cost + hw(current_target) - 1
            )
            if current_cost < sam_cost:
                sam_cost = current_cost
                sam_target = target
            current_target += modulus - 1

    # Find the cheapest chain (i.e. no depth constraints)
    min_size = size_lower_bound(target) if precomputed_values is None else 1
    if modulus is None:
        cheapest_chain = add_chain(
            target,
            None,
            sam_cost,
            squaring_cost,
            solver,
            encoding,
            thurber,
            min_size,
            precomputed_values,
        )
    else:
        cheapest_chain = add_chain_modp(
            target,
            modulus,
            None,
            sam_cost,
            squaring_cost,
            solver,
            encoding,
            thurber,
            min_size,
            precomputed_values,
        )

    # If no cheapest chain is found that satisfies these bounds, then square and multiply had the same cost
    if cheapest_chain is None:
        sam_chain = []
        for i in range(math.ceil(math.log2(sam_target))):
            sam_chain.append((2**i, 2**i))
        previous = 1
        for i in range(math.ceil(math.log2(sam_target))):
            if (sam_target >> i) & 1:
                sam_chain.append((previous, 2**i))
                previous += 2**i
        return [(sam_depth, sam_chain)]

    add_size = len(cheapest_chain)  # TODO: Check that this is indeed a valid bound
    add_cost = sum(squaring_cost if x == y else 1.0 for x, y in cheapest_chain)
    add_depth = chain_depth(cheapest_chain, precomputed_values, modulus=modulus)

    # Go through increasing depth and decrease the previous size, until we reach the cost of square and multiply
    pareto_front = []
    current_depth = sam_depth
    current_cost = sam_cost
    while current_cost > add_cost and current_depth < add_depth:
        if modulus is None:
            chain = add_chain(
                target,
                current_depth,
                current_cost,
                squaring_cost,
                solver,
                encoding,
                thurber,
                add_size,
                precomputed_values,
            )
        else:
            chain = add_chain_modp(
                target,
                modulus,
                current_depth,
                current_cost,
                squaring_cost,
                solver,
                encoding,
                thurber,
                add_size,
                precomputed_values,
            )

        if chain is not None:
            # Add to the Pareto front
            pareto_front.append((current_depth, chain))
            current_cost = sum(squaring_cost if x == y else 1.0 for x, y in chain)

        current_depth += 1

    # Add the final chain and return
    if add_cost < current_cost or len(pareto_front) == 0:
        pareto_front.append((add_depth, cheapest_chain))

    return pareto_front


def test_gen_exponentiation_front_small():  # noqa: D103
    front = gen_pareto_front(2, None, 0.75)
    assert front == [(1, [(1, 1)])]
