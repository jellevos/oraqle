"""Tools for generating short addition chains using a MaxSAT formulation."""
import math
from typing import List, Optional, Tuple

from pysat.card import CardEnc
from pysat.formula import WCNF

from oraqle.add_chains.memoization import ADDCHAIN_CACHE_PATH, cache_to_disk
from oraqle.add_chains.solving import solve, solve_with_time_limit
from oraqle.config import MAXSAT_TIMEOUT


def thurber_bounds(target: int, max_size: int) -> List[Tuple[int, int]]:
    """Returns the Thurber bounds for a given target and a maximum size of the addition chain."""
    m = target
    t = 0
    while (m % 2) == 0:
        t += 1
        m >>= 1

    bounds = []
    for step in range(max_size - t - 3 + 1):
        if ((1 << (max_size - t - step - 2) + 1) % target) == 0:
            denominator = (1 << (t + 1)) * ((1 << (max_size - t - (step + 2))) + 1)
        else:
            denominator = (1 << t) * ((1 << (max_size - t - (step + 1))) + 1)
        bound = int(math.ceil(target / denominator))
        bounds.append((bound, min(1 << step, target)))

    step = max_size - t - 2
    if step > 0:
        denominator = (1 << t) * ((1 << (max_size - t - (step + 1))) + 1)
        bound = int(math.ceil(target / denominator))
        bounds.append((bound, min(1 << step, target)))

    if max_size - t - 1 > 0:
        for step in range(max_size - t - 1, max_size + 1):
            bound = int(math.ceil(target / (1 << (max_size - step))))
            bounds.append((bound, min(1 << step, target)))

    return bounds


@cache_to_disk(ADDCHAIN_CACHE_PATH, ignore_args={"solver", "encoding", "thurber"})
def add_chain(  # noqa: PLR0912, PLR0913, PLR0915, PLR0917
    target: int,
    max_depth: Optional[int],
    strict_cost_max: float,
    squaring_cost: float,
    solver: str,
    encoding: int,
    thurber: bool,
    min_size: int,
    precomputed_values: Optional[Tuple[Tuple[int, int], ...]],
) -> Optional[List[Tuple[int, int]]]:
    """Generates a minimum-cost addition chain for a given target, abiding to the constraints.

    Parameters:
        target: The target integer.
        max_depth: The maximum depth of the addition chain
        strict_cost_max: A strict upper bound on the cost of the addition chain. I.e., cost(chain) < strict_cost_max.
        squaring_cost: The cost of doubling (squaring), compared to other additions (multiplications), which cost 1.0.
        solver: Name of the SAT solver, e.g. "glucose421" for glucose 4.2.1. See: https://pysathq.github.io/docs/html/api/solvers.html.
        encoding: The encoding to use for cardinality constraints. See: https://pysathq.github.io/docs/html/api/card.html#pysat.card.EncType.
        thurber: Whether to use the Thurber bounds, which provide lower bounds for the elements in the chain. The bounds are ignored when `precomputed_values = True`.
        min_size: The minimum size of the chain. It is always possible to use `math.ceil(math.log2(target))`.
        precomputed_values: If there are any precomputed values that can be used for free, they can be specified as a tuple of pairs (value, chain_depth).
    
    Raises:  # noqa: DOC502
        TimeoutError: If the global MAXSAT_TIMEOUT is not None, and it is reached before a maxsat instance could be solved.

    Returns:
        A minimum-cost addition chain, if it exists.
    """
    # TODO: Maybe precomputed_values should not be optional, but should be ignored if it is empty
    assert target != 0

    if target == 1:
        return []

    def x(i) -> int:
        return i

    if precomputed_values is not None:

        def z(i: int) -> int:
            offset = target + 1
            return i + offset

    def y(i, j) -> int:
        # TODO: We can make the offset tighter
        offset = (target + 1) if precomputed_values is None else 2 * (target + 1)
        assert i <= j
        return j * (j + 1) // 2 + i + offset

    def y_inv(n: int) -> Tuple[int, int]:
        offset = (target + 1) if precomputed_values is None else 2 * (target + 1)
        assert n >= offset
        n -= offset
        j = math.floor((math.sqrt(1 + 8 * n) - 1) // 2)
        i = n - j * (j + 1) // 2

        return i, j  # minus 1 so that 1 -> 0

    if max_depth is not None:

        def d(i, depth) -> int:
            offset = y(target, target)
            assert depth <= max_depth + 1
            return offset + 1 + (i - 1) * (max_depth + 1) + depth

    wcnf = WCNF()

    # x_i for i = 1,...,target represents the computed additions
    # y_i,j for i,j = 2,...,target s.t. i <= j represents that i+j is computed

    # Add constraints
    big_disjunctions = {k: [] for k in range(1, target + 1)}
    for j in range(1, target + 1):
        x_j = x(j)

        for i in range(1, min(j + 1, target + 1 - j)):
            x_i = x(i)
            y_ij = y(i, j)

            k = i + j

            # y_ij requires that x_i is set
            wcnf.append([-y_ij, x_i])
            if i != j:
                # y_ij requires that x_j is set
                wcnf.append([-y_ij, x_j])

            # x_k is set when y_ij is set
            big_disjunctions[k].append(y_ij)

            # Add objective
            wcnf.append([-y(i, j)], weight=(squaring_cost if i == j else 1))

            if max_depth is not None:
                for depth in range(max_depth + 1):
                    # d_k,depth+1 is set when d_i,depth and y_ij are set
                    wcnf.append([d(k, depth + 1), -d(i, depth), -y_ij])
                    if i != j:
                        # d_k,depth+1 is set when d_j,depth and y_ij are set
                        wcnf.append([d(k, depth + 1), -d(j, depth), -y_ij])

    if precomputed_values is not None:
        for k, k_depth in precomputed_values:
            if k == 0 or k > target:
                continue

            if max_depth is not None and k_depth > max_depth:
                continue

            # x_k is set when z_k is set
            big_disjunctions[k].append(z(k))

            if max_depth is not None:
                wcnf.append([d(k, k_depth), -z(k)])

    wcnf.append([x(target)])

    if max_depth is not None:
        wcnf.append([d(1, 0)])

    for k in range(2, target + 1):
        big_disjunctions[k].append(-x(k))
        wcnf.append(big_disjunctions[k])

        # Cut some potential additions
        if precomputed_values is None:
            # We do not use these bounds when precomputed_values is not None
            wcnf.append([x(m) for m in range((k + 1) // 2, k)])  # type: ignore

        if max_depth is not None:
            # May not exceed max_depth
            wcnf.append([-d(k, max_depth + 1)])

    # Add generalized Thurber bounds (for each step in the chain, the number must be between lower_bound and 2^step)
    # We do not use the Thurber bounds when precomputed_values is not None
    if thurber and precomputed_values is None:
        max_size = math.floor(strict_cost_max / squaring_cost)
        for lb, ub in thurber_bounds(target, max_size):
            # FIXME: These bounds seem not to help for target ~ hundreds
            wcnf.append([x(i) for i in range(lb, ub + 1)])

    # Bound the number of x that are true from below
    if max_depth is None:
        top_id = y(target, target)
    else:
        top_id = y(target, target) + 1 + (target - 1) * (max_depth + 1) + max_depth + 1
    at_least_cnf = CardEnc.atleast(
        [x(k) for k in range(2, target + 1)], bound=min_size, top_id=top_id, encoding=encoding
    )
    wcnf.extend(at_least_cnf)

    # Solve
    if MAXSAT_TIMEOUT is None:
        model = solve(wcnf, solver, strict_cost_max)
    else:
        model = solve_with_time_limit(wcnf, solver, strict_cost_max, MAXSAT_TIMEOUT)

    if model is None:
        return None

    offset = (target + 1) if precomputed_values is None else 2 * (target + 1)
    return [y_inv(n) for n in model if offset <= n <= y(target, target)]


def test_addition_chain():  # noqa: D103
    chain = add_chain(
        8,
        3,
        2.0,
        0.5,
        solver="glucose42",
        encoding=1,
        thurber=True,
        min_size=2,
        precomputed_values=None,
    )
    assert chain == [(1, 1), (2, 2), (4, 4)]


def test_addition_chain_precomputed_no_depth():  # noqa: D103
    chain = add_chain(
        8,
        None,
        2.0,
        0.5,
        solver="glucose42",
        encoding=1,
        thurber=True,
        min_size=1,
        precomputed_values=((7, 2),),
    )
    assert chain == [(1, 7)]


def test_addition_chain_precomputed_depth():  # noqa: D103
    chain = add_chain(
        8,
        3,
        2.0,
        0.5,
        solver="glucose42",
        encoding=1,
        thurber=True,
        min_size=1,
        precomputed_values=((7, 2),),
    )
    assert chain == [(1, 7)]


def test_addition_chain_precomputed_depth_too_large():  # noqa: D103
    chain = add_chain(
        8,
        3,
        2.0,
        0.5,
        solver="glucose42",
        encoding=1,
        thurber=True,
        min_size=1,
        precomputed_values=((7, 3),),
    )
    assert chain == [(1, 1), (2, 2), (4, 4)]


def test_addition_chain_precomputed_no_depth_squaring():  # noqa: D103
    chain = add_chain(
        18,
        None,
        2.0,
        0.5,
        solver="glucose42",
        encoding=1,
        thurber=True,
        min_size=1,
        precomputed_values=((9, 3),),
    )
    assert chain == [(9, 9)]


if __name__ == "__main__":
    print(add_chain(
        254,
        None,
        8.0,
        0.5,
        solver="glucose42",
        encoding=1,
        thurber=True,
        min_size=11,
        precomputed_values=None,
    ))

    print(add_chain(
        254,
        None,
        7.5,
        0.5,
        solver="glucose42",
        encoding=1,
        thurber=True,
        min_size=8,
        precomputed_values=None,
    ))
