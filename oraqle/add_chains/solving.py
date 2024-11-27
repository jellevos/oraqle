"""Tools for solving SAT formulations."""
import math
import signal
from typing import List, Optional, Sequence, Tuple

from pysat.examples.rc2 import RC2
from pysat.formula import WCNF


def solve(wcnf: WCNF, solver: str, strict_cost_max: Optional[float]) -> Optional[List[int]]:
    """This code is adapted from pysat's internal code to stop when we have reached a maximum cost.

    Returns:
        A list containing the assignment (where 3 indicates that 3=True and -3 indicates that 3=False), or None if the wcnf is unsatisfiable.
    """
    rc2 = RC2(wcnf, solver)

    if strict_cost_max is None:
        strict_cost_max = float("inf")

    while not rc2.oracle.solve(assumptions=rc2.sels + rc2.sums):  # type: ignore
        rc2.get_core()

        if not rc2.core:
            # core is empty, i.e. hard part is unsatisfiable
            return None

        rc2.process_core()

        if rc2.cost >= strict_cost_max:
            return None

    rc2.model = rc2.oracle.get_model()  # type: ignore

    # Return None if the model could not be solved
    if rc2.model is None:
        return None

    # Extract the model
    if rc2.model is None and rc2.pool.top == 0:
        # we seem to have been given an empty formula
        # so let's transform the None model returned to []
        rc2.model = []

    rc2.model = filter(lambda inp: abs(inp) in rc2.vmap.i2e, rc2.model)  # type: ignore
    rc2.model = map(lambda inp: int(math.copysign(rc2.vmap.i2e[abs(inp)], inp)), rc2.model)
    rc2.model = sorted(rc2.model, key=abs)

    return rc2.model


def extract_indices(
    sequence: List[Tuple[int, int]],
    precomputed_values: Optional[Sequence[int]] = None,
    modulus: Optional[int] = None,
) -> List[Tuple[int, int]]:
    """Returns the indices for each step of the addition chain.

    If n precomputed values are provided, then these are considered to be the first n indices after x (i.e. x has index 0, followed by 1, ..., n representing the precomputed values).
    """
    indices = {1: 0}
    offset = 1
    if precomputed_values is not None:
        for v in precomputed_values:
            indices[v] = offset
            offset += 1
    ans_sequence = []

    if modulus is None:
        for index, pair in enumerate(sequence):
            i, j = pair
            ans_sequence.append((indices[i], indices[j]))
            indices[i + j] = index + offset
    else:
        for index, pair in enumerate(sequence):
            i, j = pair
            ans_sequence.append((indices[i % modulus], indices[j % modulus]))
            indices[(i + j) % modulus] = index + offset

    return ans_sequence


def solve_with_time_limit(wcnf: WCNF, solver: str, strict_cost_max: Optional[float], timeout_secs: float) -> Optional[List[int]]:
    """This code is adapted from pysat's internal code to stop when we have reached a maximum cost.

    Raises:
        TimeoutError: When a timeout occurs (after `timeout_secs` seconds)

    Returns:
        A list containing the assignment (where 3 indicates that 3=True and -3 indicates that 3=False), or None if the wcnf is unsatisfiable.
    """
    def timeout_handler(s, f):
        raise TimeoutError
    
    # Set the timeout
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.setitimer(signal.ITIMER_REAL, timeout_secs)

    try:
        # TODO: Reduce code duplication: we only changed solve to solve_limited
        rc2 = RC2(wcnf, solver)

        if strict_cost_max is None:
            strict_cost_max = float("inf")

        while not rc2.oracle.solve_limited(assumptions=rc2.sels + rc2.sums, expect_interrupt=True):  # type: ignore
            rc2.get_core()

            if not rc2.core:
                # core is empty, i.e. hard part is unsatisfiable
                signal.setitimer(signal.ITIMER_REAL, 0)
                return None

            rc2.process_core()

            if rc2.cost >= strict_cost_max:
                signal.setitimer(signal.ITIMER_REAL, 0)
                return None

        signal.setitimer(signal.ITIMER_REAL, 0)
        rc2.model = rc2.oracle.get_model()  # type: ignore

        # Return None if the model could not be solved
        if rc2.model is None:
            return None

        # Extract the model
        if rc2.model is None and rc2.pool.top == 0:
            # we seem to have been given an empty formula
            # so let's transform the None model returned to []
            rc2.model = []

        rc2.model = filter(lambda inp: abs(inp) in rc2.vmap.i2e, rc2.model)  # type: ignore
        rc2.model = map(lambda inp: int(math.copysign(rc2.vmap.i2e[abs(inp)], inp)), rc2.model)
        rc2.model = sorted(rc2.model, key=abs)

        return rc2.model
    except TimeoutError as err:
        raise TimeoutError from err
