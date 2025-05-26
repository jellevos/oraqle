import math
from statistics import mean, stdev
import time
from typing import List, Optional, Tuple
import gurobipy as gp
from gurobipy import GRB

from oraqle.add_chains.addition_chains import add_chain, milp
from oraqle.add_chains.addition_chains_mod import hw


repeats = 3


def run_set(maxsat: bool, min_size_bound: bool, thurber_bounds: bool, skip_deep_sums: bool, max_depth: Optional[int], squaring_cost: float) -> List[Tuple[float, float]]:
    assert min_size_bound
    assert skip_deep_sums

    # FIXME: Clear the cache
    
    results = []
    for target in range(31, 240, 40):
        print(target)
        durations = []
        for _ in range(repeats):
            sam_cost = math.ceil(math.log2(target)) * squaring_cost + hw(target) - 1
            min_size = math.ceil(math.log2(target))

            start = time.monotonic()
            if maxsat:
                add_chain(target, max_depth, sam_cost, squaring_cost, "glucose421", 1, thurber_bounds, min_size, precomputed_values=None)
            else:
                milp(target, max_depth, sam_cost, squaring_cost, thurber_bounds, min_size, precomputed_values=None)
            end = time.monotonic()
            durations.append(end - start)
        results.append((mean(durations), stdev(durations)))
    
    return results



if __name__ == "__main__":
    print("MaxSAT")
    res1 = run_set(True, True, True, True, None, 1.0)
    print("> Now without")
    res2 = run_set(True, True, False, True, None, 1.0)

    print("MILP")
    res3 = run_set(False, True, True, True, None, 1.0)
    print("> Now without")
    res4 = run_set(False, True, False, True, None, 1.0)

    print(res1)
    print(res2)
    print(res3)
    print(res4)
