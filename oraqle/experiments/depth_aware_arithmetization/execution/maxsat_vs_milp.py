from importlib.resources import files
import math
import os
from statistics import mean, stdev
import time
from typing import List, Tuple

import oraqle
from oraqle.add_chains.addition_chains import add_chain, milp
from oraqle.add_chains.addition_chains_mod import hw
from oraqle.add_chains.memoization import ADDCHAIN_CACHE_FILENAME


repeats = 10


def run_set(maxsat: bool, use_sam_depth: bool, squaring_cost: float, cuts: bool) -> List[Tuple[float, float]]:
    if maxsat:
        assert cuts

    oraqle_path = files(oraqle)
    database_path = oraqle_path.joinpath(ADDCHAIN_CACHE_FILENAME + '.db')
    try:
        os.remove(str(database_path))
    except FileNotFoundError:
        pass
    
    results = []
    for target in range(31, 240, 40):
        durations = []
        for _ in range(repeats):
            sam_cost = math.ceil(math.log2(target)) * squaring_cost + hw(target) - 1
            if use_sam_depth:
                max_depth = math.ceil(math.log2(target))
            else:
                max_depth = None
            min_size = math.ceil(math.log2(target))

            start = time.monotonic()
            if maxsat:
                add_chain(target, max_depth, sam_cost, squaring_cost, "glucose421", 1, True, min_size, precomputed_values=None)
            else:
                milp(target, max_depth, sam_cost, squaring_cost, True, min_size, precomputed_values=None, include_cuts=cuts)
            end = time.monotonic()
            durations.append(end - start)
        results.append((mean(durations), stdev(durations)))
    
    return results


def generate_latex_table(res1, res2, res3, res4, res5, res6):
    rows = zip(res1, res2, res3, res4, res5, res6)
    table_lines = []

    for row in rows:
        formatted_row = [
            f"${mean:.2f} \\pm {stdev:.2f}$"
            for (mean, stdev) in row
        ]
        table_lines.append(" & ".join(formatted_row) + r" \\")
    
    return "\n".join(table_lines)


if __name__ == "__main__":
    # No depth
    limit_depth = False
    print("MILP without cuts")
    res1 = run_set(False, limit_depth, 1.0, cuts=False)
    print(res1)

    print("MILP with cuts")
    res2 = run_set(False, limit_depth, 1.0, cuts=False)
    print(res2)

    print("MaxSAT")
    res3 = run_set(True, limit_depth, 1.0, cuts=True)
    print(res3)

    # With depth
    limit_depth = True
    print("MILP without cuts")
    res4 = run_set(False, limit_depth, 1.0, cuts=False)
    print(res4)

    print("MILP with cuts")
    res5 = run_set(False, limit_depth, 1.0, cuts=False)
    print(res5)

    print("MaxSAT")
    res6 = run_set(True, limit_depth, 1.0, cuts=True)
    print(res6)

    table = generate_latex_table(res1, res2, res3, res4, res5, res6)
    print(table)
