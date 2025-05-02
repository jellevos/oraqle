import math
import time
from typing import Optional, Tuple
import gurobipy as gp
from gurobipy import GRB

from oraqle.add_chains.addition_chains import add_chain
from oraqle.add_chains.addition_chains_mod import hw


def milp(target: int, max_depth: Optional[int]) -> Tuple[int, float]:
    start = time.monotonic()
    seq = [target]

    def x(i) -> int:
        return i


    def y(i, j) -> int:
        assert i <= j
        return j * (j + 1) // 2 + i + (max(seq) + 1)


    # def find_y(n):
    #     for i in range(1, max(seq) + 1):
    #         for j in range(1, max(seq) + 1):
    #             if i > j:
    #                 continue
        
    #             if y(i, j) == n:
    #                 return (i, j)


    model = gp.Model("addition_seq")

    # x_i for i = 1,...,max(seq) represents the computed additions
    # y_i,j for i,j = 1,...,max(seq) s.t. i <= j represents that i+j is computed
    vars = [None] * (y(max(seq), max(seq)) + 1)

    for i in range(1, max(seq) + 1):
        vars[x(i)] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}")  # type: ignore

    for j in range(1, max(seq) + 1):
        for i in range(1, j + 1):
            vars[y(i, j)] = model.addVar(vtype=GRB.BINARY, name=f"y_{i},{j}")  # type: ignore
            
    # Add constraints
    big_disjunctions = {k: [] for k in range(1, max(seq) + 1)}
    for j in range(1, max(seq) + 1):
        x_j = vars[x(j)]

        for i in range(1, j + 1):
            x_i = vars[x(i)]
            y_ij = vars[y(i, j)]

            model.addConstr(y_ij <= x_i)  # type: ignore
            if i != j:
                model.addConstr(y_ij <= x_j)  # type: ignore

            # TODO: Move up
            k = i + j
            if k > max(seq):
                continue

            big_disjunctions[k].append(y_ij)

    for k in seq:
        model.addConstr(vars[x(k)] == 1)  # type: ignore

    #wcnf.append([x(1)])
    for k in range(2, max(seq) + 1):
        #big_disjunctions[k].append(-x(k))
        # print(k, big_disjunctions[k], [find_y(n) for n in big_disjunctions[k]])
        model.addConstr(sum(big_disjunctions[k]) >= vars[x(k)])  # type: ignore

        # Add objective
        #wcnf.append([-x(k)], weight=1)

    # Cut
    for k in range(2, max(seq) + 1):
        model.addConstr(sum(vars[x(m)] for m in range((k + 1) // 2, k)) >= 1)  # type: ignore


    # Solve
    #model.setObjective(sum(sum(y[i]) for i in range(2, max(seq))), GRB.MINIMIZE)  # type: ignore
    model.setObjective(sum(vars[x(i)] for i in range(2, max(seq) + 1)), GRB.MINIMIZE)  # type: ignore

    model.setParam('OutputFlag', 0)
    model.setParam('LogFile', 'gurobi.log')
    model.optimize()

    # for v in model.getVars():
    #     if v.X != 0:
    #         print('%s %g' % (v.VarName, v.X))
    # print('Obj: %g' % model.ObjVal)
    return int(model.ObjVal), time.monotonic() - start


def maxsat(target: int) -> Tuple[int, float]:
    start = time.monotonic()
    squaring_cost = 1.0
    sam_cost = math.ceil(math.log2(target)) * squaring_cost + hw(target) - 1
    chain = add_chain(target, None, sam_cost, squaring_cost, 'glucose421', 1, True, math.ceil(math.log2(target)), None)
    assert chain is not None
    return len(chain), time.monotonic() - start


if __name__ == "__main__":
    for exponent in range(11, 300, 20):
        objective_milp, time_milp = milp(exponent)
        objective_maxsat, time_maxsat = maxsat(exponent)
        assert objective_milp == objective_maxsat, f"{objective_milp} != {objective_maxsat}"
        print(exponent, time_milp, time_maxsat, time_milp / time_maxsat)
