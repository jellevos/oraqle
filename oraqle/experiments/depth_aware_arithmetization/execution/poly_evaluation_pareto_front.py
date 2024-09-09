import math
import sys

from galois import GF
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator

from oraqle.compiler.circuit import Circuit
from oraqle.compiler.nodes.abstract import SizeParetoFront
from oraqle.compiler.nodes.leafs import Input
from oraqle.compiler.polynomials.univariate import (
    UnivariatePoly,
    _eval_poly,
    _eval_poly_alternative,
    _eval_poly_divide_conquer,
)

if __name__ == "__main__":
    sys.setrecursionlimit(15000)

    shape_size = 150

    plt.figure(figsize=(3.5, 4.4))

    marker1 = (3, 2, 0)
    marker2 = (3, 2, 40)
    marker3 = (3, 2, 80)
    o_marker = "o"
    linewidth = 2.5

    squaring_cost = 1.0

    p = 127  # 31
    gf = GF(p)
    for d in [p - 1]:
        x = Input("x", gf)

        poly = UnivariatePoly.from_function(x, gf, lambda x: x % 7)
        coefficients = poly._coefficients

        # Generate points
        print("Paterson & Stockmeyer")
        depths = []
        sizes = []

        front = SizeParetoFront()

        for k in range(1, len(coefficients)):
            res, pows = _eval_poly(x, coefficients, k, gf, squaring_cost)
            circ = Circuit([res]).arithmetize()
            depths.append(circ.multiplicative_depth())
            sizes.append(circ.multiplicative_size())
            front.add(res, circ.multiplicative_depth(), circ.multiplicative_size())  # type: ignore
            print(k, circ.multiplicative_depth(), circ.multiplicative_size())

        data = {(d, s) for d, s in zip(depths, sizes)}
        plt.scatter(
            [d for d, _ in data],
            [s for _, s in data],
            marker=marker2,  # type: ignore
            zorder=10,
            alpha=0.4,
            s=shape_size,
            linewidth=linewidth,
        )

        print("Baby-step giant-step")
        depths2 = []
        sizes2 = []
        for k in range(1, len(coefficients)):
            res, pows = _eval_poly_alternative(x, coefficients, k, gf)
            circ = Circuit([res]).arithmetize()
            depths2.append(circ.multiplicative_depth())
            sizes2.append(circ.multiplicative_size())
            front.add(res, circ.multiplicative_depth(), circ.multiplicative_size())  # type: ignore

        data2 = {(d, s) for d, s in zip(depths2, sizes2)}
        plt.scatter(
            [d for d, _ in data2],
            [s for _, s in data2],
            marker=marker1,  # type: ignore
            zorder=11,
            alpha=0.45,
            s=shape_size,
            linewidth=linewidth,
        )

        print("Divide and conquer")
        depths3 = []
        sizes3 = []
        for k in range(1, len(coefficients)):
            res, pows = _eval_poly_divide_conquer(x, coefficients, k, gf, squaring_cost)
            circ = Circuit([res]).arithmetize()
            depths3.append(circ.multiplicative_depth())
            sizes3.append(circ.multiplicative_size())
            front.add(res, circ.multiplicative_depth(), circ.multiplicative_size())  # type: ignore

        data3 = {(d, s) for d, s in zip(depths3, sizes3)}
        plt.scatter(
            [d for d, _ in data3],
            [s for _, s in data3],
            marker=marker3,  # type: ignore
            zorder=11,
            alpha=0.45,
            s=shape_size,
            linewidth=linewidth,
        )

        # Plot the front
        front_initial = [(d, s) for d, s in data2 if d in front._nodes_by_depth and front._nodes_by_depth[d][0] == s]  # type: ignore
        front_advanced = [(d, s) for d, s in data if d in front._nodes_by_depth and front._nodes_by_depth[d][0] == s]  # type: ignore
        front_divconq = [(d, s) for d, s in data3 if d in front._nodes_by_depth and front._nodes_by_depth[d][0] == s]  # type: ignore

        plt.scatter(
            [d for d, _ in front_initial],
            [s for _, s in front_initial],
            marker=marker1,  # type: ignore
            zorder=10,
            color="tab:orange",
            s=shape_size,
            label="Baby-step giant-step",
            linewidth=linewidth,
        )
        plt.scatter(
            [d for d, _ in front_advanced],
            [s for _, s in front_advanced],
            marker=marker2,  # type: ignore
            zorder=10,
            color="tab:blue",
            s=shape_size,
            label="Paterson & Stockmeyer",
            linewidth=linewidth,
        )
        plt.scatter(
            [d for d, _ in front_divconq],
            [s for _, s in front_divconq],
            marker=marker3,  # type: ignore
            zorder=10,
            color="tab:green",
            s=shape_size,
            label="Divide & Conquer",
            linewidth=linewidth,
        )

        k = round(math.sqrt(d / 2))
        res, pows = _eval_poly(x, coefficients, k, gf, squaring_cost)
        circ = Circuit([res]).arithmetize()
        plt.scatter(
            circ.multiplicative_depth(),
            circ.multiplicative_size(),
            marker=o_marker,
            s=shape_size + 50,
            facecolors="none",
            edgecolors="black",
        )
        plt.text(
            circ.multiplicative_depth(),
            circ.multiplicative_size() + 0.4,
            f"k = {k}",
            ha="center",
            fontsize=8,
        )

        plt.xlim((5, 15))
        plt.ylim((15, 30))

        plt.gca().set_aspect("equal")

        plt.gca().xaxis.set_minor_locator(MultipleLocator(1))
        plt.gca().yaxis.set_minor_locator(MultipleLocator(1))

        plt.grid(True, which="both", zorder=1, alpha=0.5)

        plt.xlabel("Multiplicative depth")
        plt.ylabel("Multiplicative size")

        plt.legend(fontsize="small")

        plt.savefig("poly_eval_front_2.pdf", bbox_inches="tight")
        plt.show()
