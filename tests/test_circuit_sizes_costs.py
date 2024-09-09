"""Test file for testing circuits sizes."""

from collections import Counter

from galois import GF

from oraqle.compiler.nodes.abstract import ArithmeticNode, UnoverloadedWrapper
from oraqle.compiler.nodes.arbitrary_arithmetic import Sum
from oraqle.compiler.nodes.leafs import Constant, Input


def test_size_exponentiation_chain():
    """Test."""
    gf = GF(101)

    x = Input("x", gf)

    x = x.mul(x, flatten=False)
    x = x.mul(x, flatten=False)
    x = x.mul(x, flatten=False)

    x = x.to_arithmetic()
    assert isinstance(x, ArithmeticNode)
    assert (
        x.multiplicative_size() == 3
    ), f"((x^2)^2)^2 should be 3 multiplications, but counted {x.multiplicative_size()}"
    assert x.multiplicative_cost(0.5) == 1.5


def test_size_sum_of_products():
    """Test."""
    gf = GF(101)

    a = Input("a", gf)
    b = Input("b", gf)
    c = Input("c", gf)
    d = Input("d", gf)

    ab = a * b
    cd = c * d

    out = ab + cd
    out = out.to_arithmetic()

    assert isinstance(out, ArithmeticNode)
    assert (
        out.multiplicative_size() == 2
    ), f"a * b + c * d should be 2 multiplications, but counted {out.multiplicative_size()}"
    assert out.multiplicative_cost(0.7) == 2


def test_size_linear_function():
    """Test."""
    gf = GF(101)

    a = Input("a", gf)
    b = Input("b", gf)
    c = Input("c", gf)

    out = Sum(
        Counter({UnoverloadedWrapper(a): 1, UnoverloadedWrapper(b): 3, UnoverloadedWrapper(c): 1}),
        gf,
        gf(2),
    )

    out = out.to_arithmetic()
    assert out.multiplicative_size() == 0
    assert out.multiplicative_cost(0.5) == 0


def test_size_duplicate_nodes():
    """Test."""
    gf = GF(101)

    x = Input("x", gf)

    add1 = x.add(Constant(gf(1)))
    add2 = x.add(Constant(gf(1)))

    mul1 = x.mul(x, flatten=False)
    mul2 = x.mul(x, flatten=False)

    add3 = mul2.add(add2, flatten=False)

    mul3 = mul1.mul(add3, flatten=False)

    out = add1.add(mul3, flatten=False)

    out = out.to_arithmetic()

    assert isinstance(out, ArithmeticNode)
    assert out.multiplicative_size() == 3
    assert out.multiplicative_cost(0.7) == 2.4
