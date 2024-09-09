"""This module implements the cardio circuit that is often used in benchmarking compilers, see: https://arxiv.org/abs/2101.07078."""
from typing import Type
from galois import GF, FieldArray

from oraqle.compiler.boolean.bool_neg import Neg
from oraqle.compiler.boolean.bool_or import any_
from oraqle.compiler.circuit import Circuit
from oraqle.compiler.nodes import Input
from oraqle.compiler.nodes.abstract import Node
from oraqle.compiler.nodes.arbitrary_arithmetic import sum_


def construct_cardio_risk_circuit(gf: Type[FieldArray]) -> Node:
    """Returns the cardio circuit from https://arxiv.org/abs/2101.07078."""
    man = Input("man", gf)
    woman = Input("woman", gf)
    smoking = Input("smoking", gf)
    age = Input("age", gf)
    diabetic = Input("diabetic", gf)
    hbp = Input("hbp", gf)
    cholesterol = Input("cholesterol", gf)
    weight = Input("weight", gf)
    height = Input("height", gf)
    activity = Input("activity", gf)
    alcohol = Input("alcohol", gf)

    return sum_(
        man & (age > 50),
        woman & (age > 60),
        smoking,
        diabetic,
        hbp,
        cholesterol < 40,
        weight > (height - 90),  # This might underflow if the modulus is too small
        activity < 30,
        man & (alcohol > 3),
        Neg(man, gf) & (alcohol > 2),
    )


def construct_cardio_elevated_risk_circuit(gf: Type[FieldArray]) -> Node:
    """Returns a variant of the cardio circuit that returns a Boolean indicating whether any risk factor returned true."""
    man = Input("man", gf)
    woman = Input("woman", gf)
    smoking = Input("smoking", gf)
    age = Input("age", gf)
    diabetic = Input("diabetic", gf)
    hbp = Input("hbp", gf)
    cholesterol = Input("cholesterol", gf)
    weight = Input("weight", gf)
    height = Input("height", gf)
    activity = Input("activity", gf)
    alcohol = Input("alcohol", gf)

    return any_(
        man & (age > 50),
        woman & (age > 60),
        smoking,
        diabetic,
        hbp,
        cholesterol < 40,
        weight > (height - 90),  # This might underflow if the modulus is too small
        activity < 30,
        man & (alcohol > 3),
        Neg(man, gf) & (alcohol > 2),
    )


def test_cardio_p101():  # noqa: D103
    gf = GF(101)
    circuit = Circuit([construct_cardio_risk_circuit(gf)])

    for _, _, arithmetization in circuit.arithmetize_depth_aware():
        assert arithmetization.evaluate({
                "man": gf(1),
                "woman": gf(0),
                "age": gf(50),
                "smoking": gf(0),
                "diabetic": gf(0),
                "hbp": gf(0),
                "cholesterol": gf(45),
                "weight": gf(10),
                "height": gf(100),
                "activity": gf(90),
                "alcohol": gf(3),
            })[0] == 0
        
        assert arithmetization.evaluate({
                "man": gf(0),
                "woman": gf(1),
                "age": gf(50),
                "smoking": gf(0),
                "diabetic": gf(0),
                "hbp": gf(0),
                "cholesterol": gf(45),
                "weight": gf(10),
                "height": gf(100),
                "activity": gf(90),
                "alcohol": gf(3),
            })[0] == 1
        
        assert arithmetization.evaluate({
                "man": gf(1),
                "woman": gf(0),
                "age": gf(50),
                "smoking": gf(0),
                "diabetic": gf(0),
                "hbp": gf(0),
                "cholesterol": gf(39),
                "weight": gf(10),
                "height": gf(100),
                "activity": gf(90),
                "alcohol": gf(3),
            })[0] == 1
        
        assert arithmetization.evaluate({
                "man": gf(1),
                "woman": gf(0),
                "age": gf(50),
                "smoking": gf(1),
                "diabetic": gf(0),
                "hbp": gf(0),
                "cholesterol": gf(45),
                "weight": gf(10),
                "height": gf(100),
                "activity": gf(90),
                "alcohol": gf(3),
            })[0] == 1
