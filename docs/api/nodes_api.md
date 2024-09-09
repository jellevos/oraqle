# Nodes API
!!! warning
    In this version of Oraqle, the API is still prone to changes. Paths and names can change between any version.

## Boolean operations

??? info "AND operation"
    ::: oraqle.compiler.boolean.bool_and.And
        options:
            heading_level: 3

??? info "OR operation"
    ::: oraqle.compiler.boolean.bool_or.Or
        options:
            heading_level: 3

??? info "NEG operation"
    ::: oraqle.compiler.boolean.bool_neg.Neg
        options:
            heading_level: 3


## Arithmetic operations
These operations are fundamental arithmetic operations, so they will stay the same when they are arithmetized.


## High-level arithmetic operations

??? info "Subtraction"
    ::: oraqle.compiler.arithmetic.subtraction.Subtraction
        options:
            heading_level: 3

??? info "Exponentiation"
    ::: oraqle.compiler.arithmetic.exponentiation.Power
        options:
            heading_level: 3


## Polynomial evaluation

??? info "Univariate polynomial evaluation"
    ::: oraqle.compiler.polynomials.univariate.UnivariatePoly
        options:
            heading_level: 3


## Control flow

??? info "If-else statement"
    ::: oraqle.compiler.control_flow.conditional.IfElse
        options:
            heading_level: 3
