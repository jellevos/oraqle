# Code generation API
!!! warning
    In this version of Oraqle, the API is still prone to changes. Paths and names can change between any version.
    
The easiest way is using:
```python3
arithmetic_circuit.generate_code()
```

## Arithmetic instructions
If you want to extend the oraqle compiler, or implement your own code generation, you can use the following instructions to do so.

??? info "Abstract instruction"
    ::: oraqle.compiler.instructions.ArithmeticInstruction
        options:
            heading_level: 3

??? info "InputInstruction"
    ::: oraqle.compiler.instructions.InputInstruction
        options:
            heading_level: 3

??? info "AdditionInstruction"
    ::: oraqle.compiler.instructions.AdditionInstruction
        options:
            heading_level: 3

??? info "MultiplicationInstruction"
    ::: oraqle.compiler.instructions.MultiplicationInstruction
        options:
            heading_level: 3

??? info "ConstantAdditionInstruction"
    ::: oraqle.compiler.instructions.ConstantAdditionInstruction
        options:
            heading_level: 3

??? info "ConstantMultiplicationInstruction"
    ::: oraqle.compiler.instructions.ConstantMultiplicationInstruction
        options:
            heading_level: 3

??? info "OutputInstruction"
    ::: oraqle.compiler.instructions.OutputInstruction
        options:
            heading_level: 3


## Generating arithmetic programs
::: oraqle.compiler.instructions.ArithmeticProgram
    options:
      heading_level: 3
   

## Generating code for HElib
...
