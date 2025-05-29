"""This module contains the classes that represent instructions and programs for evaluating arithmetic circuits."""
from abc import ABC, abstractmethod
import re
from typing import Dict, List, Optional, Tuple, Type

from galois import GF, FieldArray


class ArithmeticInstruction(ABC):
    """An abstract arithmetic instruction that computes an operation in an arithmetic circuit using a stack."""

    def __init__(self, stack_index: int) -> None:
        """Initialize an instruction that writes it output to the stack at `stack_index`."""
        self._stack_index = stack_index

    @abstractmethod
    def evaluate(
        self, stack: List[Optional[FieldArray]], inputs: Dict[str, FieldArray]
    ) -> Optional[FieldArray]:
        """Executes the instruction on plaintext inputs without using encryption, keeping track of the plaintext values in the stack."""

    @abstractmethod
    def generate_code(self, stack_initialized: List[bool], decrypt_outputs: bool) -> str:
        """Generates code for this instruction, keeping track of which places of the stack are already initialized."""

    @abstractmethod
    def generate_code_openfhe(self, stack_initialized: List[Tuple[bool, int]], decrypt_outputs: bool) -> str:
        """Generates OpenFHE code for this instruction, keeping track of which places of the stack are already initialized and the number of ciphertext components in them."""


class AdditionInstruction(ArithmeticInstruction):
    """Reads two elements from the stack, adds them, and writes the result to the stack."""

    def __init__(self, stack_index: int, left_stack_index: int, right_stack_index: int) -> None:
        """Initialize an instruction that adds the elements at `left_stack_index` and `right_stack_index`, placing the result at `stack_index`."""
        self._left_stack_index = left_stack_index
        self._right_stack_index = right_stack_index
        super().__init__(stack_index)

    def evaluate(self, stack: List[Optional[FieldArray]], _inputs: Dict[str, FieldArray]) -> None:  # noqa: D102
        left = stack[self._left_stack_index]
        right = stack[self._right_stack_index]
        assert left is not None
        assert right is not None
        stack[self._stack_index] = left + right

    def generate_code(self, stack_initialized: List[bool], _decrypt_outputs: bool) -> str:  # noqa: D102
        if self._left_stack_index == self._stack_index:
            return f"stack_{self._stack_index} += stack_{self._right_stack_index};\n"
        if self._right_stack_index == self._stack_index:
            return f"stack_{self._stack_index} += stack_{self._left_stack_index};\n"

        code = ""
        if not stack_initialized[self._stack_index]:
            code += "ctxt_t "
        code += f"stack_{self._stack_index} = stack_{self._left_stack_index};\nstack_{self._stack_index} += stack_{self._right_stack_index};\n"
        stack_initialized[self._stack_index] = True
        return code
    
    def generate_code_openfhe(self, stack_initialized: List[Tuple[bool, int]], decrypt_outputs: bool) -> str:
        if self._left_stack_index == self._stack_index:
            stack_initialized[self._stack_index] = (True, max(stack_initialized[self._left_stack_index][1], stack_initialized[self._right_stack_index][1]))
            return f"context->EvalAddInPlace(stack_{self._left_stack_index}, stack_{self._right_stack_index});\n"
        if self._right_stack_index == self._stack_index:
            stack_initialized[self._stack_index] = (True, max(stack_initialized[self._left_stack_index][1], stack_initialized[self._right_stack_index][1]))
            return f"context->EvalAddInPlace(stack_{self._right_stack_index}, stack_{self._left_stack_index});\n"

        code = ""
        if not stack_initialized[self._stack_index][0]:
            code += "ctxt_t "
        code += f"stack_{self._stack_index} = context->EvalAdd(stack_{self._left_stack_index}, stack_{self._right_stack_index});\n"
        stack_initialized[self._stack_index] = (True, max(stack_initialized[self._left_stack_index][1], stack_initialized[self._right_stack_index][1]))
        return code


class MultiplicationInstruction(ArithmeticInstruction):
    """Reads two elements from the stack, multiplies them, and writes the result to the stack."""

    def __init__(self, stack_index: int, left_stack_index: int, right_stack_index: int) -> None:
        """Initialize an instruction that multiplies the elements at `left_stack_index` and `right_stack_index`, placing the result at `stack_index`."""
        self._left_stack_index = left_stack_index
        self._right_stack_index = right_stack_index
        super().__init__(stack_index)

    def evaluate(self, stack: List[Optional[FieldArray]], _inputs: Dict[str, FieldArray]) -> None:  # noqa: D102
        left = stack[self._left_stack_index]
        right = stack[self._right_stack_index]
        assert left is not None
        assert right is not None
        stack[self._stack_index] = left * right

    def generate_code(self, stack_initialized: List[bool], _decrypt_outputs: bool) -> str:  # noqa: D102
        if self._left_stack_index == self._stack_index:
            return f"stack_{self._stack_index} *= stack_{self._right_stack_index};\n"
        if self._right_stack_index == self._stack_index:
            return f"stack_{self._stack_index} *= stack_{self._left_stack_index};\n"

        code = ""
        if not stack_initialized[self._stack_index]:
            code += "ctxt_t "
        code += f"stack_{self._stack_index} = stack_{self._left_stack_index};\nstack_{self._stack_index} *= stack_{self._right_stack_index};\n"
        stack_initialized[self._stack_index] = True
        return code
    
    def generate_code_openfhe(self, stack_initialized: List[Tuple[bool, int]], decrypt_outputs: bool) -> str:
        # if self._left_stack_index == self._stack_index:
        #     return f"context->EvalAddInPlace(stack_{self._left_stack_index}, stack_{self._right_stack_index});\n"
        # if self._right_stack_index == self._stack_index:
        #     return f"context->EvalAddInPlace(stack_{self._right_stack_index}, stack_{self._left_stack_index});\n"

        code = ""
        if stack_initialized[self._left_stack_index][1] > 2:
            stack_initialized[self._left_stack_index] = (True, 2)
            code += f"context->RelinearizeInPlace(stack_{self._left_stack_index});\n"

        if stack_initialized[self._right_stack_index][1] > 2:
            stack_initialized[self._right_stack_index] = (True, 2)
            code += f"context->RelinearizeInPlace(stack_{self._right_stack_index});\n"

        if not stack_initialized[self._stack_index][0]:
            code += "ctxt_t "
        code += f"stack_{self._stack_index} = context->EvalMultNoRelin(stack_{self._left_stack_index}, stack_{self._right_stack_index});\n"
        stack_initialized[self._stack_index] = (True, 3)
        return code


class ConstantAdditionInstruction(ArithmeticInstruction):
    """Reads an element from the stack, adds a constant to it it, and writes the result to the stack."""

    def __init__(self, stack_index: int, input_stack_index: int, constant: FieldArray) -> None:
        """Initialize an instruction that adds `constant` to the element at `input_stack_index`, placing the result at `stack_index`."""
        self._input_stack_index = input_stack_index
        self._constant = constant
        super().__init__(stack_index)

    def evaluate(self, stack: List[Optional[FieldArray]], _inputs: Dict[str, FieldArray]) -> None:  # noqa: D102
        operand = stack[self._input_stack_index]
        assert operand is not None
        stack[self._stack_index] = operand + self._constant

    def generate_code(self, stack_initialized: List[bool], _decrypt_outputs: bool) -> str:  # noqa: D102
        if self._stack_index == self._input_stack_index:
            return f"stack_{self._input_stack_index} += int64_t({self._constant});\n"

        code = ""
        if not stack_initialized[self._stack_index]:
            code += "ctxt_t "
        code += f"stack_{self._stack_index} = stack_{self._input_stack_index};\nstack_{self._stack_index} += int64_t({self._constant});\n"
        stack_initialized[self._stack_index] = True
        return code
    
    def generate_code_openfhe(self, stack_initialized: List[Tuple[bool, int]], _decrypt_outputs: bool) -> str:  # noqa: D102
        plaintext = f"context->MakePackedPlaintext({{ int64_t({self._constant}) }})"
        if self._stack_index == self._input_stack_index:
            stack_initialized[self._stack_index] = (True, stack_initialized[self._input_stack_index][1])
            return f"context->EvalAddInPlace(stack_{self._input_stack_index}, {plaintext});\n"

        code = ""
        if not stack_initialized[self._stack_index][0]:
            code += "ctxt_t "
        code += f"stack_{self._stack_index} = context->EvalAdd(stack_{self._input_stack_index}, {plaintext});\n"
        stack_initialized[self._stack_index] = (True, stack_initialized[self._input_stack_index][1])
        return code


class ConstantMultiplicationInstruction(ArithmeticInstruction):
    """Reads an element from the stack, multiplies it with a constant, and writes the result to the stack."""

    def __init__(self, stack_index: int, input_stack_index: int, constant: FieldArray) -> None:
        """Initialize an instruction that multiplies the element at `input_stack_index` with `constant`, placing the result at `stack_index`."""
        self._input_stack_index = input_stack_index
        self._constant = constant
        super().__init__(stack_index)

    def evaluate(self, stack: List[Optional[FieldArray]], _inputs: Dict[str, FieldArray]) -> None:  # noqa: D102
        operand = stack[self._input_stack_index]
        assert operand is not None
        stack[self._stack_index] = operand * self._constant

    def generate_code(self, stack_initialized: List[bool], _decrypt_outputs: bool) -> str:  # noqa: D102
        if self._stack_index == self._input_stack_index:
            return f"stack_{self._input_stack_index} *= int64_t({self._constant});\n"

        code = ""
        if not stack_initialized[self._stack_index]:
            code += "ctxt_t "
        code += f"stack_{self._stack_index} = stack_{self._input_stack_index};\nstack_{self._stack_index} *= int64_t({self._constant});\n"
        stack_initialized[self._stack_index] = True
        return code
    
    def generate_code_openfhe(self, stack_initialized: List[Tuple[bool, int]], _decrypt_outputs: bool) -> str:  # noqa: D102
        # if self._stack_index == self._input_stack_index:
        #     return f"context->EvalAddInPlace(stack_{self._input_stack_index}, {self._constant}l);\n"
        plaintext = f"context->MakePackedPlaintext({{ int64_t({self._constant}) }})"

        code = ""
        if stack_initialized[self._input_stack_index][1] > 2:
            stack_initialized[self._input_stack_index] = (True, 2)
            code += f"context->RelinearizeInPlace(stack_{self._input_stack_index});\n"

        if not stack_initialized[self._stack_index][0]:
            code += "ctxt_t "
        code += f"stack_{self._stack_index} = context->EvalMult(stack_{self._input_stack_index}, {plaintext});\n"
        stack_initialized[self._stack_index] = (True, stack_initialized[self._input_stack_index][1])
        return code


class InputInstruction(ArithmeticInstruction):
    """Writes an input to the stack."""

    def __init__(self, stack_index: int, name: str) -> None:
        """Initialize an `InputInstruction` that places the input with the given `name` in the stack at index `stack_index`."""
        self._name = name
        super().__init__(stack_index)

    def evaluate(self, stack: List[Optional[FieldArray]], inputs: Dict[str, FieldArray]) -> None:  # noqa: D102
        stack[self._stack_index] = inputs[self._name]

    def generate_code(self, stack_initialized: List[bool], _decrypt_outputs: bool) -> str:  # noqa: D102
        code = ""
        if not stack_initialized[self._stack_index]:
            code += "ctxt_t "
        code += f"stack_{self._stack_index} = ciph_{self._name};\n"
        stack_initialized[self._stack_index] = True
        return code
    
    def generate_code_openfhe(self, stack_initialized: List[Tuple[bool, int]], _decrypt_outputs: bool) -> str:  # noqa: D102
        code = ""
        if not stack_initialized[self._stack_index][0]:
            code += "ctxt_t "
        code += f"stack_{self._stack_index} = ciph_{self._name};\n"
        stack_initialized[self._stack_index] = (True, 2)
        return code


class OutputInstruction(ArithmeticInstruction):
    """Outputs an element from the stack."""

    def evaluate(self, stack: List[FieldArray], _inputs: Dict[str, FieldArray]) -> FieldArray:  # noqa: D102
        return stack[self._stack_index]

    def generate_code(self, stack_initialized: List[bool], decrypt_outputs: bool) -> str:  # noqa: D102
        if decrypt_outputs:
            return f"ptxt_t decrypted(context);\nsecret_key.Decrypt(decrypted, stack_{self._stack_index});\nstd::cout << decrypted << std::endl;\n"
        else:
            return f'std::cout << "Output correctness: " << stack_{self._stack_index}.isCorrect() << std::endl;\n'

    def generate_code_openfhe(self, stack_initialized: List[Tuple[bool, int]], decrypt_outputs: bool) -> str:  # noqa: D102
        if decrypt_outputs:
            return f"ptxt_t decrypted;\ncontext->Decrypt(keys.secretKey, stack_{self._stack_index}, &decrypted);\nstd::cout << decrypted << std::endl;\n"
        else:
            return f'std::cout << "Done!" << std::endl;\n'


class ArithmeticProgram:
    """An ArithmeticProgram represents an ordered set of arithmetic operations that compute an arithmetic circuit.
    
    The easiest way to obtain an `ArithmeticProgram` of an `ArithmeticCircuit` is to call `ArithmeticCircuit.generate_program()`.
    """

    def __init__(
        self, instructions: List[ArithmeticInstruction], stack_size: int, gf: Type[FieldArray]
    ) -> None:
        """Initialize an `ArithmeticProgram` from a list of `instructions`.
        
        The user must specify an upper bound on the `stack_size` required.
        """
        self._instructions = instructions
        self._stack_size = stack_size
        self._gf = gf

    def execute(self, inputs: Dict[str, FieldArray]) -> FieldArray:
        """Executes the arithmetic program on plaintext inputs without using encryption.
        
        Raises:
            Exception: If there were no outputs in this program.
            
        Returns:
        The first output in this program.
        """
        # FIXME: Currently only supports a single output
        for input in inputs.values():
            assert isinstance(input, self._gf)

        stack: List[Optional[FieldArray]] = [None for _ in range(self._stack_size)]

        for instruction in self._instructions:
            if (output := instruction.evaluate(stack, inputs)) is not None:
                return output

        raise Exception("The program did not output anything")

    def generate_code(self, decrypt_outputs: bool) -> str:
        """Generates HElib code for this program.

        If `decrypt_outputs` is true, then the generated code will decrypt the outputs at the end of the circuit.

        Returns:
            The generated code as a string.
        """
        code = ""
        stack_initialized = [False] * self._stack_size

        for instruction in self._instructions:
            code += instruction.generate_code(stack_initialized, decrypt_outputs)

        return code
    
    def generate_code_openfhe(self, decrypt_outputs: bool) -> str:
        """Generates OpenFHE code for this program.

        If `decrypt_outputs` is true, then the generated code will decrypt the outputs at the end of the circuit.

        Returns:
            The generated code as a string.
        """
        code = ""
        stack_initialized = [(False, 2)] * self._stack_size

        for instruction in self._instructions:
            code += instruction.generate_code_openfhe(stack_initialized, decrypt_outputs)

        return code
    
    def generate_code_chunked(self, decrypt_outputs: bool, chunk_size: int) -> Tuple[str, List[Tuple[str, str]]]:
        """Generates HElib code for this program.

        If `decrypt_outputs` is true, then the generated code will decrypt the outputs at the end of the circuit.

        Returns:
            The generated code as a string.
        """
        functions = []

        # Split circuit into chunks (functions)
        for i in range(0, len(self._instructions), chunk_size):
            stack_initialized = [False] * self._stack_size
            chunk = self._instructions[i : i + chunk_size]

            code = f"void chunk_{i // chunk_size}(std::vector<ctxt_t>& ciphertexts, std::vector<ctxt_t>& stack) {{\n"
            for instruction in chunk:
                line = instruction.generate_code(stack_initialized, decrypt_outputs)
                line = re.sub(r'stack_(\d+)', r'stack(\1)', line)
                line = re.sub(r'ciph_(\d+)', r'ciphertexts(\1)', line)
                code += line
            code += "}\n"
            functions.append((f"chunk_{i // chunk_size}", code))

        # Create one function that calls all the chunks
        calling_code = "void evaluate_program(std::vector<ctxt_t>& ciphertexts, std::vector<ctxt_t>& stack) {\n"
        for func_name, _ in functions:
            calling_code += f"    {func_name}(ciphertexts, stack);\n"
        calling_code += "}\n"

        return calling_code, functions


def test_instructions_small_comparison():  # noqa: D103
    from oraqle.compiler.circuit import Circuit
    from oraqle.compiler.nodes.leafs import Input

    gf = GF(7)

    x = Input("x", gf)
    y = Input("y", gf)

    arithmetic_circuit = Circuit([x < y]).arithmetize()
    program = arithmetic_circuit.generate_program()

    for x in range(7):
        for y in range(7):
            inputs = {"x": gf(x), "y": gf(y)}
            assert arithmetic_circuit.evaluate(inputs) == program.execute(inputs)


if __name__ == "__main__":
    from oraqle.compiler.circuit import Circuit
    from oraqle.compiler.nodes.leafs import Input

    gf = GF(7)

    x = Input("x", gf)
    y = Input("y", gf)

    arithmetic_circuit = Circuit([x < y]).arithmetize()
    program = arithmetic_circuit.generate_program()

    calling, funcs = program.generate_code_chunked(True, chunk_size=10)
    print(calling)
    for _, func in funcs:
        print(func)
        print()
