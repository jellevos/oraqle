"""This module contains `ArithmeticNode`s with a single input: Constant additions and constant multiplications."""
from typing import List, Optional, Sequence, Set, Tuple

from galois import FieldArray
from pysat.formula import WCNF, IDPool
from pysat.card import CardEnc, EncType

from oraqle.compiler.graphviz import DotFile
from oraqle.compiler.instructions import (
    ArithmeticInstruction,
    ConstantAdditionInstruction,
    ConstantMultiplicationInstruction,
)
from oraqle.compiler.nodes.abstract import ArithmeticNode, CostParetoFront, ExtendedArithmeticNode, Node, ExtendedArithmeticCosts, select_stack_index
from oraqle.compiler.nodes.univariate import UnivariateNode
from oraqle.mpc.parties import PartyId

# TODO: There is (going to be) a lot of code duplication between these two classes


class ConstantUnivariateArithmetic(UnivariateNode[ArithmeticNode], ArithmeticNode):

    def __init__(self, node: ArithmeticNode, is_constant_mul: bool):
        super().__init__(node)
        self._is_constant_mul = is_constant_mul

    def _add_constraints_minimize_cost_formulation_inner(self, wcnf: WCNF, id_pool: IDPool, costs: Sequence[ExtendedArithmeticCosts], parties: int, at_most_1_enc: Optional[int]):
        # TODO: Consider reducing duplication with bivariate arithmetic

        for party_id in range(parties):
            # We can compute a value if we hold both inputs
            compute_cost = self._computational_cost(costs, PartyId(party_id))
            computable = compute_cost < float('inf')
            if computable:
                c = id_pool.id(("c", id(self), party_id))
                h_operand = id_pool.id(("h", id(self._node), party_id))
                wcnf.append([-c, h_operand])

            h = id_pool.id(("h", id(self), party_id))

            # If we do not already know this value, then
            if not PartyId(party_id) in self._known_by:
                sources = []

                # We hold h if we compute it
                if computable:
                    sources.append(c)
                
                # Or when it is sent by another party
                for other_party_id in range(parties):
                    if party_id == other_party_id:
                        continue

                    receive_cost = costs[party_id].receive(PartyId(other_party_id))

                    if receive_cost < float('inf'):
                        received = id_pool.id(("s", id(self), other_party_id, party_id))
                        sources.append(received)

                        # Add the cost for receiving a value from other_party_id
                        wcnf.append([-received], weight=receive_cost)
                
                # Add a cut: we only want to compute/receive from one source
                if at_most_1_enc is not None:
                    at_most_1 = CardEnc.atmost(sources, encoding=at_most_1_enc, vpool=id_pool)  # type: ignore
                    wcnf.extend(at_most_1)

                # Add to WCNF
                sources.append(-h)
                wcnf.append(sources)

            # We can only send if we hold the value
            for other_party_id in range(parties):
                if party_id == other_party_id:
                    continue

                send_cost = costs[party_id].send(PartyId(other_party_id))

                if send_cost < float('inf'):
                    send = id_pool.id(("s", id(self), party_id, other_party_id))
                    wcnf.append([-send, h])

                    # Prevent mutual communication of the same element
                    receive = id_pool.id(("s", id(self), other_party_id, party_id))
                    wcnf.append([-send, -receive])

                    # Add the cost for sending a value to other_party_id
                    wcnf.append([-send], weight=send_cost)

            # Add the computation cost
            if computable:
                wcnf.append([-c], weight=compute_cost)
    
    def _replace_randomness_inner(self, party_count: int) -> ExtendedArithmeticNode:
        raise NotImplementedError("TODO")
    
    def _assign_to_cluster(self, graph_builder: DotFile, party_count: int, result: List[int], id_pool: IDPool):
        if not self._assigned_to_cluster:
            for party_id in range(party_count):
                node_id = self.to_graph(graph_builder)

                for other_party_id in range(party_count):
                    s = id_pool.id(("s", id(self), other_party_id, party_id))
                    if (s-1) < len(result) and result[s - 1] > 0:
                        print("I,", party_id, "received", self, "from", other_party_id)

                c = id_pool.id(("c", id(self), party_id))
                if result[c - 1] > 0:
                    print('assigning', self, 'to', party_id)
                    graph_builder.add_node_to_cluster(node_id, party_id)

            self._node._assign_to_cluster(graph_builder, party_count, result, id_pool)

            self._assigned_to_cluster = True


class ConstantAddition(ConstantUnivariateArithmetic):
    """This node represents a multiplication of another node with a constant."""

    @property
    def _overriden_graphviz_attributes(self) -> dict:
        return {"style": "rounded,filled", "fillcolor": "grey80"}

    @property
    def _node_shape(self) -> str:
        return "square"

    @property
    def _hash_name(self) -> str:
        return f"constant_add_{self._constant}"

    @property
    def _node_label(self) -> str:
        return "+"

    def __init__(self, node: ArithmeticNode, constant: FieldArray):
        """Represents the operation `constant + node`."""
        super().__init__(node, is_constant_mul=False)
        self._constant = constant
        assert constant != 0

        self._depth_cache: Optional[int] = None

    
    def _operation_inner(self, input: FieldArray) -> FieldArray:
        return input + self._constant

    
    def multiplicative_depth(self) -> int:  # noqa: D102
        if self._depth_cache is None:
            self._depth_cache = self._node.multiplicative_depth()

        return self._depth_cache

    
    def multiplications(self) -> Set[int]:  # noqa: D102
        return self._node.multiplications()

    
    def squarings(self) -> Set[int]:  # noqa: D102
        return self._node.squarings()

    
    def create_instructions(  # noqa: D102
        self,
        instructions: List[ArithmeticInstruction],
        stack_counter: int,
        stack_occupied: List[bool],
    ) -> Tuple[int, int]:
        self._node: ArithmeticNode

        if self._instruction_cache is None:
            operand_index, stack_counter = self._node.create_instructions(
                instructions, stack_counter, stack_occupied
            )

            self._node._parent_count -= 1
            if self._node._parent_count == 0:
                stack_occupied[self._node._instruction_cache] = False  # type: ignore

            self._instruction_cache = select_stack_index(stack_occupied)

            instructions.append(
                ConstantAdditionInstruction(self._instruction_cache, operand_index, self._constant)
            )

        return self._instruction_cache, stack_counter

    
    def _arithmetize_inner(self, strategy: str) -> Node:
        return self

    
    def _arithmetize_depth_aware_inner(self, cost_of_squaring: float) -> CostParetoFront:
        front = CostParetoFront(cost_of_squaring)
        for _, _, node in self._node.arithmetize_depth_aware(cost_of_squaring):
            front.add(ConstantAddition(node, self._constant))
        return front
    
    def _expansion(self) -> Node:
        raise NotImplementedError()
    
    def _computational_cost(self, costs: Sequence[ExtendedArithmeticCosts], party_id: PartyId) -> float:
        return costs[party_id].scalar_add
    
    def to_graph(self, graph_builder: DotFile) -> int:  # noqa: D102
        if self._to_graph_cache is None:
            super().to_graph(graph_builder)
            self._to_graph_cache: int

            # TODO: Add known_by
            graph_builder.add_link(
                graph_builder.add_node(
                    label=str(self._constant), shape="circle", style="filled", fillcolor="grey92"
                ),
                self._to_graph_cache,
            )

        return self._to_graph_cache


class ConstantMultiplication(ConstantUnivariateArithmetic):
    """This node represents a multiplication of another node with a constant."""

    @property
    def _overriden_graphviz_attributes(self) -> dict:
        return {"style": "rounded,filled", "fillcolor": "grey80"}

    @property
    def _node_shape(self) -> str:
        return "square"

    @property
    def _hash_name(self) -> str:
        return f"constant_mul_{self._constant}"

    @property
    def _node_label(self) -> str:
        return "×"  # noqa: RUF001

    def __init__(self, node: ArithmeticNode, constant: FieldArray):
        """Represents the operation `constant * node`."""
        super().__init__(node, is_constant_mul=True)
        self._constant = constant
        assert constant != 0
        assert constant != 1

        self._depth_cache: Optional[int] = None

    def _operation_inner(self, input: FieldArray) -> FieldArray:
        return input * self._constant  # type: ignore

    
    def multiplicative_depth(self) -> int:  # noqa: D102
        if self._depth_cache is None:
            self._depth_cache = self._node.multiplicative_depth()  # type: ignore

        return self._depth_cache  # type: ignore

    
    def multiplications(self) -> Set[int]:  # noqa: D102
        return self._node.multiplications()  # type: ignore

    
    def squarings(self) -> Set[int]:  # noqa: D102
        return self._node.squarings()  # type: ignore

    
    def create_instructions(  # noqa: D102
        self,
        instructions: List[ArithmeticInstruction],
        stack_counter: int,
        stack_occupied: List[bool],
    ) -> Tuple[int, int]:
        self._node: ArithmeticNode

        if self._instruction_cache is None:
            operand_index, stack_counter = self._node.create_instructions(
                instructions, stack_counter, stack_occupied
            )

            self._node._parent_count -= 1
            if self._node._parent_count == 0:
                stack_occupied[self._node._instruction_cache] = False  # type: ignore

            self._instruction_cache = select_stack_index(stack_occupied)

            instructions.append(
                ConstantMultiplicationInstruction(
                    self._instruction_cache, operand_index, self._constant
                )
            )

        return self._instruction_cache, stack_counter

    
    def _arithmetize_inner(self, strategy: str) -> Node:
        return self

    
    def _arithmetize_depth_aware_inner(self, cost_of_squaring: float) -> CostParetoFront:
        front = CostParetoFront(cost_of_squaring)
        for _, _, node in self._node.arithmetize_depth_aware(cost_of_squaring):
            front.add(ConstantMultiplication(node, self._constant))
        return front
    
    def _expansion(self) -> Node:
        raise NotImplementedError()

    def _computational_cost(self, costs: Sequence[ExtendedArithmeticCosts], party_id: PartyId) -> float:
        return costs[party_id].scalar_mul
    
    def to_graph(self, graph_builder: DotFile) -> int:  # noqa: D102
        if self._to_graph_cache is None:
            super().to_graph(graph_builder)
            self._to_graph_cache: int

            # TODO: Add known_by
            graph_builder.add_link(
                graph_builder.add_node(
                    label=str(self._constant), shape="circle", style="filled", fillcolor="grey92"
                ),
                self._to_graph_cache,
            )

        return self._to_graph_cache
