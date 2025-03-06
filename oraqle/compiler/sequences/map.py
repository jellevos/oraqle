from oraqle.compiler.nodes.fp.abstract import Node
from oraqle.compiler.sequences.packed import PackedSequence


class Map(PackedSequence):

    # TODO: Maybe we should use ZpxNode?
    # TODO: For the operation, we should implement PlaceholderInputs which can be mapped to the inputs of the map.
    def __init__(self, sequence: PackedSequence, operation: Node) -> None:
        self._sequence = sequence
        self._operation = operation
        super().__init__(sequence._length, sequence._stride, sequence._offset)
