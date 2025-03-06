from oraqle.compiler.sequences.packed import PackedSequence


class Even(PackedSequence):

    def __init__(self, sequence: PackedSequence) -> None:
        self._sequence = sequence
        super().__init__(sequence._length // 2, sequence._stride * 2, sequence._offset)


class Odd(PackedSequence):

    def __init__(self, sequence: PackedSequence) -> None:
        self._sequence = sequence
        super().__init__(sequence._length // 2, sequence._stride * 2, sequence._offset + sequence._stride)
