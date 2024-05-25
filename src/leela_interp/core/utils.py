_LETTERS = {letter: i for i, letter in enumerate("abcdefgh")}


def sq2idx(sq: str, turn: bool):
    file, row = sq
    file = _LETTERS[file]
    if not turn:
        # Black's turn
        row = 9 - int(row)
    return (int(row) - 1) * 8 + file


_SQUARES = [f"{file}{rank}" for file in _LETTERS for rank in range(1, 9)]

_IDX2SQ = {
    True: {sq2idx(sq, True): sq for sq in _SQUARES},
    False: {sq2idx(sq, False): sq for sq in _SQUARES},
}


def idx2sq(idx: int, turn: bool):
    return _IDX2SQ[turn][idx]
