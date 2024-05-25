import argparse
import pickle
from pathlib import Path

import chess
import pandas as pd
import torch
import tqdm
from leela_interp import Lc0Model, LeelaBoard


def corruption_generator(board):
    """Generates small changes to a board.

    Returns a generator of (description, new_board) tuples.

    The change is one of:
    - Removing or adding a single pawn of either color.
    - Moving a single non-pawn piece to an empty square.

    Of course there are other "small" changes we could make (such as moving pawns)
    but these options give us at least one good corruption in most positions already.
    """
    # Add or remove pawns
    for i in range(64):
        piece_type = board.piece_type_at(i)
        if piece_type == chess.PAWN:
            new_board = board.copy()
            new_board.remove_piece_at(i)
            yield f"remove {chess.SQUARE_NAMES[i]}", new_board
        elif piece_type is None:
            new_board = board.copy()
            new_board.set_piece_at(i, chess.Piece(chess.PAWN, chess.WHITE))
            yield f"add white {chess.SQUARE_NAMES[i]}", new_board
            new_board = board.copy()
            new_board.set_piece_at(i, chess.Piece(chess.PAWN, chess.BLACK))
            yield f"add black {chess.SQUARE_NAMES[i]}", new_board

    # Move non-pawn pieces to other squares (we're not doing pawns just to save compute)
    for source in range(64):
        piece = board.piece_at(source)
        if piece is None or piece.piece_type == chess.PAWN:
            continue
        for target in range(64):
            # Make sure target square is empty
            if board.piece_type_at(target) is not None:
                continue

            new_board = board.copy()
            new_board.set_piece_at(target, new_board.remove_piece_at(source))
            yield f"{chess.SQUARE_NAMES[source]}{chess.SQUARE_NAMES[target]}", new_board


def get_corruptions(board: chess.Board, required_legal_move):
    """Generate corruptions of a board that lead to valid positions and preserve
    a given move as legal.

    (We later want to make sure that corruptions don't just make the previous top move
    illegal.)
    """
    rv = {}
    required_legal_move = chess.Move.from_uci(required_legal_move)

    # Add or remove individual pieces
    for key, new_board in corruption_generator(board):
        # Check if the board is still valid.
        if not new_board.is_valid():
            continue
        # Check if required_legal_move is still legal.
        if required_legal_move not in new_board.legal_moves:
            continue

        rv[key] = new_board

    return rv


def value_fn(wdl):
    """Compute win probability minus loss probability, as a metric for the value of a position."""
    if isinstance(wdl, tuple):
        return wdl[0] - wdl[2]
    else:
        return wdl[..., 0] - wdl[..., 2]


class CandidateCorruptionsDataset(torch.utils.data.Dataset):
    def __init__(self, puzzles: pd.DataFrame):
        self.puzzles = puzzles
        # self.batch_sizes = [64, 128, 256, 512, 1024]

    def __len__(self):
        return len(self.puzzles)

    def __getitem__(self, idx):
        puzzle = self.puzzles.iloc[idx]
        board = LeelaBoard.from_puzzle(puzzle)
        move = puzzle.principal_variation[0]

        corruptions = get_corruptions(board.pc_board, move)
        corruptions = [LeelaBoard.from_fen(v.fen()) for v in corruptions.values()]
        # for batch_size in self.batch_sizes:
        #     if batch_size >= len(corruptions):
        #         break
        # missing = batch_size - len(corruptions)
        # if missing < 0:
        #     corruptions = corruptions[:batch_size]
        # corruptions += [LeelaBoard() for _ in range(missing)]
        return board, corruptions


# @torch.compile(fullgraph=True)
def compute_scores(
    corrupted_policy,
    original_sparring_policy,
    sparring_corrupted_policy,
    original_wdl,
    corrupted_wdl,
    move_idx,
):
    # 1. We don't want to use cases where the probability of the correct move under the
    # sparring model decreases too much, even if it was already low. These corruptions
    # often correspond to making the move bad for "obvious" reasons (e.g. placing an
    # opponent pawn that attacks the target square of the move). We later minimize JSD
    # between clean and corrupted sparring policy, but if the probability of the top
    # move was already very low, it might not have a big enough effect on that.
    sparring_log_odds = torch.log(
        original_sparring_policy[move_idx] / (1 - original_sparring_policy[move_idx])
    )
    sparring_corrupted_log_odds = torch.log(
        sparring_corrupted_policy[:, move_idx]
        / (1 - sparring_corrupted_policy[:, move_idx])
    )
    sparring_decrease = sparring_log_odds[None] - sparring_corrupted_log_odds
    mask = (sparring_decrease < 0.2).float()

    # 2. We don't want the corruption to make the position *better*, that would indicate
    # that there's just some other move now that's even better, rather than the previous
    # move having gotten worse.
    value = value_fn(original_wdl)
    corrupted_value = value_fn(corrupted_wdl)
    value_change = corrupted_value - value
    mask *= (value_change < -0.1).float()

    # 3. We also want the new probability to be reasonably low, i.e. the previous top
    # move should now be bad. This is necessary to make activation patching useful
    # at all (if the corruption doesn't change the best move by much, there's no reason
    # to expect a clear effect from patching).
    mask *= (corrupted_policy[:, move_idx] < 0.1).float()

    # Finally, pick the corruption with the lowest JSD between the sparring policy on
    # the original and corrupted position.
    original_sparring_policy = original_sparring_policy.unsqueeze(0)
    M = (original_sparring_policy + sparring_corrupted_policy) / 2
    # This indirect way is just to compute the JSD in a way that's robust to zeros.
    kl_terms_1 = original_sparring_policy * torch.log(original_sparring_policy / M)
    kl_terms_2 = sparring_corrupted_policy * torch.log(sparring_corrupted_policy / M)
    kl_terms_1 = torch.where(original_sparring_policy > 0, kl_terms_1, 0.0)
    kl_terms_2 = torch.where(sparring_corrupted_policy > 0, kl_terms_2, 0.0)
    kl1 = torch.sum(kl_terms_1, dim=1)
    kl2 = torch.sum(kl_terms_2, dim=1)
    jsd = 0.5 * (kl1 + kl2)

    assert jsd.shape == value_change.shape, (jsd.shape, value_change.shape)

    scores = torch.where(mask.bool(), jsd, float("inf"))
    return scores.topk(1, largest=False)


def get_best_corruption(
    original_board: LeelaBoard,
    corruptions: list[LeelaBoard],
    big_model: Lc0Model,
    sparring_model: Lc0Model,
):
    # If for some reason a board has no valid corruptions, just skip it.
    if len(corruptions) == 0:
        return None

    # Insert the original board into the corruptions so we can do everything in one
    # batch.
    corruptions.append(original_board)

    policy, wdl, _ = big_model.batch_play(corruptions)
    sparring_policy, *_ = sparring_model.batch_play(corruptions)

    original_policy = policy[-1]
    original_wdl = wdl[-1]
    original_sparring_policy = sparring_policy[-1]

    corrupted_policy = policy[:-1]
    corrupted_wdl = wdl[:-1]
    sparring_corrupted_policy = sparring_policy[:-1]

    # Find the top move according to the big model
    prob, move_idx = original_policy.topk(1)
    move_idx = move_idx.item()

    # Now filter out undesirable corruptions.
    score, idx = compute_scores(
        corrupted_policy=corrupted_policy,
        original_sparring_policy=original_sparring_policy,
        sparring_corrupted_policy=sparring_corrupted_policy,
        original_wdl=original_wdl,
        corrupted_wdl=corrupted_wdl,
        move_idx=move_idx,
    )
    score = score.item()
    idx = idx.item()
    if score > 0.2:
        # If our score is too bad, we skip this board
        return None
    return corruptions[idx]


def main(args):
    torch.set_num_threads(args.num_threads)

    base_dir = Path(args.base_dir)

    try:
        with open(base_dir / "interesting_puzzles_without_corruptions.pkl", "rb") as f:
            puzzles = pickle.load(f)
    except FileNotFoundError:
        raise ValueError("Puzzles not found, run make_puzzles.py first")

    big_model = Lc0Model(base_dir / "lc0.onnx", device=args.device)
    sparring_model = Lc0Model(base_dir / "LD2.onnx", device=args.device)

    if args.n_puzzles:
        puzzles = puzzles.iloc[: args.n_puzzles]

    dataset = CandidateCorruptionsDataset(puzzles)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=None, shuffle=False, num_workers=args.num_workers
    )

    corrupted_fens = []

    for batch in tqdm.tqdm(dataloader):
        board, corruptions = batch
        corrupted_board = get_best_corruption(
            board, corruptions, big_model, sparring_model
        )
        corrupted_fens.append(
            None if corrupted_board is None else corrupted_board.fen()
        )

    mask = [x is not None for x in corrupted_fens]
    puzzles = puzzles[mask].copy()

    puzzles["corrupted_fen"] = pd.Series(
        [fen for fen in corrupted_fens if fen is not None],
        index=puzzles.index,
    )

    with open("interesting_puzzles.pkl", "wb") as f:
        pickle.dump(puzzles, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--base_dir", default=".", type=str)
    parser.add_argument("--n_puzzles", default=0, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--num_threads", default=1, type=int)
    args = parser.parse_args()
    main(args)
