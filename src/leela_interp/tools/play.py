"""Utility wrappers for running an Lc0 model.

TODO: should probably merge this with Lc0 trees at some point, but for now they
don't have batch support yet.
"""

import pandas as pd
import torch
import tqdm
from leela_interp import Lc0Model, LeelaBoard


def get_lc0_pv_probabilities(
    model: Lc0Model,
    puzzles: pd.DataFrame,
    batch_size: int = 100,
    pbar: bool | None = None,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Computes Lc0's probability for each move in the principal variation.

    Args:
        model: an LC0Model
        puzzles: a dataframe of puzzles. Will be batched automatically.
        batch_size: how many puzzles to feed into Lc0 at once.
        pbar: whether to show a progress bar. If None, determine automatically based
            on number of batches.

    Returns:
        A pandas series of lists of probabilities, where each list has one entry
        per move in the principal variation.
    """
    probs = []
    moves = []
    wdls = []
    if pbar is None:
        pbar = len(puzzles) > batch_size

    _range = tqdm.trange if pbar else range
    for i in _range(0, len(puzzles), batch_size):
        new_probs, new_moves, new_wdls = _get_lc0_pv_probabilities_single_batch(
            model, puzzles.iloc[i : i + batch_size]
        )
        probs.extend(new_probs)
        moves.extend(new_moves)
        wdls.extend(new_wdls)

    return (
        pd.Series(probs, index=puzzles.index),
        pd.Series(moves, index=puzzles.index),
        pd.Series(wdls, index=puzzles.index),
    )


def _get_lc0_pv_probabilities_single_batch(
    model: Lc0Model,
    puzzles: pd.DataFrame,
) -> tuple[list[list[float]], list[list[str]], list[list[float]]]:
    """Single batch of get_lc0_pv_probabilities, just a helper function."""
    max_len = puzzles.principal_variation.apply(len).max()
    boards = [LeelaBoard.from_puzzle(p) for _, p in puzzles.iterrows()]

    probs = [[] for _ in range(len(puzzles))]
    moves = [[] for _ in range(len(puzzles))]
    wdls = []

    for i in range(max_len):
        policies, wdl, _ = model.batch_play(boards, return_probs=True)
        if i == 0:
            wdls.extend(wdl.tolist())
        # Policies can be NaN if the board is in checkmate. We need to filter these
        # out for the allclose check.
        not_nan = ~torch.isnan(policies).any(-1)
        num_not_nan = not_nan.sum().item()
        assert isinstance(num_not_nan, int)  # make the type checker happy
        assert torch.allclose(
            policies[not_nan].sum(-1),
            torch.ones(num_not_nan, device=policies.device),
        ), policies.sum(-1)

        # Update all boards that have moves left:
        for j, board in enumerate(boards):
            pv = puzzles.iloc[j].principal_variation
            if i < len(pv):
                correct_move = pv[i]
                top_moves = model.top_moves(board, policies[j], top_k=None)
                model_move = next(iter(top_moves))
                probs[j].append(top_moves[correct_move])
                moves[j].append(model_move)
                board.push_uci(correct_move)

    return probs, moves, wdls


# TODO: we don't need this, but should have a test that checks get_lc0_pv_probabilities_batch
# against this implementation
def get_lc0_pv_probabilities_non_batched(puzzle):
    probs = []
    board = LeelaBoard.from_puzzle(puzzle)
    for move in puzzle.principal_variation:
        policy, _, _ = lc0_model.play(board, return_probs=True)
        assert torch.allclose(policy.sum(), torch.tensor(1.0))
        policy = lc0_model.policy_as_dict(board, policy)
        probs.append(policy[move])
        board.push_uci(move)

    return probs
