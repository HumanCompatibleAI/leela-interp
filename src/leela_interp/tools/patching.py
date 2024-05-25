import itertools
from typing import Callable, Literal, Optional

import pandas as pd
import torch
import tqdm
from einops import rearrange
from leela_interp.core.leela_board import LeelaBoard
from leela_interp.core.nnsight import Lc0sight


def get_move_probs(
    move_indices: torch.Tensor,
    model: Lc0sight,
    boards: list[LeelaBoard],
    legal_move_mask: Optional[torch.Tensor] = None,
):
    logits = model.output[0]
    probs = model.logits_to_probs(boards, logits, legal_move_mask=legal_move_mask)
    move_probs = probs.take_along_dim(move_indices, dim=1).squeeze(dim=1)
    return move_probs


def get_move_log_odds(
    move_indices: torch.Tensor,
    model: Lc0sight,
    boards: list[LeelaBoard],
    legal_move_mask: Optional[torch.Tensor] = None,
):
    move_probs = get_move_probs(
        move_indices, model, boards, legal_move_mask=legal_move_mask
    )
    log_odds = torch.log(move_probs / (1 - move_probs))
    return log_odds


def get_win_log_odds(move_indices, model, boards, legal_move_mask=None):
    wdl = model.output[1]
    win_probs = wdl[:, 0]
    win_log_odds = torch.log(win_probs / (1 - win_probs))
    return win_log_odds


effect_funcs = {
    "move_log_odds": get_move_log_odds,
    "move_probs": get_move_probs,
    "win_log_odds": get_win_log_odds,
}

EffectType = Literal["move_log_odds", "move_probs", "win_log_odds"]


def patch(
    patching_func: Callable,
    locations: list,
    boards: list[LeelaBoard] | LeelaBoard,
    model: Optional[Lc0sight] = None,
    effect_type: EffectType = "move_log_odds",
    output_func: Callable[
        [torch.Tensor, Lc0sight, list[LeelaBoard], Optional[torch.Tensor]], torch.Tensor
    ]
    | None = None,
    location_batch_size: int = 1,
    batch_size: int = 64,
    override_best_move_indices: torch.Tensor | None = None,
    pbar: str = "location",
):
    """Generic patching function for a batch of inputs.

    Args:
        patching_func: A function that takes a location and model and modifies
            the model's activations at that location in place. This will be called
            within an nnsight trace context, so access model activations using
            `model.some_submodule.output`.
        locations: A list of locations in the network to patch. A "location" can be
            any value, it just needs to be recognized by the `patching_func`.
            For example, this might be a tuple (layer, square) or anything else.
        boards: A list of LeelaBoards (or a single one).
        model: An Lc0sight model. If None, a new one will be loaded.
        effect_type: The type of effect on the output to compute. For example,
            "move_log_odds" (default) will compute the change in log odds of the
            original top move.
        output_func: More customizable alternative to `effect_type`, overrides
            `effect_type`. The inputs are a tensor of shape (N, ) of indices into the
            policy specifying the best moves, the model, and the boards (list of length
            N). The output should be a tensor of shape (N, ). This is called within
            an nnsight trace context, so access model activations using
            `model.some_submodule.output` (to test effect on intermediate activations)
            or `model.output` (to test effect on the final output).
        batch_size: We always batch at least all the boards together (and a `batch_size`
            smaller then `len(boards)` will be ignored). But if the number of boards
            is low, it may be more efficient to also batch over several different
            patching locations. Typically it's inefficient to set the batch size too
            high though, since there are overhead costs when using nnsight to batch
            over multiple locations. (Roughly `8 * len(boards)` should be reasonable,
            or of course less.)
        override_best_move_indices: If not None, this tensor will be passed to
            `output_func` instead of the best move indices computed for the clean run.
            Also applies if `output_func` is None and `effect_type` uses the best moves.

    Returns:
        A tensor of shape (N, M) where N is the number of boards and M is the number
        of patching locations. If only a single board is passed in (rather than a list),
        the output shape will be (M, ).
    """
    if pbar == "location":
        location_pbar = True
        batch_pbar = False
    elif pbar == "batch":
        location_pbar = False
        batch_pbar = True
    else:
        location_pbar = False
        batch_pbar = False

    # Whether we got just a single board as input:
    single = False

    if model is None:
        model = Lc0sight()

    if isinstance(boards, LeelaBoard):
        single = True
        boards = [boards]

    if output_func is None:
        output_func = effect_funcs[effect_type]
    assert output_func is not None  # type checker

    effects = []
    for i in tqdm.trange(0, len(boards), batch_size, disable=not batch_pbar):
        upper_limit = min(i + batch_size, len(boards))
        effects.append(
            _patch_single_board_batch(
                model=model,
                boards=boards[i : i + batch_size],
                output_func=output_func,
                override_best_move_indices=override_best_move_indices[
                    i : i + batch_size
                ]
                if override_best_move_indices is not None
                else None,
                locations=locations,
                location_batch_size=location_batch_size,
                patching_func=patching_func,
                batch_indices=list(range(i, upper_limit)),
                pbar=location_pbar,
            )
        )

    effects = torch.cat(effects, dim=0)

    if single:
        # Remove batch dimension
        effects = effects[0]

    return effects


def _patch_single_board_batch(
    *,
    model,
    boards,
    output_func,
    override_best_move_indices,
    locations,
    location_batch_size,
    patching_func,
    batch_indices,
    pbar,
):
    # Cache this once, it's otherwise a bottleneck on fast GPUs
    legal_move_mask = model._get_legal_move_mask(boards)

    # Clean run to get baselines
    with model.trace(boards):
        output = model.output
        logits = output[0]
        # Need to do this to zero out illegal moves before computing the top move idx:
        probs = model.logits_to_probs(boards, logits, legal_move_mask=legal_move_mask)
        best_move_indices = probs.argmax(dim=1, keepdim=True).save()
        clean_results = output_func(
            override_best_move_indices or best_move_indices,
            model,
            boards,
            legal_move_mask,
        ).save()

    best_move_indices = override_best_move_indices or best_move_indices.value
    clean_results = clean_results.value

    # Patching runs:

    all_patched_results = []
    for i in tqdm.trange(0, len(locations), location_batch_size, disable=not pbar):
        new_patched_results = []
        if location_batch_size == 1:
            location = locations[i]

            with model.trace(boards):
                patching_func(location, model, batch_indices)
                patched_results = output_func(
                    best_move_indices, model, boards, legal_move_mask
                ).save()
                new_patched_results.append(patched_results)
        else:
            with model.trace() as tracer:
                for location in locations[i : i + location_batch_size]:
                    with tracer.invoke(boards, scan=False):
                        patching_func(location, model, batch_indices)
                        patched_results = output_func(
                            best_move_indices, model, boards, legal_move_mask
                        ).save()
                        new_patched_results.append(patched_results)

        all_patched_results.extend([r.value for r in new_patched_results])

    effects = torch.stack(all_patched_results, dim=1)
    effects = effects - clean_results[:, None]
    return effects


def activation_patch(
    module_func: Callable[[int], torch.nn.Module],
    locations: list[tuple[int, ...]],
    model: Lc0sight,
    puzzles: Optional[pd.DataFrame | pd.Series] = None,
    boards: Optional[list[LeelaBoard] | LeelaBoard] = None,
    corrupted_boards: Optional[list[LeelaBoard] | LeelaBoard] = None,
    effect_type: EffectType = "move_log_odds",
    output_func: Callable[[torch.Tensor, Lc0sight, list[LeelaBoard]], torch.Tensor]
    | None = None,
    location_batch_size: int = 1,
    batch_size: int = 64,
) -> torch.Tensor:
    if puzzles is None:
        assert boards is not None
        assert corrupted_boards is not None
        if isinstance(boards, LeelaBoard):
            assert isinstance(corrupted_boards, LeelaBoard)
        else:
            assert isinstance(boards, list)
            assert isinstance(corrupted_boards, list)

    elif isinstance(puzzles, pd.Series):
        assert boards is None
        assert corrupted_boards is None
        boards = LeelaBoard.from_puzzle(puzzles)
        corrupted_boards = LeelaBoard.from_fen(puzzles.corrupted_fen)
    else:
        assert isinstance(puzzles, pd.DataFrame)
        assert boards is None
        assert corrupted_boards is None
        boards = [LeelaBoard.from_puzzle(puzzle) for _, puzzle in puzzles.iterrows()]
        corrupted_boards = [
            LeelaBoard.from_fen(puzzle.corrupted_fen)
            for _, puzzle in puzzles.iterrows()
        ]

    # Corrupted run to get activations
    if isinstance(corrupted_boards, LeelaBoard):
        corrupted_boards = [corrupted_boards]

    single = False
    if isinstance(boards, LeelaBoard):
        single = True
        boards = [boards]

    effects = []
    for i in range(0, len(corrupted_boards), batch_size):
        effects.append(
            _activation_patch_single_board_batch(
                model=model,
                boards=boards[i : i + batch_size],
                corrupted_boards=corrupted_boards[i : i + batch_size],
                output_func=output_func,
                locations=locations,
                location_batch_size=location_batch_size,
                module_func=module_func,
                effect_type=effect_type,
            )
        )

    effects = torch.cat(effects, dim=0)
    if single:
        # Remove batch dimension
        effects = effects[0]
    return effects


def _activation_patch_single_board_batch(
    *,
    model,
    boards,
    corrupted_boards,
    output_func,
    locations,
    location_batch_size,
    effect_type,
    module_func,
):
    with model.trace(corrupted_boards):
        corrupted_activations = [
            module_func(layer).output.save() for layer in range(model.N_LAYERS)
        ]

    corrupted_activations = [act.value for act in corrupted_activations]

    def _activation_patching_func(location, model, batch_indices):
        layer, *rest = location
        act = module_func(layer).output
        rest = tuple(rest)
        act[(slice(None),) + rest] = corrupted_activations[layer][(slice(None),) + rest]

    return patch(
        _activation_patching_func,
        locations,
        boards=boards,
        model=model,
        effect_type=effect_type,
        output_func=output_func,
        location_batch_size=location_batch_size,
        batch_size=len(boards),
    )


def residual_stream_activation_patch(
    model: Lc0sight,
    puzzles: Optional[pd.DataFrame | pd.Series] = None,
    boards: Optional[list[LeelaBoard] | LeelaBoard] = None,
    corrupted_boards: Optional[list[LeelaBoard] | LeelaBoard] = None,
    effect_type: EffectType = "move_log_odds",
    output_func: Callable[[torch.Tensor, Lc0sight, list[LeelaBoard]], torch.Tensor]
    | None = None,
    layers: Optional[list[int]] = None,
    squares: Optional[list[int]] = None,
    batch_size: int = 64,
    location_batch_size: int = 1,
    module_func: Callable[[int], torch.nn.Module] | None = None,
) -> torch.Tensor:
    if module_func is None:
        module_func = model.residual_stream

    if layers is None:
        layers = list(range(model.N_LAYERS))

    if squares is None:
        squares = list(range(64))

    locations = list(itertools.product(layers, squares))

    effects = activation_patch(
        module_func,
        locations,
        puzzles=puzzles,
        boards=boards,
        corrupted_boards=corrupted_boards,
        model=model,
        effect_type=effect_type,
        output_func=output_func,
        batch_size=batch_size,
        location_batch_size=location_batch_size,
    )

    return rearrange(
        effects,
        "... (layer square) -> ... layer square",
        layer=len(layers),
        square=len(squares),
    )
