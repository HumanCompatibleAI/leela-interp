import argparse
import pickle
from pathlib import Path

import chess
import torch
from leela_interp import (
    Lc0sight,
    LeelaBoard,
    patching,
)
from leela_interp.tools.piece_movement_heads import (
    bishop_heads,
    knight_heads,
    rook_heads,
)


def main(args):
    torch.set_num_threads(args.num_threads)

    base_dir = Path(args.base_dir)
    model = Lc0sight(base_dir / "lc0.onnx", device=args.device)

    save_dir = base_dir / "results/piece_movement_heads"
    save_dir.mkdir(parents=True, exist_ok=True)

    try:
        with open(base_dir / "interesting_puzzles.pkl", "rb") as f:
            puzzles = pickle.load(f)
    except FileNotFoundError:
        raise ValueError("Puzzles not found, run make_corruptions.py first")

    if args.n_puzzles:
        puzzles = puzzles.iloc[: args.n_puzzles]

    mask = []
    piece_types = []
    ablation_indices = []
    protected_indices = []
    boards = []
    for _, puzzle in puzzles.iterrows():
        # Just make sure there's still something interesting after move 3,
        # this is probably overly aggressive
        if len(puzzle.principal_variation) < 5:
            mask.append(False)
            continue

        board = LeelaBoard.from_puzzle(puzzle)

        ablation_square = puzzle.principal_variation[2][2:4]
        ablation_idx = board.sq2idx(ablation_square)

        # We don't want to ablate the flow with the source square to avoid confounding
        protected_square = puzzle.principal_variation[2][0:2]
        protected_idx = board.sq2idx(protected_square)

        third_source_square = puzzle.principal_variation[2][0:2]
        piece_type = board.pc_board.piece_type_at(
            board.idx2chess_sq(board.sq2idx(third_source_square))
        )
        piece_color = board.pc_board.color_at(
            board.idx2chess_sq(board.sq2idx(third_source_square))
        )
        if piece_color != board.pc_board.turn:
            mask.append(False)
            continue

        if piece_type not in {chess.BISHOP, chess.KNIGHT, chess.ROOK}:
            mask.append(False)
            continue

        mask.append(True)
        piece_types.append(piece_type)
        ablation_indices.append(ablation_idx)
        protected_indices.append(protected_idx)
        boards.append(board)

    mask = torch.tensor(mask, device=args.device)
    piece_types = torch.tensor(piece_types, device=args.device)
    ablation_indices = torch.tensor(ablation_indices, device=args.device)
    protected_indices = torch.tensor(protected_indices, device=args.device)

    def make_patching_func(query_or_key, other_piece=False, random_square=False):
        def patching_func(location, model, batch_indices):
            bishop_cases = piece_types[batch_indices] == chess.BISHOP
            knight_cases = piece_types[batch_indices] == chess.KNIGHT
            rook_cases = piece_types[batch_indices] == chess.ROOK

            all_cases = [bishop_cases, knight_cases, rook_cases]
            if other_piece:
                # Ablate all heads for *other* piece types
                all_cases = [~x for x in all_cases]

            if random_square:
                ablation_indices_batch = torch.randint(
                    64, (len(batch_indices),), device=args.device
                )
            else:
                ablation_indices_batch = ablation_indices[batch_indices]

            protected_indices_batch = protected_indices[batch_indices]

            for cases, heads in zip(
                all_cases,
                [bishop_heads, knight_heads, rook_heads],
            ):
                for layer in range(15):
                    # True for entries we will zero-ablate
                    ablation_mask = torch.zeros(
                        len(batch_indices),
                        24,
                        64,
                        64,
                        dtype=torch.bool,
                        device=args.device,
                    )

                    layer_heads = [head for _layer, head in heads if _layer == layer]
                    for head in layer_heads:
                        # Set ablation mask on the query or key square we want to ablate
                        # but remove it again on protected pair.
                        if query_or_key == "query":
                            ablation_mask[
                                cases, head, ablation_indices_batch[cases]
                            ] = True
                            ablation_mask[
                                cases, head, :, protected_indices_batch[cases]
                            ] = False
                        elif query_or_key == "key":
                            ablation_mask[
                                cases, head, :, ablation_indices_batch[cases]
                            ] = True
                            ablation_mask[
                                cases, head, protected_indices_batch[cases], :
                            ] = False
                        else:
                            raise ValueError("Unknown query_or_key")

                    # Actual intervention:
                    model.attention_scores(layer).output[ablation_mask] = 0

        return patching_func

    effects = {}
    kwarg_dict = {
        "Main ablation": {},
        "Other piece types": {"other_piece": True},
        "Random square": {"random_square": True},
    }

    for query_or_key in ["key"] + ["query"] * args.query:
        for name, kwargs in kwarg_dict.items():
            effects[(query_or_key, name)] = -patching.patch(
                patching_func=make_patching_func(query_or_key, **kwargs),
                locations=[None],
                boards=boards,
                model=model,
                pbar="batch",
                batch_size=args.batch_size,
            )

    torch.save(
        {
            "mask": mask,
            "piece_types": piece_types,
            "ablation_indices": ablation_indices,
            "effects": effects,
        },
        save_dir / "effects.pt",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--base_dir", default=".", type=str)
    parser.add_argument("--n_puzzles", default=0, type=int)
    parser.add_argument("--num_threads", default=1, type=int)
    # By default, we only ablate with the third target as the key square.
    # Pass in --query to also ablate with the query square and separately store results.
    parser.add_argument("--query", action="store_true")
    args = parser.parse_args()
    main(args)
