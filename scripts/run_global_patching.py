import argparse
import itertools
import pickle
from pathlib import Path

import torch
from einops import rearrange
from leela_interp import Lc0sight
from leela_interp.tools import patching


def main(args):
    if not (args.residual_stream or args.attention):
        raise ValueError(
            "At least one of --residual_stream or --attention must be specified"
        )
    torch.set_num_threads(args.num_threads)

    base_dir = Path(args.base_dir)
    model = Lc0sight(base_dir / "lc0.onnx", device=args.device)

    save_dir = base_dir / "results/global_patching"
    save_dir.mkdir(parents=True, exist_ok=True)

    try:
        with open(base_dir / "interesting_puzzles.pkl", "rb") as f:
            puzzles = pickle.load(f)
    except FileNotFoundError:
        raise ValueError("Corrupted puzzles not found, run make_corruptions.py first")

    if args.n_puzzles:
        puzzles = puzzles.iloc[: args.n_puzzles]

    if args.residual_stream:
        # Ablate one square in the residual stream at a time
        effects = patching.residual_stream_activation_patch(
            model=model,
            # The puzzles we loaded already specify corrupted board positions
            puzzles=puzzles,
            batch_size=args.batch_size,
        )
        torch.save(effects, save_dir / "residual_stream_results.pt")

    if args.attention:
        # Ablate one attention head at a time
        locations = list(itertools.product(range(15), range(24)))
        effects = patching.activation_patch(
            model=model,
            module_func=model.headwise_attention_output,
            locations=locations,
            puzzles=puzzles,
            batch_size=args.batch_size,
        )
        effects = rearrange(
            effects, "batch (layer head) -> batch layer head", layer=15, head=24
        )
        torch.save(effects, save_dir / "attention_head_results.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--base_dir", default=".", type=str)
    parser.add_argument("--n_puzzles", default=0, type=int)
    parser.add_argument("--num_threads", default=1, type=int)
    parser.add_argument("--residual_stream", action="store_true")
    parser.add_argument("--attention", action="store_true")
    args = parser.parse_args()
    main(args)
