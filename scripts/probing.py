import argparse
import pickle
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
import tqdm
from leela_interp import ActivationCache, Lc0Model, LeelaBoard
from leela_interp.tools import probing


def train_probe(
    X: np.ndarray,
    y: list[int] | np.ndarray,
    z_squares: list[int] | np.ndarray,
    *,
    n_epochs,
    lr,
    batch_size,
    weight_decay,
    k,
    device,
):
    """Train a bilinear probe to predict y from X and Z.

    Args:
        X: should have shape (n_boards, 64, d).
        y: should have shape (n_boards,) and values in {0, ..., 63}.
        z_squares: should have shape (n_boards,) and values in {0, ..., 63}.

    Returns:
        A trained bilinear probe.
    """
    Z = X[np.arange(len(X)), z_squares, :]
    assert Z.shape == (len(X), 768)

    dataset = probing.ProbeData.create(X=X, y=y, Z=Z)
    split_data = dataset.split(val_split=0)

    probe = probing.BilinearSquarePredictor()
    probe.train(
        split_data,
        n_epochs=n_epochs,
        lr=lr,
        batch_size=batch_size,
        weight_decay=weight_decay,
        k=k,
        device=device,
        pbar=False,
    )

    return probe


def collect_data(
    puzzles: pd.DataFrame,
    activations: ActivationCache,
    activation_name: str,
    prediction_target: Literal["target", "source"],
    n_train: int,
    train=True,
):
    ys = []
    z_squares = []
    if train:
        boards = activations.boards[:n_train]
        puzzles = puzzles.iloc[:n_train]
    else:
        boards = activations.boards[n_train:]
        puzzles = puzzles.iloc[n_train:]

    for board, (_, puzzle) in zip(boards, puzzles.iterrows()):
        # Important check to make sure we're not accidentally using activations
        # from different puzzles:
        assert board.fen() == LeelaBoard.from_puzzle(puzzle).fen()
        if prediction_target == "target":
            # Predict the third move target from the first move target:
            y = puzzle.principal_variation[2][2:4]
            z = puzzle.principal_variation[0][2:4]
        elif prediction_target == "source":
            # Predict the third move source from the third move target:
            y = puzzle.principal_variation[2][:2]
            z = puzzle.principal_variation[2][2:4]
        ys.append(board.sq2idx(y))
        z_squares.append(board.sq2idx(z))

    if train:
        X = activations[activation_name][:n_train]
    else:
        X = activations[activation_name][n_train:]
    assert X.shape[1:] == (64, 768), X.shape

    ys = np.array(ys)
    z_squares = np.array(z_squares)

    return X, ys, z_squares


def eval_probe(target_probe, source_probe, puzzles, activations, name, n_train):
    X, target_y, z_squares = collect_data(
        puzzles, activations, name, "target", n_train=n_train, train=False
    )
    _, source_y, _ = collect_data(
        puzzles, activations, name, "source", n_train=n_train, train=False
    )

    # Predict target squares:
    Z = X[np.arange(len(X)), z_squares, :]
    assert Z.shape == (len(X), 768)
    target_squares = target_probe.predict(X, Z)

    # Predict source squares:
    Z = X[np.arange(len(X)), target_squares, :]
    assert Z.shape == (len(X), 768)
    source_squares = source_probe.predict(X, Z)

    source_accuracy = (source_squares == source_y).mean()
    target_accuracy = (target_squares == target_y).mean()
    accuracy = ((source_squares == source_y) * (target_squares == target_y)).mean()

    return source_accuracy, target_accuracy, accuracy


def train_probes(activations, puzzles, n_train, hparams):
    target_probes = []
    for layer in range(15):
        print(f"Layer {layer}")
        name = f"encoder{layer}/ln2"
        X, y, z_squares = collect_data(
            puzzles, activations, name, "target", n_train=n_train
        )
        probe = train_probe(X, y, z_squares, **hparams)
        target_probes.append(probe)

    source_probes = []
    for layer in range(15):
        print(f"Layer {layer}")
        name = f"encoder{layer}/ln2"
        X, y, z_squares = collect_data(
            puzzles, activations, name, "source", n_train=n_train
        )
        probe = train_probe(X, y, z_squares, **hparams)
        source_probes.append(probe)

    return target_probes, source_probes


def eval_probes(target_probes, source_probes, puzzles, activations, n_train, path):
    accuracies = []
    source_accuracies = []
    target_accuracies = []

    for layer, target_probe, source_probe in zip(
        tqdm.trange(15), target_probes, source_probes
    ):
        name = f"encoder{layer}/ln2"
        source_accuracy, target_accuracy, accuracy = eval_probe(
            target_probe, source_probe, puzzles, activations, name, n_train
        )
        accuracies.append(accuracy)
        source_accuracies.append(source_accuracy)
        target_accuracies.append(target_accuracy)

    with open(path, "wb") as f:
        pickle.dump(
            {
                "accuracies": accuracies,
                "source_accuracies": source_accuracies,
                "target_accuracies": target_accuracies,
            },
            f,
        )


def main(args):
    if not (args.main or args.random_model):
        raise ValueError("Please specify at least one of --main or --random_model")

    torch.set_num_threads(args.num_threads)

    base_dir = Path(args.base_dir)

    try:
        with open(base_dir / "interesting_puzzles_without_corruptions.pkl", "rb") as f:
            puzzles = pickle.load(f)
    except FileNotFoundError:
        raise ValueError("Puzzles not found, run make_puzzles.py first")

    if args.split == "all":
        pass
    elif args.split == "different_targets":
        puzzles = puzzles[puzzles["different_targets"]]
    elif args.split == "same_targets":
        puzzles = puzzles[~puzzles["different_targets"]]
    else:
        raise ValueError(
            f"Unknown split: {args.split}, "
            "expected 'all', 'different_targets', or 'same_targets'"
        )

    if args.n_puzzles > 0:
        puzzles = puzzles.iloc[: args.n_puzzles]

    # Use 70% of puzzles for training, rest for testing
    n_train = int(len(puzzles) * 0.7)
    print(f"Using {len(puzzles)} puzzles total, {n_train} for training.")

    hparams = {
        "n_epochs": 5,
        "lr": 1e-2,
        "weight_decay": 0,
        "k": 32,
        "batch_size": 64,
        "device": args.device,
    }

    if args.main:
        model = Lc0Model(base_dir / "lc0.onnx", device=args.device)

        activations = ActivationCache.capture(
            model=model,
            boards=[LeelaBoard.from_puzzle(p) for _, p in puzzles.iterrows()],
            # There's a typo in Lc0, so we mirror it; "rehape" is deliberate
            names=["attn_body/ma_gating/rehape2"]
            + [f"encoder{layer}/ln2" for layer in range(15)],
            n_samples=len(puzzles),
            # Uncomment to store activations on disk (they're about 70GB).
            # Without a path, they'll be kept in memory, which is faster but uses 70GB of RAM.
            # path="residual_activations.zarr",
            overwrite=True,
        )

        for seed in range(args.n_seeds):
            torch.manual_seed(seed)

            target_probes, source_probes = train_probes(
                activations, puzzles, n_train, hparams
            )

            save_dir = base_dir / f"results/probing/{args.split}/{seed}"
            save_dir.mkdir(parents=True, exist_ok=True)

            with open(save_dir / "target_probes.pkl", "wb") as f:
                pickle.dump(target_probes, f)
            with open(save_dir / "source_probes.pkl", "wb") as f:
                pickle.dump(source_probes, f)

            eval_probes(
                target_probes,
                source_probes,
                puzzles,
                activations,
                n_train,
                save_dir / "main.pkl",
            )

        # Free up memory in case we're running the random model next
        del activations

    if args.random_model:
        random_model = Lc0Model(onnx_model_path="lc0-random.onnx", device=args.device)
        activations = ActivationCache.capture(
            boards=[LeelaBoard.from_puzzle(p) for _, p in puzzles.iterrows()],
            names=["attn_body/ma_gating/rehape2"]
            + [f"encoder{layer}/ln2" for layer in range(15)],
            n_samples=len(puzzles),
            # path="random_activations.zarr",
            store_boards=True,
            overwrite=True,
            model=random_model,
        )

        for seed in range(args.n_seeds):
            torch.manual_seed(seed)

            save_dir = base_dir / f"results/probing/{args.split}/{seed}"
            save_dir.mkdir(parents=True, exist_ok=True)

            target_probes, source_probes = train_probes(
                activations, puzzles, n_train, hparams
            )
            eval_probes(
                target_probes,
                source_probes,
                puzzles,
                activations,
                n_train,
                save_dir / "random_model.pkl",
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--n_seeds", default=1, type=int)
    parser.add_argument("--base_dir", default=".", type=str)
    parser.add_argument("--n_puzzles", default=0, type=int)
    parser.add_argument("--num_threads", default=1, type=int)
    parser.add_argument("--main", action="store_true")
    parser.add_argument("--random_model", action="store_true")
    parser.add_argument("--split", default="all", type=str)
    args = parser.parse_args()
    main(args)
