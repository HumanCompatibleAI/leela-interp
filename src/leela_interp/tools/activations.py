import pickle
import shutil
import warnings
from pathlib import Path
from typing import Iterable

import numpy as np
import tqdm
import zarr
from einops import rearrange
from leela_interp.core.lc0 import Lc0Model
from leela_interp.core.leela_board import LeelaBoard
from leela_interp.core.nnsight import Lc0sight


class ActivationCache:
    def __init__(
        self, data: zarr.hierarchy.Group, boards: list[LeelaBoard] | None = None
    ):
        self.data = data
        if boards is not None:
            assert len(boards) == data.attrs["n_samples"]
        self.boards = boards

    def __getitem__(self, name: str) -> zarr.core.Array:
        return self.data[name]

    def numpy(self, name: str) -> np.ndarray:
        return self[name][:]

    @property
    def names(self) -> tuple[str, ...]:
        # TODO: this is broken because zarr uses / as a separator just like Lc0
        return tuple(self.data.keys())

    @classmethod
    def load(cls, path: str | Path):
        with zarr.open(str(path), mode="r") as data:
            data = data

        try:
            with open(str(path) + "_boards.pkl", "rb") as f:
                boards = pickle.load(f)
        except FileNotFoundError:
            boards = None

        return cls(data=data, boards=boards)

    def __len__(self):
        return self.data.attrs["n_samples"]

    @classmethod
    def capture(
        cls,
        boards: Iterable[LeelaBoard],
        names: list[str],
        n_samples: int,
        model: Lc0Model | Lc0sight,
        batch_size: int = 1024,
        path: str | Path | None = None,
        store_boards: bool = True,
        pbar: str | None = "tqdm",
        overwrite: bool = False,
    ) -> "ActivationCache":
        if not names:
            raise ValueError("At least one activation name must be specified")

        if path is None:
            store = zarr.MemoryStore()
        else:
            if Path(path).exists():
                if not overwrite:
                    raise FileExistsError(f"File {path} already exists")
                else:
                    shutil.rmtree(path)
            store = zarr.DirectoryStore(str(path))

        data = zarr.group(store=store)

        board_iter = iter(boards)

        if store_boards:
            # We want to make sure to only store the first n_samples boards,
            # to avoid running a potentially very long generator.
            if not isinstance(boards, list):
                boards = []
                for i in range(n_samples):
                    try:
                        board = next(board_iter)
                        boards.append(board)
                    except StopIteration:
                        warnings.warn(
                            f"{n_samples} samples requested, but fewer boards available. "
                            f"Returning only {i} samples."
                        )
                        break
            else:
                boards = boards[:n_samples]

            if path:
                with open(str(path) + "_boards.pkl", "wb") as f:
                    pickle.dump(boards, f)

            # Given that we've already run the generator once, it's more efficient
            # to just use the stored list from now on.
            board_iter = iter(boards)

        if pbar == "tqdm":
            iterator = tqdm.trange(0, n_samples, batch_size)
        elif pbar == "print":

            def print_iterator():
                for i in range(0, n_samples, batch_size):
                    print(f"{i}/{n_samples} boards")
                    yield i

            iterator = print_iterator()

        else:
            iterator = range(0, n_samples, batch_size)

        for i in iterator:
            # We need to make sure the final batch isn't too big:
            batch_size = min(batch_size, n_samples - i)
            if batch_size <= 0:
                break

            try:
                with model.capturing(names) as new_activations:
                    model.batch_play([next(board_iter) for _ in range(batch_size)])
            except StopIteration:
                warnings.warn(
                    f"{n_samples} samples requested, but fewer boards available. "
                    f"Returning only {i} samples."
                )
                break

            for name in names:
                new_activations[name] = new_activations[name].cpu().numpy()

                # There's a typo in Lc0, so we mirror it; "rehape" is deliberate
                if (
                    name.endswith(("ln1", "ln2"))
                    or name == "attn_body/ma_gating/rehape2"
                ):
                    # Reshape residual stream activations into a more reasonable shape
                    # with batch as a separate axis
                    new_activations[name] = rearrange(
                        new_activations[name],
                        "(batch square) hidden -> batch square hidden",
                        square=64,
                    )

                if name not in data:
                    # shape without batch dimension:
                    shape = new_activations[name].shape[1:]
                    # TODO: should probably manually specify chunk shape
                    data[name] = zarr.empty(
                        shape=(n_samples, *shape), dtype=np.float32, chunks=True
                    )
                data[name][i : i + batch_size] = new_activations[name]

        data.attrs["n_samples"] = len(data[names[0]])

        return cls(data=data, boards=boards if store_boards else None)  # type: ignore
