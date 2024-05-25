"""This file implements plotting chess boards with heatmaps/arrows.

You'll usually use `LeelaBoard.plot()` instead of using `IcebergBoard` directly.
You might need to add more fonts to the `FONTS` list if none of the current ones exist
on your system.
"""

import chess
import chess.svg
import iceberg as ice
import matplotlib
import numpy as np
import torch
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from .utils import idx2sq

# Tried in order, at least one of these must exist on your system to add captions
# to chess boards:
FONTS = ["Monaco", "DejaVu Sans Mono"]


def palette(
    values: np.ndarray, cmap="viridis", zero_center=False, upper_ratio: float = 1.0
):
    if not isinstance(values, np.ndarray):
        raise TypeError("values must be a numpy array")
    if zero_center:
        max_val = max(abs(values.min()), abs(values.max()))
        norm = Normalize(vmin=-max_val, vmax=max_val)
    else:
        vmin = values.min()
        diff = values.max() - vmin
        total_range = diff / upper_ratio
        norm = Normalize(vmin=vmin, vmax=vmin + total_range)
    mappable = ScalarMappable(norm=norm, cmap=cmap)
    return [
        matplotlib.colors.to_hex(mappable.to_rgba(value)) for value in values
    ], mappable


CHESS_NAME_TO_SQUARE = {chess.square_name(i): chess.Square(i) for i in range(64)}


class IcebergBoard(ice.DrawableWithChild):
    board: chess.Board
    heatmap: np.ndarray | torch.Tensor | list[str] | dict[str, str | float] | None = (
        None
    )
    # Can't call it "move" because that's an iceberg method
    next_moves: str | list[str] | None = None
    highlight: str | None = None
    caption: str | None = None
    cmap: str = "YlOrRd"
    mappable: ScalarMappable | None = None
    zero_center: bool = False
    arrows: dict[str, str] | None = None
    attn_map: np.ndarray | torch.Tensor | None = None
    show_lastmove: bool = True

    def setup(self):
        fill = {}
        arrows = []

        if isinstance(self.heatmap, dict):
            if self.heatmap and isinstance(next(iter(self.heatmap.values())), float):
                if self.mappable is None:
                    colormap_values, mappable = palette(
                        np.array(list(self.heatmap.values())),
                        cmap=self.cmap,
                        zero_center=self.zero_center,
                    )
                else:
                    colormap_values = [
                        matplotlib.colors.to_hex(self.mappable.to_rgba(value))
                        for value in self.heatmap.values()
                    ]
                self.heatmap = {
                    k: colormap_values[i] for i, k in enumerate(self.heatmap.keys())
                }

            fill = {CHESS_NAME_TO_SQUARE[k]: v for k, v in self.heatmap.items()}
        elif self.heatmap is not None:
            if isinstance(self.heatmap, torch.Tensor):
                self.heatmap = self.heatmap.cpu().numpy()

            if isinstance(self.heatmap, np.ndarray):
                assert self.heatmap.shape == (64,)
                if self.mappable is None:
                    colormap_values, mappable = palette(
                        self.heatmap, cmap=self.cmap, zero_center=self.zero_center
                    )
                else:
                    colormap_values = [
                        matplotlib.colors.to_hex(self.mappable.to_rgba(value))
                        for value in self.heatmap
                    ]
            else:
                assert isinstance(self.heatmap, list)
                assert len(self.heatmap) == 64
                assert isinstance(self.heatmap[0], str)
                colormap_values = self.heatmap

            fill = {
                CHESS_NAME_TO_SQUARE[idx2sq(idx, self.board.turn)]: colormap_values[idx]
                for idx in range(64)
            }

        if self.next_moves is not None:
            next_moves = self.next_moves
            if isinstance(next_moves, str):
                next_moves = [next_moves]
            arrows = []
            for move in next_moves:
                from_square = CHESS_NAME_TO_SQUARE[move[:2]]
                to_square = CHESS_NAME_TO_SQUARE[move[2:4]]
                arrows.append(chess.svg.Arrow(from_square, to_square, color="blue"))

        if self.attn_map is not None:
            for value in self.attn_map:
                from_square = CHESS_NAME_TO_SQUARE[value[0][:2]]
                to_square = CHESS_NAME_TO_SQUARE[value[0][2:4]]
                opacity = value[1]
                arrows.append(chess.svg.Arrow(from_square, to_square, color="red"))
                arrows[-1].opacity = opacity

        if self.arrows is not None:
            for move, color in self.arrows.items():
                from_square = CHESS_NAME_TO_SQUARE[move[:2]]
                to_square = CHESS_NAME_TO_SQUARE[move[2:4]]
                arrows.append(chess.svg.Arrow(from_square, to_square, color=color))

        lastmove = self.board.peek() if self.board.move_stack else None

        svg = chess.svg.board(
            board=self.board,
            size=390,
            lastmove=lastmove if self.show_lastmove else None,
            # We abuse the check functionality as a nice distinctive way to highlight
            # a square even in the presence of a heatmap.
            check=None
            if self.highlight is None
            else CHESS_NAME_TO_SQUARE[self.highlight],
            fill=fill,
            arrows=arrows,
            colors={
                "square light": "#f5f5f5",
                "square dark": "#cfcfcf",
                "square light lastmove": "#cfcfff",
                "square dark lastmove": "#a0a0ff",
            },
        )

        child = ice.SVG(raw_svg=svg)

        # Add an invisible grid of squares to we can reference them later.
        square_size = 44
        squares = [
            ice.Rectangle(
                ice.Bounds.from_size(square_size, square_size), border_color=None
            )
            for _ in range(64)
        ]
        squares_in_rows = [
            [squares[i * 8 + (7 - j)] for j in range(8)] for i in range(8)
        ]
        grid = ice.Grid(children_matrix=squares_in_rows, gap=0)
        dx, dy = 15, 15
        grid = grid.move(dx, dy)
        child = ice.Anchor([child, grid])
        self._squares = {
            chess.Square(i): square for i, square in enumerate(squares[::-1])
        }

        if self.caption is not None:
            text = None
            for font in FONTS:
                try:
                    text = ice.Text(
                        self.caption,
                        font_style=ice.FontStyle(
                            family=font,
                            size=16,
                            color=ice.Colors.BLACK,
                        ),
                        width=child.bounds.width,
                    )
                    break
                except ValueError as e:
                    if "Invalid font family" not in str(e):
                        raise
            if text is None:
                raise ValueError(f"Couldn't find a valid font, tried {FONTS}")

            child = ice.Arrange(
                [text, child],
                gap=10,
                arrange_direction=ice.Arrange.Direction.VERTICAL,
            )

        self.set_child(child)

    def square(self, square: chess.Square) -> ice.Rectangle:
        return self._squares[square]
