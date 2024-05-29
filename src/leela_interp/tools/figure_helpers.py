from typing import Any

import iceberg as ice
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import skia
from scipy.stats import binom

EFFECTS_CMAP = mpl.colors.LinearSegmentedColormap.from_list(
    "my_colormap", ["white", "#ffa600"]
)
EFFECTS_CMAP_2 = "YlOrBr"

EFFECTS_UPPER_RATIO = 1.0

FONT_FAMILY = "Monaco"

COLORS = [
    "#00b894",
    "#0984e3",
    "#d63031",
    "#495057",
]
COLOR_DICT = {
    "first_target": COLORS[0],
    "third_target": COLORS[1],
    "corrupted": COLORS[2],
    "other": COLORS[3],
}
PLOT_FACE_COLOR = "#F8F9FA"

BEST_MOVE_COLOR = "#2d3436"

LINE_WIDTH = 0.5
ERROR_ALPHA = 0.3

PUZZLE_LOC = 19612

# Width of the LaTeX document
TEXT_WIDTH = 5.50107

# Small gap between figures
HALF_WIDTH = 0.47 * TEXT_WIDTH


def set(fast=False):
    params: dict[str, Any] = {
        "font.family": "serif",
        "font.serif": "Times New Roman",
        "text.usetex": not fast,
    }

    SMALL_SIZE = 8
    MEDIUM_SIZE = 10
    BIGGER_SIZE = 12

    plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
    plt.rc("axes", titlesize=MEDIUM_SIZE)  # fontsize of the axes title
    plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title

    plt.rcParams.update(params)


def get_width(fraction=1.0):
    return TEXT_WIDTH * fraction


def save(path, fig=None):
    if fig is None:
        fig = plt.gcf()

    plt.tight_layout()
    fig.savefig(path)
    plt.show()


def percentile_errors(
    data: np.ndarray, percentiles: list[int] | np.ndarray, confidence: float = 0.95
) -> np.ndarray:
    # data should be a 1D array of i.i.d. samples from some distribution
    # percentiles is a list of percentiles for which we want to know the error bar
    # Returns an array of shape (len(percentiles), 2) with the lower and upper bounds of the confidence intervals
    assert data.ndim == 1

    n = len(data)
    data_sorted = np.sort(data)
    bounds = np.zeros((len(percentiles), 2))

    for i, p in enumerate(percentiles):
        # Compute the binomial distribution parameters
        alpha = (1 - confidence) / 2
        lower_bound_index = int(binom.ppf(alpha, n, p / 100))
        upper_bound_index = int(binom.ppf(1 - alpha, n, p / 100))

        # Ensure indices are within the range of data indices
        lower_bound_index = max(0, min(lower_bound_index, n - 1))
        upper_bound_index = max(0, min(upper_bound_index, n - 1))

        # Set the lower and upper bounds
        bounds[i, 0] = data_sorted[lower_bound_index]
        bounds[i, 1] = data_sorted[upper_bound_index]

    return bounds


def plot_percentiles(
    data: dict[str, np.ndarray],
    resolution: int = 1000,
    zoom_start: int | None = None,
    zoom_width_ratio: float = 1,
    zoom_resolution=None,
    colors=None,
    title: str = "",
    figsize=(8, 4),
    zoom_tick_frequency=2,
    tick_frequency=20,
    y_lower=None,
    y_upper=None,
    y_ticks=None,
    confidence=0.95,
):
    if colors is None:
        colors = {}
    if zoom_resolution is None:
        zoom_resolution = resolution

    print_percentiles = list(range(0, 81, 10)) + [85, 90, 95, 98, 99, 100]

    # Create subplots
    if zoom_start is None:
        fig, ax1 = plt.subplots(1, 1, figsize=figsize)
    else:
        fig, (ax1, ax2) = plt.subplots(
            1, 2, figsize=figsize, sharey=False, width_ratios=[1, zoom_width_ratio]
        )

    # Plot full range percentiles
    xs = np.linspace(0, 100, resolution + 1)
    for name, series in data.items():
        assert series.ndim == 1
        percentiles = np.percentile(series, xs)
        errors = percentile_errors(series, xs, confidence=confidence)
        ax1.plot(
            xs,
            percentiles,
            label=name,
            color=colors.get(name, None),
            linewidth=LINE_WIDTH,
        )
        ax1.fill_between(
            xs,
            errors[:, 0],
            errors[:, 1],
            color=colors.get(name, None),
            alpha=ERROR_ALPHA,
            linewidth=0,
        )
        for i in print_percentiles:
            print(
                f"{i}th percentile ({name}): {percentiles[i * resolution // 100]:.2f} +- {(errors[i * resolution // 100, 1] - errors[i * resolution // 100, 0]) / 2:.2f}"
            )

    ax1.legend()
    ax1.set_xlabel("Percentile")
    ax1.set_ylabel("Log odds reduction")
    ax1.set_title(title)
    ax1.set_xlim(-1, 101)
    ax1.set_xticks(np.arange(0, 100, tick_frequency), minor=True)
    if y_ticks is not None:
        ax1.set_yticks(y_ticks)
    ax1.set_yticks(np.arange(0, 10, 1), minor=True)
    ax1.tick_params(which="minor", length=0)
    if y_lower is not None:
        ax1.set_ylim(bottom=y_lower)
    if y_upper is not None:
        ax1.set_ylim(top=y_upper)

    if zoom_start is not None:
        # Plot zoomed-in range
        zoom_xs = np.linspace(zoom_start - 0.5, 100, resolution + 1)
        # Recompute these with higher resolution
        for name, series in data.items():
            percentiles = np.percentile(series, zoom_xs)
            errors = percentile_errors(series, zoom_xs, confidence=confidence)
            ax2.plot(
                zoom_xs,
                percentiles,
                label=name,
                color=colors.get(name, None),
                linewidth=LINE_WIDTH,
            )
            ax2.fill_between(
                zoom_xs,
                errors[:, 0],
                errors[:, 1],
                color=colors.get(name, None),
                alpha=ERROR_ALPHA,
                linewidth=0,
            )

        # ax2.legend()
        ax2.set_xlabel("Percentile")
        # ax2.set_ylabel("Log odds reduction")
        # ax2.set_title(f"Zoomed in ({zoom_start}th to 100th percentile)")
        ax2.set_title("Zoomed in")
        ax2.set_xticks(range(zoom_start, 101, zoom_tick_frequency))
        ax2.set_xlim(zoom_start - 0.5, 100.5)
        ax2.set_ylim(ax1.get_ylim())
        ax2.set_yticks(np.arange(0, 10, 1), minor=True)
        ax2.set_yticklabels([])
        ax2.tick_params(axis="y", which="both", length=0)

    axes = [ax1]
    if zoom_start is not None:
        axes.append(ax2)
    for ax in axes:
        # ax.spines[["right", "top", "left"]].set_visible(False)
        # ax.spines["bottom"].set_position("zero")
        ax.spines[:].set_visible(False)
        ax.set_facecolor(PLOT_FACE_COLOR)
        ax.grid(linestyle="--")
        ax.grid(which="minor", alpha=0.3, linestyle="--")

    plt.tight_layout()
    return fig


class SkiaPath(ice.Path):
    skia_path: skia.Path
    path_style: ice.PathStyle

    def __init__(
        self,
        skia_path: skia.Path,
        path_style: ice.PathStyle,
    ):
        self.init_from_fields(skia_path=skia_path, path_style=path_style)

    def setup(self):
        self.set_path(self.skia_path, self.path_style)


class HatchedRectangle(ice.Drawable):
    """A rectangle.

    Args:
        rectangle: The bounds of the rectangle.
        border_color: The color of the border.
        fill_color: The color of the fill.
        border_thickness: The thickness of the border.
        anti_alias: Whether to use anti-aliasing.
        border_position: The position of the border.
        border_radius: The radius of the border.
        dont_modify_bounds: Whether to modify the bounds of the rectangle to account for the border.
    """

    rectangle: ice.Bounds
    border_color: ice.Color = None
    fill_color: ice.Color = None
    border_thickness: float = 1.0
    anti_alias: bool = True
    border_position: ice.BorderPosition = ice.BorderPosition.INSIDE
    border_radius: float | tuple[float, float] = 0.0
    dont_modify_bounds: bool = False
    hatched: bool = False
    hatched_spacing: float = 10
    hatched_thickness: float = 1
    hatched_angle: float = 45
    partial_end: float = 1.0

    def __init__(
        self,
        rectangle: ice.Bounds,
        border_color: ice.Color = None,
        fill_color: ice.Color = None,
        border_thickness: float = 1.0,
        anti_alias: bool = True,
        border_position: ice.BorderPosition = ice.BorderPosition.INSIDE,
        border_radius: float | tuple[float, float] = 0.0,
        dont_modify_bounds: bool = False,
        hatched: bool = False,
        hatched_spacing: float = 10,
        hatched_thickness: float = 1,
        hatched_angle: float = 45,
        partial_end: float = 1.0,
    ):
        self.init_from_fields(
            rectangle=rectangle,
            border_color=border_color,
            fill_color=fill_color,
            border_thickness=border_thickness,
            anti_alias=anti_alias,
            border_position=border_position,
            border_radius=border_radius,
            dont_modify_bounds=dont_modify_bounds,
            hatched=hatched,
            hatched_spacing=hatched_spacing,
            hatched_thickness=hatched_thickness,
            hatched_angle=hatched_angle,
            partial_end=partial_end,
        )

    def setup(
        self,
    ) -> None:
        self._border_paint = ice.PathStyle(
            color=self.border_color,
            thickness=self.border_thickness,
            # dashed=True,
            # dash_intervals=[2, 2],
        ).skia_paint

        self._fill_paint = (
            skia.Paint(
                Style=skia.Paint.kFill_Style,
                AntiAlias=self.anti_alias,
                Color4f=self.fill_color.to_skia(),
            )
            if self.fill_color
            else None
        )

        self._passed_bounds = self.rectangle
        self._bounds = self.rectangle
        self._skia_rect = self.rectangle.to_skia()
        self._border_skia_rect = self.rectangle.inset(
            self.border_thickness / 2, self.border_thickness / 2
        ).to_skia()

        # Increase the bounds to account for the border.
        if self.border_position == ice.BorderPosition.CENTER:
            self._bounds = self._bounds.inset(
                -self.border_thickness / 2, -self.border_thickness / 2
            )
            self._border_skia_rect = self.rectangle.to_skia()
        elif self.border_position == ice.BorderPosition.OUTSIDE:
            self._bounds = self._bounds.inset(
                -self.border_thickness, -self.border_thickness
            )
            self._border_skia_rect = self.rectangle.inset(
                -self.border_thickness / 2, -self.border_thickness / 2
            ).to_skia()

        border_rect_path = skia.Path()
        border_rect_path.addRoundRect(self._border_skia_rect, self.border_radius_tuple)
        self.border_rect_path = border_rect_path

        fill_rect_path = skia.Path()
        fill_rect_path.addRoundRect(self._skia_rect, self.border_radius_tuple)
        self.fill_rect_path = fill_rect_path

        hatched_spacing = self.hatched_spacing
        hatched_thickness = self.hatched_thickness
        hatched_lines = skia.Path()
        hatched_mult = 4
        x, y = self._border_skia_rect.left(), self._border_skia_rect.top()
        w, h = self._border_skia_rect.width(), self._border_skia_rect.height()
        cx, cy = x + w / 2, y + h / 2
        sx = cx - w / 2 * hatched_mult
        ex = cx + w / 2 * hatched_mult
        sy = cy - h / 2 * hatched_mult
        ey = cy + h / 2 * hatched_mult

        hatched_ys = np.arange(sy, ey, hatched_spacing)

        # Compute partial ends of all lines.
        # When partial_end is 1, all lines should have partial_end 1.

        partial_end_thresh = 1 / len(hatched_ys)

        for i, hatch_y in enumerate(hatched_ys):
            # left, top, right, bottom
            left = sx
            top = hatch_y - hatched_thickness / 2
            right = (ex - sx) * max(
                1, self.partial_end * (i + 1) / len(hatched_ys)
            ) + sx
            bottom = hatch_y + hatched_thickness / 2

            hatched_lines.addRect(
                left,
                top,
                right,
                bottom,
            )

        hatched_lines.transform(skia.Matrix().setRotate(self.hatched_angle, cx, cy))

        path_builder = skia.OpBuilder()
        path_builder.add(hatched_lines, skia.kUnion_PathOp)
        path_builder.add(border_rect_path, skia.kIntersect_PathOp)
        new_path = path_builder.resolve()

        self.hatched_lines = new_path

    @property
    def bounds(self) -> ice.Bounds:
        if self.dont_modify_bounds:
            return self._passed_bounds

        return self._bounds

    @property
    def border_radius_tuple(self) -> tuple[float, float]:
        # Return 8 numbers for the 4 corners.
        if isinstance(self.border_radius, tuple):
            a, b, c, d = self.border_radius
            return a, a, b, b, c, c, d, d

        # Return 4 numbers for the 2 corners.
        return (self.border_radius,) * 8

    def draw(self, canvas):
        if self._fill_paint and not self.hatched:
            # canvas.drawRoundRect(self._skia_rect, rx, ry, self._fill_paint)
            canvas.drawPath(self.fill_rect_path, self._fill_paint)
        if self.hatched:
            canvas.drawPath(self.hatched_lines, self._fill_paint)
        if self._border_paint:
            canvas.drawPath(self.border_rect_path, self._border_paint)
            # canvas.drawRoundRect(self._border_skia_rect, rx, ry, self._border_paint)


class PolicyBar(ice.DrawableWithChild):
    numbers: list[float]
    bar_labels: list[str]
    numbers_changed: list[float] = None
    bar_width: float = 30
    bar_height: float = 80
    bar_gap: float = 10
    bar_color: ice.Color = ice.Color.from_hex("#b2bec3")
    hatched_color: ice.Color = ice.Color.from_hex("#d63031")
    line_width: float = 2
    label_gap: float = 10
    label_font_size: float = 13
    label_font_family: str = "Fira Mono"
    line_overflow: float = 10
    ellipses: bool = True
    ellipses_gap: float = 20
    arrow_gap: float = 2
    min_height: float = 0.05
    use_tex: bool = False
    move_scale: float = 1
    hatched_end: float = 1

    def setup(self):
        numbers = [max(number, self.min_height) for number in self.numbers]
        numbers_changed = (
            [max(number, self.min_height) for number in self.numbers_changed]
            if self.numbers_changed is not None
            else None
        )
        bars = [
            HatchedRectangle(
                ice.Bounds.from_size(self.bar_height * number, self.bar_width + 2 * 2),
                border_position=ice.BorderPosition.INSIDE,
                fill_color=self.bar_color,
                border_color=self.bar_color,
                border_thickness=2,
                border_radius=(0, 5, 5, 0),
            )
            for number in numbers
        ]
        if numbers_changed is not None:
            assert len(self.numbers) == len(self.numbers_changed)

            bars_ghost = [
                HatchedRectangle(
                    ice.Bounds.from_size(
                        self.bar_height * number,
                        self.bar_width,
                    ),
                    fill_color=self.hatched_color,
                    border_color=self.hatched_color,
                    border_thickness=2,
                    border_radius=(0, 5, 5, 0),
                    hatched=True,
                    border_position=ice.BorderPosition.OUTSIDE,
                    hatched_angle=-45,
                    hatched_spacing=5,
                    partial_end=self.hatched_end,
                )
                for number in numbers_changed
            ]

        bars_arranged = bars[0]
        with bars_arranged:
            for bar in bars[1:]:
                bars_arranged += bar.pad_top(self.bar_gap).relative_to(
                    bars_arranged,
                    ice.TOP_LEFT,
                    ice.BOTTOM_LEFT,
                )

        if self.numbers_changed is not None:
            with bars_arranged:
                for i, bar_ghost in enumerate(bars_ghost):
                    bars_arranged += bar_ghost.relative_to(
                        bars[i],
                        ice.MIDDLE_LEFT,
                        ice.MIDDLE_LEFT,
                    )

            # Add arrows pointing to the new values.
            with bars_arranged:
                for i, (number_before, number_after) in enumerate(
                    zip(self.numbers, self.numbers_changed)
                ):
                    if abs(number_before - number_after) < 2e-2:
                        continue

                    sx, sy = bars[i].relative_bounds.corners[ice.MIDDLE_RIGHT]
                    ex, ey = bars_ghost[i].relative_bounds.corners[ice.MIDDLE_RIGHT]

                    if number_before < number_after:
                        sx += self.arrow_gap
                        ex -= self.arrow_gap
                    else:
                        ex += self.arrow_gap
                        sx -= self.arrow_gap

                    arrow = ice.Arrow(
                        (sx, sy),
                        (ex, ey),
                        line_path_style=ice.PathStyle(
                            color=self.hatched_color, thickness=2
                        ),
                        arrow_head_style=ice.ArrowHeadStyle.FILLED_TRIANGLE,
                        head_length=3,
                    )
                    if number_before < number_after:
                        arrow_placeholder = ice.Line(
                            (sx, sy),
                            (ex, ey),
                            path_style=ice.PathStyle(color=ice.WHITE, thickness=2),
                        )
                        bars_arranged += arrow_placeholder.scale(1, 4)
                    bars_arranged += arrow

        with bars_arranged:
            sx, sy = bars_arranged.relative_bounds.corners[ice.TOP_LEFT]
            ex, ey = bars_arranged.relative_bounds.corners[ice.BOTTOM_LEFT]

            line = ice.Line(
                (sx, sy - self.line_overflow),
                (ex, ey + self.line_overflow),
                path_style=ice.PathStyle(color=ice.BLACK, thickness=self.line_width),
            )
            bars_arranged += line.move(1, 0)

        last_text = None
        with bars_arranged:
            for i, label in enumerate(self.bar_labels):
                if self.use_tex:
                    text = ice.Tex(
                        tex=f"\\wmove{{{label}}}", preamble="\\usepackage{xskak}"
                    ).scale(self.bar_width / 15 * self.move_scale)
                else:
                    text = ice.Text(
                        label,
                        ice.FontStyle(
                            self.label_font_family, size=self.label_font_size
                        ),
                    )
                last_text = text
                bars_arranged += text.pad_right(self.label_gap).relative_to(
                    bars[i],
                    ice.MIDDLE_RIGHT,
                    ice.MIDDLE_LEFT,
                )

        if self.ellipses:
            ellipsis = ice.MathTex("\\ldots").scale(2)
            ellipsis = ice.Transform(child=ellipsis, rotation=90)

            with bars_arranged:
                bars_arranged += ellipsis.relative_to(
                    last_text,
                    ice.DOWN * self.ellipses_gap,
                )

        self.set_child(bars_arranged)
