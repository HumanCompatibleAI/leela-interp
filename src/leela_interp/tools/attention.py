import matplotlib
import numpy as np
import torch
from leela_interp import LeelaBoard
from leela_interp.core.iceberg_board import palette

TORCH_INDEX = int | slice | list[int] | torch.Tensor | np.ndarray


def attention_attribution(
    boards: list[LeelaBoard],
    layer: int,
    head: int,
    model,
    mode="policy-grad",
    return_pt=False,
    top_move_indices: torch.Tensor | None = None,
) -> np.ndarray | torch.Tensor:
    name = f"encoder{layer}/mha/QK/softmax"
    with model.capturing(
        [name], gradients=(mode in {"policy-grad", "value-grad"})
    ) as activations:
        policy, wdl, _ = model.batch_play(boards, return_probs=True)

    if mode == "attention":
        result = activations[name][:, head]
    elif mode == "policy-grad":
        if top_move_indices is None:
            top_move_indices = policy.argmax(dim=-1)
        top_move_probs = policy.gather(-1, top_move_indices.unsqueeze(-1)).squeeze(-1)
        top_move_probs.backward(torch.ones_like(top_move_probs))
        result = activations[name][:, head] * activations[name].grad[:, head]
    elif mode == "value-grad":
        win_probs = wdl[:, 0]
        win_probs.backward(torch.ones_like(win_probs))
        result = activations[name][:, head] * activations[name].grad[:, head]
    else:
        raise ValueError(
            f"Unknown mode: {mode}. "
            "Must be one of 'attention', 'policy-grad', 'value-grad'."
        )
    if return_pt:
        return result.detach()
    else:
        return result.detach().cpu().numpy()


def top_k_attributions(
    attributions: torch.Tensor, board: LeelaBoard, k: int = 5, color_mappable=None
):
    assert attributions.shape == (64, 64)
    values, indices = torch.topk(attributions.abs().view(-1), k)
    query_indices, key_indices = np.unravel_index(
        indices.cpu().numpy(), attributions.shape
    )
    query_squares = [board.idx2sq(idx) for idx in query_indices]
    key_squares = [board.idx2sq(idx) for idx in key_indices]
    colors = {}
    values = {}

    if color_mappable is None:
        _, color_mappable = palette(
            attributions.cpu().numpy().ravel(), cmap="bwr", zero_center=True
        )
    for i, (query, key) in enumerate(zip(query_squares, key_squares)):
        value = attributions.view(-1)[indices[i]].item()
        values[f"{key}{query}"] = value
        rgba_color = color_mappable.to_rgba(value, alpha=0.8)
        colors[f"{key}{query}"] = (
            matplotlib.colors.rgb2hex(rgba_color[:3]) + f"{int(rgba_color[3]*255):02x}"
        )

    return values, colors
