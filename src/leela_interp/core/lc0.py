import contextlib
import functools
import os
from typing import Callable

import chess
import numpy as np
import onnx
import onnx2torch
import torch
from einops import rearrange

from .forward_pass_implementation import forward as modified_forward
from .leela_board import LeelaBoard


class Lc0Model(torch.nn.Module):
    POLICY_OUTPUT_SIZE = 1858
    D_MODEL = 768
    N_LAYERS = 15
    N_HEADS = 24

    def __init__(
        self,
        onnx_model_path=None,
        device=None,
        skip_model_check=True,
        sparring_use_history=False,
    ):
        super().__init__()
        if onnx_model_path is None:
            # Get from environment variable.
            onnx_model_path = os.environ.get("LC0_MODEL_PATH", None)

        if device is None:
            device = os.environ.get("DEVICE", "cpu")

        print(f"Using device: {device}")

        if onnx_model_path is None:
            raise ValueError(
                "onnx_model_path must be specified or LC0_MODEL_PATH environment variable must be set."
            )

        self._onnx_model_path = onnx_model_path
        self._device = device
        self._activations = {}
        self._hooks = []

        onnx_model = onnx.load(onnx_model_path)

        # Find the metadata for no_history.
        self._no_history = False
        for metadata in onnx_model.metadata_props:
            if metadata.key == "no_history":
                self._no_history = metadata.value.lower() == "true"
                break

        if not skip_model_check:
            onnx.checker.check_model(onnx_model, full_check=True)

        self._lc0_model = onnx2torch.convert(onnx_model)

        # Add some modules to expose residual activations in a more reasonable shape.
        post_attention = [torch.nn.Identity() for _ in range(self.N_LAYERS)]
        post_mlp = [torch.nn.Identity() for _ in range(self.N_LAYERS)]
        attention_output = [torch.nn.Identity() for _ in range(self.N_LAYERS)]
        mlp_output = [torch.nn.Identity() for _ in range(self.N_LAYERS)]
        self._lc0_model.post_attention = torch.nn.ModuleList(post_attention)
        self._lc0_model.post_mlp = torch.nn.ModuleList(post_mlp)
        self._lc0_model.attention_output = torch.nn.ModuleList(attention_output)
        self._lc0_model.mlp_output = torch.nn.ModuleList(mlp_output)

        self._lc0_model.to(device)

        self._is_sparring = not hasattr(
            self._lc0_model.initializers, "onnx_initializer_465"
        )
        self._sparring_use_history = sparring_use_history

        # Move all shapes to the CPU. This speeds things up because otherwise they'll
        # have to implicitly be moved there on every forward pass.
        if not self._is_sparring:
            for i in range(466):
                initializer = getattr(
                    self._lc0_model.initializers, f"onnx_initializer_{i}"
                )
                if (
                    # A bit of a hack but this detects the shapes Lc0 stores:
                    isinstance(initializer, torch.Tensor)
                    and (initializer.dtype in {torch.int64, torch.int32})
                    and (
                        # Shapes:
                        (initializer.ndim == 1 and initializer[0].item() == -1)
                        # Slices:
                        or initializer.shape == (1,)
                    )
                ):
                    setattr(
                        self._lc0_model.initializers,
                        f"onnx_initializer_{i}",
                        initializer.cpu(),
                    )
        # Disable gradient calculation and set the model to eval mode.
        # TODO: I don't think this is needed or does anything, the parameters aren't
        # pytorch parameters anyway, but luckily also don't need gradients by default.
        self._lc0_model.requires_grad_(False)
        self._lc0_model.eval()

    def forward(self, x, original_forward: bool = False):
        """Input/output behavior should be the same no matter `original_forward`.
        The modified forward pass (default) just exposes some additional activations
        via `nn.Identity` modules.
        """
        if self._is_sparring or original_forward:
            return self._lc0_model(x)
        return modified_forward(self._lc0_model, x)

    @property
    def onnx_model_path(self):
        """The path to the ONNX model."""
        return self._onnx_model_path

    @property
    def device(self):
        """The device on which the model is stored."""
        return self._device

    def _register_hooks(
        self,
        module_names: list[str],
        modifier_hooks: dict[str, Callable],
        gradients: bool,
    ):
        for name in module_names:
            module = self.modules[name]
            hook = module.register_forward_hook(
                self._get_activation(name, compute_gradient=gradients)
            )
            self._hooks.append(hook)

        # Register modifier hooks second so we get the unmodified activations if
        # we requested both.
        for name, hook in modifier_hooks.items():
            module = self.modules[name]
            # We assume the hook takes the module name as its first argument, before
            # the typical model, input, output arguments.
            hook = module.register_forward_hook(functools.partial(hook, name))
            self._hooks.append(hook)

    def _get_activation(self, name: str, compute_gradient: bool):
        def hook(model, input, output):
            if isinstance(output, torch.Tensor):
                if compute_gradient:
                    output.requires_grad = True
                    self._activations[name] = output
                    return output
                self._activations[name] = output.detach()

        return hook

    def _unregister_hooks(self):
        for hook in self._hooks:
            hook.remove()

        self._hooks = []

    @contextlib.contextmanager
    def capturing(
        self,
        module_names=None,
        modifier_hooks: dict[str, Callable] | None = None,
        gradients: bool = False,
    ):
        """Deprecated way to capture activations/gradients and do interventions.

        Based on manually adding pytorch hooks. Using the `Lc0sight` nnsight interface
        is more convenient in most cases, but these manual hooks might be useful
        if there are nnsight issues in a specific case.
        """
        assert self._activations == {}, "Activations are already being captured."
        assert self._hooks == [], "Hooks are already registered."

        if module_names is None:
            module_names = self.module_names

        if modifier_hooks is None:
            modifier_hooks = {}

        self._register_hooks(module_names, modifier_hooks, gradients)

        try:
            yield self._activations
        finally:
            self._unregister_hooks()
            self._activations = {}

    def make_inputs(self, boards: list[LeelaBoard]) -> torch.Tensor:
        if self._is_sparring and not self._sparring_use_history:
            boards = [
                LeelaBoard.from_fen(board.pc_board.fen(), history_synthesis="repeat")
                for board in boards
            ]

        return torch.tensor(
            np.array([board.lcz_features(self._no_history) for board in boards]),
            dtype=torch.float32,
            device=self.device,
        )

    def batch_play(
        self, boards: list[LeelaBoard], return_probs=True
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        model_inputs = self.make_inputs(boards)
        policy, wdl, *mlh = self(model_inputs)

        if return_probs:
            policy = self.logits_to_probs(boards, policy)
        return policy, wdl, mlh[0] if mlh else None

    def logits_to_probs(
        self,
        boards: list[LeelaBoard] | LeelaBoard,
        logits: torch.Tensor,
        legal_move_mask: torch.Tensor | None = None,  # Just to speed things up
    ) -> torch.Tensor:
        if isinstance(boards, LeelaBoard):
            boards = [boards]
        try:
            if logits.ndim == 1:
                logits = logits[None]
        except AttributeError:
            # For some reason this happens in nnsight tracing blocks, haven't debugged
            # yet. For now we just don't support logits without a batch dimension
            # there.
            pass

        if legal_move_mask is None:
            legal_move_mask = self._get_legal_move_mask(boards)
        logits[~legal_move_mask] = -torch.inf
        probs = torch.softmax(logits, dim=-1)
        return probs

    def _get_legal_move_mask(self, boards: list[LeelaBoard]) -> torch.Tensor:
        legal_move_mask = torch.zeros(
            (len(boards), self.POLICY_OUTPUT_SIZE), dtype=torch.bool
        )
        for i, board in enumerate(boards):
            legal_indices = torch.tensor(
                self.legal_moves(board)[0], device=self._device, dtype=torch.long
            )
            # Note: don't use scatter_ here because there's a bug on MPS:
            # https://github.com/pytorch/pytorch/issues/115152
            legal_move_mask[i, legal_indices] = True

        return legal_move_mask

    def play(
        self, board: LeelaBoard, return_probs: bool = True
    ) -> tuple[torch.Tensor, tuple[float, float, float], float | None]:
        """Returns the policy, WDL, and MLH for the given board.

        Args:
            board: The board to evaluate.
            return_probs: If True, return the policy as probabilities instead of logits.

        Returns:
            A tuple containing the policy, WDL, and MLH. The policy is the logits.
        """

        policy, wdl, mlh = self.batch_play([board], return_probs=return_probs)
        return (
            policy[0],
            tuple(x.item() for x in wdl[0]),
            mlh[0].item() if mlh is not None else None,
        )

    def pretty_play(self, board: LeelaBoard, top_k: int | None = 5):
        policy, wdl, _ = self.play(board, return_probs=True)
        print(
            "\n".join(
                f"{board.pc_board.san(chess.Move.from_uci(move))}: {prob:.2%}"
                for move, prob in self.top_moves(board, policy, top_k=top_k).items()
            )
        )
        print(f"W: {wdl[0]:.2%}, D: {wdl[1]:.2%}, L: {wdl[2]:.2%}")

    def legal_moves(self, board: LeelaBoard) -> tuple[list[int], list[str]]:
        legal_uci = [m.uci() for m in board.generate_legal_moves()]
        return board.batch_uci2idx(legal_uci), legal_uci

    def policy_as_dict(
        self, board: LeelaBoard, policy: torch.Tensor
    ) -> dict[str, float]:
        assert policy.shape == (self.POLICY_OUTPUT_SIZE,)
        legal_indices, legal_uci = self.legal_moves(board)
        return {
            uci: policy[index].item() for index, uci in zip(legal_indices, legal_uci)
        }

    def top_moves(
        self, board: LeelaBoard, policy: torch.Tensor, top_k: int | None = 1
    ) -> dict[str, float]:
        assert policy.shape == (self.POLICY_OUTPUT_SIZE,)
        legal_indices, legal_uci = self.legal_moves(board)
        if top_k is None:
            top_k = len(legal_indices)
        assert top_k > 0
        top_k_indices = torch.argsort(policy[legal_indices])[-top_k:].flip(0)
        return {legal_uci[i]: policy[legal_indices[i]].item() for i in top_k_indices}

    @property
    def model(self):
        """The underlying PyTorch model."""
        return self._lc0_model

    @property
    def module_names(self):
        """A list of the names of all modules in the model."""
        return [name for name, _ in self.model.named_modules()]

    @property
    def modules(self):
        """A list of all modules in the model."""
        return {name: module for name, module in self.model.named_modules()}

    def policy_head(self, x):
        encoder14_ln2 = rearrange(
            x, "batch squares d -> (batch squares) d", squares=64, d=self.D_MODEL
        )
        self = self.model

        # Copied from LC0Model.model.code
        initializers_onnx_initializer_432 = self.initializers.onnx_initializer_432
        policy_dense1_matmul = getattr(self, "policy/dense1/matmul")(
            encoder14_ln2, initializers_onnx_initializer_432
        )
        initializers_onnx_initializer_432 = None
        initializers_onnx_initializer_433 = self.initializers.onnx_initializer_433
        policy_dense1_add = getattr(self, "policy/dense1/add")(
            policy_dense1_matmul, initializers_onnx_initializer_433
        )
        policy_dense1_matmul = initializers_onnx_initializer_433 = None
        policy_dense1_mish_softplus = getattr(self, "policy/dense1/mish/softplus")(
            policy_dense1_add
        )
        policy_dense1_mish_tanh = getattr(self, "policy/dense1/mish/tanh")(
            policy_dense1_mish_softplus
        )
        policy_dense1_mish_softplus = None
        policy_dense1_mish = getattr(self, "policy/dense1/mish")(
            policy_dense1_mish_tanh, policy_dense1_add
        )
        policy_dense1_mish_tanh = policy_dense1_add = None
        initializers_onnx_initializer_434 = self.initializers.onnx_initializer_434
        policy_q_matmul = getattr(self, "policy/Q/matmul")(
            policy_dense1_mish, initializers_onnx_initializer_434
        )
        initializers_onnx_initializer_434 = None
        initializers_onnx_initializer_435 = self.initializers.onnx_initializer_435
        policy_q_add = getattr(self, "policy/Q/add")(
            policy_q_matmul, initializers_onnx_initializer_435
        )
        policy_q_matmul = initializers_onnx_initializer_435 = None
        initializers_onnx_initializer_436 = self.initializers.onnx_initializer_436
        policy_q_reshape = getattr(self, "policy/Q/reshape")(
            policy_q_add, initializers_onnx_initializer_436
        )
        policy_q_add = initializers_onnx_initializer_436 = None
        initializers_onnx_initializer_437 = self.initializers.onnx_initializer_437
        policy_k_matmul = getattr(self, "policy/K/matmul")(
            policy_dense1_mish, initializers_onnx_initializer_437
        )
        policy_dense1_mish = initializers_onnx_initializer_437 = None
        initializers_onnx_initializer_438 = self.initializers.onnx_initializer_438
        policy_k_add = getattr(self, "policy/K/add")(
            policy_k_matmul, initializers_onnx_initializer_438
        )
        policy_k_matmul = initializers_onnx_initializer_438 = None
        initializers_onnx_initializer_439 = self.initializers.onnx_initializer_439
        policy_k_reshape = getattr(self, "policy/K/reshape")(
            policy_k_add, initializers_onnx_initializer_439
        )
        policy_k_add = initializers_onnx_initializer_439 = None
        policy_k_transpose = getattr(self, "policy/K/transpose")(policy_k_reshape)
        policy_matmul = getattr(self, "policy/matmul")(
            policy_q_reshape, policy_k_transpose
        )
        policy_q_reshape = policy_k_transpose = None
        initializers_onnx_initializer_440 = self.initializers.onnx_initializer_440
        policy_scale = getattr(self, "policy/scale")(
            policy_matmul, initializers_onnx_initializer_440
        )
        policy_matmul = initializers_onnx_initializer_440 = None
        initializers_onnx_initializer_441 = self.initializers.onnx_initializer_441
        initializers_onnx_initializer_442 = self.initializers.onnx_initializer_442
        policy_promotion_slice = getattr(self, "policy/promotion/slice")(
            policy_k_reshape,
            initializers_onnx_initializer_441,
            initializers_onnx_initializer_442,
        )
        policy_k_reshape = initializers_onnx_initializer_441 = (
            initializers_onnx_initializer_442
        ) = None
        initializers_onnx_initializer_443 = self.initializers.onnx_initializer_443
        policy_promotion_matmul = getattr(self, "policy/promotion/matmul")(
            policy_promotion_slice, initializers_onnx_initializer_443
        )
        policy_promotion_slice = initializers_onnx_initializer_443 = None
        policy_promotion_transpose = getattr(self, "policy/promotion/transpose")(
            policy_promotion_matmul
        )
        policy_promotion_matmul = None
        initializers_onnx_initializer_444 = self.initializers.onnx_initializer_444
        policy_promotion_split = getattr(self, "policy/promotion/split")(
            policy_promotion_transpose, initializers_onnx_initializer_444
        )
        policy_promotion_transpose = initializers_onnx_initializer_444 = None
        getitem = policy_promotion_split[0]
        getitem_1 = policy_promotion_split[1]
        policy_promotion_split = None
        policy_promotion_add = getattr(self, "policy/promotion/add")(getitem, getitem_1)
        getitem = getitem_1 = None
        policy_promotion_transpose2 = getattr(self, "policy/promotion/transpose2")(
            policy_promotion_add
        )
        policy_promotion_add = None
        initializers_onnx_initializer_445 = self.initializers.onnx_initializer_445
        policy_promotion_reshape = getattr(self, "policy/promotion/reshape")(
            policy_promotion_transpose2, initializers_onnx_initializer_445
        )
        policy_promotion_transpose2 = initializers_onnx_initializer_445 = None
        initializers_onnx_initializer_446 = self.initializers.onnx_initializer_446
        initializers_onnx_initializer_447 = self.initializers.onnx_initializer_447
        policy_promotion_slice2 = getattr(self, "policy/promotion/slice2")(
            policy_scale,
            initializers_onnx_initializer_446,
            initializers_onnx_initializer_447,
        )
        initializers_onnx_initializer_446 = initializers_onnx_initializer_447 = None
        initializers_onnx_initializer_448 = self.initializers.onnx_initializer_448
        policy_promotion_reshape2 = getattr(self, "policy/promotion/reshape2")(
            policy_promotion_slice2, initializers_onnx_initializer_448
        )
        policy_promotion_slice2 = initializers_onnx_initializer_448 = None
        policy_promotion_concat = getattr(self, "policy/promotion/concat")(
            policy_promotion_reshape2,
            policy_promotion_reshape2,
            policy_promotion_reshape2,
        )
        policy_promotion_reshape2 = None
        initializers_onnx_initializer_449 = self.initializers.onnx_initializer_449
        policy_promotion_reshape3 = getattr(self, "policy/promotion/reshape3")(
            policy_promotion_concat, initializers_onnx_initializer_449
        )
        policy_promotion_concat = initializers_onnx_initializer_449 = None
        policy_promotion_add2 = getattr(self, "policy/promotion/add2")(
            policy_promotion_reshape3, policy_promotion_reshape
        )
        policy_promotion_reshape3 = policy_promotion_reshape = None
        initializers_onnx_initializer_450 = self.initializers.onnx_initializer_450
        policy_promotion_reshape4 = getattr(self, "policy/promotion/reshape4")(
            policy_promotion_add2, initializers_onnx_initializer_450
        )
        policy_promotion_add2 = initializers_onnx_initializer_450 = None
        policy_concat = getattr(self, "policy/concat")(
            policy_scale, policy_promotion_reshape4
        )
        policy_scale = policy_promotion_reshape4 = None
        initializers_onnx_initializer_451 = self.initializers.onnx_initializer_451
        policy_reshape = getattr(self, "policy/reshape")(
            policy_concat, initializers_onnx_initializer_451
        )
        policy_concat = initializers_onnx_initializer_451 = None
        initializers_onnx_initializer_452 = self.initializers.onnx_initializer_452
        output_policy = getattr(self, "output/policy")(
            policy_reshape, initializers_onnx_initializer_452
        )
        return output_policy

    def wdl_head(self, x, return_logits: bool = False):
        encoder14_ln2 = rearrange(
            x, "batch squares d -> (batch squares) d", squares=64, d=self.D_MODEL
        )
        self = self.model

        # Copied from LC0Model.model.code
        initializers_onnx_initializer_453 = self.initializers.onnx_initializer_453
        value_embed_matmul = getattr(self, "value/embed/matmul")(
            encoder14_ln2, initializers_onnx_initializer_453
        )
        initializers_onnx_initializer_453 = None
        initializers_onnx_initializer_454 = self.initializers.onnx_initializer_454
        value_embed_add = getattr(self, "value/embed/add")(
            value_embed_matmul, initializers_onnx_initializer_454
        )
        value_embed_matmul = initializers_onnx_initializer_454 = None
        value_embed_mish_softplus = getattr(self, "value/embed/mish/softplus")(
            value_embed_add
        )
        value_embed_mish_tanh = getattr(self, "value/embed/mish/tanh")(
            value_embed_mish_softplus
        )
        value_embed_mish_softplus = None
        value_embed_mish = getattr(self, "value/embed/mish")(
            value_embed_mish_tanh, value_embed_add
        )
        value_embed_mish_tanh = value_embed_add = None
        initializers_onnx_initializer_455 = self.initializers.onnx_initializer_455
        value_reshape = getattr(self, "value/reshape")(
            value_embed_mish, initializers_onnx_initializer_455
        )
        value_embed_mish = initializers_onnx_initializer_455 = None
        initializers_onnx_initializer_456 = self.initializers.onnx_initializer_456
        value_dense1_matmul = getattr(self, "value/dense1/matmul")(
            value_reshape, initializers_onnx_initializer_456
        )
        value_reshape = initializers_onnx_initializer_456 = None
        initializers_onnx_initializer_457 = self.initializers.onnx_initializer_457
        value_dense1_add = getattr(self, "value/dense1/add")(
            value_dense1_matmul, initializers_onnx_initializer_457
        )
        value_dense1_matmul = initializers_onnx_initializer_457 = None
        value_dense1_mish_softplus = getattr(self, "value/dense1/mish/softplus")(
            value_dense1_add
        )
        value_dense1_mish_tanh = getattr(self, "value/dense1/mish/tanh")(
            value_dense1_mish_softplus
        )
        value_dense1_mish_softplus = None
        value_dense1_mish = getattr(self, "value/dense1/mish")(
            value_dense1_mish_tanh, value_dense1_add
        )
        value_dense1_mish_tanh = value_dense1_add = None
        initializers_onnx_initializer_458 = self.initializers.onnx_initializer_458
        value_dense2_matmul = getattr(self, "value/dense2/matmul")(
            value_dense1_mish, initializers_onnx_initializer_458
        )
        value_dense1_mish = initializers_onnx_initializer_458 = None
        initializers_onnx_initializer_459 = self.initializers.onnx_initializer_459
        value_dense2_add = getattr(self, "value/dense2/add")(
            value_dense2_matmul, initializers_onnx_initializer_459
        )
        value_dense2_matmul = initializers_onnx_initializer_459 = None
        if return_logits:
            # skip the final softmax
            return value_dense2_add
        output_wdl = getattr(self, "output/wdl")(value_dense2_add)
        return output_wdl

    def mlh_head(self, x):
        encoder14_ln2 = rearrange(
            x, "batch squares d -> (batch squares) d", squares=64, d=self.D_MODEL
        )
        self = self.model

        # Copied from LC0Model.model.code
        initializers_onnx_initializer_460 = self.initializers.onnx_initializer_460
        mlh_embed_matmul = getattr(self, "mlh/embed/matmul")(
            encoder14_ln2, initializers_onnx_initializer_460
        )
        encoder14_ln2 = initializers_onnx_initializer_460 = None
        initializers_onnx_initializer_461 = self.initializers.onnx_initializer_461
        mlh_embed_add = getattr(self, "mlh/embed/add")(
            mlh_embed_matmul, initializers_onnx_initializer_461
        )
        mlh_embed_matmul = initializers_onnx_initializer_461 = None
        mlh_embed_mish_softplus = getattr(self, "mlh/embed/mish/softplus")(
            mlh_embed_add
        )
        mlh_embed_mish_tanh = getattr(self, "mlh/embed/mish/tanh")(
            mlh_embed_mish_softplus
        )
        mlh_embed_mish_softplus = None
        mlh_embed_mish = getattr(self, "mlh/embed/mish")(
            mlh_embed_mish_tanh, mlh_embed_add
        )
        mlh_embed_mish_tanh = mlh_embed_add = None
        initializers_onnx_initializer_462 = self.initializers.onnx_initializer_462
        mlh_reshape = getattr(self, "mlh/reshape")(
            mlh_embed_mish, initializers_onnx_initializer_462
        )
        mlh_embed_mish = initializers_onnx_initializer_462 = None
        initializers_onnx_initializer_463 = self.initializers.onnx_initializer_463
        mlh_dense1_matmul = getattr(self, "mlh/dense1/matmul")(
            mlh_reshape, initializers_onnx_initializer_463
        )
        mlh_reshape = initializers_onnx_initializer_463 = None
        initializers_onnx_initializer_464 = self.initializers.onnx_initializer_464
        mlh_dense1_add = getattr(self, "mlh/dense1/add")(
            mlh_dense1_matmul, initializers_onnx_initializer_464
        )
        mlh_dense1_matmul = initializers_onnx_initializer_464 = None
        mlh_dense1_mish_softplus = getattr(self, "mlh/dense1/mish/softplus")(
            mlh_dense1_add
        )
        mlh_dense1_mish_tanh = getattr(self, "mlh/dense1/mish/tanh")(
            mlh_dense1_mish_softplus
        )
        mlh_dense1_mish_softplus = None
        mlh_dense1_mish = getattr(self, "mlh/dense1/mish")(
            mlh_dense1_mish_tanh, mlh_dense1_add
        )
        mlh_dense1_mish_tanh = mlh_dense1_add = None
        initializers_onnx_initializer_465 = self.initializers.onnx_initializer_465
        mlh_dense2_matmul = getattr(self, "mlh/dense2/matmul")(
            mlh_dense1_mish, initializers_onnx_initializer_465
        )
        mlh_dense1_mish = initializers_onnx_initializer_465 = None
        initializers_onnx_initializer_466 = self.initializers.onnx_initializer_466
        mlh_dense2_add = getattr(self, "mlh/dense2/add")(
            mlh_dense2_matmul, initializers_onnx_initializer_466
        )
        mlh_dense2_matmul = initializers_onnx_initializer_466 = None
        mlh_dense2_mish_softplus = getattr(self, "mlh/dense2/mish/softplus")(
            mlh_dense2_add
        )
        mlh_dense2_mish_tanh = getattr(self, "mlh/dense2/mish/tanh")(
            mlh_dense2_mish_softplus
        )
        mlh_dense2_mish_softplus = None
        mlh_dense2_mish = getattr(self, "mlh/dense2/mish")(
            mlh_dense2_mish_tanh, mlh_dense2_add
        )
        mlh_dense2_mish_tanh = mlh_dense2_add = None
        output_mlh = getattr(self, "output/mlh")(mlh_dense2_mish)
        return output_mlh

    def final_attn_layer(self, x):
        encoder13_ln2 = rearrange(
            x, "batch squares d -> (batch squares) d", squares=64, d=self.D_MODEL
        )
        self = self.model

        initializers_onnx_initializer_404 = self.initializers.onnx_initializer_404
        encoder14_mha_q_w = getattr(self, "encoder14/mha/Q/w")(
            encoder13_ln2, initializers_onnx_initializer_404
        )
        initializers_onnx_initializer_404 = None
        initializers_onnx_initializer_405 = self.initializers.onnx_initializer_405
        encoder14_mha_q_b = getattr(self, "encoder14/mha/Q/b")(
            encoder14_mha_q_w, initializers_onnx_initializer_405
        )
        encoder14_mha_q_w = initializers_onnx_initializer_405 = None
        initializers_onnx_initializer_406 = self.initializers.onnx_initializer_406
        encoder14_mha_q_reshape = getattr(self, "encoder14/mha/Q/reshape")(
            encoder14_mha_q_b, initializers_onnx_initializer_406
        )
        encoder14_mha_q_b = initializers_onnx_initializer_406 = None
        encoder14_mha_q_transpose = getattr(self, "encoder14/mha/Q/transpose")(
            encoder14_mha_q_reshape
        )
        encoder14_mha_q_reshape = None
        initializers_onnx_initializer_407 = self.initializers.onnx_initializer_407
        encoder14_mha_k_w = getattr(self, "encoder14/mha/K/w")(
            encoder13_ln2, initializers_onnx_initializer_407
        )
        initializers_onnx_initializer_407 = None
        initializers_onnx_initializer_408 = self.initializers.onnx_initializer_408
        encoder14_mha_k_b = getattr(self, "encoder14/mha/K/b")(
            encoder14_mha_k_w, initializers_onnx_initializer_408
        )
        encoder14_mha_k_w = initializers_onnx_initializer_408 = None
        initializers_onnx_initializer_409 = self.initializers.onnx_initializer_409
        encoder14_mha_k_reshape = getattr(self, "encoder14/mha/K/reshape")(
            encoder14_mha_k_b, initializers_onnx_initializer_409
        )
        encoder14_mha_k_b = initializers_onnx_initializer_409 = None
        encoder14_mha_k_transpose = getattr(self, "encoder14/mha/K/transpose")(
            encoder14_mha_k_reshape
        )
        encoder14_mha_k_reshape = None
        initializers_onnx_initializer_410 = self.initializers.onnx_initializer_410
        encoder14_mha_v_w = getattr(self, "encoder14/mha/V/w")(
            encoder13_ln2, initializers_onnx_initializer_410
        )
        initializers_onnx_initializer_410 = None
        initializers_onnx_initializer_411 = self.initializers.onnx_initializer_411
        encoder14_mha_v_b = getattr(self, "encoder14/mha/V/b")(
            encoder14_mha_v_w, initializers_onnx_initializer_411
        )
        encoder14_mha_v_w = initializers_onnx_initializer_411 = None
        initializers_onnx_initializer_412 = self.initializers.onnx_initializer_412
        encoder14_mha_v_reshape = getattr(self, "encoder14/mha/V/reshape")(
            encoder14_mha_v_b, initializers_onnx_initializer_412
        )
        encoder14_mha_v_b = initializers_onnx_initializer_412 = None
        encoder14_mha_v_transpose = getattr(self, "encoder14/mha/V/transpose")(
            encoder14_mha_v_reshape
        )
        encoder14_mha_v_reshape = None
        encoder14_mha_qk_matmul = getattr(self, "encoder14/mha/QK/matmul")(
            encoder14_mha_q_transpose, encoder14_mha_k_transpose
        )
        encoder14_mha_q_transpose = encoder14_mha_k_transpose = None
        initializers_onnx_initializer_413 = self.initializers.onnx_initializer_413
        encoder14_mha_qk_scale = getattr(self, "encoder14/mha/QK/scale")(
            encoder14_mha_qk_matmul, initializers_onnx_initializer_413
        )
        encoder14_mha_qk_matmul = initializers_onnx_initializer_413 = None
        initializers_onnx_initializer_414 = self.initializers.onnx_initializer_414
        encoder14_smolgen_compress = getattr(self, "encoder14/smolgen/compress")(
            encoder13_ln2, initializers_onnx_initializer_414
        )
        initializers_onnx_initializer_414 = None
        initializers_onnx_initializer_415 = self.initializers.onnx_initializer_415
        encoder14_smolgen_compress_reshape = getattr(
            self, "encoder14/smolgen/compress/reshape"
        )(encoder14_smolgen_compress, initializers_onnx_initializer_415)
        encoder14_smolgen_compress = initializers_onnx_initializer_415 = None
        initializers_onnx_initializer_416 = self.initializers.onnx_initializer_416
        encoder14_smolgen_dense1_w = getattr(self, "encoder14/smolgen/dense1/w")(
            encoder14_smolgen_compress_reshape, initializers_onnx_initializer_416
        )
        encoder14_smolgen_compress_reshape = initializers_onnx_initializer_416 = None
        initializers_onnx_initializer_417 = self.initializers.onnx_initializer_417
        encoder14_smolgen_dense1_b = getattr(self, "encoder14/smolgen/dense1/b")(
            encoder14_smolgen_dense1_w, initializers_onnx_initializer_417
        )
        encoder14_smolgen_dense1_w = initializers_onnx_initializer_417 = None
        encoder14_smolgen_dense1_swish_sigmoid = getattr(
            self, "encoder14/smolgen/dense1/swish/sigmoid"
        )(encoder14_smolgen_dense1_b)
        encoder14_smolgen_dense1_swish = getattr(
            self, "encoder14/smolgen/dense1/swish"
        )(encoder14_smolgen_dense1_swish_sigmoid, encoder14_smolgen_dense1_b)
        encoder14_smolgen_dense1_swish_sigmoid = encoder14_smolgen_dense1_b = None
        encoder14_smolgen_ln1 = getattr(self, "encoder14/smolgen/ln1")(
            encoder14_smolgen_dense1_swish
        )
        encoder14_smolgen_dense1_swish = None
        initializers_onnx_initializer_418 = self.initializers.onnx_initializer_418
        encoder14_smolgen_dense2_w = getattr(self, "encoder14/smolgen/dense2/w")(
            encoder14_smolgen_ln1, initializers_onnx_initializer_418
        )
        encoder14_smolgen_ln1 = initializers_onnx_initializer_418 = None
        initializers_onnx_initializer_419 = self.initializers.onnx_initializer_419
        encoder14_smolgen_dense2_b = getattr(self, "encoder14/smolgen/dense2/b")(
            encoder14_smolgen_dense2_w, initializers_onnx_initializer_419
        )
        encoder14_smolgen_dense2_w = initializers_onnx_initializer_419 = None
        encoder14_smolgen_dense2_swish_sigmoid = getattr(
            self, "encoder14/smolgen/dense2/swish/sigmoid"
        )(encoder14_smolgen_dense2_b)
        encoder14_smolgen_dense2_swish = getattr(
            self, "encoder14/smolgen/dense2/swish"
        )(encoder14_smolgen_dense2_swish_sigmoid, encoder14_smolgen_dense2_b)
        encoder14_smolgen_dense2_swish_sigmoid = encoder14_smolgen_dense2_b = None
        encoder14_smolgen_ln2 = getattr(self, "encoder14/smolgen/ln2")(
            encoder14_smolgen_dense2_swish
        )
        encoder14_smolgen_dense2_swish = None
        initializers_onnx_initializer_420 = self.initializers.onnx_initializer_420
        encoder14_smolgen_gen_from_reshape = getattr(
            self, "encoder14/smolgen/gen_from/reshape"
        )(encoder14_smolgen_ln2, initializers_onnx_initializer_420)
        encoder14_smolgen_ln2 = initializers_onnx_initializer_420 = None
        initializers_onnx_initializer_421 = self.initializers.onnx_initializer_421
        encoder14_smolgen_smol_weight_gen = getattr(
            self, "encoder14/smolgen/smol_weight_gen"
        )(encoder14_smolgen_gen_from_reshape, initializers_onnx_initializer_421)
        encoder14_smolgen_gen_from_reshape = initializers_onnx_initializer_421 = None
        initializers_onnx_initializer_422 = self.initializers.onnx_initializer_422
        encoder14_smolgen_out_reshape = getattr(self, "encoder14/smolgen/out/reshape")(
            encoder14_smolgen_smol_weight_gen, initializers_onnx_initializer_422
        )
        encoder14_smolgen_smol_weight_gen = initializers_onnx_initializer_422 = None
        encoder14_smolgen_weights = getattr(self, "encoder14/smolgen_weights")(
            encoder14_mha_qk_scale, encoder14_smolgen_out_reshape
        )
        encoder14_mha_qk_scale = encoder14_smolgen_out_reshape = None
        encoder14_mha_qk_softmax = getattr(self, "encoder14/mha/QK/softmax")(
            encoder14_smolgen_weights
        )
        encoder14_smolgen_weights = None
        encoder14_mha_qkv_matmul = getattr(self, "encoder14/mha/QKV/matmul")(
            encoder14_mha_qk_softmax, encoder14_mha_v_transpose
        )
        encoder14_mha_qk_softmax = encoder14_mha_v_transpose = None
        encoder14_mha_out_transpose = getattr(self, "encoder14/mha/out/transpose")(
            encoder14_mha_qkv_matmul
        )
        encoder14_mha_qkv_matmul = None
        initializers_onnx_initializer_423 = self.initializers.onnx_initializer_423
        encoder14_mha_out_reshape = getattr(self, "encoder14/mha/out/reshape")(
            encoder14_mha_out_transpose, initializers_onnx_initializer_423
        )
        encoder14_mha_out_transpose = initializers_onnx_initializer_423 = None
        initializers_onnx_initializer_424 = self.initializers.onnx_initializer_424
        encoder14_mha_out_dense_w = getattr(self, "encoder14/mha/out/dense/w")(
            encoder14_mha_out_reshape, initializers_onnx_initializer_424
        )
        encoder14_mha_out_reshape = initializers_onnx_initializer_424 = None
        initializers_onnx_initializer_425 = self.initializers.onnx_initializer_425
        encoder14_mha_out_dense_b = getattr(self, "encoder14/mha/out/dense/b")(
            encoder14_mha_out_dense_w, initializers_onnx_initializer_425
        )
        encoder14_mha_out_dense_w = initializers_onnx_initializer_425 = None
        initializers_onnx_initializer_426 = self.initializers.onnx_initializer_426
        encoder14_alpha_input = getattr(self, "encoder14/alpha*input")(
            encoder13_ln2, initializers_onnx_initializer_426
        )
        encoder13_ln2 = initializers_onnx_initializer_426 = None
        encoder14_mha_out_skip = getattr(self, "encoder14/mha/out/skip")(
            encoder14_mha_out_dense_b, encoder14_alpha_input
        )
        encoder14_mha_out_dense_b = encoder14_alpha_input = None
        encoder14_ln1 = getattr(self, "encoder14/ln1")(encoder14_mha_out_skip)
        encoder14_mha_out_skip = None
        return rearrange(
            encoder14_ln1, "(batch squares) d -> batch squares d", squares=64
        )

    def final_mlp_layer(self, x):
        encoder14_ln1 = rearrange(
            x, "batch squares d -> (batch squares) d", squares=64, d=self.D_MODEL
        )
        self = self.model

        initializers_onnx_initializer_427 = self.initializers.onnx_initializer_427
        encoder14_ffn_dense1_w = getattr(self, "encoder14/ffn/dense1/w")(
            encoder14_ln1, initializers_onnx_initializer_427
        )
        initializers_onnx_initializer_427 = None
        initializers_onnx_initializer_428 = self.initializers.onnx_initializer_428
        encoder14_ffn_dense1_b = getattr(self, "encoder14/ffn/dense1/b")(
            encoder14_ffn_dense1_w, initializers_onnx_initializer_428
        )
        encoder14_ffn_dense1_w = initializers_onnx_initializer_428 = None
        encoder14_ffn_dense1_sqrrelu_relu = getattr(
            self, "encoder14/ffn/dense1/sqrrelu/relu"
        )(encoder14_ffn_dense1_b)
        encoder14_ffn_dense1_b = None
        encoder14_ffn_dense1_sqrrelu_sqr = getattr(
            self, "encoder14/ffn/dense1/sqrrelu/sqr"
        )(encoder14_ffn_dense1_sqrrelu_relu, encoder14_ffn_dense1_sqrrelu_relu)
        encoder14_ffn_dense1_sqrrelu_relu = None
        initializers_onnx_initializer_429 = self.initializers.onnx_initializer_429
        encoder14_ffn_dense2_w = getattr(self, "encoder14/ffn/dense2/w")(
            encoder14_ffn_dense1_sqrrelu_sqr, initializers_onnx_initializer_429
        )
        encoder14_ffn_dense1_sqrrelu_sqr = initializers_onnx_initializer_429 = None
        initializers_onnx_initializer_430 = self.initializers.onnx_initializer_430
        encoder14_ffn_dense2_b = getattr(self, "encoder14/ffn/dense2/b")(
            encoder14_ffn_dense2_w, initializers_onnx_initializer_430
        )
        encoder14_ffn_dense2_w = initializers_onnx_initializer_430 = None
        initializers_onnx_initializer_431 = self.initializers.onnx_initializer_431
        encoder14_alpha_out1 = getattr(self, "encoder14/alpha*out1")(
            encoder14_ln1, initializers_onnx_initializer_431
        )
        encoder14_ln1 = initializers_onnx_initializer_431 = None
        encoder14_ffn_skip = getattr(self, "encoder14/ffn/skip")(
            encoder14_ffn_dense2_b, encoder14_alpha_out1
        )
        encoder14_ffn_dense2_b = encoder14_alpha_out1 = None
        encoder14_ln2 = getattr(self, "encoder14/ln2")(encoder14_ffn_skip)
        encoder14_ffn_skip = None

        return rearrange(
            encoder14_ln2, "(batch squares) d -> batch squares d", squares=64
        )
