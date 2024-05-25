"""This file contains the implementation of Leela's forward pass.

It is based on code auto-generated from the ONNX model using `onnx2torch`, but we made
a few additions to make it compatible with `nnsight` (basically exposing residual
activations with a batch dimension---the ONNX implementations flattens batch and
sequence dimension in residual stream activations and MLP/attention outputs).

You don't need to call this manually, it's called by `LC0Model.forward`.
You can access the unmodified forward pass code under `LC0Model._lc0_model.code`.
"""


def forward(self, input_1):
    ########################################################################
    # Embedding stuff
    ########################################################################
    attn_body_transpose = getattr(self, "attn_body/transpose")(input_1)
    input_1 = None
    initializers_onnx_initializer_0 = self.initializers.onnx_initializer_0
    attn_body_reshape = getattr(self, "attn_body/reshape")(
        attn_body_transpose, initializers_onnx_initializer_0
    )
    attn_body_transpose = initializers_onnx_initializer_0 = None
    attn_body_shape = getattr(self, "attn_body/shape")(attn_body_reshape)
    initializers_onnx_initializer_1 = self.initializers.onnx_initializer_1
    initializers_onnx_initializer_2 = self.initializers.onnx_initializer_2
    attn_body_batch = getattr(self, "attn_body/batch")(
        attn_body_shape,
        initializers_onnx_initializer_1,
        initializers_onnx_initializer_2,
    )
    attn_body_shape = initializers_onnx_initializer_1 = (
        initializers_onnx_initializer_2
    ) = None
    initializers_onnx_initializer_3 = self.initializers.onnx_initializer_3
    attn_body_pos_encoding_shape = getattr(self, "attn_body/pos_encoding_shape")(
        attn_body_batch, initializers_onnx_initializer_3
    )
    attn_body_batch = initializers_onnx_initializer_3 = None
    initializers_onnx_initializer_4 = self.initializers.onnx_initializer_4
    attn_body_expand = getattr(self, "attn_body/expand")(
        initializers_onnx_initializer_4, attn_body_pos_encoding_shape
    )
    initializers_onnx_initializer_4 = attn_body_pos_encoding_shape = None
    attn_body_padded_input = getattr(self, "attn_body/padded_input")(
        attn_body_reshape, attn_body_expand
    )
    attn_body_reshape = attn_body_expand = None
    initializers_onnx_initializer_5 = self.initializers.onnx_initializer_5
    attn_body_reshape2 = getattr(self, "attn_body/reshape2")(
        attn_body_padded_input, initializers_onnx_initializer_5
    )
    attn_body_padded_input = initializers_onnx_initializer_5 = None
    initializers_onnx_initializer_6 = self.initializers.onnx_initializer_6
    attn_body_matmul = getattr(self, "attn_body/matmul")(
        attn_body_reshape2, initializers_onnx_initializer_6
    )
    attn_body_reshape2 = initializers_onnx_initializer_6 = None
    initializers_onnx_initializer_7 = self.initializers.onnx_initializer_7
    attn_body_add = getattr(self, "attn_body/add")(
        attn_body_matmul, initializers_onnx_initializer_7
    )
    attn_body_matmul = initializers_onnx_initializer_7 = None
    attn_body_mish_softplus = getattr(self, "attn_body/mish/softplus")(attn_body_add)
    attn_body_mish_tanh = getattr(self, "attn_body/mish/tanh")(attn_body_mish_softplus)
    attn_body_mish_softplus = None
    attn_body_mish = getattr(self, "attn_body/mish")(attn_body_mish_tanh, attn_body_add)
    attn_body_mish_tanh = attn_body_add = None
    initializers_onnx_initializer_8 = self.initializers.onnx_initializer_8
    attn_body_ma_gating_rehape1 = getattr(self, "attn_body/ma_gating/rehape1")(
        attn_body_mish, initializers_onnx_initializer_8
    )
    attn_body_mish = initializers_onnx_initializer_8 = None
    initializers_onnx_initializer_9 = self.initializers.onnx_initializer_9
    ip_mul_gate = self.ip_mul_gate(
        attn_body_ma_gating_rehape1, initializers_onnx_initializer_9
    )
    attn_body_ma_gating_rehape1 = initializers_onnx_initializer_9 = None
    initializers_onnx_initializer_10 = self.initializers.onnx_initializer_10
    ip_add_gate = self.ip_add_gate(ip_mul_gate, initializers_onnx_initializer_10)
    ip_mul_gate = initializers_onnx_initializer_10 = None
    initializers_onnx_initializer_11 = self.initializers.onnx_initializer_11
    attn_body_ma_gating_rehape2 = getattr(self, "attn_body/ma_gating/rehape2")(
        ip_add_gate, initializers_onnx_initializer_11
    )
    ip_add_gate = initializers_onnx_initializer_11 = None

    ########################################################################
    # Layer 0
    ########################################################################
    initializers_onnx_initializer_12 = self.initializers.onnx_initializer_12
    encoder0_mha_q_w = getattr(self, "encoder0/mha/Q/w")(
        attn_body_ma_gating_rehape2, initializers_onnx_initializer_12
    )
    initializers_onnx_initializer_12 = None
    initializers_onnx_initializer_13 = self.initializers.onnx_initializer_13
    encoder0_mha_q_b = getattr(self, "encoder0/mha/Q/b")(
        encoder0_mha_q_w, initializers_onnx_initializer_13
    )
    encoder0_mha_q_w = initializers_onnx_initializer_13 = None
    initializers_onnx_initializer_14 = self.initializers.onnx_initializer_14
    encoder0_mha_q_reshape = getattr(self, "encoder0/mha/Q/reshape")(
        encoder0_mha_q_b, initializers_onnx_initializer_14
    )
    encoder0_mha_q_b = initializers_onnx_initializer_14 = None
    encoder0_mha_q_transpose = getattr(self, "encoder0/mha/Q/transpose")(
        encoder0_mha_q_reshape
    )
    encoder0_mha_q_reshape = None
    initializers_onnx_initializer_15 = self.initializers.onnx_initializer_15
    encoder0_mha_k_w = getattr(self, "encoder0/mha/K/w")(
        attn_body_ma_gating_rehape2, initializers_onnx_initializer_15
    )
    initializers_onnx_initializer_15 = None
    initializers_onnx_initializer_16 = self.initializers.onnx_initializer_16
    encoder0_mha_k_b = getattr(self, "encoder0/mha/K/b")(
        encoder0_mha_k_w, initializers_onnx_initializer_16
    )
    encoder0_mha_k_w = initializers_onnx_initializer_16 = None
    initializers_onnx_initializer_17 = self.initializers.onnx_initializer_17
    encoder0_mha_k_reshape = getattr(self, "encoder0/mha/K/reshape")(
        encoder0_mha_k_b, initializers_onnx_initializer_17
    )
    encoder0_mha_k_b = initializers_onnx_initializer_17 = None
    encoder0_mha_k_transpose = getattr(self, "encoder0/mha/K/transpose")(
        encoder0_mha_k_reshape
    )
    encoder0_mha_k_reshape = None
    initializers_onnx_initializer_18 = self.initializers.onnx_initializer_18
    encoder0_mha_v_w = getattr(self, "encoder0/mha/V/w")(
        attn_body_ma_gating_rehape2, initializers_onnx_initializer_18
    )
    initializers_onnx_initializer_18 = None
    initializers_onnx_initializer_19 = self.initializers.onnx_initializer_19
    encoder0_mha_v_b = getattr(self, "encoder0/mha/V/b")(
        encoder0_mha_v_w, initializers_onnx_initializer_19
    )
    encoder0_mha_v_w = initializers_onnx_initializer_19 = None
    initializers_onnx_initializer_20 = self.initializers.onnx_initializer_20
    encoder0_mha_v_reshape = getattr(self, "encoder0/mha/V/reshape")(
        encoder0_mha_v_b, initializers_onnx_initializer_20
    )
    encoder0_mha_v_b = initializers_onnx_initializer_20 = None
    encoder0_mha_v_transpose = getattr(self, "encoder0/mha/V/transpose")(
        encoder0_mha_v_reshape
    )
    encoder0_mha_v_reshape = None
    encoder0_mha_qk_matmul = getattr(self, "encoder0/mha/QK/matmul")(
        encoder0_mha_q_transpose, encoder0_mha_k_transpose
    )
    encoder0_mha_q_transpose = encoder0_mha_k_transpose = None
    initializers_onnx_initializer_21 = self.initializers.onnx_initializer_21
    encoder0_mha_qk_scale = getattr(self, "encoder0/mha/QK/scale")(
        encoder0_mha_qk_matmul, initializers_onnx_initializer_21
    )
    encoder0_mha_qk_matmul = initializers_onnx_initializer_21 = None
    initializers_onnx_initializer_22 = self.initializers.onnx_initializer_22
    encoder0_smolgen_compress = getattr(self, "encoder0/smolgen/compress")(
        attn_body_ma_gating_rehape2, initializers_onnx_initializer_22
    )
    initializers_onnx_initializer_22 = None
    initializers_onnx_initializer_23 = self.initializers.onnx_initializer_23
    encoder0_smolgen_compress_reshape = getattr(
        self, "encoder0/smolgen/compress/reshape"
    )(encoder0_smolgen_compress, initializers_onnx_initializer_23)
    encoder0_smolgen_compress = initializers_onnx_initializer_23 = None
    initializers_onnx_initializer_24 = self.initializers.onnx_initializer_24
    encoder0_smolgen_dense1_w = getattr(self, "encoder0/smolgen/dense1/w")(
        encoder0_smolgen_compress_reshape, initializers_onnx_initializer_24
    )
    encoder0_smolgen_compress_reshape = initializers_onnx_initializer_24 = None
    initializers_onnx_initializer_25 = self.initializers.onnx_initializer_25
    encoder0_smolgen_dense1_b = getattr(self, "encoder0/smolgen/dense1/b")(
        encoder0_smolgen_dense1_w, initializers_onnx_initializer_25
    )
    encoder0_smolgen_dense1_w = initializers_onnx_initializer_25 = None
    encoder0_smolgen_dense1_swish_sigmoid = getattr(
        self, "encoder0/smolgen/dense1/swish/sigmoid"
    )(encoder0_smolgen_dense1_b)
    encoder0_smolgen_dense1_swish = getattr(self, "encoder0/smolgen/dense1/swish")(
        encoder0_smolgen_dense1_swish_sigmoid, encoder0_smolgen_dense1_b
    )
    encoder0_smolgen_dense1_swish_sigmoid = encoder0_smolgen_dense1_b = None
    encoder0_smolgen_ln1 = getattr(self, "encoder0/smolgen/ln1")(
        encoder0_smolgen_dense1_swish
    )
    encoder0_smolgen_dense1_swish = None
    initializers_onnx_initializer_26 = self.initializers.onnx_initializer_26
    encoder0_smolgen_dense2_w = getattr(self, "encoder0/smolgen/dense2/w")(
        encoder0_smolgen_ln1, initializers_onnx_initializer_26
    )
    encoder0_smolgen_ln1 = initializers_onnx_initializer_26 = None
    initializers_onnx_initializer_27 = self.initializers.onnx_initializer_27
    encoder0_smolgen_dense2_b = getattr(self, "encoder0/smolgen/dense2/b")(
        encoder0_smolgen_dense2_w, initializers_onnx_initializer_27
    )
    encoder0_smolgen_dense2_w = initializers_onnx_initializer_27 = None
    encoder0_smolgen_dense2_swish_sigmoid = getattr(
        self, "encoder0/smolgen/dense2/swish/sigmoid"
    )(encoder0_smolgen_dense2_b)
    encoder0_smolgen_dense2_swish = getattr(self, "encoder0/smolgen/dense2/swish")(
        encoder0_smolgen_dense2_swish_sigmoid, encoder0_smolgen_dense2_b
    )
    encoder0_smolgen_dense2_swish_sigmoid = encoder0_smolgen_dense2_b = None
    encoder0_smolgen_ln2 = getattr(self, "encoder0/smolgen/ln2")(
        encoder0_smolgen_dense2_swish
    )
    encoder0_smolgen_dense2_swish = None
    initializers_onnx_initializer_28 = self.initializers.onnx_initializer_28
    encoder0_smolgen_gen_from_reshape = getattr(
        self, "encoder0/smolgen/gen_from/reshape"
    )(encoder0_smolgen_ln2, initializers_onnx_initializer_28)
    encoder0_smolgen_ln2 = initializers_onnx_initializer_28 = None
    initializers_onnx_initializer_29 = self.initializers.onnx_initializer_29
    encoder0_smolgen_smol_weight_gen = getattr(
        self, "encoder0/smolgen/smol_weight_gen"
    )(encoder0_smolgen_gen_from_reshape, initializers_onnx_initializer_29)
    encoder0_smolgen_gen_from_reshape = initializers_onnx_initializer_29 = None
    initializers_onnx_initializer_30 = self.initializers.onnx_initializer_30
    encoder0_smolgen_out_reshape = getattr(self, "encoder0/smolgen/out/reshape")(
        encoder0_smolgen_smol_weight_gen, initializers_onnx_initializer_30
    )
    encoder0_smolgen_smol_weight_gen = initializers_onnx_initializer_30 = None
    encoder0_smolgen_weights = getattr(self, "encoder0/smolgen_weights")(
        encoder0_mha_qk_scale, encoder0_smolgen_out_reshape
    )
    encoder0_mha_qk_scale = encoder0_smolgen_out_reshape = None
    encoder0_mha_qk_softmax = getattr(self, "encoder0/mha/QK/softmax")(
        encoder0_smolgen_weights
    )
    encoder0_smolgen_weights = None
    encoder0_mha_qkv_matmul = getattr(self, "encoder0/mha/QKV/matmul")(
        encoder0_mha_qk_softmax, encoder0_mha_v_transpose
    )
    encoder0_mha_qk_softmax = encoder0_mha_v_transpose = None
    encoder0_mha_out_transpose = getattr(self, "encoder0/mha/out/transpose")(
        encoder0_mha_qkv_matmul
    )
    encoder0_mha_qkv_matmul = None
    initializers_onnx_initializer_31 = self.initializers.onnx_initializer_31
    encoder0_mha_out_reshape = getattr(self, "encoder0/mha/out/reshape")(
        encoder0_mha_out_transpose, initializers_onnx_initializer_31
    )
    encoder0_mha_out_transpose = initializers_onnx_initializer_31 = None
    initializers_onnx_initializer_32 = self.initializers.onnx_initializer_32
    encoder0_mha_out_dense_w = getattr(self, "encoder0/mha/out/dense/w")(
        encoder0_mha_out_reshape, initializers_onnx_initializer_32
    )
    encoder0_mha_out_reshape = initializers_onnx_initializer_32 = None
    initializers_onnx_initializer_33 = self.initializers.onnx_initializer_33
    encoder0_mha_out_dense_b = getattr(self, "encoder0/mha/out/dense/b")(
        encoder0_mha_out_dense_w, initializers_onnx_initializer_33
    )
    encoder0_mha_out_dense_w = initializers_onnx_initializer_33 = None

    encoder0_mha_out_dense_b = self.attention_output[0](
        encoder0_mha_out_dense_b.view(-1, 64, 768)
    ).view(-1, 768)

    initializers_onnx_initializer_34 = self.initializers.onnx_initializer_34
    encoder0_alpha_input = getattr(self, "encoder0/alpha*input")(
        attn_body_ma_gating_rehape2, initializers_onnx_initializer_34
    )
    attn_body_ma_gating_rehape2 = initializers_onnx_initializer_34 = None
    encoder0_mha_out_skip = getattr(self, "encoder0/mha/out/skip")(
        encoder0_mha_out_dense_b, encoder0_alpha_input
    )
    encoder0_mha_out_dense_b = encoder0_alpha_input = None
    encoder0_ln1 = getattr(self, "encoder0/ln1")(encoder0_mha_out_skip)
    encoder0_mha_out_skip = None

    ########################################################################
    # Make residuals accessible with squares as a separate dim,
    # this is important for nnsight.
    ########################################################################
    # This is just an nn.Identity layer
    residual_reasonable_shape = self.post_attention[0](
        encoder0_ln1.reshape(-1, 64, 768)
    )
    encoder0_ln1 = residual_reasonable_shape.reshape(-1, 768)
    residual_reasonable_shape = None

    initializers_onnx_initializer_35 = self.initializers.onnx_initializer_35
    encoder0_ffn_dense1_w = getattr(self, "encoder0/ffn/dense1/w")(
        encoder0_ln1, initializers_onnx_initializer_35
    )
    initializers_onnx_initializer_35 = None
    initializers_onnx_initializer_36 = self.initializers.onnx_initializer_36
    encoder0_ffn_dense1_b = getattr(self, "encoder0/ffn/dense1/b")(
        encoder0_ffn_dense1_w, initializers_onnx_initializer_36
    )
    encoder0_ffn_dense1_w = initializers_onnx_initializer_36 = None
    encoder0_ffn_dense1_sqrrelu_relu = getattr(
        self, "encoder0/ffn/dense1/sqrrelu/relu"
    )(encoder0_ffn_dense1_b)
    encoder0_ffn_dense1_b = None
    encoder0_ffn_dense1_sqrrelu_sqr = getattr(self, "encoder0/ffn/dense1/sqrrelu/sqr")(
        encoder0_ffn_dense1_sqrrelu_relu, encoder0_ffn_dense1_sqrrelu_relu
    )
    encoder0_ffn_dense1_sqrrelu_relu = None
    initializers_onnx_initializer_37 = self.initializers.onnx_initializer_37
    encoder0_ffn_dense2_w = getattr(self, "encoder0/ffn/dense2/w")(
        encoder0_ffn_dense1_sqrrelu_sqr, initializers_onnx_initializer_37
    )
    encoder0_ffn_dense1_sqrrelu_sqr = initializers_onnx_initializer_37 = None
    initializers_onnx_initializer_38 = self.initializers.onnx_initializer_38
    encoder0_ffn_dense2_b = getattr(self, "encoder0/ffn/dense2/b")(
        encoder0_ffn_dense2_w, initializers_onnx_initializer_38
    )

    encoder0_ffn_dense2_b = self.mlp_output[0](
        encoder0_ffn_dense2_b.view(-1, 64, 768)
    ).view(-1, 768)

    encoder0_ffn_dense2_w = initializers_onnx_initializer_38 = None
    initializers_onnx_initializer_39 = self.initializers.onnx_initializer_39
    encoder0_alpha_out1 = getattr(self, "encoder0/alpha*out1")(
        encoder0_ln1, initializers_onnx_initializer_39
    )
    encoder0_ln1 = initializers_onnx_initializer_39 = None
    encoder0_ffn_skip = getattr(self, "encoder0/ffn/skip")(
        encoder0_ffn_dense2_b, encoder0_alpha_out1
    )
    encoder0_ffn_dense2_b = encoder0_alpha_out1 = None
    encoder0_ln2 = getattr(self, "encoder0/ln2")(encoder0_ffn_skip)
    encoder0_ffn_skip = None

    ########################################################################
    # Make residuals accessible with squares as a separate dim,
    # this is important for nnsight.
    ########################################################################
    # This is just an nn.Identity layer
    residual_reasonable_shape = self.post_mlp[0](encoder0_ln2.reshape(-1, 64, 768))
    encoder0_ln2 = residual_reasonable_shape.reshape(-1, 768)
    residual_reasonable_shape = None

    ########################################################################
    # Layer 1
    ########################################################################
    initializers_onnx_initializer_40 = self.initializers.onnx_initializer_40
    encoder1_mha_q_w = getattr(self, "encoder1/mha/Q/w")(
        encoder0_ln2, initializers_onnx_initializer_40
    )
    initializers_onnx_initializer_40 = None
    initializers_onnx_initializer_41 = self.initializers.onnx_initializer_41
    encoder1_mha_q_b = getattr(self, "encoder1/mha/Q/b")(
        encoder1_mha_q_w, initializers_onnx_initializer_41
    )
    encoder1_mha_q_w = initializers_onnx_initializer_41 = None
    initializers_onnx_initializer_42 = self.initializers.onnx_initializer_42
    encoder1_mha_q_reshape = getattr(self, "encoder1/mha/Q/reshape")(
        encoder1_mha_q_b, initializers_onnx_initializer_42
    )
    encoder1_mha_q_b = initializers_onnx_initializer_42 = None
    encoder1_mha_q_transpose = getattr(self, "encoder1/mha/Q/transpose")(
        encoder1_mha_q_reshape
    )
    encoder1_mha_q_reshape = None
    initializers_onnx_initializer_43 = self.initializers.onnx_initializer_43
    encoder1_mha_k_w = getattr(self, "encoder1/mha/K/w")(
        encoder0_ln2, initializers_onnx_initializer_43
    )
    initializers_onnx_initializer_43 = None
    initializers_onnx_initializer_44 = self.initializers.onnx_initializer_44
    encoder1_mha_k_b = getattr(self, "encoder1/mha/K/b")(
        encoder1_mha_k_w, initializers_onnx_initializer_44
    )
    encoder1_mha_k_w = initializers_onnx_initializer_44 = None
    initializers_onnx_initializer_45 = self.initializers.onnx_initializer_45
    encoder1_mha_k_reshape = getattr(self, "encoder1/mha/K/reshape")(
        encoder1_mha_k_b, initializers_onnx_initializer_45
    )
    encoder1_mha_k_b = initializers_onnx_initializer_45 = None
    encoder1_mha_k_transpose = getattr(self, "encoder1/mha/K/transpose")(
        encoder1_mha_k_reshape
    )
    encoder1_mha_k_reshape = None
    initializers_onnx_initializer_46 = self.initializers.onnx_initializer_46
    encoder1_mha_v_w = getattr(self, "encoder1/mha/V/w")(
        encoder0_ln2, initializers_onnx_initializer_46
    )
    initializers_onnx_initializer_46 = None
    initializers_onnx_initializer_47 = self.initializers.onnx_initializer_47
    encoder1_mha_v_b = getattr(self, "encoder1/mha/V/b")(
        encoder1_mha_v_w, initializers_onnx_initializer_47
    )
    encoder1_mha_v_w = initializers_onnx_initializer_47 = None
    initializers_onnx_initializer_48 = self.initializers.onnx_initializer_48
    encoder1_mha_v_reshape = getattr(self, "encoder1/mha/V/reshape")(
        encoder1_mha_v_b, initializers_onnx_initializer_48
    )
    encoder1_mha_v_b = initializers_onnx_initializer_48 = None
    encoder1_mha_v_transpose = getattr(self, "encoder1/mha/V/transpose")(
        encoder1_mha_v_reshape
    )
    encoder1_mha_v_reshape = None
    encoder1_mha_qk_matmul = getattr(self, "encoder1/mha/QK/matmul")(
        encoder1_mha_q_transpose, encoder1_mha_k_transpose
    )
    encoder1_mha_q_transpose = encoder1_mha_k_transpose = None
    initializers_onnx_initializer_49 = self.initializers.onnx_initializer_49
    encoder1_mha_qk_scale = getattr(self, "encoder1/mha/QK/scale")(
        encoder1_mha_qk_matmul, initializers_onnx_initializer_49
    )
    encoder1_mha_qk_matmul = initializers_onnx_initializer_49 = None
    initializers_onnx_initializer_50 = self.initializers.onnx_initializer_50
    encoder1_smolgen_compress = getattr(self, "encoder1/smolgen/compress")(
        encoder0_ln2, initializers_onnx_initializer_50
    )
    initializers_onnx_initializer_50 = None
    initializers_onnx_initializer_51 = self.initializers.onnx_initializer_51
    encoder1_smolgen_compress_reshape = getattr(
        self, "encoder1/smolgen/compress/reshape"
    )(encoder1_smolgen_compress, initializers_onnx_initializer_51)
    encoder1_smolgen_compress = initializers_onnx_initializer_51 = None
    initializers_onnx_initializer_52 = self.initializers.onnx_initializer_52
    encoder1_smolgen_dense1_w = getattr(self, "encoder1/smolgen/dense1/w")(
        encoder1_smolgen_compress_reshape, initializers_onnx_initializer_52
    )
    encoder1_smolgen_compress_reshape = initializers_onnx_initializer_52 = None
    initializers_onnx_initializer_53 = self.initializers.onnx_initializer_53
    encoder1_smolgen_dense1_b = getattr(self, "encoder1/smolgen/dense1/b")(
        encoder1_smolgen_dense1_w, initializers_onnx_initializer_53
    )
    encoder1_smolgen_dense1_w = initializers_onnx_initializer_53 = None
    encoder1_smolgen_dense1_swish_sigmoid = getattr(
        self, "encoder1/smolgen/dense1/swish/sigmoid"
    )(encoder1_smolgen_dense1_b)
    encoder1_smolgen_dense1_swish = getattr(self, "encoder1/smolgen/dense1/swish")(
        encoder1_smolgen_dense1_swish_sigmoid, encoder1_smolgen_dense1_b
    )
    encoder1_smolgen_dense1_swish_sigmoid = encoder1_smolgen_dense1_b = None
    encoder1_smolgen_ln1 = getattr(self, "encoder1/smolgen/ln1")(
        encoder1_smolgen_dense1_swish
    )
    encoder1_smolgen_dense1_swish = None
    initializers_onnx_initializer_54 = self.initializers.onnx_initializer_54
    encoder1_smolgen_dense2_w = getattr(self, "encoder1/smolgen/dense2/w")(
        encoder1_smolgen_ln1, initializers_onnx_initializer_54
    )
    encoder1_smolgen_ln1 = initializers_onnx_initializer_54 = None
    initializers_onnx_initializer_55 = self.initializers.onnx_initializer_55
    encoder1_smolgen_dense2_b = getattr(self, "encoder1/smolgen/dense2/b")(
        encoder1_smolgen_dense2_w, initializers_onnx_initializer_55
    )
    encoder1_smolgen_dense2_w = initializers_onnx_initializer_55 = None
    encoder1_smolgen_dense2_swish_sigmoid = getattr(
        self, "encoder1/smolgen/dense2/swish/sigmoid"
    )(encoder1_smolgen_dense2_b)
    encoder1_smolgen_dense2_swish = getattr(self, "encoder1/smolgen/dense2/swish")(
        encoder1_smolgen_dense2_swish_sigmoid, encoder1_smolgen_dense2_b
    )
    encoder1_smolgen_dense2_swish_sigmoid = encoder1_smolgen_dense2_b = None
    encoder1_smolgen_ln2 = getattr(self, "encoder1/smolgen/ln2")(
        encoder1_smolgen_dense2_swish
    )
    encoder1_smolgen_dense2_swish = None
    initializers_onnx_initializer_56 = self.initializers.onnx_initializer_56
    encoder1_smolgen_gen_from_reshape = getattr(
        self, "encoder1/smolgen/gen_from/reshape"
    )(encoder1_smolgen_ln2, initializers_onnx_initializer_56)
    encoder1_smolgen_ln2 = initializers_onnx_initializer_56 = None
    initializers_onnx_initializer_57 = self.initializers.onnx_initializer_57
    encoder1_smolgen_smol_weight_gen = getattr(
        self, "encoder1/smolgen/smol_weight_gen"
    )(encoder1_smolgen_gen_from_reshape, initializers_onnx_initializer_57)
    encoder1_smolgen_gen_from_reshape = initializers_onnx_initializer_57 = None
    initializers_onnx_initializer_58 = self.initializers.onnx_initializer_58
    encoder1_smolgen_out_reshape = getattr(self, "encoder1/smolgen/out/reshape")(
        encoder1_smolgen_smol_weight_gen, initializers_onnx_initializer_58
    )
    encoder1_smolgen_smol_weight_gen = initializers_onnx_initializer_58 = None
    encoder1_smolgen_weights = getattr(self, "encoder1/smolgen_weights")(
        encoder1_mha_qk_scale, encoder1_smolgen_out_reshape
    )
    encoder1_mha_qk_scale = encoder1_smolgen_out_reshape = None
    encoder1_mha_qk_softmax = getattr(self, "encoder1/mha/QK/softmax")(
        encoder1_smolgen_weights
    )
    encoder1_smolgen_weights = None
    encoder1_mha_qkv_matmul = getattr(self, "encoder1/mha/QKV/matmul")(
        encoder1_mha_qk_softmax, encoder1_mha_v_transpose
    )
    encoder1_mha_qk_softmax = encoder1_mha_v_transpose = None
    encoder1_mha_out_transpose = getattr(self, "encoder1/mha/out/transpose")(
        encoder1_mha_qkv_matmul
    )
    encoder1_mha_qkv_matmul = None
    initializers_onnx_initializer_59 = self.initializers.onnx_initializer_59
    encoder1_mha_out_reshape = getattr(self, "encoder1/mha/out/reshape")(
        encoder1_mha_out_transpose, initializers_onnx_initializer_59
    )
    encoder1_mha_out_transpose = initializers_onnx_initializer_59 = None
    initializers_onnx_initializer_60 = self.initializers.onnx_initializer_60
    encoder1_mha_out_dense_w = getattr(self, "encoder1/mha/out/dense/w")(
        encoder1_mha_out_reshape, initializers_onnx_initializer_60
    )
    encoder1_mha_out_reshape = initializers_onnx_initializer_60 = None
    initializers_onnx_initializer_61 = self.initializers.onnx_initializer_61
    encoder1_mha_out_dense_b = getattr(self, "encoder1/mha/out/dense/b")(
        encoder1_mha_out_dense_w, initializers_onnx_initializer_61
    )
    encoder1_mha_out_dense_w = initializers_onnx_initializer_61 = None

    encoder1_mha_out_dense_b = self.attention_output[1](
        encoder1_mha_out_dense_b.view(-1, 64, 768)
    ).view(-1, 768)

    initializers_onnx_initializer_62 = self.initializers.onnx_initializer_62
    encoder1_alpha_input = getattr(self, "encoder1/alpha*input")(
        encoder0_ln2, initializers_onnx_initializer_62
    )
    encoder0_ln2 = initializers_onnx_initializer_62 = None
    encoder1_mha_out_skip = getattr(self, "encoder1/mha/out/skip")(
        encoder1_mha_out_dense_b, encoder1_alpha_input
    )
    encoder1_mha_out_dense_b = encoder1_alpha_input = None
    encoder1_ln1 = getattr(self, "encoder1/ln1")(encoder1_mha_out_skip)
    encoder1_mha_out_skip = None

    ########################################################################
    # Make residuals accessible with squares as a separate dim,
    # this is important for nnsight.
    ########################################################################
    # This is just an nn.Identity layer
    residual_reasonable_shape = self.post_attention[1](
        encoder1_ln1.reshape(-1, 64, 768)
    )
    encoder1_ln1 = residual_reasonable_shape.reshape(-1, 768)
    residual_reasonable_shape = None

    initializers_onnx_initializer_63 = self.initializers.onnx_initializer_63
    encoder1_ffn_dense1_w = getattr(self, "encoder1/ffn/dense1/w")(
        encoder1_ln1, initializers_onnx_initializer_63
    )
    initializers_onnx_initializer_63 = None
    initializers_onnx_initializer_64 = self.initializers.onnx_initializer_64
    encoder1_ffn_dense1_b = getattr(self, "encoder1/ffn/dense1/b")(
        encoder1_ffn_dense1_w, initializers_onnx_initializer_64
    )
    encoder1_ffn_dense1_w = initializers_onnx_initializer_64 = None
    encoder1_ffn_dense1_sqrrelu_relu = getattr(
        self, "encoder1/ffn/dense1/sqrrelu/relu"
    )(encoder1_ffn_dense1_b)
    encoder1_ffn_dense1_b = None
    encoder1_ffn_dense1_sqrrelu_sqr = getattr(self, "encoder1/ffn/dense1/sqrrelu/sqr")(
        encoder1_ffn_dense1_sqrrelu_relu, encoder1_ffn_dense1_sqrrelu_relu
    )
    encoder1_ffn_dense1_sqrrelu_relu = None
    initializers_onnx_initializer_65 = self.initializers.onnx_initializer_65
    encoder1_ffn_dense2_w = getattr(self, "encoder1/ffn/dense2/w")(
        encoder1_ffn_dense1_sqrrelu_sqr, initializers_onnx_initializer_65
    )
    encoder1_ffn_dense1_sqrrelu_sqr = initializers_onnx_initializer_65 = None
    initializers_onnx_initializer_66 = self.initializers.onnx_initializer_66
    encoder1_ffn_dense2_b = getattr(self, "encoder1/ffn/dense2/b")(
        encoder1_ffn_dense2_w, initializers_onnx_initializer_66
    )

    encoder1_ffn_dense2_b = self.mlp_output[1](
        encoder1_ffn_dense2_b.view(-1, 64, 768)
    ).view(-1, 768)

    encoder1_ffn_dense2_w = initializers_onnx_initializer_66 = None
    initializers_onnx_initializer_67 = self.initializers.onnx_initializer_67
    encoder1_alpha_out1 = getattr(self, "encoder1/alpha*out1")(
        encoder1_ln1, initializers_onnx_initializer_67
    )
    encoder1_ln1 = initializers_onnx_initializer_67 = None
    encoder1_ffn_skip = getattr(self, "encoder1/ffn/skip")(
        encoder1_ffn_dense2_b, encoder1_alpha_out1
    )
    encoder1_ffn_dense2_b = encoder1_alpha_out1 = None
    encoder1_ln2 = getattr(self, "encoder1/ln2")(encoder1_ffn_skip)
    encoder1_ffn_skip = None

    ########################################################################
    # Make residuals accessible with squares as a separate dim,
    # this is important for nnsight.
    ########################################################################
    # This is just an nn.Identity layer
    residual_reasonable_shape = self.post_mlp[1](encoder1_ln2.reshape(-1, 64, 768))
    encoder1_ln2 = residual_reasonable_shape.reshape(-1, 768)
    residual_reasonable_shape = None

    ########################################################################
    # Layer 2
    ########################################################################
    initializers_onnx_initializer_68 = self.initializers.onnx_initializer_68
    encoder2_mha_q_w = getattr(self, "encoder2/mha/Q/w")(
        encoder1_ln2, initializers_onnx_initializer_68
    )
    initializers_onnx_initializer_68 = None
    initializers_onnx_initializer_69 = self.initializers.onnx_initializer_69
    encoder2_mha_q_b = getattr(self, "encoder2/mha/Q/b")(
        encoder2_mha_q_w, initializers_onnx_initializer_69
    )
    encoder2_mha_q_w = initializers_onnx_initializer_69 = None
    initializers_onnx_initializer_70 = self.initializers.onnx_initializer_70
    encoder2_mha_q_reshape = getattr(self, "encoder2/mha/Q/reshape")(
        encoder2_mha_q_b, initializers_onnx_initializer_70
    )
    encoder2_mha_q_b = initializers_onnx_initializer_70 = None
    encoder2_mha_q_transpose = getattr(self, "encoder2/mha/Q/transpose")(
        encoder2_mha_q_reshape
    )
    encoder2_mha_q_reshape = None
    initializers_onnx_initializer_71 = self.initializers.onnx_initializer_71
    encoder2_mha_k_w = getattr(self, "encoder2/mha/K/w")(
        encoder1_ln2, initializers_onnx_initializer_71
    )
    initializers_onnx_initializer_71 = None
    initializers_onnx_initializer_72 = self.initializers.onnx_initializer_72
    encoder2_mha_k_b = getattr(self, "encoder2/mha/K/b")(
        encoder2_mha_k_w, initializers_onnx_initializer_72
    )
    encoder2_mha_k_w = initializers_onnx_initializer_72 = None
    initializers_onnx_initializer_73 = self.initializers.onnx_initializer_73
    encoder2_mha_k_reshape = getattr(self, "encoder2/mha/K/reshape")(
        encoder2_mha_k_b, initializers_onnx_initializer_73
    )
    encoder2_mha_k_b = initializers_onnx_initializer_73 = None
    encoder2_mha_k_transpose = getattr(self, "encoder2/mha/K/transpose")(
        encoder2_mha_k_reshape
    )
    encoder2_mha_k_reshape = None
    initializers_onnx_initializer_74 = self.initializers.onnx_initializer_74
    encoder2_mha_v_w = getattr(self, "encoder2/mha/V/w")(
        encoder1_ln2, initializers_onnx_initializer_74
    )
    initializers_onnx_initializer_74 = None
    initializers_onnx_initializer_75 = self.initializers.onnx_initializer_75
    encoder2_mha_v_b = getattr(self, "encoder2/mha/V/b")(
        encoder2_mha_v_w, initializers_onnx_initializer_75
    )
    encoder2_mha_v_w = initializers_onnx_initializer_75 = None
    initializers_onnx_initializer_76 = self.initializers.onnx_initializer_76
    encoder2_mha_v_reshape = getattr(self, "encoder2/mha/V/reshape")(
        encoder2_mha_v_b, initializers_onnx_initializer_76
    )
    encoder2_mha_v_b = initializers_onnx_initializer_76 = None
    encoder2_mha_v_transpose = getattr(self, "encoder2/mha/V/transpose")(
        encoder2_mha_v_reshape
    )
    encoder2_mha_v_reshape = None
    encoder2_mha_qk_matmul = getattr(self, "encoder2/mha/QK/matmul")(
        encoder2_mha_q_transpose, encoder2_mha_k_transpose
    )
    encoder2_mha_q_transpose = encoder2_mha_k_transpose = None
    initializers_onnx_initializer_77 = self.initializers.onnx_initializer_77
    encoder2_mha_qk_scale = getattr(self, "encoder2/mha/QK/scale")(
        encoder2_mha_qk_matmul, initializers_onnx_initializer_77
    )
    encoder2_mha_qk_matmul = initializers_onnx_initializer_77 = None
    initializers_onnx_initializer_78 = self.initializers.onnx_initializer_78
    encoder2_smolgen_compress = getattr(self, "encoder2/smolgen/compress")(
        encoder1_ln2, initializers_onnx_initializer_78
    )
    initializers_onnx_initializer_78 = None
    initializers_onnx_initializer_79 = self.initializers.onnx_initializer_79
    encoder2_smolgen_compress_reshape = getattr(
        self, "encoder2/smolgen/compress/reshape"
    )(encoder2_smolgen_compress, initializers_onnx_initializer_79)
    encoder2_smolgen_compress = initializers_onnx_initializer_79 = None
    initializers_onnx_initializer_80 = self.initializers.onnx_initializer_80
    encoder2_smolgen_dense1_w = getattr(self, "encoder2/smolgen/dense1/w")(
        encoder2_smolgen_compress_reshape, initializers_onnx_initializer_80
    )
    encoder2_smolgen_compress_reshape = initializers_onnx_initializer_80 = None
    initializers_onnx_initializer_81 = self.initializers.onnx_initializer_81
    encoder2_smolgen_dense1_b = getattr(self, "encoder2/smolgen/dense1/b")(
        encoder2_smolgen_dense1_w, initializers_onnx_initializer_81
    )
    encoder2_smolgen_dense1_w = initializers_onnx_initializer_81 = None
    encoder2_smolgen_dense1_swish_sigmoid = getattr(
        self, "encoder2/smolgen/dense1/swish/sigmoid"
    )(encoder2_smolgen_dense1_b)
    encoder2_smolgen_dense1_swish = getattr(self, "encoder2/smolgen/dense1/swish")(
        encoder2_smolgen_dense1_swish_sigmoid, encoder2_smolgen_dense1_b
    )
    encoder2_smolgen_dense1_swish_sigmoid = encoder2_smolgen_dense1_b = None
    encoder2_smolgen_ln1 = getattr(self, "encoder2/smolgen/ln1")(
        encoder2_smolgen_dense1_swish
    )
    encoder2_smolgen_dense1_swish = None
    initializers_onnx_initializer_82 = self.initializers.onnx_initializer_82
    encoder2_smolgen_dense2_w = getattr(self, "encoder2/smolgen/dense2/w")(
        encoder2_smolgen_ln1, initializers_onnx_initializer_82
    )
    encoder2_smolgen_ln1 = initializers_onnx_initializer_82 = None
    initializers_onnx_initializer_83 = self.initializers.onnx_initializer_83
    encoder2_smolgen_dense2_b = getattr(self, "encoder2/smolgen/dense2/b")(
        encoder2_smolgen_dense2_w, initializers_onnx_initializer_83
    )
    encoder2_smolgen_dense2_w = initializers_onnx_initializer_83 = None
    encoder2_smolgen_dense2_swish_sigmoid = getattr(
        self, "encoder2/smolgen/dense2/swish/sigmoid"
    )(encoder2_smolgen_dense2_b)
    encoder2_smolgen_dense2_swish = getattr(self, "encoder2/smolgen/dense2/swish")(
        encoder2_smolgen_dense2_swish_sigmoid, encoder2_smolgen_dense2_b
    )
    encoder2_smolgen_dense2_swish_sigmoid = encoder2_smolgen_dense2_b = None
    encoder2_smolgen_ln2 = getattr(self, "encoder2/smolgen/ln2")(
        encoder2_smolgen_dense2_swish
    )
    encoder2_smolgen_dense2_swish = None
    initializers_onnx_initializer_84 = self.initializers.onnx_initializer_84
    encoder2_smolgen_gen_from_reshape = getattr(
        self, "encoder2/smolgen/gen_from/reshape"
    )(encoder2_smolgen_ln2, initializers_onnx_initializer_84)
    encoder2_smolgen_ln2 = initializers_onnx_initializer_84 = None
    initializers_onnx_initializer_85 = self.initializers.onnx_initializer_85
    encoder2_smolgen_smol_weight_gen = getattr(
        self, "encoder2/smolgen/smol_weight_gen"
    )(encoder2_smolgen_gen_from_reshape, initializers_onnx_initializer_85)
    encoder2_smolgen_gen_from_reshape = initializers_onnx_initializer_85 = None
    initializers_onnx_initializer_86 = self.initializers.onnx_initializer_86
    encoder2_smolgen_out_reshape = getattr(self, "encoder2/smolgen/out/reshape")(
        encoder2_smolgen_smol_weight_gen, initializers_onnx_initializer_86
    )
    encoder2_smolgen_smol_weight_gen = initializers_onnx_initializer_86 = None
    encoder2_smolgen_weights = getattr(self, "encoder2/smolgen_weights")(
        encoder2_mha_qk_scale, encoder2_smolgen_out_reshape
    )
    encoder2_mha_qk_scale = encoder2_smolgen_out_reshape = None
    encoder2_mha_qk_softmax = getattr(self, "encoder2/mha/QK/softmax")(
        encoder2_smolgen_weights
    )
    encoder2_smolgen_weights = None
    encoder2_mha_qkv_matmul = getattr(self, "encoder2/mha/QKV/matmul")(
        encoder2_mha_qk_softmax, encoder2_mha_v_transpose
    )
    encoder2_mha_qk_softmax = encoder2_mha_v_transpose = None
    encoder2_mha_out_transpose = getattr(self, "encoder2/mha/out/transpose")(
        encoder2_mha_qkv_matmul
    )
    encoder2_mha_qkv_matmul = None
    initializers_onnx_initializer_87 = self.initializers.onnx_initializer_87
    encoder2_mha_out_reshape = getattr(self, "encoder2/mha/out/reshape")(
        encoder2_mha_out_transpose, initializers_onnx_initializer_87
    )
    encoder2_mha_out_transpose = initializers_onnx_initializer_87 = None
    initializers_onnx_initializer_88 = self.initializers.onnx_initializer_88
    encoder2_mha_out_dense_w = getattr(self, "encoder2/mha/out/dense/w")(
        encoder2_mha_out_reshape, initializers_onnx_initializer_88
    )
    encoder2_mha_out_reshape = initializers_onnx_initializer_88 = None
    initializers_onnx_initializer_89 = self.initializers.onnx_initializer_89
    encoder2_mha_out_dense_b = getattr(self, "encoder2/mha/out/dense/b")(
        encoder2_mha_out_dense_w, initializers_onnx_initializer_89
    )
    encoder2_mha_out_dense_w = initializers_onnx_initializer_89 = None

    encoder2_mha_out_dense_b = self.attention_output[2](
        encoder2_mha_out_dense_b.view(-1, 64, 768)
    ).view(-1, 768)

    initializers_onnx_initializer_90 = self.initializers.onnx_initializer_90
    encoder2_alpha_input = getattr(self, "encoder2/alpha*input")(
        encoder1_ln2, initializers_onnx_initializer_90
    )
    encoder1_ln2 = initializers_onnx_initializer_90 = None
    encoder2_mha_out_skip = getattr(self, "encoder2/mha/out/skip")(
        encoder2_mha_out_dense_b, encoder2_alpha_input
    )
    encoder2_mha_out_dense_b = encoder2_alpha_input = None
    encoder2_ln1 = getattr(self, "encoder2/ln1")(encoder2_mha_out_skip)
    encoder2_mha_out_skip = None

    ########################################################################
    # Make residuals accessible with squares as a separate dim,
    # this is important for nnsight.
    ########################################################################
    # This is just an nn.Identity layer
    residual_reasonable_shape = self.post_attention[2](
        encoder2_ln1.reshape(-1, 64, 768)
    )
    encoder2_ln1 = residual_reasonable_shape.reshape(-1, 768)
    residual_reasonable_shape = None

    initializers_onnx_initializer_91 = self.initializers.onnx_initializer_91
    encoder2_ffn_dense1_w = getattr(self, "encoder2/ffn/dense1/w")(
        encoder2_ln1, initializers_onnx_initializer_91
    )
    initializers_onnx_initializer_91 = None
    initializers_onnx_initializer_92 = self.initializers.onnx_initializer_92
    encoder2_ffn_dense1_b = getattr(self, "encoder2/ffn/dense1/b")(
        encoder2_ffn_dense1_w, initializers_onnx_initializer_92
    )
    encoder2_ffn_dense1_w = initializers_onnx_initializer_92 = None
    encoder2_ffn_dense1_sqrrelu_relu = getattr(
        self, "encoder2/ffn/dense1/sqrrelu/relu"
    )(encoder2_ffn_dense1_b)
    encoder2_ffn_dense1_b = None
    encoder2_ffn_dense1_sqrrelu_sqr = getattr(self, "encoder2/ffn/dense1/sqrrelu/sqr")(
        encoder2_ffn_dense1_sqrrelu_relu, encoder2_ffn_dense1_sqrrelu_relu
    )
    encoder2_ffn_dense1_sqrrelu_relu = None
    initializers_onnx_initializer_93 = self.initializers.onnx_initializer_93
    encoder2_ffn_dense2_w = getattr(self, "encoder2/ffn/dense2/w")(
        encoder2_ffn_dense1_sqrrelu_sqr, initializers_onnx_initializer_93
    )
    encoder2_ffn_dense1_sqrrelu_sqr = initializers_onnx_initializer_93 = None
    initializers_onnx_initializer_94 = self.initializers.onnx_initializer_94
    encoder2_ffn_dense2_b = getattr(self, "encoder2/ffn/dense2/b")(
        encoder2_ffn_dense2_w, initializers_onnx_initializer_94
    )

    encoder2_ffn_dense2_b = self.mlp_output[2](
        encoder2_ffn_dense2_b.view(-1, 64, 768)
    ).view(-1, 768)

    encoder2_ffn_dense2_w = initializers_onnx_initializer_94 = None
    initializers_onnx_initializer_95 = self.initializers.onnx_initializer_95
    encoder2_alpha_out1 = getattr(self, "encoder2/alpha*out1")(
        encoder2_ln1, initializers_onnx_initializer_95
    )
    encoder2_ln1 = initializers_onnx_initializer_95 = None
    encoder2_ffn_skip = getattr(self, "encoder2/ffn/skip")(
        encoder2_ffn_dense2_b, encoder2_alpha_out1
    )
    encoder2_ffn_dense2_b = encoder2_alpha_out1 = None
    encoder2_ln2 = getattr(self, "encoder2/ln2")(encoder2_ffn_skip)
    encoder2_ffn_skip = None

    ########################################################################
    # Make residuals accessible with squares as a separate dim,
    # this is important for nnsight.
    ########################################################################
    # This is just an nn.Identity layer
    residual_reasonable_shape = self.post_mlp[2](encoder2_ln2.reshape(-1, 64, 768))
    encoder2_ln2 = residual_reasonable_shape.reshape(-1, 768)
    residual_reasonable_shape = None

    ########################################################################
    # Layer 3
    ########################################################################
    initializers_onnx_initializer_96 = self.initializers.onnx_initializer_96
    encoder3_mha_q_w = getattr(self, "encoder3/mha/Q/w")(
        encoder2_ln2, initializers_onnx_initializer_96
    )
    initializers_onnx_initializer_96 = None
    initializers_onnx_initializer_97 = self.initializers.onnx_initializer_97
    encoder3_mha_q_b = getattr(self, "encoder3/mha/Q/b")(
        encoder3_mha_q_w, initializers_onnx_initializer_97
    )
    encoder3_mha_q_w = initializers_onnx_initializer_97 = None
    initializers_onnx_initializer_98 = self.initializers.onnx_initializer_98
    encoder3_mha_q_reshape = getattr(self, "encoder3/mha/Q/reshape")(
        encoder3_mha_q_b, initializers_onnx_initializer_98
    )
    encoder3_mha_q_b = initializers_onnx_initializer_98 = None
    encoder3_mha_q_transpose = getattr(self, "encoder3/mha/Q/transpose")(
        encoder3_mha_q_reshape
    )
    encoder3_mha_q_reshape = None
    initializers_onnx_initializer_99 = self.initializers.onnx_initializer_99
    encoder3_mha_k_w = getattr(self, "encoder3/mha/K/w")(
        encoder2_ln2, initializers_onnx_initializer_99
    )
    initializers_onnx_initializer_99 = None
    initializers_onnx_initializer_100 = self.initializers.onnx_initializer_100
    encoder3_mha_k_b = getattr(self, "encoder3/mha/K/b")(
        encoder3_mha_k_w, initializers_onnx_initializer_100
    )
    encoder3_mha_k_w = initializers_onnx_initializer_100 = None
    initializers_onnx_initializer_101 = self.initializers.onnx_initializer_101
    encoder3_mha_k_reshape = getattr(self, "encoder3/mha/K/reshape")(
        encoder3_mha_k_b, initializers_onnx_initializer_101
    )
    encoder3_mha_k_b = initializers_onnx_initializer_101 = None
    encoder3_mha_k_transpose = getattr(self, "encoder3/mha/K/transpose")(
        encoder3_mha_k_reshape
    )
    encoder3_mha_k_reshape = None
    initializers_onnx_initializer_102 = self.initializers.onnx_initializer_102
    encoder3_mha_v_w = getattr(self, "encoder3/mha/V/w")(
        encoder2_ln2, initializers_onnx_initializer_102
    )
    initializers_onnx_initializer_102 = None
    initializers_onnx_initializer_103 = self.initializers.onnx_initializer_103
    encoder3_mha_v_b = getattr(self, "encoder3/mha/V/b")(
        encoder3_mha_v_w, initializers_onnx_initializer_103
    )
    encoder3_mha_v_w = initializers_onnx_initializer_103 = None
    initializers_onnx_initializer_104 = self.initializers.onnx_initializer_104
    encoder3_mha_v_reshape = getattr(self, "encoder3/mha/V/reshape")(
        encoder3_mha_v_b, initializers_onnx_initializer_104
    )
    encoder3_mha_v_b = initializers_onnx_initializer_104 = None
    encoder3_mha_v_transpose = getattr(self, "encoder3/mha/V/transpose")(
        encoder3_mha_v_reshape
    )
    encoder3_mha_v_reshape = None
    encoder3_mha_qk_matmul = getattr(self, "encoder3/mha/QK/matmul")(
        encoder3_mha_q_transpose, encoder3_mha_k_transpose
    )
    encoder3_mha_q_transpose = encoder3_mha_k_transpose = None
    initializers_onnx_initializer_105 = self.initializers.onnx_initializer_105
    encoder3_mha_qk_scale = getattr(self, "encoder3/mha/QK/scale")(
        encoder3_mha_qk_matmul, initializers_onnx_initializer_105
    )
    encoder3_mha_qk_matmul = initializers_onnx_initializer_105 = None
    initializers_onnx_initializer_106 = self.initializers.onnx_initializer_106
    encoder3_smolgen_compress = getattr(self, "encoder3/smolgen/compress")(
        encoder2_ln2, initializers_onnx_initializer_106
    )
    initializers_onnx_initializer_106 = None
    initializers_onnx_initializer_107 = self.initializers.onnx_initializer_107
    encoder3_smolgen_compress_reshape = getattr(
        self, "encoder3/smolgen/compress/reshape"
    )(encoder3_smolgen_compress, initializers_onnx_initializer_107)
    encoder3_smolgen_compress = initializers_onnx_initializer_107 = None
    initializers_onnx_initializer_108 = self.initializers.onnx_initializer_108
    encoder3_smolgen_dense1_w = getattr(self, "encoder3/smolgen/dense1/w")(
        encoder3_smolgen_compress_reshape, initializers_onnx_initializer_108
    )
    encoder3_smolgen_compress_reshape = initializers_onnx_initializer_108 = None
    initializers_onnx_initializer_109 = self.initializers.onnx_initializer_109
    encoder3_smolgen_dense1_b = getattr(self, "encoder3/smolgen/dense1/b")(
        encoder3_smolgen_dense1_w, initializers_onnx_initializer_109
    )
    encoder3_smolgen_dense1_w = initializers_onnx_initializer_109 = None
    encoder3_smolgen_dense1_swish_sigmoid = getattr(
        self, "encoder3/smolgen/dense1/swish/sigmoid"
    )(encoder3_smolgen_dense1_b)
    encoder3_smolgen_dense1_swish = getattr(self, "encoder3/smolgen/dense1/swish")(
        encoder3_smolgen_dense1_swish_sigmoid, encoder3_smolgen_dense1_b
    )
    encoder3_smolgen_dense1_swish_sigmoid = encoder3_smolgen_dense1_b = None
    encoder3_smolgen_ln1 = getattr(self, "encoder3/smolgen/ln1")(
        encoder3_smolgen_dense1_swish
    )
    encoder3_smolgen_dense1_swish = None
    initializers_onnx_initializer_110 = self.initializers.onnx_initializer_110
    encoder3_smolgen_dense2_w = getattr(self, "encoder3/smolgen/dense2/w")(
        encoder3_smolgen_ln1, initializers_onnx_initializer_110
    )
    encoder3_smolgen_ln1 = initializers_onnx_initializer_110 = None
    initializers_onnx_initializer_111 = self.initializers.onnx_initializer_111
    encoder3_smolgen_dense2_b = getattr(self, "encoder3/smolgen/dense2/b")(
        encoder3_smolgen_dense2_w, initializers_onnx_initializer_111
    )
    encoder3_smolgen_dense2_w = initializers_onnx_initializer_111 = None
    encoder3_smolgen_dense2_swish_sigmoid = getattr(
        self, "encoder3/smolgen/dense2/swish/sigmoid"
    )(encoder3_smolgen_dense2_b)
    encoder3_smolgen_dense2_swish = getattr(self, "encoder3/smolgen/dense2/swish")(
        encoder3_smolgen_dense2_swish_sigmoid, encoder3_smolgen_dense2_b
    )
    encoder3_smolgen_dense2_swish_sigmoid = encoder3_smolgen_dense2_b = None
    encoder3_smolgen_ln2 = getattr(self, "encoder3/smolgen/ln2")(
        encoder3_smolgen_dense2_swish
    )
    encoder3_smolgen_dense2_swish = None
    initializers_onnx_initializer_112 = self.initializers.onnx_initializer_112
    encoder3_smolgen_gen_from_reshape = getattr(
        self, "encoder3/smolgen/gen_from/reshape"
    )(encoder3_smolgen_ln2, initializers_onnx_initializer_112)
    encoder3_smolgen_ln2 = initializers_onnx_initializer_112 = None
    initializers_onnx_initializer_113 = self.initializers.onnx_initializer_113
    encoder3_smolgen_smol_weight_gen = getattr(
        self, "encoder3/smolgen/smol_weight_gen"
    )(encoder3_smolgen_gen_from_reshape, initializers_onnx_initializer_113)
    encoder3_smolgen_gen_from_reshape = initializers_onnx_initializer_113 = None
    initializers_onnx_initializer_114 = self.initializers.onnx_initializer_114
    encoder3_smolgen_out_reshape = getattr(self, "encoder3/smolgen/out/reshape")(
        encoder3_smolgen_smol_weight_gen, initializers_onnx_initializer_114
    )
    encoder3_smolgen_smol_weight_gen = initializers_onnx_initializer_114 = None
    encoder3_smolgen_weights = getattr(self, "encoder3/smolgen_weights")(
        encoder3_mha_qk_scale, encoder3_smolgen_out_reshape
    )
    encoder3_mha_qk_scale = encoder3_smolgen_out_reshape = None
    encoder3_mha_qk_softmax = getattr(self, "encoder3/mha/QK/softmax")(
        encoder3_smolgen_weights
    )
    encoder3_smolgen_weights = None
    encoder3_mha_qkv_matmul = getattr(self, "encoder3/mha/QKV/matmul")(
        encoder3_mha_qk_softmax, encoder3_mha_v_transpose
    )
    encoder3_mha_qk_softmax = encoder3_mha_v_transpose = None
    encoder3_mha_out_transpose = getattr(self, "encoder3/mha/out/transpose")(
        encoder3_mha_qkv_matmul
    )
    encoder3_mha_qkv_matmul = None
    initializers_onnx_initializer_115 = self.initializers.onnx_initializer_115
    encoder3_mha_out_reshape = getattr(self, "encoder3/mha/out/reshape")(
        encoder3_mha_out_transpose, initializers_onnx_initializer_115
    )
    encoder3_mha_out_transpose = initializers_onnx_initializer_115 = None
    initializers_onnx_initializer_116 = self.initializers.onnx_initializer_116
    encoder3_mha_out_dense_w = getattr(self, "encoder3/mha/out/dense/w")(
        encoder3_mha_out_reshape, initializers_onnx_initializer_116
    )
    encoder3_mha_out_reshape = initializers_onnx_initializer_116 = None
    initializers_onnx_initializer_117 = self.initializers.onnx_initializer_117
    encoder3_mha_out_dense_b = getattr(self, "encoder3/mha/out/dense/b")(
        encoder3_mha_out_dense_w, initializers_onnx_initializer_117
    )

    encoder3_mha_out_dense_b = self.attention_output[3](
        encoder3_mha_out_dense_b.view(-1, 64, 768)
    ).view(-1, 768)

    encoder3_mha_out_dense_w = initializers_onnx_initializer_117 = None
    initializers_onnx_initializer_118 = self.initializers.onnx_initializer_118
    encoder3_alpha_input = getattr(self, "encoder3/alpha*input")(
        encoder2_ln2, initializers_onnx_initializer_118
    )
    encoder2_ln2 = initializers_onnx_initializer_118 = None
    encoder3_mha_out_skip = getattr(self, "encoder3/mha/out/skip")(
        encoder3_mha_out_dense_b, encoder3_alpha_input
    )
    encoder3_mha_out_dense_b = encoder3_alpha_input = None
    encoder3_ln1 = getattr(self, "encoder3/ln1")(encoder3_mha_out_skip)
    encoder3_mha_out_skip = None

    ########################################################################
    # Make residuals accessible with squares as a separate dim,
    # this is important for nnsight.
    ########################################################################
    # This is just an nn.Identity layer
    residual_reasonable_shape = self.post_attention[3](
        encoder3_ln1.reshape(-1, 64, 768)
    )
    encoder3_ln1 = residual_reasonable_shape.reshape(-1, 768)
    residual_reasonable_shape = None

    initializers_onnx_initializer_119 = self.initializers.onnx_initializer_119
    encoder3_ffn_dense1_w = getattr(self, "encoder3/ffn/dense1/w")(
        encoder3_ln1, initializers_onnx_initializer_119
    )
    initializers_onnx_initializer_119 = None
    initializers_onnx_initializer_120 = self.initializers.onnx_initializer_120
    encoder3_ffn_dense1_b = getattr(self, "encoder3/ffn/dense1/b")(
        encoder3_ffn_dense1_w, initializers_onnx_initializer_120
    )
    encoder3_ffn_dense1_w = initializers_onnx_initializer_120 = None
    encoder3_ffn_dense1_sqrrelu_relu = getattr(
        self, "encoder3/ffn/dense1/sqrrelu/relu"
    )(encoder3_ffn_dense1_b)
    encoder3_ffn_dense1_b = None
    encoder3_ffn_dense1_sqrrelu_sqr = getattr(self, "encoder3/ffn/dense1/sqrrelu/sqr")(
        encoder3_ffn_dense1_sqrrelu_relu, encoder3_ffn_dense1_sqrrelu_relu
    )
    encoder3_ffn_dense1_sqrrelu_relu = None
    initializers_onnx_initializer_121 = self.initializers.onnx_initializer_121
    encoder3_ffn_dense2_w = getattr(self, "encoder3/ffn/dense2/w")(
        encoder3_ffn_dense1_sqrrelu_sqr, initializers_onnx_initializer_121
    )
    encoder3_ffn_dense1_sqrrelu_sqr = initializers_onnx_initializer_121 = None
    initializers_onnx_initializer_122 = self.initializers.onnx_initializer_122
    encoder3_ffn_dense2_b = getattr(self, "encoder3/ffn/dense2/b")(
        encoder3_ffn_dense2_w, initializers_onnx_initializer_122
    )

    encoder3_ffn_dense2_b = self.mlp_output[3](
        encoder3_ffn_dense2_b.view(-1, 64, 768)
    ).view(-1, 768)

    encoder3_ffn_dense2_w = initializers_onnx_initializer_122 = None
    initializers_onnx_initializer_123 = self.initializers.onnx_initializer_123
    encoder3_alpha_out1 = getattr(self, "encoder3/alpha*out1")(
        encoder3_ln1, initializers_onnx_initializer_123
    )
    encoder3_ln1 = initializers_onnx_initializer_123 = None
    encoder3_ffn_skip = getattr(self, "encoder3/ffn/skip")(
        encoder3_ffn_dense2_b, encoder3_alpha_out1
    )
    encoder3_ffn_dense2_b = encoder3_alpha_out1 = None
    encoder3_ln2 = getattr(self, "encoder3/ln2")(encoder3_ffn_skip)
    encoder3_ffn_skip = None

    ########################################################################
    # Make residuals accessible with squares as a separate dim,
    # this is important for nnsight.
    ########################################################################
    # This is just an nn.Identity layer
    residual_reasonable_shape = self.post_mlp[3](encoder3_ln2.reshape(-1, 64, 768))
    encoder3_ln2 = residual_reasonable_shape.reshape(-1, 768)
    residual_reasonable_shape = None

    ########################################################################
    # Layer 4
    ########################################################################
    initializers_onnx_initializer_124 = self.initializers.onnx_initializer_124
    encoder4_mha_q_w = getattr(self, "encoder4/mha/Q/w")(
        encoder3_ln2, initializers_onnx_initializer_124
    )
    initializers_onnx_initializer_124 = None
    initializers_onnx_initializer_125 = self.initializers.onnx_initializer_125
    encoder4_mha_q_b = getattr(self, "encoder4/mha/Q/b")(
        encoder4_mha_q_w, initializers_onnx_initializer_125
    )
    encoder4_mha_q_w = initializers_onnx_initializer_125 = None
    initializers_onnx_initializer_126 = self.initializers.onnx_initializer_126
    encoder4_mha_q_reshape = getattr(self, "encoder4/mha/Q/reshape")(
        encoder4_mha_q_b, initializers_onnx_initializer_126
    )
    encoder4_mha_q_b = initializers_onnx_initializer_126 = None
    encoder4_mha_q_transpose = getattr(self, "encoder4/mha/Q/transpose")(
        encoder4_mha_q_reshape
    )
    encoder4_mha_q_reshape = None
    initializers_onnx_initializer_127 = self.initializers.onnx_initializer_127
    encoder4_mha_k_w = getattr(self, "encoder4/mha/K/w")(
        encoder3_ln2, initializers_onnx_initializer_127
    )
    initializers_onnx_initializer_127 = None
    initializers_onnx_initializer_128 = self.initializers.onnx_initializer_128
    encoder4_mha_k_b = getattr(self, "encoder4/mha/K/b")(
        encoder4_mha_k_w, initializers_onnx_initializer_128
    )
    encoder4_mha_k_w = initializers_onnx_initializer_128 = None
    initializers_onnx_initializer_129 = self.initializers.onnx_initializer_129
    encoder4_mha_k_reshape = getattr(self, "encoder4/mha/K/reshape")(
        encoder4_mha_k_b, initializers_onnx_initializer_129
    )
    encoder4_mha_k_b = initializers_onnx_initializer_129 = None
    encoder4_mha_k_transpose = getattr(self, "encoder4/mha/K/transpose")(
        encoder4_mha_k_reshape
    )
    encoder4_mha_k_reshape = None
    initializers_onnx_initializer_130 = self.initializers.onnx_initializer_130
    encoder4_mha_v_w = getattr(self, "encoder4/mha/V/w")(
        encoder3_ln2, initializers_onnx_initializer_130
    )
    initializers_onnx_initializer_130 = None
    initializers_onnx_initializer_131 = self.initializers.onnx_initializer_131
    encoder4_mha_v_b = getattr(self, "encoder4/mha/V/b")(
        encoder4_mha_v_w, initializers_onnx_initializer_131
    )
    encoder4_mha_v_w = initializers_onnx_initializer_131 = None
    initializers_onnx_initializer_132 = self.initializers.onnx_initializer_132
    encoder4_mha_v_reshape = getattr(self, "encoder4/mha/V/reshape")(
        encoder4_mha_v_b, initializers_onnx_initializer_132
    )
    encoder4_mha_v_b = initializers_onnx_initializer_132 = None
    encoder4_mha_v_transpose = getattr(self, "encoder4/mha/V/transpose")(
        encoder4_mha_v_reshape
    )
    encoder4_mha_v_reshape = None
    encoder4_mha_qk_matmul = getattr(self, "encoder4/mha/QK/matmul")(
        encoder4_mha_q_transpose, encoder4_mha_k_transpose
    )
    encoder4_mha_q_transpose = encoder4_mha_k_transpose = None
    initializers_onnx_initializer_133 = self.initializers.onnx_initializer_133
    encoder4_mha_qk_scale = getattr(self, "encoder4/mha/QK/scale")(
        encoder4_mha_qk_matmul, initializers_onnx_initializer_133
    )
    encoder4_mha_qk_matmul = initializers_onnx_initializer_133 = None
    initializers_onnx_initializer_134 = self.initializers.onnx_initializer_134
    encoder4_smolgen_compress = getattr(self, "encoder4/smolgen/compress")(
        encoder3_ln2, initializers_onnx_initializer_134
    )
    initializers_onnx_initializer_134 = None
    initializers_onnx_initializer_135 = self.initializers.onnx_initializer_135
    encoder4_smolgen_compress_reshape = getattr(
        self, "encoder4/smolgen/compress/reshape"
    )(encoder4_smolgen_compress, initializers_onnx_initializer_135)
    encoder4_smolgen_compress = initializers_onnx_initializer_135 = None
    initializers_onnx_initializer_136 = self.initializers.onnx_initializer_136
    encoder4_smolgen_dense1_w = getattr(self, "encoder4/smolgen/dense1/w")(
        encoder4_smolgen_compress_reshape, initializers_onnx_initializer_136
    )
    encoder4_smolgen_compress_reshape = initializers_onnx_initializer_136 = None
    initializers_onnx_initializer_137 = self.initializers.onnx_initializer_137
    encoder4_smolgen_dense1_b = getattr(self, "encoder4/smolgen/dense1/b")(
        encoder4_smolgen_dense1_w, initializers_onnx_initializer_137
    )
    encoder4_smolgen_dense1_w = initializers_onnx_initializer_137 = None
    encoder4_smolgen_dense1_swish_sigmoid = getattr(
        self, "encoder4/smolgen/dense1/swish/sigmoid"
    )(encoder4_smolgen_dense1_b)
    encoder4_smolgen_dense1_swish = getattr(self, "encoder4/smolgen/dense1/swish")(
        encoder4_smolgen_dense1_swish_sigmoid, encoder4_smolgen_dense1_b
    )
    encoder4_smolgen_dense1_swish_sigmoid = encoder4_smolgen_dense1_b = None
    encoder4_smolgen_ln1 = getattr(self, "encoder4/smolgen/ln1")(
        encoder4_smolgen_dense1_swish
    )
    encoder4_smolgen_dense1_swish = None
    initializers_onnx_initializer_138 = self.initializers.onnx_initializer_138
    encoder4_smolgen_dense2_w = getattr(self, "encoder4/smolgen/dense2/w")(
        encoder4_smolgen_ln1, initializers_onnx_initializer_138
    )
    encoder4_smolgen_ln1 = initializers_onnx_initializer_138 = None
    initializers_onnx_initializer_139 = self.initializers.onnx_initializer_139
    encoder4_smolgen_dense2_b = getattr(self, "encoder4/smolgen/dense2/b")(
        encoder4_smolgen_dense2_w, initializers_onnx_initializer_139
    )
    encoder4_smolgen_dense2_w = initializers_onnx_initializer_139 = None
    encoder4_smolgen_dense2_swish_sigmoid = getattr(
        self, "encoder4/smolgen/dense2/swish/sigmoid"
    )(encoder4_smolgen_dense2_b)
    encoder4_smolgen_dense2_swish = getattr(self, "encoder4/smolgen/dense2/swish")(
        encoder4_smolgen_dense2_swish_sigmoid, encoder4_smolgen_dense2_b
    )
    encoder4_smolgen_dense2_swish_sigmoid = encoder4_smolgen_dense2_b = None
    encoder4_smolgen_ln2 = getattr(self, "encoder4/smolgen/ln2")(
        encoder4_smolgen_dense2_swish
    )
    encoder4_smolgen_dense2_swish = None
    initializers_onnx_initializer_140 = self.initializers.onnx_initializer_140
    encoder4_smolgen_gen_from_reshape = getattr(
        self, "encoder4/smolgen/gen_from/reshape"
    )(encoder4_smolgen_ln2, initializers_onnx_initializer_140)
    encoder4_smolgen_ln2 = initializers_onnx_initializer_140 = None
    initializers_onnx_initializer_141 = self.initializers.onnx_initializer_141
    encoder4_smolgen_smol_weight_gen = getattr(
        self, "encoder4/smolgen/smol_weight_gen"
    )(encoder4_smolgen_gen_from_reshape, initializers_onnx_initializer_141)
    encoder4_smolgen_gen_from_reshape = initializers_onnx_initializer_141 = None
    initializers_onnx_initializer_142 = self.initializers.onnx_initializer_142
    encoder4_smolgen_out_reshape = getattr(self, "encoder4/smolgen/out/reshape")(
        encoder4_smolgen_smol_weight_gen, initializers_onnx_initializer_142
    )
    encoder4_smolgen_smol_weight_gen = initializers_onnx_initializer_142 = None
    encoder4_smolgen_weights = getattr(self, "encoder4/smolgen_weights")(
        encoder4_mha_qk_scale, encoder4_smolgen_out_reshape
    )
    encoder4_mha_qk_scale = encoder4_smolgen_out_reshape = None
    encoder4_mha_qk_softmax = getattr(self, "encoder4/mha/QK/softmax")(
        encoder4_smolgen_weights
    )
    encoder4_smolgen_weights = None
    encoder4_mha_qkv_matmul = getattr(self, "encoder4/mha/QKV/matmul")(
        encoder4_mha_qk_softmax, encoder4_mha_v_transpose
    )
    encoder4_mha_qk_softmax = encoder4_mha_v_transpose = None
    encoder4_mha_out_transpose = getattr(self, "encoder4/mha/out/transpose")(
        encoder4_mha_qkv_matmul
    )
    encoder4_mha_qkv_matmul = None
    initializers_onnx_initializer_143 = self.initializers.onnx_initializer_143
    encoder4_mha_out_reshape = getattr(self, "encoder4/mha/out/reshape")(
        encoder4_mha_out_transpose, initializers_onnx_initializer_143
    )
    encoder4_mha_out_transpose = initializers_onnx_initializer_143 = None
    initializers_onnx_initializer_144 = self.initializers.onnx_initializer_144
    encoder4_mha_out_dense_w = getattr(self, "encoder4/mha/out/dense/w")(
        encoder4_mha_out_reshape, initializers_onnx_initializer_144
    )
    encoder4_mha_out_reshape = initializers_onnx_initializer_144 = None
    initializers_onnx_initializer_145 = self.initializers.onnx_initializer_145
    encoder4_mha_out_dense_b = getattr(self, "encoder4/mha/out/dense/b")(
        encoder4_mha_out_dense_w, initializers_onnx_initializer_145
    )

    encoder4_mha_out_dense_b = self.attention_output[4](
        encoder4_mha_out_dense_b.view(-1, 64, 768)
    ).view(-1, 768)

    encoder4_mha_out_dense_w = initializers_onnx_initializer_145 = None
    initializers_onnx_initializer_146 = self.initializers.onnx_initializer_146
    encoder4_alpha_input = getattr(self, "encoder4/alpha*input")(
        encoder3_ln2, initializers_onnx_initializer_146
    )
    encoder3_ln2 = initializers_onnx_initializer_146 = None
    encoder4_mha_out_skip = getattr(self, "encoder4/mha/out/skip")(
        encoder4_mha_out_dense_b, encoder4_alpha_input
    )
    encoder4_mha_out_dense_b = encoder4_alpha_input = None
    encoder4_ln1 = getattr(self, "encoder4/ln1")(encoder4_mha_out_skip)
    encoder4_mha_out_skip = None

    ########################################################################
    # Make residuals accessible with squares as a separate dim,
    # this is important for nnsight.
    ########################################################################
    # This is just an nn.Identity layer
    residual_reasonable_shape = self.post_attention[4](
        encoder4_ln1.reshape(-1, 64, 768)
    )
    encoder4_ln1 = residual_reasonable_shape.reshape(-1, 768)
    residual_reasonable_shape = None

    initializers_onnx_initializer_147 = self.initializers.onnx_initializer_147
    encoder4_ffn_dense1_w = getattr(self, "encoder4/ffn/dense1/w")(
        encoder4_ln1, initializers_onnx_initializer_147
    )
    initializers_onnx_initializer_147 = None
    initializers_onnx_initializer_148 = self.initializers.onnx_initializer_148
    encoder4_ffn_dense1_b = getattr(self, "encoder4/ffn/dense1/b")(
        encoder4_ffn_dense1_w, initializers_onnx_initializer_148
    )
    encoder4_ffn_dense1_w = initializers_onnx_initializer_148 = None
    encoder4_ffn_dense1_sqrrelu_relu = getattr(
        self, "encoder4/ffn/dense1/sqrrelu/relu"
    )(encoder4_ffn_dense1_b)
    encoder4_ffn_dense1_b = None
    encoder4_ffn_dense1_sqrrelu_sqr = getattr(self, "encoder4/ffn/dense1/sqrrelu/sqr")(
        encoder4_ffn_dense1_sqrrelu_relu, encoder4_ffn_dense1_sqrrelu_relu
    )
    encoder4_ffn_dense1_sqrrelu_relu = None
    initializers_onnx_initializer_149 = self.initializers.onnx_initializer_149
    encoder4_ffn_dense2_w = getattr(self, "encoder4/ffn/dense2/w")(
        encoder4_ffn_dense1_sqrrelu_sqr, initializers_onnx_initializer_149
    )
    encoder4_ffn_dense1_sqrrelu_sqr = initializers_onnx_initializer_149 = None
    initializers_onnx_initializer_150 = self.initializers.onnx_initializer_150
    encoder4_ffn_dense2_b = getattr(self, "encoder4/ffn/dense2/b")(
        encoder4_ffn_dense2_w, initializers_onnx_initializer_150
    )

    encoder4_ffn_dense2_b = self.mlp_output[4](
        encoder4_ffn_dense2_b.view(-1, 64, 768)
    ).view(-1, 768)

    encoder4_ffn_dense2_w = initializers_onnx_initializer_150 = None
    initializers_onnx_initializer_151 = self.initializers.onnx_initializer_151
    encoder4_alpha_out1 = getattr(self, "encoder4/alpha*out1")(
        encoder4_ln1, initializers_onnx_initializer_151
    )
    encoder4_ln1 = initializers_onnx_initializer_151 = None
    encoder4_ffn_skip = getattr(self, "encoder4/ffn/skip")(
        encoder4_ffn_dense2_b, encoder4_alpha_out1
    )
    encoder4_ffn_dense2_b = encoder4_alpha_out1 = None
    encoder4_ln2 = getattr(self, "encoder4/ln2")(encoder4_ffn_skip)
    encoder4_ffn_skip = None

    ########################################################################
    # Make residuals accessible with squares as a separate dim,
    # this is important for nnsight.
    ########################################################################
    # This is just an nn.Identity layer
    residual_reasonable_shape = self.post_mlp[4](encoder4_ln2.reshape(-1, 64, 768))
    encoder4_ln2 = residual_reasonable_shape.reshape(-1, 768)
    residual_reasonable_shape = None

    ########################################################################
    # Layer 5
    ########################################################################
    initializers_onnx_initializer_152 = self.initializers.onnx_initializer_152
    encoder5_mha_q_w = getattr(self, "encoder5/mha/Q/w")(
        encoder4_ln2, initializers_onnx_initializer_152
    )
    initializers_onnx_initializer_152 = None
    initializers_onnx_initializer_153 = self.initializers.onnx_initializer_153
    encoder5_mha_q_b = getattr(self, "encoder5/mha/Q/b")(
        encoder5_mha_q_w, initializers_onnx_initializer_153
    )
    encoder5_mha_q_w = initializers_onnx_initializer_153 = None
    initializers_onnx_initializer_154 = self.initializers.onnx_initializer_154
    encoder5_mha_q_reshape = getattr(self, "encoder5/mha/Q/reshape")(
        encoder5_mha_q_b, initializers_onnx_initializer_154
    )
    encoder5_mha_q_b = initializers_onnx_initializer_154 = None
    encoder5_mha_q_transpose = getattr(self, "encoder5/mha/Q/transpose")(
        encoder5_mha_q_reshape
    )
    encoder5_mha_q_reshape = None
    initializers_onnx_initializer_155 = self.initializers.onnx_initializer_155
    encoder5_mha_k_w = getattr(self, "encoder5/mha/K/w")(
        encoder4_ln2, initializers_onnx_initializer_155
    )
    initializers_onnx_initializer_155 = None
    initializers_onnx_initializer_156 = self.initializers.onnx_initializer_156
    encoder5_mha_k_b = getattr(self, "encoder5/mha/K/b")(
        encoder5_mha_k_w, initializers_onnx_initializer_156
    )
    encoder5_mha_k_w = initializers_onnx_initializer_156 = None
    initializers_onnx_initializer_157 = self.initializers.onnx_initializer_157
    encoder5_mha_k_reshape = getattr(self, "encoder5/mha/K/reshape")(
        encoder5_mha_k_b, initializers_onnx_initializer_157
    )
    encoder5_mha_k_b = initializers_onnx_initializer_157 = None
    encoder5_mha_k_transpose = getattr(self, "encoder5/mha/K/transpose")(
        encoder5_mha_k_reshape
    )
    encoder5_mha_k_reshape = None
    initializers_onnx_initializer_158 = self.initializers.onnx_initializer_158
    encoder5_mha_v_w = getattr(self, "encoder5/mha/V/w")(
        encoder4_ln2, initializers_onnx_initializer_158
    )
    initializers_onnx_initializer_158 = None
    initializers_onnx_initializer_159 = self.initializers.onnx_initializer_159
    encoder5_mha_v_b = getattr(self, "encoder5/mha/V/b")(
        encoder5_mha_v_w, initializers_onnx_initializer_159
    )
    encoder5_mha_v_w = initializers_onnx_initializer_159 = None
    initializers_onnx_initializer_160 = self.initializers.onnx_initializer_160
    encoder5_mha_v_reshape = getattr(self, "encoder5/mha/V/reshape")(
        encoder5_mha_v_b, initializers_onnx_initializer_160
    )
    encoder5_mha_v_b = initializers_onnx_initializer_160 = None
    encoder5_mha_v_transpose = getattr(self, "encoder5/mha/V/transpose")(
        encoder5_mha_v_reshape
    )
    encoder5_mha_v_reshape = None
    encoder5_mha_qk_matmul = getattr(self, "encoder5/mha/QK/matmul")(
        encoder5_mha_q_transpose, encoder5_mha_k_transpose
    )
    encoder5_mha_q_transpose = encoder5_mha_k_transpose = None
    initializers_onnx_initializer_161 = self.initializers.onnx_initializer_161
    encoder5_mha_qk_scale = getattr(self, "encoder5/mha/QK/scale")(
        encoder5_mha_qk_matmul, initializers_onnx_initializer_161
    )
    encoder5_mha_qk_matmul = initializers_onnx_initializer_161 = None
    initializers_onnx_initializer_162 = self.initializers.onnx_initializer_162
    encoder5_smolgen_compress = getattr(self, "encoder5/smolgen/compress")(
        encoder4_ln2, initializers_onnx_initializer_162
    )
    initializers_onnx_initializer_162 = None
    initializers_onnx_initializer_163 = self.initializers.onnx_initializer_163
    encoder5_smolgen_compress_reshape = getattr(
        self, "encoder5/smolgen/compress/reshape"
    )(encoder5_smolgen_compress, initializers_onnx_initializer_163)
    encoder5_smolgen_compress = initializers_onnx_initializer_163 = None
    initializers_onnx_initializer_164 = self.initializers.onnx_initializer_164
    encoder5_smolgen_dense1_w = getattr(self, "encoder5/smolgen/dense1/w")(
        encoder5_smolgen_compress_reshape, initializers_onnx_initializer_164
    )
    encoder5_smolgen_compress_reshape = initializers_onnx_initializer_164 = None
    initializers_onnx_initializer_165 = self.initializers.onnx_initializer_165
    encoder5_smolgen_dense1_b = getattr(self, "encoder5/smolgen/dense1/b")(
        encoder5_smolgen_dense1_w, initializers_onnx_initializer_165
    )
    encoder5_smolgen_dense1_w = initializers_onnx_initializer_165 = None
    encoder5_smolgen_dense1_swish_sigmoid = getattr(
        self, "encoder5/smolgen/dense1/swish/sigmoid"
    )(encoder5_smolgen_dense1_b)
    encoder5_smolgen_dense1_swish = getattr(self, "encoder5/smolgen/dense1/swish")(
        encoder5_smolgen_dense1_swish_sigmoid, encoder5_smolgen_dense1_b
    )
    encoder5_smolgen_dense1_swish_sigmoid = encoder5_smolgen_dense1_b = None
    encoder5_smolgen_ln1 = getattr(self, "encoder5/smolgen/ln1")(
        encoder5_smolgen_dense1_swish
    )
    encoder5_smolgen_dense1_swish = None
    initializers_onnx_initializer_166 = self.initializers.onnx_initializer_166
    encoder5_smolgen_dense2_w = getattr(self, "encoder5/smolgen/dense2/w")(
        encoder5_smolgen_ln1, initializers_onnx_initializer_166
    )
    encoder5_smolgen_ln1 = initializers_onnx_initializer_166 = None
    initializers_onnx_initializer_167 = self.initializers.onnx_initializer_167
    encoder5_smolgen_dense2_b = getattr(self, "encoder5/smolgen/dense2/b")(
        encoder5_smolgen_dense2_w, initializers_onnx_initializer_167
    )
    encoder5_smolgen_dense2_w = initializers_onnx_initializer_167 = None
    encoder5_smolgen_dense2_swish_sigmoid = getattr(
        self, "encoder5/smolgen/dense2/swish/sigmoid"
    )(encoder5_smolgen_dense2_b)
    encoder5_smolgen_dense2_swish = getattr(self, "encoder5/smolgen/dense2/swish")(
        encoder5_smolgen_dense2_swish_sigmoid, encoder5_smolgen_dense2_b
    )
    encoder5_smolgen_dense2_swish_sigmoid = encoder5_smolgen_dense2_b = None
    encoder5_smolgen_ln2 = getattr(self, "encoder5/smolgen/ln2")(
        encoder5_smolgen_dense2_swish
    )
    encoder5_smolgen_dense2_swish = None
    initializers_onnx_initializer_168 = self.initializers.onnx_initializer_168
    encoder5_smolgen_gen_from_reshape = getattr(
        self, "encoder5/smolgen/gen_from/reshape"
    )(encoder5_smolgen_ln2, initializers_onnx_initializer_168)
    encoder5_smolgen_ln2 = initializers_onnx_initializer_168 = None
    initializers_onnx_initializer_169 = self.initializers.onnx_initializer_169
    encoder5_smolgen_smol_weight_gen = getattr(
        self, "encoder5/smolgen/smol_weight_gen"
    )(encoder5_smolgen_gen_from_reshape, initializers_onnx_initializer_169)
    encoder5_smolgen_gen_from_reshape = initializers_onnx_initializer_169 = None
    initializers_onnx_initializer_170 = self.initializers.onnx_initializer_170
    encoder5_smolgen_out_reshape = getattr(self, "encoder5/smolgen/out/reshape")(
        encoder5_smolgen_smol_weight_gen, initializers_onnx_initializer_170
    )
    encoder5_smolgen_smol_weight_gen = initializers_onnx_initializer_170 = None
    encoder5_smolgen_weights = getattr(self, "encoder5/smolgen_weights")(
        encoder5_mha_qk_scale, encoder5_smolgen_out_reshape
    )
    encoder5_mha_qk_scale = encoder5_smolgen_out_reshape = None
    encoder5_mha_qk_softmax = getattr(self, "encoder5/mha/QK/softmax")(
        encoder5_smolgen_weights
    )
    encoder5_smolgen_weights = None
    encoder5_mha_qkv_matmul = getattr(self, "encoder5/mha/QKV/matmul")(
        encoder5_mha_qk_softmax, encoder5_mha_v_transpose
    )
    encoder5_mha_qk_softmax = encoder5_mha_v_transpose = None
    encoder5_mha_out_transpose = getattr(self, "encoder5/mha/out/transpose")(
        encoder5_mha_qkv_matmul
    )
    encoder5_mha_qkv_matmul = None
    initializers_onnx_initializer_171 = self.initializers.onnx_initializer_171
    encoder5_mha_out_reshape = getattr(self, "encoder5/mha/out/reshape")(
        encoder5_mha_out_transpose, initializers_onnx_initializer_171
    )
    encoder5_mha_out_transpose = initializers_onnx_initializer_171 = None
    initializers_onnx_initializer_172 = self.initializers.onnx_initializer_172
    encoder5_mha_out_dense_w = getattr(self, "encoder5/mha/out/dense/w")(
        encoder5_mha_out_reshape, initializers_onnx_initializer_172
    )
    encoder5_mha_out_reshape = initializers_onnx_initializer_172 = None
    initializers_onnx_initializer_173 = self.initializers.onnx_initializer_173
    encoder5_mha_out_dense_b = getattr(self, "encoder5/mha/out/dense/b")(
        encoder5_mha_out_dense_w, initializers_onnx_initializer_173
    )

    encoder5_mha_out_dense_b = self.attention_output[5](
        encoder5_mha_out_dense_b.view(-1, 64, 768)
    ).view(-1, 768)

    encoder5_mha_out_dense_w = initializers_onnx_initializer_173 = None
    initializers_onnx_initializer_174 = self.initializers.onnx_initializer_174
    encoder5_alpha_input = getattr(self, "encoder5/alpha*input")(
        encoder4_ln2, initializers_onnx_initializer_174
    )
    encoder4_ln2 = initializers_onnx_initializer_174 = None
    encoder5_mha_out_skip = getattr(self, "encoder5/mha/out/skip")(
        encoder5_mha_out_dense_b, encoder5_alpha_input
    )
    encoder5_mha_out_dense_b = encoder5_alpha_input = None
    encoder5_ln1 = getattr(self, "encoder5/ln1")(encoder5_mha_out_skip)
    encoder5_mha_out_skip = None

    ########################################################################
    # Make residuals accessible with squares as a separate dim,
    # this is important for nnsight.
    ########################################################################
    # This is just an nn.Identity layer
    residual_reasonable_shape = self.post_attention[5](
        encoder5_ln1.reshape(-1, 64, 768)
    )
    encoder5_ln1 = residual_reasonable_shape.reshape(-1, 768)
    residual_reasonable_shape = None

    initializers_onnx_initializer_175 = self.initializers.onnx_initializer_175
    encoder5_ffn_dense1_w = getattr(self, "encoder5/ffn/dense1/w")(
        encoder5_ln1, initializers_onnx_initializer_175
    )
    initializers_onnx_initializer_175 = None
    initializers_onnx_initializer_176 = self.initializers.onnx_initializer_176
    encoder5_ffn_dense1_b = getattr(self, "encoder5/ffn/dense1/b")(
        encoder5_ffn_dense1_w, initializers_onnx_initializer_176
    )
    encoder5_ffn_dense1_w = initializers_onnx_initializer_176 = None
    encoder5_ffn_dense1_sqrrelu_relu = getattr(
        self, "encoder5/ffn/dense1/sqrrelu/relu"
    )(encoder5_ffn_dense1_b)
    encoder5_ffn_dense1_b = None
    encoder5_ffn_dense1_sqrrelu_sqr = getattr(self, "encoder5/ffn/dense1/sqrrelu/sqr")(
        encoder5_ffn_dense1_sqrrelu_relu, encoder5_ffn_dense1_sqrrelu_relu
    )
    encoder5_ffn_dense1_sqrrelu_relu = None
    initializers_onnx_initializer_177 = self.initializers.onnx_initializer_177
    encoder5_ffn_dense2_w = getattr(self, "encoder5/ffn/dense2/w")(
        encoder5_ffn_dense1_sqrrelu_sqr, initializers_onnx_initializer_177
    )
    encoder5_ffn_dense1_sqrrelu_sqr = initializers_onnx_initializer_177 = None
    initializers_onnx_initializer_178 = self.initializers.onnx_initializer_178
    encoder5_ffn_dense2_b = getattr(self, "encoder5/ffn/dense2/b")(
        encoder5_ffn_dense2_w, initializers_onnx_initializer_178
    )

    encoder5_ffn_dense2_b = self.mlp_output[5](
        encoder5_ffn_dense2_b.view(-1, 64, 768)
    ).view(-1, 768)

    encoder5_ffn_dense2_w = initializers_onnx_initializer_178 = None
    initializers_onnx_initializer_179 = self.initializers.onnx_initializer_179
    encoder5_alpha_out1 = getattr(self, "encoder5/alpha*out1")(
        encoder5_ln1, initializers_onnx_initializer_179
    )
    encoder5_ln1 = initializers_onnx_initializer_179 = None
    encoder5_ffn_skip = getattr(self, "encoder5/ffn/skip")(
        encoder5_ffn_dense2_b, encoder5_alpha_out1
    )
    encoder5_ffn_dense2_b = encoder5_alpha_out1 = None
    encoder5_ln2 = getattr(self, "encoder5/ln2")(encoder5_ffn_skip)
    encoder5_ffn_skip = None

    ########################################################################
    # Make residuals accessible with squares as a separate dim,
    # this is important for nnsight.
    ########################################################################
    # This is just an nn.Identity layer
    residual_reasonable_shape = self.post_mlp[5](encoder5_ln2.reshape(-1, 64, 768))
    encoder5_ln2 = residual_reasonable_shape.reshape(-1, 768)
    residual_reasonable_shape = None

    ########################################################################
    # Layer 6
    ########################################################################
    initializers_onnx_initializer_180 = self.initializers.onnx_initializer_180
    encoder6_mha_q_w = getattr(self, "encoder6/mha/Q/w")(
        encoder5_ln2, initializers_onnx_initializer_180
    )
    initializers_onnx_initializer_180 = None
    initializers_onnx_initializer_181 = self.initializers.onnx_initializer_181
    encoder6_mha_q_b = getattr(self, "encoder6/mha/Q/b")(
        encoder6_mha_q_w, initializers_onnx_initializer_181
    )
    encoder6_mha_q_w = initializers_onnx_initializer_181 = None
    initializers_onnx_initializer_182 = self.initializers.onnx_initializer_182
    encoder6_mha_q_reshape = getattr(self, "encoder6/mha/Q/reshape")(
        encoder6_mha_q_b, initializers_onnx_initializer_182
    )
    encoder6_mha_q_b = initializers_onnx_initializer_182 = None
    encoder6_mha_q_transpose = getattr(self, "encoder6/mha/Q/transpose")(
        encoder6_mha_q_reshape
    )
    encoder6_mha_q_reshape = None
    initializers_onnx_initializer_183 = self.initializers.onnx_initializer_183
    encoder6_mha_k_w = getattr(self, "encoder6/mha/K/w")(
        encoder5_ln2, initializers_onnx_initializer_183
    )
    initializers_onnx_initializer_183 = None
    initializers_onnx_initializer_184 = self.initializers.onnx_initializer_184
    encoder6_mha_k_b = getattr(self, "encoder6/mha/K/b")(
        encoder6_mha_k_w, initializers_onnx_initializer_184
    )
    encoder6_mha_k_w = initializers_onnx_initializer_184 = None
    initializers_onnx_initializer_185 = self.initializers.onnx_initializer_185
    encoder6_mha_k_reshape = getattr(self, "encoder6/mha/K/reshape")(
        encoder6_mha_k_b, initializers_onnx_initializer_185
    )
    encoder6_mha_k_b = initializers_onnx_initializer_185 = None
    encoder6_mha_k_transpose = getattr(self, "encoder6/mha/K/transpose")(
        encoder6_mha_k_reshape
    )
    encoder6_mha_k_reshape = None
    initializers_onnx_initializer_186 = self.initializers.onnx_initializer_186
    encoder6_mha_v_w = getattr(self, "encoder6/mha/V/w")(
        encoder5_ln2, initializers_onnx_initializer_186
    )
    initializers_onnx_initializer_186 = None
    initializers_onnx_initializer_187 = self.initializers.onnx_initializer_187
    encoder6_mha_v_b = getattr(self, "encoder6/mha/V/b")(
        encoder6_mha_v_w, initializers_onnx_initializer_187
    )
    encoder6_mha_v_w = initializers_onnx_initializer_187 = None
    initializers_onnx_initializer_188 = self.initializers.onnx_initializer_188
    encoder6_mha_v_reshape = getattr(self, "encoder6/mha/V/reshape")(
        encoder6_mha_v_b, initializers_onnx_initializer_188
    )
    encoder6_mha_v_b = initializers_onnx_initializer_188 = None
    encoder6_mha_v_transpose = getattr(self, "encoder6/mha/V/transpose")(
        encoder6_mha_v_reshape
    )
    encoder6_mha_v_reshape = None
    encoder6_mha_qk_matmul = getattr(self, "encoder6/mha/QK/matmul")(
        encoder6_mha_q_transpose, encoder6_mha_k_transpose
    )
    encoder6_mha_q_transpose = encoder6_mha_k_transpose = None
    initializers_onnx_initializer_189 = self.initializers.onnx_initializer_189
    encoder6_mha_qk_scale = getattr(self, "encoder6/mha/QK/scale")(
        encoder6_mha_qk_matmul, initializers_onnx_initializer_189
    )
    encoder6_mha_qk_matmul = initializers_onnx_initializer_189 = None
    initializers_onnx_initializer_190 = self.initializers.onnx_initializer_190
    encoder6_smolgen_compress = getattr(self, "encoder6/smolgen/compress")(
        encoder5_ln2, initializers_onnx_initializer_190
    )
    initializers_onnx_initializer_190 = None
    initializers_onnx_initializer_191 = self.initializers.onnx_initializer_191
    encoder6_smolgen_compress_reshape = getattr(
        self, "encoder6/smolgen/compress/reshape"
    )(encoder6_smolgen_compress, initializers_onnx_initializer_191)
    encoder6_smolgen_compress = initializers_onnx_initializer_191 = None
    initializers_onnx_initializer_192 = self.initializers.onnx_initializer_192
    encoder6_smolgen_dense1_w = getattr(self, "encoder6/smolgen/dense1/w")(
        encoder6_smolgen_compress_reshape, initializers_onnx_initializer_192
    )
    encoder6_smolgen_compress_reshape = initializers_onnx_initializer_192 = None
    initializers_onnx_initializer_193 = self.initializers.onnx_initializer_193
    encoder6_smolgen_dense1_b = getattr(self, "encoder6/smolgen/dense1/b")(
        encoder6_smolgen_dense1_w, initializers_onnx_initializer_193
    )
    encoder6_smolgen_dense1_w = initializers_onnx_initializer_193 = None
    encoder6_smolgen_dense1_swish_sigmoid = getattr(
        self, "encoder6/smolgen/dense1/swish/sigmoid"
    )(encoder6_smolgen_dense1_b)
    encoder6_smolgen_dense1_swish = getattr(self, "encoder6/smolgen/dense1/swish")(
        encoder6_smolgen_dense1_swish_sigmoid, encoder6_smolgen_dense1_b
    )
    encoder6_smolgen_dense1_swish_sigmoid = encoder6_smolgen_dense1_b = None
    encoder6_smolgen_ln1 = getattr(self, "encoder6/smolgen/ln1")(
        encoder6_smolgen_dense1_swish
    )
    encoder6_smolgen_dense1_swish = None
    initializers_onnx_initializer_194 = self.initializers.onnx_initializer_194
    encoder6_smolgen_dense2_w = getattr(self, "encoder6/smolgen/dense2/w")(
        encoder6_smolgen_ln1, initializers_onnx_initializer_194
    )
    encoder6_smolgen_ln1 = initializers_onnx_initializer_194 = None
    initializers_onnx_initializer_195 = self.initializers.onnx_initializer_195
    encoder6_smolgen_dense2_b = getattr(self, "encoder6/smolgen/dense2/b")(
        encoder6_smolgen_dense2_w, initializers_onnx_initializer_195
    )
    encoder6_smolgen_dense2_w = initializers_onnx_initializer_195 = None
    encoder6_smolgen_dense2_swish_sigmoid = getattr(
        self, "encoder6/smolgen/dense2/swish/sigmoid"
    )(encoder6_smolgen_dense2_b)
    encoder6_smolgen_dense2_swish = getattr(self, "encoder6/smolgen/dense2/swish")(
        encoder6_smolgen_dense2_swish_sigmoid, encoder6_smolgen_dense2_b
    )
    encoder6_smolgen_dense2_swish_sigmoid = encoder6_smolgen_dense2_b = None
    encoder6_smolgen_ln2 = getattr(self, "encoder6/smolgen/ln2")(
        encoder6_smolgen_dense2_swish
    )
    encoder6_smolgen_dense2_swish = None
    initializers_onnx_initializer_196 = self.initializers.onnx_initializer_196
    encoder6_smolgen_gen_from_reshape = getattr(
        self, "encoder6/smolgen/gen_from/reshape"
    )(encoder6_smolgen_ln2, initializers_onnx_initializer_196)
    encoder6_smolgen_ln2 = initializers_onnx_initializer_196 = None
    initializers_onnx_initializer_197 = self.initializers.onnx_initializer_197
    encoder6_smolgen_smol_weight_gen = getattr(
        self, "encoder6/smolgen/smol_weight_gen"
    )(encoder6_smolgen_gen_from_reshape, initializers_onnx_initializer_197)
    encoder6_smolgen_gen_from_reshape = initializers_onnx_initializer_197 = None
    initializers_onnx_initializer_198 = self.initializers.onnx_initializer_198
    encoder6_smolgen_out_reshape = getattr(self, "encoder6/smolgen/out/reshape")(
        encoder6_smolgen_smol_weight_gen, initializers_onnx_initializer_198
    )
    encoder6_smolgen_smol_weight_gen = initializers_onnx_initializer_198 = None
    encoder6_smolgen_weights = getattr(self, "encoder6/smolgen_weights")(
        encoder6_mha_qk_scale, encoder6_smolgen_out_reshape
    )
    encoder6_mha_qk_scale = encoder6_smolgen_out_reshape = None
    encoder6_mha_qk_softmax = getattr(self, "encoder6/mha/QK/softmax")(
        encoder6_smolgen_weights
    )
    encoder6_smolgen_weights = None
    encoder6_mha_qkv_matmul = getattr(self, "encoder6/mha/QKV/matmul")(
        encoder6_mha_qk_softmax, encoder6_mha_v_transpose
    )
    encoder6_mha_qk_softmax = encoder6_mha_v_transpose = None
    encoder6_mha_out_transpose = getattr(self, "encoder6/mha/out/transpose")(
        encoder6_mha_qkv_matmul
    )
    encoder6_mha_qkv_matmul = None
    initializers_onnx_initializer_199 = self.initializers.onnx_initializer_199
    encoder6_mha_out_reshape = getattr(self, "encoder6/mha/out/reshape")(
        encoder6_mha_out_transpose, initializers_onnx_initializer_199
    )
    encoder6_mha_out_transpose = initializers_onnx_initializer_199 = None
    initializers_onnx_initializer_200 = self.initializers.onnx_initializer_200
    encoder6_mha_out_dense_w = getattr(self, "encoder6/mha/out/dense/w")(
        encoder6_mha_out_reshape, initializers_onnx_initializer_200
    )
    encoder6_mha_out_reshape = initializers_onnx_initializer_200 = None
    initializers_onnx_initializer_201 = self.initializers.onnx_initializer_201
    encoder6_mha_out_dense_b = getattr(self, "encoder6/mha/out/dense/b")(
        encoder6_mha_out_dense_w, initializers_onnx_initializer_201
    )

    encoder6_mha_out_dense_b = self.attention_output[6](
        encoder6_mha_out_dense_b.view(-1, 64, 768)
    ).view(-1, 768)

    encoder6_mha_out_dense_w = initializers_onnx_initializer_201 = None
    initializers_onnx_initializer_202 = self.initializers.onnx_initializer_202
    encoder6_alpha_input = getattr(self, "encoder6/alpha*input")(
        encoder5_ln2, initializers_onnx_initializer_202
    )
    encoder5_ln2 = initializers_onnx_initializer_202 = None
    encoder6_mha_out_skip = getattr(self, "encoder6/mha/out/skip")(
        encoder6_mha_out_dense_b, encoder6_alpha_input
    )
    encoder6_mha_out_dense_b = encoder6_alpha_input = None
    encoder6_ln1 = getattr(self, "encoder6/ln1")(encoder6_mha_out_skip)
    encoder6_mha_out_skip = None

    ########################################################################
    # Make residuals accessible with squares as a separate dim,
    # this is important for nnsight.
    ########################################################################
    # This is just an nn.Identity layer
    residual_reasonable_shape = self.post_attention[6](
        encoder6_ln1.reshape(-1, 64, 768)
    )
    encoder6_ln1 = residual_reasonable_shape.reshape(-1, 768)
    residual_reasonable_shape = None

    initializers_onnx_initializer_203 = self.initializers.onnx_initializer_203
    encoder6_ffn_dense1_w = getattr(self, "encoder6/ffn/dense1/w")(
        encoder6_ln1, initializers_onnx_initializer_203
    )
    initializers_onnx_initializer_203 = None
    initializers_onnx_initializer_204 = self.initializers.onnx_initializer_204
    encoder6_ffn_dense1_b = getattr(self, "encoder6/ffn/dense1/b")(
        encoder6_ffn_dense1_w, initializers_onnx_initializer_204
    )
    encoder6_ffn_dense1_w = initializers_onnx_initializer_204 = None
    encoder6_ffn_dense1_sqrrelu_relu = getattr(
        self, "encoder6/ffn/dense1/sqrrelu/relu"
    )(encoder6_ffn_dense1_b)
    encoder6_ffn_dense1_b = None
    encoder6_ffn_dense1_sqrrelu_sqr = getattr(self, "encoder6/ffn/dense1/sqrrelu/sqr")(
        encoder6_ffn_dense1_sqrrelu_relu, encoder6_ffn_dense1_sqrrelu_relu
    )
    encoder6_ffn_dense1_sqrrelu_relu = None
    initializers_onnx_initializer_205 = self.initializers.onnx_initializer_205
    encoder6_ffn_dense2_w = getattr(self, "encoder6/ffn/dense2/w")(
        encoder6_ffn_dense1_sqrrelu_sqr, initializers_onnx_initializer_205
    )
    encoder6_ffn_dense1_sqrrelu_sqr = initializers_onnx_initializer_205 = None
    initializers_onnx_initializer_206 = self.initializers.onnx_initializer_206
    encoder6_ffn_dense2_b = getattr(self, "encoder6/ffn/dense2/b")(
        encoder6_ffn_dense2_w, initializers_onnx_initializer_206
    )

    encoder6_ffn_dense2_b = self.mlp_output[6](
        encoder6_ffn_dense2_b.view(-1, 64, 768)
    ).view(-1, 768)

    encoder6_ffn_dense2_w = initializers_onnx_initializer_206 = None
    initializers_onnx_initializer_207 = self.initializers.onnx_initializer_207
    encoder6_alpha_out1 = getattr(self, "encoder6/alpha*out1")(
        encoder6_ln1, initializers_onnx_initializer_207
    )
    encoder6_ln1 = initializers_onnx_initializer_207 = None
    encoder6_ffn_skip = getattr(self, "encoder6/ffn/skip")(
        encoder6_ffn_dense2_b, encoder6_alpha_out1
    )
    encoder6_ffn_dense2_b = encoder6_alpha_out1 = None
    encoder6_ln2 = getattr(self, "encoder6/ln2")(encoder6_ffn_skip)
    encoder6_ffn_skip = None

    ########################################################################
    # Make residuals accessible with squares as a separate dim,
    # this is important for nnsight.
    ########################################################################
    # This is just an nn.Identity layer
    residual_reasonable_shape = self.post_mlp[6](encoder6_ln2.reshape(-1, 64, 768))
    encoder6_ln2 = residual_reasonable_shape.reshape(-1, 768)
    residual_reasonable_shape = None

    ########################################################################
    # Layer 7
    ########################################################################
    initializers_onnx_initializer_208 = self.initializers.onnx_initializer_208
    encoder7_mha_q_w = getattr(self, "encoder7/mha/Q/w")(
        encoder6_ln2, initializers_onnx_initializer_208
    )
    initializers_onnx_initializer_208 = None
    initializers_onnx_initializer_209 = self.initializers.onnx_initializer_209
    encoder7_mha_q_b = getattr(self, "encoder7/mha/Q/b")(
        encoder7_mha_q_w, initializers_onnx_initializer_209
    )
    encoder7_mha_q_w = initializers_onnx_initializer_209 = None
    initializers_onnx_initializer_210 = self.initializers.onnx_initializer_210
    encoder7_mha_q_reshape = getattr(self, "encoder7/mha/Q/reshape")(
        encoder7_mha_q_b, initializers_onnx_initializer_210
    )
    encoder7_mha_q_b = initializers_onnx_initializer_210 = None
    encoder7_mha_q_transpose = getattr(self, "encoder7/mha/Q/transpose")(
        encoder7_mha_q_reshape
    )
    encoder7_mha_q_reshape = None
    initializers_onnx_initializer_211 = self.initializers.onnx_initializer_211
    encoder7_mha_k_w = getattr(self, "encoder7/mha/K/w")(
        encoder6_ln2, initializers_onnx_initializer_211
    )
    initializers_onnx_initializer_211 = None
    initializers_onnx_initializer_212 = self.initializers.onnx_initializer_212
    encoder7_mha_k_b = getattr(self, "encoder7/mha/K/b")(
        encoder7_mha_k_w, initializers_onnx_initializer_212
    )
    encoder7_mha_k_w = initializers_onnx_initializer_212 = None
    initializers_onnx_initializer_213 = self.initializers.onnx_initializer_213
    encoder7_mha_k_reshape = getattr(self, "encoder7/mha/K/reshape")(
        encoder7_mha_k_b, initializers_onnx_initializer_213
    )
    encoder7_mha_k_b = initializers_onnx_initializer_213 = None
    encoder7_mha_k_transpose = getattr(self, "encoder7/mha/K/transpose")(
        encoder7_mha_k_reshape
    )
    encoder7_mha_k_reshape = None
    initializers_onnx_initializer_214 = self.initializers.onnx_initializer_214
    encoder7_mha_v_w = getattr(self, "encoder7/mha/V/w")(
        encoder6_ln2, initializers_onnx_initializer_214
    )
    initializers_onnx_initializer_214 = None
    initializers_onnx_initializer_215 = self.initializers.onnx_initializer_215
    encoder7_mha_v_b = getattr(self, "encoder7/mha/V/b")(
        encoder7_mha_v_w, initializers_onnx_initializer_215
    )
    encoder7_mha_v_w = initializers_onnx_initializer_215 = None
    initializers_onnx_initializer_216 = self.initializers.onnx_initializer_216
    encoder7_mha_v_reshape = getattr(self, "encoder7/mha/V/reshape")(
        encoder7_mha_v_b, initializers_onnx_initializer_216
    )
    encoder7_mha_v_b = initializers_onnx_initializer_216 = None
    encoder7_mha_v_transpose = getattr(self, "encoder7/mha/V/transpose")(
        encoder7_mha_v_reshape
    )
    encoder7_mha_v_reshape = None
    encoder7_mha_qk_matmul = getattr(self, "encoder7/mha/QK/matmul")(
        encoder7_mha_q_transpose, encoder7_mha_k_transpose
    )
    encoder7_mha_q_transpose = encoder7_mha_k_transpose = None
    initializers_onnx_initializer_217 = self.initializers.onnx_initializer_217
    encoder7_mha_qk_scale = getattr(self, "encoder7/mha/QK/scale")(
        encoder7_mha_qk_matmul, initializers_onnx_initializer_217
    )
    encoder7_mha_qk_matmul = initializers_onnx_initializer_217 = None
    initializers_onnx_initializer_218 = self.initializers.onnx_initializer_218
    encoder7_smolgen_compress = getattr(self, "encoder7/smolgen/compress")(
        encoder6_ln2, initializers_onnx_initializer_218
    )
    initializers_onnx_initializer_218 = None
    initializers_onnx_initializer_219 = self.initializers.onnx_initializer_219
    encoder7_smolgen_compress_reshape = getattr(
        self, "encoder7/smolgen/compress/reshape"
    )(encoder7_smolgen_compress, initializers_onnx_initializer_219)
    encoder7_smolgen_compress = initializers_onnx_initializer_219 = None
    initializers_onnx_initializer_220 = self.initializers.onnx_initializer_220
    encoder7_smolgen_dense1_w = getattr(self, "encoder7/smolgen/dense1/w")(
        encoder7_smolgen_compress_reshape, initializers_onnx_initializer_220
    )
    encoder7_smolgen_compress_reshape = initializers_onnx_initializer_220 = None
    initializers_onnx_initializer_221 = self.initializers.onnx_initializer_221
    encoder7_smolgen_dense1_b = getattr(self, "encoder7/smolgen/dense1/b")(
        encoder7_smolgen_dense1_w, initializers_onnx_initializer_221
    )
    encoder7_smolgen_dense1_w = initializers_onnx_initializer_221 = None
    encoder7_smolgen_dense1_swish_sigmoid = getattr(
        self, "encoder7/smolgen/dense1/swish/sigmoid"
    )(encoder7_smolgen_dense1_b)
    encoder7_smolgen_dense1_swish = getattr(self, "encoder7/smolgen/dense1/swish")(
        encoder7_smolgen_dense1_swish_sigmoid, encoder7_smolgen_dense1_b
    )
    encoder7_smolgen_dense1_swish_sigmoid = encoder7_smolgen_dense1_b = None
    encoder7_smolgen_ln1 = getattr(self, "encoder7/smolgen/ln1")(
        encoder7_smolgen_dense1_swish
    )
    encoder7_smolgen_dense1_swish = None
    initializers_onnx_initializer_222 = self.initializers.onnx_initializer_222
    encoder7_smolgen_dense2_w = getattr(self, "encoder7/smolgen/dense2/w")(
        encoder7_smolgen_ln1, initializers_onnx_initializer_222
    )
    encoder7_smolgen_ln1 = initializers_onnx_initializer_222 = None
    initializers_onnx_initializer_223 = self.initializers.onnx_initializer_223
    encoder7_smolgen_dense2_b = getattr(self, "encoder7/smolgen/dense2/b")(
        encoder7_smolgen_dense2_w, initializers_onnx_initializer_223
    )
    encoder7_smolgen_dense2_w = initializers_onnx_initializer_223 = None
    encoder7_smolgen_dense2_swish_sigmoid = getattr(
        self, "encoder7/smolgen/dense2/swish/sigmoid"
    )(encoder7_smolgen_dense2_b)
    encoder7_smolgen_dense2_swish = getattr(self, "encoder7/smolgen/dense2/swish")(
        encoder7_smolgen_dense2_swish_sigmoid, encoder7_smolgen_dense2_b
    )
    encoder7_smolgen_dense2_swish_sigmoid = encoder7_smolgen_dense2_b = None
    encoder7_smolgen_ln2 = getattr(self, "encoder7/smolgen/ln2")(
        encoder7_smolgen_dense2_swish
    )
    encoder7_smolgen_dense2_swish = None
    initializers_onnx_initializer_224 = self.initializers.onnx_initializer_224
    encoder7_smolgen_gen_from_reshape = getattr(
        self, "encoder7/smolgen/gen_from/reshape"
    )(encoder7_smolgen_ln2, initializers_onnx_initializer_224)
    encoder7_smolgen_ln2 = initializers_onnx_initializer_224 = None
    initializers_onnx_initializer_225 = self.initializers.onnx_initializer_225
    encoder7_smolgen_smol_weight_gen = getattr(
        self, "encoder7/smolgen/smol_weight_gen"
    )(encoder7_smolgen_gen_from_reshape, initializers_onnx_initializer_225)
    encoder7_smolgen_gen_from_reshape = initializers_onnx_initializer_225 = None
    initializers_onnx_initializer_226 = self.initializers.onnx_initializer_226
    encoder7_smolgen_out_reshape = getattr(self, "encoder7/smolgen/out/reshape")(
        encoder7_smolgen_smol_weight_gen, initializers_onnx_initializer_226
    )
    encoder7_smolgen_smol_weight_gen = initializers_onnx_initializer_226 = None
    encoder7_smolgen_weights = getattr(self, "encoder7/smolgen_weights")(
        encoder7_mha_qk_scale, encoder7_smolgen_out_reshape
    )
    encoder7_mha_qk_scale = encoder7_smolgen_out_reshape = None
    encoder7_mha_qk_softmax = getattr(self, "encoder7/mha/QK/softmax")(
        encoder7_smolgen_weights
    )
    encoder7_smolgen_weights = None
    encoder7_mha_qkv_matmul = getattr(self, "encoder7/mha/QKV/matmul")(
        encoder7_mha_qk_softmax, encoder7_mha_v_transpose
    )
    encoder7_mha_qk_softmax = encoder7_mha_v_transpose = None
    encoder7_mha_out_transpose = getattr(self, "encoder7/mha/out/transpose")(
        encoder7_mha_qkv_matmul
    )
    encoder7_mha_qkv_matmul = None
    initializers_onnx_initializer_227 = self.initializers.onnx_initializer_227
    encoder7_mha_out_reshape = getattr(self, "encoder7/mha/out/reshape")(
        encoder7_mha_out_transpose, initializers_onnx_initializer_227
    )
    encoder7_mha_out_transpose = initializers_onnx_initializer_227 = None
    initializers_onnx_initializer_228 = self.initializers.onnx_initializer_228
    encoder7_mha_out_dense_w = getattr(self, "encoder7/mha/out/dense/w")(
        encoder7_mha_out_reshape, initializers_onnx_initializer_228
    )
    encoder7_mha_out_reshape = initializers_onnx_initializer_228 = None
    initializers_onnx_initializer_229 = self.initializers.onnx_initializer_229
    encoder7_mha_out_dense_b = getattr(self, "encoder7/mha/out/dense/b")(
        encoder7_mha_out_dense_w, initializers_onnx_initializer_229
    )

    encoder7_mha_out_dense_b = self.attention_output[7](
        encoder7_mha_out_dense_b.view(-1, 64, 768)
    ).view(-1, 768)

    encoder7_mha_out_dense_w = initializers_onnx_initializer_229 = None
    initializers_onnx_initializer_230 = self.initializers.onnx_initializer_230
    encoder7_alpha_input = getattr(self, "encoder7/alpha*input")(
        encoder6_ln2, initializers_onnx_initializer_230
    )
    encoder6_ln2 = initializers_onnx_initializer_230 = None
    encoder7_mha_out_skip = getattr(self, "encoder7/mha/out/skip")(
        encoder7_mha_out_dense_b, encoder7_alpha_input
    )
    encoder7_mha_out_dense_b = encoder7_alpha_input = None
    encoder7_ln1 = getattr(self, "encoder7/ln1")(encoder7_mha_out_skip)
    encoder7_mha_out_skip = None

    ########################################################################
    # Make residuals accessible with squares as a separate dim,
    # this is important for nnsight.
    ########################################################################
    # This is just an nn.Identity layer
    residual_reasonable_shape = self.post_attention[7](
        encoder7_ln1.reshape(-1, 64, 768)
    )
    encoder7_ln1 = residual_reasonable_shape.reshape(-1, 768)
    residual_reasonable_shape = None

    initializers_onnx_initializer_231 = self.initializers.onnx_initializer_231
    encoder7_ffn_dense1_w = getattr(self, "encoder7/ffn/dense1/w")(
        encoder7_ln1, initializers_onnx_initializer_231
    )
    initializers_onnx_initializer_231 = None
    initializers_onnx_initializer_232 = self.initializers.onnx_initializer_232
    encoder7_ffn_dense1_b = getattr(self, "encoder7/ffn/dense1/b")(
        encoder7_ffn_dense1_w, initializers_onnx_initializer_232
    )
    encoder7_ffn_dense1_w = initializers_onnx_initializer_232 = None
    encoder7_ffn_dense1_sqrrelu_relu = getattr(
        self, "encoder7/ffn/dense1/sqrrelu/relu"
    )(encoder7_ffn_dense1_b)
    encoder7_ffn_dense1_b = None
    encoder7_ffn_dense1_sqrrelu_sqr = getattr(self, "encoder7/ffn/dense1/sqrrelu/sqr")(
        encoder7_ffn_dense1_sqrrelu_relu, encoder7_ffn_dense1_sqrrelu_relu
    )
    encoder7_ffn_dense1_sqrrelu_relu = None
    initializers_onnx_initializer_233 = self.initializers.onnx_initializer_233
    encoder7_ffn_dense2_w = getattr(self, "encoder7/ffn/dense2/w")(
        encoder7_ffn_dense1_sqrrelu_sqr, initializers_onnx_initializer_233
    )
    encoder7_ffn_dense1_sqrrelu_sqr = initializers_onnx_initializer_233 = None
    initializers_onnx_initializer_234 = self.initializers.onnx_initializer_234
    encoder7_ffn_dense2_b = getattr(self, "encoder7/ffn/dense2/b")(
        encoder7_ffn_dense2_w, initializers_onnx_initializer_234
    )

    encoder7_ffn_dense2_b = self.mlp_output[7](
        encoder7_ffn_dense2_b.view(-1, 64, 768)
    ).view(-1, 768)

    encoder7_ffn_dense2_w = initializers_onnx_initializer_234 = None
    initializers_onnx_initializer_235 = self.initializers.onnx_initializer_235
    encoder7_alpha_out1 = getattr(self, "encoder7/alpha*out1")(
        encoder7_ln1, initializers_onnx_initializer_235
    )
    encoder7_ln1 = initializers_onnx_initializer_235 = None
    encoder7_ffn_skip = getattr(self, "encoder7/ffn/skip")(
        encoder7_ffn_dense2_b, encoder7_alpha_out1
    )
    encoder7_ffn_dense2_b = encoder7_alpha_out1 = None
    encoder7_ln2 = getattr(self, "encoder7/ln2")(encoder7_ffn_skip)
    encoder7_ffn_skip = None

    ########################################################################
    # Make residuals accessible with squares as a separate dim,
    # this is important for nnsight.
    ########################################################################
    # This is just an nn.Identity layer
    residual_reasonable_shape = self.post_mlp[7](encoder7_ln2.reshape(-1, 64, 768))
    encoder7_ln2 = residual_reasonable_shape.reshape(-1, 768)
    residual_reasonable_shape = None

    ########################################################################
    # Layer 8
    ########################################################################
    initializers_onnx_initializer_236 = self.initializers.onnx_initializer_236
    encoder8_mha_q_w = getattr(self, "encoder8/mha/Q/w")(
        encoder7_ln2, initializers_onnx_initializer_236
    )
    initializers_onnx_initializer_236 = None
    initializers_onnx_initializer_237 = self.initializers.onnx_initializer_237
    encoder8_mha_q_b = getattr(self, "encoder8/mha/Q/b")(
        encoder8_mha_q_w, initializers_onnx_initializer_237
    )
    encoder8_mha_q_w = initializers_onnx_initializer_237 = None
    initializers_onnx_initializer_238 = self.initializers.onnx_initializer_238
    encoder8_mha_q_reshape = getattr(self, "encoder8/mha/Q/reshape")(
        encoder8_mha_q_b, initializers_onnx_initializer_238
    )
    encoder8_mha_q_b = initializers_onnx_initializer_238 = None
    encoder8_mha_q_transpose = getattr(self, "encoder8/mha/Q/transpose")(
        encoder8_mha_q_reshape
    )
    encoder8_mha_q_reshape = None
    initializers_onnx_initializer_239 = self.initializers.onnx_initializer_239
    encoder8_mha_k_w = getattr(self, "encoder8/mha/K/w")(
        encoder7_ln2, initializers_onnx_initializer_239
    )
    initializers_onnx_initializer_239 = None
    initializers_onnx_initializer_240 = self.initializers.onnx_initializer_240
    encoder8_mha_k_b = getattr(self, "encoder8/mha/K/b")(
        encoder8_mha_k_w, initializers_onnx_initializer_240
    )
    encoder8_mha_k_w = initializers_onnx_initializer_240 = None
    initializers_onnx_initializer_241 = self.initializers.onnx_initializer_241
    encoder8_mha_k_reshape = getattr(self, "encoder8/mha/K/reshape")(
        encoder8_mha_k_b, initializers_onnx_initializer_241
    )
    encoder8_mha_k_b = initializers_onnx_initializer_241 = None
    encoder8_mha_k_transpose = getattr(self, "encoder8/mha/K/transpose")(
        encoder8_mha_k_reshape
    )
    encoder8_mha_k_reshape = None
    initializers_onnx_initializer_242 = self.initializers.onnx_initializer_242
    encoder8_mha_v_w = getattr(self, "encoder8/mha/V/w")(
        encoder7_ln2, initializers_onnx_initializer_242
    )
    initializers_onnx_initializer_242 = None
    initializers_onnx_initializer_243 = self.initializers.onnx_initializer_243
    encoder8_mha_v_b = getattr(self, "encoder8/mha/V/b")(
        encoder8_mha_v_w, initializers_onnx_initializer_243
    )
    encoder8_mha_v_w = initializers_onnx_initializer_243 = None
    initializers_onnx_initializer_244 = self.initializers.onnx_initializer_244
    encoder8_mha_v_reshape = getattr(self, "encoder8/mha/V/reshape")(
        encoder8_mha_v_b, initializers_onnx_initializer_244
    )
    encoder8_mha_v_b = initializers_onnx_initializer_244 = None
    encoder8_mha_v_transpose = getattr(self, "encoder8/mha/V/transpose")(
        encoder8_mha_v_reshape
    )
    encoder8_mha_v_reshape = None
    encoder8_mha_qk_matmul = getattr(self, "encoder8/mha/QK/matmul")(
        encoder8_mha_q_transpose, encoder8_mha_k_transpose
    )
    encoder8_mha_q_transpose = encoder8_mha_k_transpose = None
    initializers_onnx_initializer_245 = self.initializers.onnx_initializer_245
    encoder8_mha_qk_scale = getattr(self, "encoder8/mha/QK/scale")(
        encoder8_mha_qk_matmul, initializers_onnx_initializer_245
    )
    encoder8_mha_qk_matmul = initializers_onnx_initializer_245 = None
    initializers_onnx_initializer_246 = self.initializers.onnx_initializer_246
    encoder8_smolgen_compress = getattr(self, "encoder8/smolgen/compress")(
        encoder7_ln2, initializers_onnx_initializer_246
    )
    initializers_onnx_initializer_246 = None
    initializers_onnx_initializer_247 = self.initializers.onnx_initializer_247
    encoder8_smolgen_compress_reshape = getattr(
        self, "encoder8/smolgen/compress/reshape"
    )(encoder8_smolgen_compress, initializers_onnx_initializer_247)
    encoder8_smolgen_compress = initializers_onnx_initializer_247 = None
    initializers_onnx_initializer_248 = self.initializers.onnx_initializer_248
    encoder8_smolgen_dense1_w = getattr(self, "encoder8/smolgen/dense1/w")(
        encoder8_smolgen_compress_reshape, initializers_onnx_initializer_248
    )
    encoder8_smolgen_compress_reshape = initializers_onnx_initializer_248 = None
    initializers_onnx_initializer_249 = self.initializers.onnx_initializer_249
    encoder8_smolgen_dense1_b = getattr(self, "encoder8/smolgen/dense1/b")(
        encoder8_smolgen_dense1_w, initializers_onnx_initializer_249
    )
    encoder8_smolgen_dense1_w = initializers_onnx_initializer_249 = None
    encoder8_smolgen_dense1_swish_sigmoid = getattr(
        self, "encoder8/smolgen/dense1/swish/sigmoid"
    )(encoder8_smolgen_dense1_b)
    encoder8_smolgen_dense1_swish = getattr(self, "encoder8/smolgen/dense1/swish")(
        encoder8_smolgen_dense1_swish_sigmoid, encoder8_smolgen_dense1_b
    )
    encoder8_smolgen_dense1_swish_sigmoid = encoder8_smolgen_dense1_b = None
    encoder8_smolgen_ln1 = getattr(self, "encoder8/smolgen/ln1")(
        encoder8_smolgen_dense1_swish
    )
    encoder8_smolgen_dense1_swish = None
    initializers_onnx_initializer_250 = self.initializers.onnx_initializer_250
    encoder8_smolgen_dense2_w = getattr(self, "encoder8/smolgen/dense2/w")(
        encoder8_smolgen_ln1, initializers_onnx_initializer_250
    )
    encoder8_smolgen_ln1 = initializers_onnx_initializer_250 = None
    initializers_onnx_initializer_251 = self.initializers.onnx_initializer_251
    encoder8_smolgen_dense2_b = getattr(self, "encoder8/smolgen/dense2/b")(
        encoder8_smolgen_dense2_w, initializers_onnx_initializer_251
    )
    encoder8_smolgen_dense2_w = initializers_onnx_initializer_251 = None
    encoder8_smolgen_dense2_swish_sigmoid = getattr(
        self, "encoder8/smolgen/dense2/swish/sigmoid"
    )(encoder8_smolgen_dense2_b)
    encoder8_smolgen_dense2_swish = getattr(self, "encoder8/smolgen/dense2/swish")(
        encoder8_smolgen_dense2_swish_sigmoid, encoder8_smolgen_dense2_b
    )
    encoder8_smolgen_dense2_swish_sigmoid = encoder8_smolgen_dense2_b = None
    encoder8_smolgen_ln2 = getattr(self, "encoder8/smolgen/ln2")(
        encoder8_smolgen_dense2_swish
    )
    encoder8_smolgen_dense2_swish = None
    initializers_onnx_initializer_252 = self.initializers.onnx_initializer_252
    encoder8_smolgen_gen_from_reshape = getattr(
        self, "encoder8/smolgen/gen_from/reshape"
    )(encoder8_smolgen_ln2, initializers_onnx_initializer_252)
    encoder8_smolgen_ln2 = initializers_onnx_initializer_252 = None
    initializers_onnx_initializer_253 = self.initializers.onnx_initializer_253
    encoder8_smolgen_smol_weight_gen = getattr(
        self, "encoder8/smolgen/smol_weight_gen"
    )(encoder8_smolgen_gen_from_reshape, initializers_onnx_initializer_253)
    encoder8_smolgen_gen_from_reshape = initializers_onnx_initializer_253 = None
    initializers_onnx_initializer_254 = self.initializers.onnx_initializer_254
    encoder8_smolgen_out_reshape = getattr(self, "encoder8/smolgen/out/reshape")(
        encoder8_smolgen_smol_weight_gen, initializers_onnx_initializer_254
    )
    encoder8_smolgen_smol_weight_gen = initializers_onnx_initializer_254 = None
    encoder8_smolgen_weights = getattr(self, "encoder8/smolgen_weights")(
        encoder8_mha_qk_scale, encoder8_smolgen_out_reshape
    )
    encoder8_mha_qk_scale = encoder8_smolgen_out_reshape = None
    encoder8_mha_qk_softmax = getattr(self, "encoder8/mha/QK/softmax")(
        encoder8_smolgen_weights
    )
    encoder8_smolgen_weights = None
    encoder8_mha_qkv_matmul = getattr(self, "encoder8/mha/QKV/matmul")(
        encoder8_mha_qk_softmax, encoder8_mha_v_transpose
    )
    encoder8_mha_qk_softmax = encoder8_mha_v_transpose = None
    encoder8_mha_out_transpose = getattr(self, "encoder8/mha/out/transpose")(
        encoder8_mha_qkv_matmul
    )
    encoder8_mha_qkv_matmul = None
    initializers_onnx_initializer_255 = self.initializers.onnx_initializer_255
    encoder8_mha_out_reshape = getattr(self, "encoder8/mha/out/reshape")(
        encoder8_mha_out_transpose, initializers_onnx_initializer_255
    )
    encoder8_mha_out_transpose = initializers_onnx_initializer_255 = None
    initializers_onnx_initializer_256 = self.initializers.onnx_initializer_256
    encoder8_mha_out_dense_w = getattr(self, "encoder8/mha/out/dense/w")(
        encoder8_mha_out_reshape, initializers_onnx_initializer_256
    )
    encoder8_mha_out_reshape = initializers_onnx_initializer_256 = None
    initializers_onnx_initializer_257 = self.initializers.onnx_initializer_257
    encoder8_mha_out_dense_b = getattr(self, "encoder8/mha/out/dense/b")(
        encoder8_mha_out_dense_w, initializers_onnx_initializer_257
    )

    encoder8_mha_out_dense_b = self.attention_output[8](
        encoder8_mha_out_dense_b.view(-1, 64, 768)
    ).view(-1, 768)

    encoder8_mha_out_dense_w = initializers_onnx_initializer_257 = None
    initializers_onnx_initializer_258 = self.initializers.onnx_initializer_258
    encoder8_alpha_input = getattr(self, "encoder8/alpha*input")(
        encoder7_ln2, initializers_onnx_initializer_258
    )
    encoder7_ln2 = initializers_onnx_initializer_258 = None
    encoder8_mha_out_skip = getattr(self, "encoder8/mha/out/skip")(
        encoder8_mha_out_dense_b, encoder8_alpha_input
    )
    encoder8_mha_out_dense_b = encoder8_alpha_input = None
    encoder8_ln1 = getattr(self, "encoder8/ln1")(encoder8_mha_out_skip)
    encoder8_mha_out_skip = None

    ########################################################################
    # Make residuals accessible with squares as a separate dim,
    # this is important for nnsight.
    ########################################################################
    # This is just an nn.Identity layer
    residual_reasonable_shape = self.post_attention[8](
        encoder8_ln1.reshape(-1, 64, 768)
    )
    encoder8_ln1 = residual_reasonable_shape.reshape(-1, 768)
    residual_reasonable_shape = None

    initializers_onnx_initializer_259 = self.initializers.onnx_initializer_259
    encoder8_ffn_dense1_w = getattr(self, "encoder8/ffn/dense1/w")(
        encoder8_ln1, initializers_onnx_initializer_259
    )
    initializers_onnx_initializer_259 = None
    initializers_onnx_initializer_260 = self.initializers.onnx_initializer_260
    encoder8_ffn_dense1_b = getattr(self, "encoder8/ffn/dense1/b")(
        encoder8_ffn_dense1_w, initializers_onnx_initializer_260
    )
    encoder8_ffn_dense1_w = initializers_onnx_initializer_260 = None
    encoder8_ffn_dense1_sqrrelu_relu = getattr(
        self, "encoder8/ffn/dense1/sqrrelu/relu"
    )(encoder8_ffn_dense1_b)
    encoder8_ffn_dense1_b = None
    encoder8_ffn_dense1_sqrrelu_sqr = getattr(self, "encoder8/ffn/dense1/sqrrelu/sqr")(
        encoder8_ffn_dense1_sqrrelu_relu, encoder8_ffn_dense1_sqrrelu_relu
    )
    encoder8_ffn_dense1_sqrrelu_relu = None
    initializers_onnx_initializer_261 = self.initializers.onnx_initializer_261
    encoder8_ffn_dense2_w = getattr(self, "encoder8/ffn/dense2/w")(
        encoder8_ffn_dense1_sqrrelu_sqr, initializers_onnx_initializer_261
    )
    encoder8_ffn_dense1_sqrrelu_sqr = initializers_onnx_initializer_261 = None
    initializers_onnx_initializer_262 = self.initializers.onnx_initializer_262
    encoder8_ffn_dense2_b = getattr(self, "encoder8/ffn/dense2/b")(
        encoder8_ffn_dense2_w, initializers_onnx_initializer_262
    )

    encoder8_ffn_dense2_b = self.mlp_output[8](
        encoder8_ffn_dense2_b.view(-1, 64, 768)
    ).view(-1, 768)

    encoder8_ffn_dense2_w = initializers_onnx_initializer_262 = None
    initializers_onnx_initializer_263 = self.initializers.onnx_initializer_263
    encoder8_alpha_out1 = getattr(self, "encoder8/alpha*out1")(
        encoder8_ln1, initializers_onnx_initializer_263
    )
    encoder8_ln1 = initializers_onnx_initializer_263 = None
    encoder8_ffn_skip = getattr(self, "encoder8/ffn/skip")(
        encoder8_ffn_dense2_b, encoder8_alpha_out1
    )
    encoder8_ffn_dense2_b = encoder8_alpha_out1 = None
    encoder8_ln2 = getattr(self, "encoder8/ln2")(encoder8_ffn_skip)
    encoder8_ffn_skip = None

    ########################################################################
    # Make residuals accessible with squares as a separate dim,
    # this is important for nnsight.
    ########################################################################
    # This is just an nn.Identity layer
    residual_reasonable_shape = self.post_mlp[8](encoder8_ln2.reshape(-1, 64, 768))
    encoder8_ln2 = residual_reasonable_shape.reshape(-1, 768)
    residual_reasonable_shape = None

    ########################################################################
    # Layer 9
    ########################################################################
    initializers_onnx_initializer_264 = self.initializers.onnx_initializer_264
    encoder9_mha_q_w = getattr(self, "encoder9/mha/Q/w")(
        encoder8_ln2, initializers_onnx_initializer_264
    )
    initializers_onnx_initializer_264 = None
    initializers_onnx_initializer_265 = self.initializers.onnx_initializer_265
    encoder9_mha_q_b = getattr(self, "encoder9/mha/Q/b")(
        encoder9_mha_q_w, initializers_onnx_initializer_265
    )
    encoder9_mha_q_w = initializers_onnx_initializer_265 = None
    initializers_onnx_initializer_266 = self.initializers.onnx_initializer_266
    encoder9_mha_q_reshape = getattr(self, "encoder9/mha/Q/reshape")(
        encoder9_mha_q_b, initializers_onnx_initializer_266
    )
    encoder9_mha_q_b = initializers_onnx_initializer_266 = None
    encoder9_mha_q_transpose = getattr(self, "encoder9/mha/Q/transpose")(
        encoder9_mha_q_reshape
    )
    encoder9_mha_q_reshape = None
    initializers_onnx_initializer_267 = self.initializers.onnx_initializer_267
    encoder9_mha_k_w = getattr(self, "encoder9/mha/K/w")(
        encoder8_ln2, initializers_onnx_initializer_267
    )
    initializers_onnx_initializer_267 = None
    initializers_onnx_initializer_268 = self.initializers.onnx_initializer_268
    encoder9_mha_k_b = getattr(self, "encoder9/mha/K/b")(
        encoder9_mha_k_w, initializers_onnx_initializer_268
    )
    encoder9_mha_k_w = initializers_onnx_initializer_268 = None
    initializers_onnx_initializer_269 = self.initializers.onnx_initializer_269
    encoder9_mha_k_reshape = getattr(self, "encoder9/mha/K/reshape")(
        encoder9_mha_k_b, initializers_onnx_initializer_269
    )
    encoder9_mha_k_b = initializers_onnx_initializer_269 = None
    encoder9_mha_k_transpose = getattr(self, "encoder9/mha/K/transpose")(
        encoder9_mha_k_reshape
    )
    encoder9_mha_k_reshape = None
    initializers_onnx_initializer_270 = self.initializers.onnx_initializer_270
    encoder9_mha_v_w = getattr(self, "encoder9/mha/V/w")(
        encoder8_ln2, initializers_onnx_initializer_270
    )
    initializers_onnx_initializer_270 = None
    initializers_onnx_initializer_271 = self.initializers.onnx_initializer_271
    encoder9_mha_v_b = getattr(self, "encoder9/mha/V/b")(
        encoder9_mha_v_w, initializers_onnx_initializer_271
    )
    encoder9_mha_v_w = initializers_onnx_initializer_271 = None
    initializers_onnx_initializer_272 = self.initializers.onnx_initializer_272
    encoder9_mha_v_reshape = getattr(self, "encoder9/mha/V/reshape")(
        encoder9_mha_v_b, initializers_onnx_initializer_272
    )
    encoder9_mha_v_b = initializers_onnx_initializer_272 = None
    encoder9_mha_v_transpose = getattr(self, "encoder9/mha/V/transpose")(
        encoder9_mha_v_reshape
    )
    encoder9_mha_v_reshape = None
    encoder9_mha_qk_matmul = getattr(self, "encoder9/mha/QK/matmul")(
        encoder9_mha_q_transpose, encoder9_mha_k_transpose
    )
    encoder9_mha_q_transpose = encoder9_mha_k_transpose = None
    initializers_onnx_initializer_273 = self.initializers.onnx_initializer_273
    encoder9_mha_qk_scale = getattr(self, "encoder9/mha/QK/scale")(
        encoder9_mha_qk_matmul, initializers_onnx_initializer_273
    )
    encoder9_mha_qk_matmul = initializers_onnx_initializer_273 = None
    initializers_onnx_initializer_274 = self.initializers.onnx_initializer_274
    encoder9_smolgen_compress = getattr(self, "encoder9/smolgen/compress")(
        encoder8_ln2, initializers_onnx_initializer_274
    )
    initializers_onnx_initializer_274 = None
    initializers_onnx_initializer_275 = self.initializers.onnx_initializer_275
    encoder9_smolgen_compress_reshape = getattr(
        self, "encoder9/smolgen/compress/reshape"
    )(encoder9_smolgen_compress, initializers_onnx_initializer_275)
    encoder9_smolgen_compress = initializers_onnx_initializer_275 = None
    initializers_onnx_initializer_276 = self.initializers.onnx_initializer_276
    encoder9_smolgen_dense1_w = getattr(self, "encoder9/smolgen/dense1/w")(
        encoder9_smolgen_compress_reshape, initializers_onnx_initializer_276
    )
    encoder9_smolgen_compress_reshape = initializers_onnx_initializer_276 = None
    initializers_onnx_initializer_277 = self.initializers.onnx_initializer_277
    encoder9_smolgen_dense1_b = getattr(self, "encoder9/smolgen/dense1/b")(
        encoder9_smolgen_dense1_w, initializers_onnx_initializer_277
    )
    encoder9_smolgen_dense1_w = initializers_onnx_initializer_277 = None
    encoder9_smolgen_dense1_swish_sigmoid = getattr(
        self, "encoder9/smolgen/dense1/swish/sigmoid"
    )(encoder9_smolgen_dense1_b)
    encoder9_smolgen_dense1_swish = getattr(self, "encoder9/smolgen/dense1/swish")(
        encoder9_smolgen_dense1_swish_sigmoid, encoder9_smolgen_dense1_b
    )
    encoder9_smolgen_dense1_swish_sigmoid = encoder9_smolgen_dense1_b = None
    encoder9_smolgen_ln1 = getattr(self, "encoder9/smolgen/ln1")(
        encoder9_smolgen_dense1_swish
    )
    encoder9_smolgen_dense1_swish = None
    initializers_onnx_initializer_278 = self.initializers.onnx_initializer_278
    encoder9_smolgen_dense2_w = getattr(self, "encoder9/smolgen/dense2/w")(
        encoder9_smolgen_ln1, initializers_onnx_initializer_278
    )
    encoder9_smolgen_ln1 = initializers_onnx_initializer_278 = None
    initializers_onnx_initializer_279 = self.initializers.onnx_initializer_279
    encoder9_smolgen_dense2_b = getattr(self, "encoder9/smolgen/dense2/b")(
        encoder9_smolgen_dense2_w, initializers_onnx_initializer_279
    )
    encoder9_smolgen_dense2_w = initializers_onnx_initializer_279 = None
    encoder9_smolgen_dense2_swish_sigmoid = getattr(
        self, "encoder9/smolgen/dense2/swish/sigmoid"
    )(encoder9_smolgen_dense2_b)
    encoder9_smolgen_dense2_swish = getattr(self, "encoder9/smolgen/dense2/swish")(
        encoder9_smolgen_dense2_swish_sigmoid, encoder9_smolgen_dense2_b
    )
    encoder9_smolgen_dense2_swish_sigmoid = encoder9_smolgen_dense2_b = None
    encoder9_smolgen_ln2 = getattr(self, "encoder9/smolgen/ln2")(
        encoder9_smolgen_dense2_swish
    )
    encoder9_smolgen_dense2_swish = None
    initializers_onnx_initializer_280 = self.initializers.onnx_initializer_280
    encoder9_smolgen_gen_from_reshape = getattr(
        self, "encoder9/smolgen/gen_from/reshape"
    )(encoder9_smolgen_ln2, initializers_onnx_initializer_280)
    encoder9_smolgen_ln2 = initializers_onnx_initializer_280 = None
    initializers_onnx_initializer_281 = self.initializers.onnx_initializer_281
    encoder9_smolgen_smol_weight_gen = getattr(
        self, "encoder9/smolgen/smol_weight_gen"
    )(encoder9_smolgen_gen_from_reshape, initializers_onnx_initializer_281)
    encoder9_smolgen_gen_from_reshape = initializers_onnx_initializer_281 = None
    initializers_onnx_initializer_282 = self.initializers.onnx_initializer_282
    encoder9_smolgen_out_reshape = getattr(self, "encoder9/smolgen/out/reshape")(
        encoder9_smolgen_smol_weight_gen, initializers_onnx_initializer_282
    )
    encoder9_smolgen_smol_weight_gen = initializers_onnx_initializer_282 = None
    encoder9_smolgen_weights = getattr(self, "encoder9/smolgen_weights")(
        encoder9_mha_qk_scale, encoder9_smolgen_out_reshape
    )
    encoder9_mha_qk_scale = encoder9_smolgen_out_reshape = None
    encoder9_mha_qk_softmax = getattr(self, "encoder9/mha/QK/softmax")(
        encoder9_smolgen_weights
    )
    encoder9_smolgen_weights = None
    encoder9_mha_qkv_matmul = getattr(self, "encoder9/mha/QKV/matmul")(
        encoder9_mha_qk_softmax, encoder9_mha_v_transpose
    )
    encoder9_mha_qk_softmax = encoder9_mha_v_transpose = None
    encoder9_mha_out_transpose = getattr(self, "encoder9/mha/out/transpose")(
        encoder9_mha_qkv_matmul
    )
    encoder9_mha_qkv_matmul = None
    initializers_onnx_initializer_283 = self.initializers.onnx_initializer_283
    encoder9_mha_out_reshape = getattr(self, "encoder9/mha/out/reshape")(
        encoder9_mha_out_transpose, initializers_onnx_initializer_283
    )
    encoder9_mha_out_transpose = initializers_onnx_initializer_283 = None
    initializers_onnx_initializer_284 = self.initializers.onnx_initializer_284
    encoder9_mha_out_dense_w = getattr(self, "encoder9/mha/out/dense/w")(
        encoder9_mha_out_reshape, initializers_onnx_initializer_284
    )
    encoder9_mha_out_reshape = initializers_onnx_initializer_284 = None
    initializers_onnx_initializer_285 = self.initializers.onnx_initializer_285
    encoder9_mha_out_dense_b = getattr(self, "encoder9/mha/out/dense/b")(
        encoder9_mha_out_dense_w, initializers_onnx_initializer_285
    )

    encoder9_mha_out_dense_b = self.attention_output[9](
        encoder9_mha_out_dense_b.view(-1, 64, 768)
    ).view(-1, 768)

    encoder9_mha_out_dense_w = initializers_onnx_initializer_285 = None
    initializers_onnx_initializer_286 = self.initializers.onnx_initializer_286
    encoder9_alpha_input = getattr(self, "encoder9/alpha*input")(
        encoder8_ln2, initializers_onnx_initializer_286
    )
    encoder8_ln2 = initializers_onnx_initializer_286 = None
    encoder9_mha_out_skip = getattr(self, "encoder9/mha/out/skip")(
        encoder9_mha_out_dense_b, encoder9_alpha_input
    )
    encoder9_mha_out_dense_b = encoder9_alpha_input = None
    encoder9_ln1 = getattr(self, "encoder9/ln1")(encoder9_mha_out_skip)
    encoder9_mha_out_skip = None

    ########################################################################
    # Make residuals accessible with squares as a separate dim,
    # this is important for nnsight.
    ########################################################################
    # This is just an nn.Identity layer
    residual_reasonable_shape = self.post_attention[9](
        encoder9_ln1.reshape(-1, 64, 768)
    )
    encoder9_ln1 = residual_reasonable_shape.reshape(-1, 768)
    residual_reasonable_shape = None

    initializers_onnx_initializer_287 = self.initializers.onnx_initializer_287
    encoder9_ffn_dense1_w = getattr(self, "encoder9/ffn/dense1/w")(
        encoder9_ln1, initializers_onnx_initializer_287
    )
    initializers_onnx_initializer_287 = None
    initializers_onnx_initializer_288 = self.initializers.onnx_initializer_288
    encoder9_ffn_dense1_b = getattr(self, "encoder9/ffn/dense1/b")(
        encoder9_ffn_dense1_w, initializers_onnx_initializer_288
    )
    encoder9_ffn_dense1_w = initializers_onnx_initializer_288 = None
    encoder9_ffn_dense1_sqrrelu_relu = getattr(
        self, "encoder9/ffn/dense1/sqrrelu/relu"
    )(encoder9_ffn_dense1_b)
    encoder9_ffn_dense1_b = None
    encoder9_ffn_dense1_sqrrelu_sqr = getattr(self, "encoder9/ffn/dense1/sqrrelu/sqr")(
        encoder9_ffn_dense1_sqrrelu_relu, encoder9_ffn_dense1_sqrrelu_relu
    )
    encoder9_ffn_dense1_sqrrelu_relu = None
    initializers_onnx_initializer_289 = self.initializers.onnx_initializer_289
    encoder9_ffn_dense2_w = getattr(self, "encoder9/ffn/dense2/w")(
        encoder9_ffn_dense1_sqrrelu_sqr, initializers_onnx_initializer_289
    )
    encoder9_ffn_dense1_sqrrelu_sqr = initializers_onnx_initializer_289 = None
    initializers_onnx_initializer_290 = self.initializers.onnx_initializer_290
    encoder9_ffn_dense2_b = getattr(self, "encoder9/ffn/dense2/b")(
        encoder9_ffn_dense2_w, initializers_onnx_initializer_290
    )

    encoder9_ffn_dense2_b = self.mlp_output[9](
        encoder9_ffn_dense2_b.view(-1, 64, 768)
    ).view(-1, 768)

    encoder9_ffn_dense2_w = initializers_onnx_initializer_290 = None
    initializers_onnx_initializer_291 = self.initializers.onnx_initializer_291
    encoder9_alpha_out1 = getattr(self, "encoder9/alpha*out1")(
        encoder9_ln1, initializers_onnx_initializer_291
    )
    encoder9_ln1 = initializers_onnx_initializer_291 = None
    encoder9_ffn_skip = getattr(self, "encoder9/ffn/skip")(
        encoder9_ffn_dense2_b, encoder9_alpha_out1
    )
    encoder9_ffn_dense2_b = encoder9_alpha_out1 = None
    encoder9_ln2 = getattr(self, "encoder9/ln2")(encoder9_ffn_skip)
    encoder9_ffn_skip = None

    ########################################################################
    # Make residuals accessible with squares as a separate dim,
    # this is important for nnsight.
    ########################################################################
    # This is just an nn.Identity layer
    residual_reasonable_shape = self.post_mlp[9](encoder9_ln2.reshape(-1, 64, 768))
    encoder9_ln2 = residual_reasonable_shape.reshape(-1, 768)
    residual_reasonable_shape = None

    ########################################################################
    # Layer 10
    ########################################################################
    initializers_onnx_initializer_292 = self.initializers.onnx_initializer_292
    encoder10_mha_q_w = getattr(self, "encoder10/mha/Q/w")(
        encoder9_ln2, initializers_onnx_initializer_292
    )
    initializers_onnx_initializer_292 = None
    initializers_onnx_initializer_293 = self.initializers.onnx_initializer_293
    encoder10_mha_q_b = getattr(self, "encoder10/mha/Q/b")(
        encoder10_mha_q_w, initializers_onnx_initializer_293
    )
    encoder10_mha_q_w = initializers_onnx_initializer_293 = None
    initializers_onnx_initializer_294 = self.initializers.onnx_initializer_294
    encoder10_mha_q_reshape = getattr(self, "encoder10/mha/Q/reshape")(
        encoder10_mha_q_b, initializers_onnx_initializer_294
    )
    encoder10_mha_q_b = initializers_onnx_initializer_294 = None
    encoder10_mha_q_transpose = getattr(self, "encoder10/mha/Q/transpose")(
        encoder10_mha_q_reshape
    )
    encoder10_mha_q_reshape = None
    initializers_onnx_initializer_295 = self.initializers.onnx_initializer_295
    encoder10_mha_k_w = getattr(self, "encoder10/mha/K/w")(
        encoder9_ln2, initializers_onnx_initializer_295
    )
    initializers_onnx_initializer_295 = None
    initializers_onnx_initializer_296 = self.initializers.onnx_initializer_296
    encoder10_mha_k_b = getattr(self, "encoder10/mha/K/b")(
        encoder10_mha_k_w, initializers_onnx_initializer_296
    )
    encoder10_mha_k_w = initializers_onnx_initializer_296 = None
    initializers_onnx_initializer_297 = self.initializers.onnx_initializer_297
    encoder10_mha_k_reshape = getattr(self, "encoder10/mha/K/reshape")(
        encoder10_mha_k_b, initializers_onnx_initializer_297
    )
    encoder10_mha_k_b = initializers_onnx_initializer_297 = None
    encoder10_mha_k_transpose = getattr(self, "encoder10/mha/K/transpose")(
        encoder10_mha_k_reshape
    )
    encoder10_mha_k_reshape = None
    initializers_onnx_initializer_298 = self.initializers.onnx_initializer_298
    encoder10_mha_v_w = getattr(self, "encoder10/mha/V/w")(
        encoder9_ln2, initializers_onnx_initializer_298
    )
    initializers_onnx_initializer_298 = None
    initializers_onnx_initializer_299 = self.initializers.onnx_initializer_299
    encoder10_mha_v_b = getattr(self, "encoder10/mha/V/b")(
        encoder10_mha_v_w, initializers_onnx_initializer_299
    )
    encoder10_mha_v_w = initializers_onnx_initializer_299 = None
    initializers_onnx_initializer_300 = self.initializers.onnx_initializer_300
    encoder10_mha_v_reshape = getattr(self, "encoder10/mha/V/reshape")(
        encoder10_mha_v_b, initializers_onnx_initializer_300
    )
    encoder10_mha_v_b = initializers_onnx_initializer_300 = None
    encoder10_mha_v_transpose = getattr(self, "encoder10/mha/V/transpose")(
        encoder10_mha_v_reshape
    )
    encoder10_mha_v_reshape = None
    encoder10_mha_qk_matmul = getattr(self, "encoder10/mha/QK/matmul")(
        encoder10_mha_q_transpose, encoder10_mha_k_transpose
    )
    encoder10_mha_q_transpose = encoder10_mha_k_transpose = None
    initializers_onnx_initializer_301 = self.initializers.onnx_initializer_301
    encoder10_mha_qk_scale = getattr(self, "encoder10/mha/QK/scale")(
        encoder10_mha_qk_matmul, initializers_onnx_initializer_301
    )
    encoder10_mha_qk_matmul = initializers_onnx_initializer_301 = None
    initializers_onnx_initializer_302 = self.initializers.onnx_initializer_302
    encoder10_smolgen_compress = getattr(self, "encoder10/smolgen/compress")(
        encoder9_ln2, initializers_onnx_initializer_302
    )
    initializers_onnx_initializer_302 = None
    initializers_onnx_initializer_303 = self.initializers.onnx_initializer_303
    encoder10_smolgen_compress_reshape = getattr(
        self, "encoder10/smolgen/compress/reshape"
    )(encoder10_smolgen_compress, initializers_onnx_initializer_303)
    encoder10_smolgen_compress = initializers_onnx_initializer_303 = None
    initializers_onnx_initializer_304 = self.initializers.onnx_initializer_304
    encoder10_smolgen_dense1_w = getattr(self, "encoder10/smolgen/dense1/w")(
        encoder10_smolgen_compress_reshape, initializers_onnx_initializer_304
    )
    encoder10_smolgen_compress_reshape = initializers_onnx_initializer_304 = None
    initializers_onnx_initializer_305 = self.initializers.onnx_initializer_305
    encoder10_smolgen_dense1_b = getattr(self, "encoder10/smolgen/dense1/b")(
        encoder10_smolgen_dense1_w, initializers_onnx_initializer_305
    )
    encoder10_smolgen_dense1_w = initializers_onnx_initializer_305 = None
    encoder10_smolgen_dense1_swish_sigmoid = getattr(
        self, "encoder10/smolgen/dense1/swish/sigmoid"
    )(encoder10_smolgen_dense1_b)
    encoder10_smolgen_dense1_swish = getattr(self, "encoder10/smolgen/dense1/swish")(
        encoder10_smolgen_dense1_swish_sigmoid, encoder10_smolgen_dense1_b
    )
    encoder10_smolgen_dense1_swish_sigmoid = encoder10_smolgen_dense1_b = None
    encoder10_smolgen_ln1 = getattr(self, "encoder10/smolgen/ln1")(
        encoder10_smolgen_dense1_swish
    )
    encoder10_smolgen_dense1_swish = None
    initializers_onnx_initializer_306 = self.initializers.onnx_initializer_306
    encoder10_smolgen_dense2_w = getattr(self, "encoder10/smolgen/dense2/w")(
        encoder10_smolgen_ln1, initializers_onnx_initializer_306
    )
    encoder10_smolgen_ln1 = initializers_onnx_initializer_306 = None
    initializers_onnx_initializer_307 = self.initializers.onnx_initializer_307
    encoder10_smolgen_dense2_b = getattr(self, "encoder10/smolgen/dense2/b")(
        encoder10_smolgen_dense2_w, initializers_onnx_initializer_307
    )
    encoder10_smolgen_dense2_w = initializers_onnx_initializer_307 = None
    encoder10_smolgen_dense2_swish_sigmoid = getattr(
        self, "encoder10/smolgen/dense2/swish/sigmoid"
    )(encoder10_smolgen_dense2_b)
    encoder10_smolgen_dense2_swish = getattr(self, "encoder10/smolgen/dense2/swish")(
        encoder10_smolgen_dense2_swish_sigmoid, encoder10_smolgen_dense2_b
    )
    encoder10_smolgen_dense2_swish_sigmoid = encoder10_smolgen_dense2_b = None
    encoder10_smolgen_ln2 = getattr(self, "encoder10/smolgen/ln2")(
        encoder10_smolgen_dense2_swish
    )
    encoder10_smolgen_dense2_swish = None
    initializers_onnx_initializer_308 = self.initializers.onnx_initializer_308
    encoder10_smolgen_gen_from_reshape = getattr(
        self, "encoder10/smolgen/gen_from/reshape"
    )(encoder10_smolgen_ln2, initializers_onnx_initializer_308)
    encoder10_smolgen_ln2 = initializers_onnx_initializer_308 = None
    initializers_onnx_initializer_309 = self.initializers.onnx_initializer_309
    encoder10_smolgen_smol_weight_gen = getattr(
        self, "encoder10/smolgen/smol_weight_gen"
    )(encoder10_smolgen_gen_from_reshape, initializers_onnx_initializer_309)
    encoder10_smolgen_gen_from_reshape = initializers_onnx_initializer_309 = None
    initializers_onnx_initializer_310 = self.initializers.onnx_initializer_310
    encoder10_smolgen_out_reshape = getattr(self, "encoder10/smolgen/out/reshape")(
        encoder10_smolgen_smol_weight_gen, initializers_onnx_initializer_310
    )
    encoder10_smolgen_smol_weight_gen = initializers_onnx_initializer_310 = None
    encoder10_smolgen_weights = getattr(self, "encoder10/smolgen_weights")(
        encoder10_mha_qk_scale, encoder10_smolgen_out_reshape
    )
    encoder10_mha_qk_scale = encoder10_smolgen_out_reshape = None
    encoder10_mha_qk_softmax = getattr(self, "encoder10/mha/QK/softmax")(
        encoder10_smolgen_weights
    )
    encoder10_smolgen_weights = None
    encoder10_mha_qkv_matmul = getattr(self, "encoder10/mha/QKV/matmul")(
        encoder10_mha_qk_softmax, encoder10_mha_v_transpose
    )
    encoder10_mha_qk_softmax = encoder10_mha_v_transpose = None
    encoder10_mha_out_transpose = getattr(self, "encoder10/mha/out/transpose")(
        encoder10_mha_qkv_matmul
    )
    encoder10_mha_qkv_matmul = None
    initializers_onnx_initializer_311 = self.initializers.onnx_initializer_311
    encoder10_mha_out_reshape = getattr(self, "encoder10/mha/out/reshape")(
        encoder10_mha_out_transpose, initializers_onnx_initializer_311
    )
    encoder10_mha_out_transpose = initializers_onnx_initializer_311 = None
    initializers_onnx_initializer_312 = self.initializers.onnx_initializer_312
    encoder10_mha_out_dense_w = getattr(self, "encoder10/mha/out/dense/w")(
        encoder10_mha_out_reshape, initializers_onnx_initializer_312
    )
    encoder10_mha_out_reshape = initializers_onnx_initializer_312 = None
    initializers_onnx_initializer_313 = self.initializers.onnx_initializer_313
    encoder10_mha_out_dense_b = getattr(self, "encoder10/mha/out/dense/b")(
        encoder10_mha_out_dense_w, initializers_onnx_initializer_313
    )

    encoder10_mha_out_dense_b = self.attention_output[10](
        encoder10_mha_out_dense_b.view(-1, 64, 768)
    ).view(-1, 768)

    encoder10_mha_out_dense_w = initializers_onnx_initializer_313 = None
    initializers_onnx_initializer_314 = self.initializers.onnx_initializer_314
    encoder10_alpha_input = getattr(self, "encoder10/alpha*input")(
        encoder9_ln2, initializers_onnx_initializer_314
    )
    encoder9_ln2 = initializers_onnx_initializer_314 = None
    encoder10_mha_out_skip = getattr(self, "encoder10/mha/out/skip")(
        encoder10_mha_out_dense_b, encoder10_alpha_input
    )
    encoder10_mha_out_dense_b = encoder10_alpha_input = None
    encoder10_ln1 = getattr(self, "encoder10/ln1")(encoder10_mha_out_skip)
    encoder10_mha_out_skip = None

    ########################################################################
    # Make residuals accessible with squares as a separate dim,
    # this is important for nnsight.
    ########################################################################
    # This is just an nn.Identity layer
    residual_reasonable_shape = self.post_attention[10](
        encoder10_ln1.reshape(-1, 64, 768)
    )
    encoder10_ln1 = residual_reasonable_shape.reshape(-1, 768)
    residual_reasonable_shape = None

    initializers_onnx_initializer_315 = self.initializers.onnx_initializer_315
    encoder10_ffn_dense1_w = getattr(self, "encoder10/ffn/dense1/w")(
        encoder10_ln1, initializers_onnx_initializer_315
    )
    initializers_onnx_initializer_315 = None
    initializers_onnx_initializer_316 = self.initializers.onnx_initializer_316
    encoder10_ffn_dense1_b = getattr(self, "encoder10/ffn/dense1/b")(
        encoder10_ffn_dense1_w, initializers_onnx_initializer_316
    )
    encoder10_ffn_dense1_w = initializers_onnx_initializer_316 = None
    encoder10_ffn_dense1_sqrrelu_relu = getattr(
        self, "encoder10/ffn/dense1/sqrrelu/relu"
    )(encoder10_ffn_dense1_b)
    encoder10_ffn_dense1_b = None
    encoder10_ffn_dense1_sqrrelu_sqr = getattr(
        self, "encoder10/ffn/dense1/sqrrelu/sqr"
    )(encoder10_ffn_dense1_sqrrelu_relu, encoder10_ffn_dense1_sqrrelu_relu)
    encoder10_ffn_dense1_sqrrelu_relu = None
    initializers_onnx_initializer_317 = self.initializers.onnx_initializer_317
    encoder10_ffn_dense2_w = getattr(self, "encoder10/ffn/dense2/w")(
        encoder10_ffn_dense1_sqrrelu_sqr, initializers_onnx_initializer_317
    )
    encoder10_ffn_dense1_sqrrelu_sqr = initializers_onnx_initializer_317 = None
    initializers_onnx_initializer_318 = self.initializers.onnx_initializer_318
    encoder10_ffn_dense2_b = getattr(self, "encoder10/ffn/dense2/b")(
        encoder10_ffn_dense2_w, initializers_onnx_initializer_318
    )

    encoder10_ffn_dense2_b = self.mlp_output[10](
        encoder10_ffn_dense2_b.view(-1, 64, 768)
    ).view(-1, 768)

    encoder10_ffn_dense2_w = initializers_onnx_initializer_318 = None
    initializers_onnx_initializer_319 = self.initializers.onnx_initializer_319
    encoder10_alpha_out1 = getattr(self, "encoder10/alpha*out1")(
        encoder10_ln1, initializers_onnx_initializer_319
    )
    encoder10_ln1 = initializers_onnx_initializer_319 = None
    encoder10_ffn_skip = getattr(self, "encoder10/ffn/skip")(
        encoder10_ffn_dense2_b, encoder10_alpha_out1
    )
    encoder10_ffn_dense2_b = encoder10_alpha_out1 = None
    encoder10_ln2 = getattr(self, "encoder10/ln2")(encoder10_ffn_skip)
    encoder10_ffn_skip = None

    ########################################################################
    # Make residuals accessible with squares as a separate dim,
    # this is important for nnsight.
    ########################################################################
    # This is just an nn.Identity layer
    residual_reasonable_shape = self.post_mlp[10](encoder10_ln2.reshape(-1, 64, 768))
    encoder10_ln2 = residual_reasonable_shape.reshape(-1, 768)
    residual_reasonable_shape = None

    ########################################################################
    # Layer 11
    ########################################################################
    initializers_onnx_initializer_320 = self.initializers.onnx_initializer_320
    encoder11_mha_q_w = getattr(self, "encoder11/mha/Q/w")(
        encoder10_ln2, initializers_onnx_initializer_320
    )
    initializers_onnx_initializer_320 = None
    initializers_onnx_initializer_321 = self.initializers.onnx_initializer_321
    encoder11_mha_q_b = getattr(self, "encoder11/mha/Q/b")(
        encoder11_mha_q_w, initializers_onnx_initializer_321
    )
    encoder11_mha_q_w = initializers_onnx_initializer_321 = None
    initializers_onnx_initializer_322 = self.initializers.onnx_initializer_322
    encoder11_mha_q_reshape = getattr(self, "encoder11/mha/Q/reshape")(
        encoder11_mha_q_b, initializers_onnx_initializer_322
    )
    encoder11_mha_q_b = initializers_onnx_initializer_322 = None
    encoder11_mha_q_transpose = getattr(self, "encoder11/mha/Q/transpose")(
        encoder11_mha_q_reshape
    )
    encoder11_mha_q_reshape = None
    initializers_onnx_initializer_323 = self.initializers.onnx_initializer_323
    encoder11_mha_k_w = getattr(self, "encoder11/mha/K/w")(
        encoder10_ln2, initializers_onnx_initializer_323
    )
    initializers_onnx_initializer_323 = None
    initializers_onnx_initializer_324 = self.initializers.onnx_initializer_324
    encoder11_mha_k_b = getattr(self, "encoder11/mha/K/b")(
        encoder11_mha_k_w, initializers_onnx_initializer_324
    )
    encoder11_mha_k_w = initializers_onnx_initializer_324 = None
    initializers_onnx_initializer_325 = self.initializers.onnx_initializer_325
    encoder11_mha_k_reshape = getattr(self, "encoder11/mha/K/reshape")(
        encoder11_mha_k_b, initializers_onnx_initializer_325
    )
    encoder11_mha_k_b = initializers_onnx_initializer_325 = None
    encoder11_mha_k_transpose = getattr(self, "encoder11/mha/K/transpose")(
        encoder11_mha_k_reshape
    )
    encoder11_mha_k_reshape = None
    initializers_onnx_initializer_326 = self.initializers.onnx_initializer_326
    encoder11_mha_v_w = getattr(self, "encoder11/mha/V/w")(
        encoder10_ln2, initializers_onnx_initializer_326
    )
    initializers_onnx_initializer_326 = None
    initializers_onnx_initializer_327 = self.initializers.onnx_initializer_327
    encoder11_mha_v_b = getattr(self, "encoder11/mha/V/b")(
        encoder11_mha_v_w, initializers_onnx_initializer_327
    )
    encoder11_mha_v_w = initializers_onnx_initializer_327 = None
    initializers_onnx_initializer_328 = self.initializers.onnx_initializer_328
    encoder11_mha_v_reshape = getattr(self, "encoder11/mha/V/reshape")(
        encoder11_mha_v_b, initializers_onnx_initializer_328
    )
    encoder11_mha_v_b = initializers_onnx_initializer_328 = None
    encoder11_mha_v_transpose = getattr(self, "encoder11/mha/V/transpose")(
        encoder11_mha_v_reshape
    )
    encoder11_mha_v_reshape = None
    encoder11_mha_qk_matmul = getattr(self, "encoder11/mha/QK/matmul")(
        encoder11_mha_q_transpose, encoder11_mha_k_transpose
    )
    encoder11_mha_q_transpose = encoder11_mha_k_transpose = None
    initializers_onnx_initializer_329 = self.initializers.onnx_initializer_329
    encoder11_mha_qk_scale = getattr(self, "encoder11/mha/QK/scale")(
        encoder11_mha_qk_matmul, initializers_onnx_initializer_329
    )
    encoder11_mha_qk_matmul = initializers_onnx_initializer_329 = None
    initializers_onnx_initializer_330 = self.initializers.onnx_initializer_330
    encoder11_smolgen_compress = getattr(self, "encoder11/smolgen/compress")(
        encoder10_ln2, initializers_onnx_initializer_330
    )
    initializers_onnx_initializer_330 = None
    initializers_onnx_initializer_331 = self.initializers.onnx_initializer_331
    encoder11_smolgen_compress_reshape = getattr(
        self, "encoder11/smolgen/compress/reshape"
    )(encoder11_smolgen_compress, initializers_onnx_initializer_331)
    encoder11_smolgen_compress = initializers_onnx_initializer_331 = None
    initializers_onnx_initializer_332 = self.initializers.onnx_initializer_332
    encoder11_smolgen_dense1_w = getattr(self, "encoder11/smolgen/dense1/w")(
        encoder11_smolgen_compress_reshape, initializers_onnx_initializer_332
    )
    encoder11_smolgen_compress_reshape = initializers_onnx_initializer_332 = None
    initializers_onnx_initializer_333 = self.initializers.onnx_initializer_333
    encoder11_smolgen_dense1_b = getattr(self, "encoder11/smolgen/dense1/b")(
        encoder11_smolgen_dense1_w, initializers_onnx_initializer_333
    )
    encoder11_smolgen_dense1_w = initializers_onnx_initializer_333 = None
    encoder11_smolgen_dense1_swish_sigmoid = getattr(
        self, "encoder11/smolgen/dense1/swish/sigmoid"
    )(encoder11_smolgen_dense1_b)
    encoder11_smolgen_dense1_swish = getattr(self, "encoder11/smolgen/dense1/swish")(
        encoder11_smolgen_dense1_swish_sigmoid, encoder11_smolgen_dense1_b
    )
    encoder11_smolgen_dense1_swish_sigmoid = encoder11_smolgen_dense1_b = None
    encoder11_smolgen_ln1 = getattr(self, "encoder11/smolgen/ln1")(
        encoder11_smolgen_dense1_swish
    )
    encoder11_smolgen_dense1_swish = None
    initializers_onnx_initializer_334 = self.initializers.onnx_initializer_334
    encoder11_smolgen_dense2_w = getattr(self, "encoder11/smolgen/dense2/w")(
        encoder11_smolgen_ln1, initializers_onnx_initializer_334
    )
    encoder11_smolgen_ln1 = initializers_onnx_initializer_334 = None
    initializers_onnx_initializer_335 = self.initializers.onnx_initializer_335
    encoder11_smolgen_dense2_b = getattr(self, "encoder11/smolgen/dense2/b")(
        encoder11_smolgen_dense2_w, initializers_onnx_initializer_335
    )
    encoder11_smolgen_dense2_w = initializers_onnx_initializer_335 = None
    encoder11_smolgen_dense2_swish_sigmoid = getattr(
        self, "encoder11/smolgen/dense2/swish/sigmoid"
    )(encoder11_smolgen_dense2_b)
    encoder11_smolgen_dense2_swish = getattr(self, "encoder11/smolgen/dense2/swish")(
        encoder11_smolgen_dense2_swish_sigmoid, encoder11_smolgen_dense2_b
    )
    encoder11_smolgen_dense2_swish_sigmoid = encoder11_smolgen_dense2_b = None
    encoder11_smolgen_ln2 = getattr(self, "encoder11/smolgen/ln2")(
        encoder11_smolgen_dense2_swish
    )
    encoder11_smolgen_dense2_swish = None
    initializers_onnx_initializer_336 = self.initializers.onnx_initializer_336
    encoder11_smolgen_gen_from_reshape = getattr(
        self, "encoder11/smolgen/gen_from/reshape"
    )(encoder11_smolgen_ln2, initializers_onnx_initializer_336)
    encoder11_smolgen_ln2 = initializers_onnx_initializer_336 = None
    initializers_onnx_initializer_337 = self.initializers.onnx_initializer_337
    encoder11_smolgen_smol_weight_gen = getattr(
        self, "encoder11/smolgen/smol_weight_gen"
    )(encoder11_smolgen_gen_from_reshape, initializers_onnx_initializer_337)
    encoder11_smolgen_gen_from_reshape = initializers_onnx_initializer_337 = None
    initializers_onnx_initializer_338 = self.initializers.onnx_initializer_338
    encoder11_smolgen_out_reshape = getattr(self, "encoder11/smolgen/out/reshape")(
        encoder11_smolgen_smol_weight_gen, initializers_onnx_initializer_338
    )
    encoder11_smolgen_smol_weight_gen = initializers_onnx_initializer_338 = None
    encoder11_smolgen_weights = getattr(self, "encoder11/smolgen_weights")(
        encoder11_mha_qk_scale, encoder11_smolgen_out_reshape
    )
    encoder11_mha_qk_scale = encoder11_smolgen_out_reshape = None
    encoder11_mha_qk_softmax = getattr(self, "encoder11/mha/QK/softmax")(
        encoder11_smolgen_weights
    )
    encoder11_smolgen_weights = None
    encoder11_mha_qkv_matmul = getattr(self, "encoder11/mha/QKV/matmul")(
        encoder11_mha_qk_softmax, encoder11_mha_v_transpose
    )
    encoder11_mha_qk_softmax = encoder11_mha_v_transpose = None
    encoder11_mha_out_transpose = getattr(self, "encoder11/mha/out/transpose")(
        encoder11_mha_qkv_matmul
    )
    encoder11_mha_qkv_matmul = None
    initializers_onnx_initializer_339 = self.initializers.onnx_initializer_339
    encoder11_mha_out_reshape = getattr(self, "encoder11/mha/out/reshape")(
        encoder11_mha_out_transpose, initializers_onnx_initializer_339
    )
    encoder11_mha_out_transpose = initializers_onnx_initializer_339 = None
    initializers_onnx_initializer_340 = self.initializers.onnx_initializer_340
    encoder11_mha_out_dense_w = getattr(self, "encoder11/mha/out/dense/w")(
        encoder11_mha_out_reshape, initializers_onnx_initializer_340
    )
    encoder11_mha_out_reshape = initializers_onnx_initializer_340 = None
    initializers_onnx_initializer_341 = self.initializers.onnx_initializer_341
    encoder11_mha_out_dense_b = getattr(self, "encoder11/mha/out/dense/b")(
        encoder11_mha_out_dense_w, initializers_onnx_initializer_341
    )

    encoder11_mha_out_dense_b = self.attention_output[11](
        encoder11_mha_out_dense_b.view(-1, 64, 768)
    ).view(-1, 768)

    encoder11_mha_out_dense_w = initializers_onnx_initializer_341 = None
    initializers_onnx_initializer_342 = self.initializers.onnx_initializer_342
    encoder11_alpha_input = getattr(self, "encoder11/alpha*input")(
        encoder10_ln2, initializers_onnx_initializer_342
    )
    encoder10_ln2 = initializers_onnx_initializer_342 = None
    encoder11_mha_out_skip = getattr(self, "encoder11/mha/out/skip")(
        encoder11_mha_out_dense_b, encoder11_alpha_input
    )
    encoder11_mha_out_dense_b = encoder11_alpha_input = None
    encoder11_ln1 = getattr(self, "encoder11/ln1")(encoder11_mha_out_skip)
    encoder11_mha_out_skip = None

    ########################################################################
    # Make residuals accessible with squares as a separate dim,
    # this is important for nnsight.
    ########################################################################
    # This is just an nn.Identity layer
    residual_reasonable_shape = self.post_attention[11](
        encoder11_ln1.reshape(-1, 64, 768)
    )
    encoder11_ln1 = residual_reasonable_shape.reshape(-1, 768)
    residual_reasonable_shape = None

    initializers_onnx_initializer_343 = self.initializers.onnx_initializer_343
    encoder11_ffn_dense1_w = getattr(self, "encoder11/ffn/dense1/w")(
        encoder11_ln1, initializers_onnx_initializer_343
    )
    initializers_onnx_initializer_343 = None
    initializers_onnx_initializer_344 = self.initializers.onnx_initializer_344
    encoder11_ffn_dense1_b = getattr(self, "encoder11/ffn/dense1/b")(
        encoder11_ffn_dense1_w, initializers_onnx_initializer_344
    )
    encoder11_ffn_dense1_w = initializers_onnx_initializer_344 = None
    encoder11_ffn_dense1_sqrrelu_relu = getattr(
        self, "encoder11/ffn/dense1/sqrrelu/relu"
    )(encoder11_ffn_dense1_b)
    encoder11_ffn_dense1_b = None
    encoder11_ffn_dense1_sqrrelu_sqr = getattr(
        self, "encoder11/ffn/dense1/sqrrelu/sqr"
    )(encoder11_ffn_dense1_sqrrelu_relu, encoder11_ffn_dense1_sqrrelu_relu)
    encoder11_ffn_dense1_sqrrelu_relu = None
    initializers_onnx_initializer_345 = self.initializers.onnx_initializer_345
    encoder11_ffn_dense2_w = getattr(self, "encoder11/ffn/dense2/w")(
        encoder11_ffn_dense1_sqrrelu_sqr, initializers_onnx_initializer_345
    )
    encoder11_ffn_dense1_sqrrelu_sqr = initializers_onnx_initializer_345 = None
    initializers_onnx_initializer_346 = self.initializers.onnx_initializer_346
    encoder11_ffn_dense2_b = getattr(self, "encoder11/ffn/dense2/b")(
        encoder11_ffn_dense2_w, initializers_onnx_initializer_346
    )

    encoder11_ffn_dense2_b = self.mlp_output[11](
        encoder11_ffn_dense2_b.view(-1, 64, 768)
    ).view(-1, 768)

    encoder11_ffn_dense2_w = initializers_onnx_initializer_346 = None
    initializers_onnx_initializer_347 = self.initializers.onnx_initializer_347
    encoder11_alpha_out1 = getattr(self, "encoder11/alpha*out1")(
        encoder11_ln1, initializers_onnx_initializer_347
    )
    encoder11_ln1 = initializers_onnx_initializer_347 = None
    encoder11_ffn_skip = getattr(self, "encoder11/ffn/skip")(
        encoder11_ffn_dense2_b, encoder11_alpha_out1
    )
    encoder11_ffn_dense2_b = encoder11_alpha_out1 = None
    encoder11_ln2 = getattr(self, "encoder11/ln2")(encoder11_ffn_skip)
    encoder11_ffn_skip = None

    ########################################################################
    # Make residuals accessible with squares as a separate dim,
    # this is important for nnsight.
    ########################################################################
    # This is just an nn.Identity layer
    residual_reasonable_shape = self.post_mlp[11](encoder11_ln2.reshape(-1, 64, 768))
    encoder11_ln2 = residual_reasonable_shape.reshape(-1, 768)
    residual_reasonable_shape = None

    ########################################################################
    # Layer 12
    ########################################################################
    initializers_onnx_initializer_348 = self.initializers.onnx_initializer_348
    encoder12_mha_q_w = getattr(self, "encoder12/mha/Q/w")(
        encoder11_ln2, initializers_onnx_initializer_348
    )
    initializers_onnx_initializer_348 = None
    initializers_onnx_initializer_349 = self.initializers.onnx_initializer_349
    encoder12_mha_q_b = getattr(self, "encoder12/mha/Q/b")(
        encoder12_mha_q_w, initializers_onnx_initializer_349
    )
    encoder12_mha_q_w = initializers_onnx_initializer_349 = None
    initializers_onnx_initializer_350 = self.initializers.onnx_initializer_350
    encoder12_mha_q_reshape = getattr(self, "encoder12/mha/Q/reshape")(
        encoder12_mha_q_b, initializers_onnx_initializer_350
    )
    encoder12_mha_q_b = initializers_onnx_initializer_350 = None
    encoder12_mha_q_transpose = getattr(self, "encoder12/mha/Q/transpose")(
        encoder12_mha_q_reshape
    )
    encoder12_mha_q_reshape = None
    initializers_onnx_initializer_351 = self.initializers.onnx_initializer_351
    encoder12_mha_k_w = getattr(self, "encoder12/mha/K/w")(
        encoder11_ln2, initializers_onnx_initializer_351
    )
    initializers_onnx_initializer_351 = None
    initializers_onnx_initializer_352 = self.initializers.onnx_initializer_352
    encoder12_mha_k_b = getattr(self, "encoder12/mha/K/b")(
        encoder12_mha_k_w, initializers_onnx_initializer_352
    )
    encoder12_mha_k_w = initializers_onnx_initializer_352 = None
    initializers_onnx_initializer_353 = self.initializers.onnx_initializer_353
    encoder12_mha_k_reshape = getattr(self, "encoder12/mha/K/reshape")(
        encoder12_mha_k_b, initializers_onnx_initializer_353
    )
    encoder12_mha_k_b = initializers_onnx_initializer_353 = None
    encoder12_mha_k_transpose = getattr(self, "encoder12/mha/K/transpose")(
        encoder12_mha_k_reshape
    )
    encoder12_mha_k_reshape = None
    initializers_onnx_initializer_354 = self.initializers.onnx_initializer_354
    encoder12_mha_v_w = getattr(self, "encoder12/mha/V/w")(
        encoder11_ln2, initializers_onnx_initializer_354
    )
    initializers_onnx_initializer_354 = None
    initializers_onnx_initializer_355 = self.initializers.onnx_initializer_355
    encoder12_mha_v_b = getattr(self, "encoder12/mha/V/b")(
        encoder12_mha_v_w, initializers_onnx_initializer_355
    )
    encoder12_mha_v_w = initializers_onnx_initializer_355 = None
    initializers_onnx_initializer_356 = self.initializers.onnx_initializer_356
    encoder12_mha_v_reshape = getattr(self, "encoder12/mha/V/reshape")(
        encoder12_mha_v_b, initializers_onnx_initializer_356
    )
    encoder12_mha_v_b = initializers_onnx_initializer_356 = None
    encoder12_mha_v_transpose = getattr(self, "encoder12/mha/V/transpose")(
        encoder12_mha_v_reshape
    )
    encoder12_mha_v_reshape = None
    encoder12_mha_qk_matmul = getattr(self, "encoder12/mha/QK/matmul")(
        encoder12_mha_q_transpose, encoder12_mha_k_transpose
    )
    encoder12_mha_q_transpose = encoder12_mha_k_transpose = None
    initializers_onnx_initializer_357 = self.initializers.onnx_initializer_357
    encoder12_mha_qk_scale = getattr(self, "encoder12/mha/QK/scale")(
        encoder12_mha_qk_matmul, initializers_onnx_initializer_357
    )
    encoder12_mha_qk_matmul = initializers_onnx_initializer_357 = None
    initializers_onnx_initializer_358 = self.initializers.onnx_initializer_358
    encoder12_smolgen_compress = getattr(self, "encoder12/smolgen/compress")(
        encoder11_ln2, initializers_onnx_initializer_358
    )
    initializers_onnx_initializer_358 = None
    initializers_onnx_initializer_359 = self.initializers.onnx_initializer_359
    encoder12_smolgen_compress_reshape = getattr(
        self, "encoder12/smolgen/compress/reshape"
    )(encoder12_smolgen_compress, initializers_onnx_initializer_359)
    encoder12_smolgen_compress = initializers_onnx_initializer_359 = None
    initializers_onnx_initializer_360 = self.initializers.onnx_initializer_360
    encoder12_smolgen_dense1_w = getattr(self, "encoder12/smolgen/dense1/w")(
        encoder12_smolgen_compress_reshape, initializers_onnx_initializer_360
    )
    encoder12_smolgen_compress_reshape = initializers_onnx_initializer_360 = None
    initializers_onnx_initializer_361 = self.initializers.onnx_initializer_361
    encoder12_smolgen_dense1_b = getattr(self, "encoder12/smolgen/dense1/b")(
        encoder12_smolgen_dense1_w, initializers_onnx_initializer_361
    )
    encoder12_smolgen_dense1_w = initializers_onnx_initializer_361 = None
    encoder12_smolgen_dense1_swish_sigmoid = getattr(
        self, "encoder12/smolgen/dense1/swish/sigmoid"
    )(encoder12_smolgen_dense1_b)
    encoder12_smolgen_dense1_swish = getattr(self, "encoder12/smolgen/dense1/swish")(
        encoder12_smolgen_dense1_swish_sigmoid, encoder12_smolgen_dense1_b
    )
    encoder12_smolgen_dense1_swish_sigmoid = encoder12_smolgen_dense1_b = None
    encoder12_smolgen_ln1 = getattr(self, "encoder12/smolgen/ln1")(
        encoder12_smolgen_dense1_swish
    )
    encoder12_smolgen_dense1_swish = None
    initializers_onnx_initializer_362 = self.initializers.onnx_initializer_362
    encoder12_smolgen_dense2_w = getattr(self, "encoder12/smolgen/dense2/w")(
        encoder12_smolgen_ln1, initializers_onnx_initializer_362
    )
    encoder12_smolgen_ln1 = initializers_onnx_initializer_362 = None
    initializers_onnx_initializer_363 = self.initializers.onnx_initializer_363
    encoder12_smolgen_dense2_b = getattr(self, "encoder12/smolgen/dense2/b")(
        encoder12_smolgen_dense2_w, initializers_onnx_initializer_363
    )
    encoder12_smolgen_dense2_w = initializers_onnx_initializer_363 = None
    encoder12_smolgen_dense2_swish_sigmoid = getattr(
        self, "encoder12/smolgen/dense2/swish/sigmoid"
    )(encoder12_smolgen_dense2_b)
    encoder12_smolgen_dense2_swish = getattr(self, "encoder12/smolgen/dense2/swish")(
        encoder12_smolgen_dense2_swish_sigmoid, encoder12_smolgen_dense2_b
    )
    encoder12_smolgen_dense2_swish_sigmoid = encoder12_smolgen_dense2_b = None
    encoder12_smolgen_ln2 = getattr(self, "encoder12/smolgen/ln2")(
        encoder12_smolgen_dense2_swish
    )
    encoder12_smolgen_dense2_swish = None
    initializers_onnx_initializer_364 = self.initializers.onnx_initializer_364
    encoder12_smolgen_gen_from_reshape = getattr(
        self, "encoder12/smolgen/gen_from/reshape"
    )(encoder12_smolgen_ln2, initializers_onnx_initializer_364)
    encoder12_smolgen_ln2 = initializers_onnx_initializer_364 = None
    initializers_onnx_initializer_365 = self.initializers.onnx_initializer_365
    encoder12_smolgen_smol_weight_gen = getattr(
        self, "encoder12/smolgen/smol_weight_gen"
    )(encoder12_smolgen_gen_from_reshape, initializers_onnx_initializer_365)
    encoder12_smolgen_gen_from_reshape = initializers_onnx_initializer_365 = None
    initializers_onnx_initializer_366 = self.initializers.onnx_initializer_366
    encoder12_smolgen_out_reshape = getattr(self, "encoder12/smolgen/out/reshape")(
        encoder12_smolgen_smol_weight_gen, initializers_onnx_initializer_366
    )
    encoder12_smolgen_smol_weight_gen = initializers_onnx_initializer_366 = None
    encoder12_smolgen_weights = getattr(self, "encoder12/smolgen_weights")(
        encoder12_mha_qk_scale, encoder12_smolgen_out_reshape
    )
    encoder12_mha_qk_scale = encoder12_smolgen_out_reshape = None
    encoder12_mha_qk_softmax = getattr(self, "encoder12/mha/QK/softmax")(
        encoder12_smolgen_weights
    )
    encoder12_smolgen_weights = None
    encoder12_mha_qkv_matmul = getattr(self, "encoder12/mha/QKV/matmul")(
        encoder12_mha_qk_softmax, encoder12_mha_v_transpose
    )
    encoder12_mha_qk_softmax = encoder12_mha_v_transpose = None
    encoder12_mha_out_transpose = getattr(self, "encoder12/mha/out/transpose")(
        encoder12_mha_qkv_matmul
    )
    encoder12_mha_qkv_matmul = None
    initializers_onnx_initializer_367 = self.initializers.onnx_initializer_367
    encoder12_mha_out_reshape = getattr(self, "encoder12/mha/out/reshape")(
        encoder12_mha_out_transpose, initializers_onnx_initializer_367
    )
    encoder12_mha_out_transpose = initializers_onnx_initializer_367 = None
    initializers_onnx_initializer_368 = self.initializers.onnx_initializer_368
    encoder12_mha_out_dense_w = getattr(self, "encoder12/mha/out/dense/w")(
        encoder12_mha_out_reshape, initializers_onnx_initializer_368
    )
    encoder12_mha_out_reshape = initializers_onnx_initializer_368 = None
    initializers_onnx_initializer_369 = self.initializers.onnx_initializer_369
    encoder12_mha_out_dense_b = getattr(self, "encoder12/mha/out/dense/b")(
        encoder12_mha_out_dense_w, initializers_onnx_initializer_369
    )

    encoder12_mha_out_dense_b = self.attention_output[12](
        encoder12_mha_out_dense_b.view(-1, 64, 768)
    ).view(-1, 768)

    encoder12_mha_out_dense_w = initializers_onnx_initializer_369 = None
    initializers_onnx_initializer_370 = self.initializers.onnx_initializer_370
    encoder12_alpha_input = getattr(self, "encoder12/alpha*input")(
        encoder11_ln2, initializers_onnx_initializer_370
    )
    encoder11_ln2 = initializers_onnx_initializer_370 = None
    encoder12_mha_out_skip = getattr(self, "encoder12/mha/out/skip")(
        encoder12_mha_out_dense_b, encoder12_alpha_input
    )
    encoder12_mha_out_dense_b = encoder12_alpha_input = None
    encoder12_ln1 = getattr(self, "encoder12/ln1")(encoder12_mha_out_skip)
    encoder12_mha_out_skip = None

    ########################################################################
    # Make residuals accessible with squares as a separate dim,
    # this is important for nnsight.
    ########################################################################
    # This is just an nn.Identity layer
    residual_reasonable_shape = self.post_attention[12](
        encoder12_ln1.reshape(-1, 64, 768)
    )
    encoder12_ln1 = residual_reasonable_shape.reshape(-1, 768)
    residual_reasonable_shape = None

    initializers_onnx_initializer_371 = self.initializers.onnx_initializer_371
    encoder12_ffn_dense1_w = getattr(self, "encoder12/ffn/dense1/w")(
        encoder12_ln1, initializers_onnx_initializer_371
    )
    initializers_onnx_initializer_371 = None
    initializers_onnx_initializer_372 = self.initializers.onnx_initializer_372
    encoder12_ffn_dense1_b = getattr(self, "encoder12/ffn/dense1/b")(
        encoder12_ffn_dense1_w, initializers_onnx_initializer_372
    )
    encoder12_ffn_dense1_w = initializers_onnx_initializer_372 = None
    encoder12_ffn_dense1_sqrrelu_relu = getattr(
        self, "encoder12/ffn/dense1/sqrrelu/relu"
    )(encoder12_ffn_dense1_b)
    encoder12_ffn_dense1_b = None
    encoder12_ffn_dense1_sqrrelu_sqr = getattr(
        self, "encoder12/ffn/dense1/sqrrelu/sqr"
    )(encoder12_ffn_dense1_sqrrelu_relu, encoder12_ffn_dense1_sqrrelu_relu)
    encoder12_ffn_dense1_sqrrelu_relu = None
    initializers_onnx_initializer_373 = self.initializers.onnx_initializer_373
    encoder12_ffn_dense2_w = getattr(self, "encoder12/ffn/dense2/w")(
        encoder12_ffn_dense1_sqrrelu_sqr, initializers_onnx_initializer_373
    )
    encoder12_ffn_dense1_sqrrelu_sqr = initializers_onnx_initializer_373 = None
    initializers_onnx_initializer_374 = self.initializers.onnx_initializer_374
    encoder12_ffn_dense2_b = getattr(self, "encoder12/ffn/dense2/b")(
        encoder12_ffn_dense2_w, initializers_onnx_initializer_374
    )

    encoder12_ffn_dense2_b = self.mlp_output[12](
        encoder12_ffn_dense2_b.view(-1, 64, 768)
    ).view(-1, 768)

    encoder12_ffn_dense2_w = initializers_onnx_initializer_374 = None
    initializers_onnx_initializer_375 = self.initializers.onnx_initializer_375
    encoder12_alpha_out1 = getattr(self, "encoder12/alpha*out1")(
        encoder12_ln1, initializers_onnx_initializer_375
    )
    encoder12_ln1 = initializers_onnx_initializer_375 = None
    encoder12_ffn_skip = getattr(self, "encoder12/ffn/skip")(
        encoder12_ffn_dense2_b, encoder12_alpha_out1
    )
    encoder12_ffn_dense2_b = encoder12_alpha_out1 = None
    encoder12_ln2 = getattr(self, "encoder12/ln2")(encoder12_ffn_skip)
    encoder12_ffn_skip = None

    ########################################################################
    # Make residuals accessible with squares as a separate dim,
    # this is important for nnsight.
    ########################################################################
    # This is just an nn.Identity layer
    residual_reasonable_shape = self.post_mlp[12](encoder12_ln2.reshape(-1, 64, 768))
    encoder12_ln2 = residual_reasonable_shape.reshape(-1, 768)
    residual_reasonable_shape = None

    ########################################################################
    # Layer 13
    ########################################################################
    initializers_onnx_initializer_376 = self.initializers.onnx_initializer_376
    encoder13_mha_q_w = getattr(self, "encoder13/mha/Q/w")(
        encoder12_ln2, initializers_onnx_initializer_376
    )
    initializers_onnx_initializer_376 = None
    initializers_onnx_initializer_377 = self.initializers.onnx_initializer_377
    encoder13_mha_q_b = getattr(self, "encoder13/mha/Q/b")(
        encoder13_mha_q_w, initializers_onnx_initializer_377
    )
    encoder13_mha_q_w = initializers_onnx_initializer_377 = None
    initializers_onnx_initializer_378 = self.initializers.onnx_initializer_378
    encoder13_mha_q_reshape = getattr(self, "encoder13/mha/Q/reshape")(
        encoder13_mha_q_b, initializers_onnx_initializer_378
    )
    encoder13_mha_q_b = initializers_onnx_initializer_378 = None
    encoder13_mha_q_transpose = getattr(self, "encoder13/mha/Q/transpose")(
        encoder13_mha_q_reshape
    )
    encoder13_mha_q_reshape = None
    initializers_onnx_initializer_379 = self.initializers.onnx_initializer_379
    encoder13_mha_k_w = getattr(self, "encoder13/mha/K/w")(
        encoder12_ln2, initializers_onnx_initializer_379
    )
    initializers_onnx_initializer_379 = None
    initializers_onnx_initializer_380 = self.initializers.onnx_initializer_380
    encoder13_mha_k_b = getattr(self, "encoder13/mha/K/b")(
        encoder13_mha_k_w, initializers_onnx_initializer_380
    )
    encoder13_mha_k_w = initializers_onnx_initializer_380 = None
    initializers_onnx_initializer_381 = self.initializers.onnx_initializer_381
    encoder13_mha_k_reshape = getattr(self, "encoder13/mha/K/reshape")(
        encoder13_mha_k_b, initializers_onnx_initializer_381
    )
    encoder13_mha_k_b = initializers_onnx_initializer_381 = None
    encoder13_mha_k_transpose = getattr(self, "encoder13/mha/K/transpose")(
        encoder13_mha_k_reshape
    )
    encoder13_mha_k_reshape = None
    initializers_onnx_initializer_382 = self.initializers.onnx_initializer_382
    encoder13_mha_v_w = getattr(self, "encoder13/mha/V/w")(
        encoder12_ln2, initializers_onnx_initializer_382
    )
    initializers_onnx_initializer_382 = None
    initializers_onnx_initializer_383 = self.initializers.onnx_initializer_383
    encoder13_mha_v_b = getattr(self, "encoder13/mha/V/b")(
        encoder13_mha_v_w, initializers_onnx_initializer_383
    )
    encoder13_mha_v_w = initializers_onnx_initializer_383 = None
    initializers_onnx_initializer_384 = self.initializers.onnx_initializer_384
    encoder13_mha_v_reshape = getattr(self, "encoder13/mha/V/reshape")(
        encoder13_mha_v_b, initializers_onnx_initializer_384
    )
    encoder13_mha_v_b = initializers_onnx_initializer_384 = None
    encoder13_mha_v_transpose = getattr(self, "encoder13/mha/V/transpose")(
        encoder13_mha_v_reshape
    )
    encoder13_mha_v_reshape = None
    encoder13_mha_qk_matmul = getattr(self, "encoder13/mha/QK/matmul")(
        encoder13_mha_q_transpose, encoder13_mha_k_transpose
    )
    encoder13_mha_q_transpose = encoder13_mha_k_transpose = None
    initializers_onnx_initializer_385 = self.initializers.onnx_initializer_385
    encoder13_mha_qk_scale = getattr(self, "encoder13/mha/QK/scale")(
        encoder13_mha_qk_matmul, initializers_onnx_initializer_385
    )
    encoder13_mha_qk_matmul = initializers_onnx_initializer_385 = None
    initializers_onnx_initializer_386 = self.initializers.onnx_initializer_386
    encoder13_smolgen_compress = getattr(self, "encoder13/smolgen/compress")(
        encoder12_ln2, initializers_onnx_initializer_386
    )
    initializers_onnx_initializer_386 = None
    initializers_onnx_initializer_387 = self.initializers.onnx_initializer_387
    encoder13_smolgen_compress_reshape = getattr(
        self, "encoder13/smolgen/compress/reshape"
    )(encoder13_smolgen_compress, initializers_onnx_initializer_387)
    encoder13_smolgen_compress = initializers_onnx_initializer_387 = None
    initializers_onnx_initializer_388 = self.initializers.onnx_initializer_388
    encoder13_smolgen_dense1_w = getattr(self, "encoder13/smolgen/dense1/w")(
        encoder13_smolgen_compress_reshape, initializers_onnx_initializer_388
    )
    encoder13_smolgen_compress_reshape = initializers_onnx_initializer_388 = None
    initializers_onnx_initializer_389 = self.initializers.onnx_initializer_389
    encoder13_smolgen_dense1_b = getattr(self, "encoder13/smolgen/dense1/b")(
        encoder13_smolgen_dense1_w, initializers_onnx_initializer_389
    )
    encoder13_smolgen_dense1_w = initializers_onnx_initializer_389 = None
    encoder13_smolgen_dense1_swish_sigmoid = getattr(
        self, "encoder13/smolgen/dense1/swish/sigmoid"
    )(encoder13_smolgen_dense1_b)
    encoder13_smolgen_dense1_swish = getattr(self, "encoder13/smolgen/dense1/swish")(
        encoder13_smolgen_dense1_swish_sigmoid, encoder13_smolgen_dense1_b
    )
    encoder13_smolgen_dense1_swish_sigmoid = encoder13_smolgen_dense1_b = None
    encoder13_smolgen_ln1 = getattr(self, "encoder13/smolgen/ln1")(
        encoder13_smolgen_dense1_swish
    )
    encoder13_smolgen_dense1_swish = None
    initializers_onnx_initializer_390 = self.initializers.onnx_initializer_390
    encoder13_smolgen_dense2_w = getattr(self, "encoder13/smolgen/dense2/w")(
        encoder13_smolgen_ln1, initializers_onnx_initializer_390
    )
    encoder13_smolgen_ln1 = initializers_onnx_initializer_390 = None
    initializers_onnx_initializer_391 = self.initializers.onnx_initializer_391
    encoder13_smolgen_dense2_b = getattr(self, "encoder13/smolgen/dense2/b")(
        encoder13_smolgen_dense2_w, initializers_onnx_initializer_391
    )
    encoder13_smolgen_dense2_w = initializers_onnx_initializer_391 = None
    encoder13_smolgen_dense2_swish_sigmoid = getattr(
        self, "encoder13/smolgen/dense2/swish/sigmoid"
    )(encoder13_smolgen_dense2_b)
    encoder13_smolgen_dense2_swish = getattr(self, "encoder13/smolgen/dense2/swish")(
        encoder13_smolgen_dense2_swish_sigmoid, encoder13_smolgen_dense2_b
    )
    encoder13_smolgen_dense2_swish_sigmoid = encoder13_smolgen_dense2_b = None
    encoder13_smolgen_ln2 = getattr(self, "encoder13/smolgen/ln2")(
        encoder13_smolgen_dense2_swish
    )
    encoder13_smolgen_dense2_swish = None
    initializers_onnx_initializer_392 = self.initializers.onnx_initializer_392
    encoder13_smolgen_gen_from_reshape = getattr(
        self, "encoder13/smolgen/gen_from/reshape"
    )(encoder13_smolgen_ln2, initializers_onnx_initializer_392)
    encoder13_smolgen_ln2 = initializers_onnx_initializer_392 = None
    initializers_onnx_initializer_393 = self.initializers.onnx_initializer_393
    encoder13_smolgen_smol_weight_gen = getattr(
        self, "encoder13/smolgen/smol_weight_gen"
    )(encoder13_smolgen_gen_from_reshape, initializers_onnx_initializer_393)
    encoder13_smolgen_gen_from_reshape = initializers_onnx_initializer_393 = None
    initializers_onnx_initializer_394 = self.initializers.onnx_initializer_394
    encoder13_smolgen_out_reshape = getattr(self, "encoder13/smolgen/out/reshape")(
        encoder13_smolgen_smol_weight_gen, initializers_onnx_initializer_394
    )
    encoder13_smolgen_smol_weight_gen = initializers_onnx_initializer_394 = None
    encoder13_smolgen_weights = getattr(self, "encoder13/smolgen_weights")(
        encoder13_mha_qk_scale, encoder13_smolgen_out_reshape
    )
    encoder13_mha_qk_scale = encoder13_smolgen_out_reshape = None
    encoder13_mha_qk_softmax = getattr(self, "encoder13/mha/QK/softmax")(
        encoder13_smolgen_weights
    )
    encoder13_smolgen_weights = None
    encoder13_mha_qkv_matmul = getattr(self, "encoder13/mha/QKV/matmul")(
        encoder13_mha_qk_softmax, encoder13_mha_v_transpose
    )
    encoder13_mha_qk_softmax = encoder13_mha_v_transpose = None
    encoder13_mha_out_transpose = getattr(self, "encoder13/mha/out/transpose")(
        encoder13_mha_qkv_matmul
    )
    encoder13_mha_qkv_matmul = None
    initializers_onnx_initializer_395 = self.initializers.onnx_initializer_395
    encoder13_mha_out_reshape = getattr(self, "encoder13/mha/out/reshape")(
        encoder13_mha_out_transpose, initializers_onnx_initializer_395
    )
    encoder13_mha_out_transpose = initializers_onnx_initializer_395 = None
    initializers_onnx_initializer_396 = self.initializers.onnx_initializer_396
    encoder13_mha_out_dense_w = getattr(self, "encoder13/mha/out/dense/w")(
        encoder13_mha_out_reshape, initializers_onnx_initializer_396
    )
    encoder13_mha_out_reshape = initializers_onnx_initializer_396 = None
    initializers_onnx_initializer_397 = self.initializers.onnx_initializer_397
    encoder13_mha_out_dense_b = getattr(self, "encoder13/mha/out/dense/b")(
        encoder13_mha_out_dense_w, initializers_onnx_initializer_397
    )

    encoder13_mha_out_dense_b = self.attention_output[13](
        encoder13_mha_out_dense_b.view(-1, 64, 768)
    ).view(-1, 768)

    encoder13_mha_out_dense_w = initializers_onnx_initializer_397 = None
    initializers_onnx_initializer_398 = self.initializers.onnx_initializer_398
    encoder13_alpha_input = getattr(self, "encoder13/alpha*input")(
        encoder12_ln2, initializers_onnx_initializer_398
    )
    encoder12_ln2 = initializers_onnx_initializer_398 = None
    encoder13_mha_out_skip = getattr(self, "encoder13/mha/out/skip")(
        encoder13_mha_out_dense_b, encoder13_alpha_input
    )
    encoder13_mha_out_dense_b = encoder13_alpha_input = None
    encoder13_ln1 = getattr(self, "encoder13/ln1")(encoder13_mha_out_skip)
    encoder13_mha_out_skip = None

    ########################################################################
    # Make residuals accessible with squares as a separate dim,
    # this is important for nnsight.
    ########################################################################
    # This is just an nn.Identity layer
    residual_reasonable_shape = self.post_attention[13](
        encoder13_ln1.reshape(-1, 64, 768)
    )
    encoder13_ln1 = residual_reasonable_shape.reshape(-1, 768)
    residual_reasonable_shape = None

    initializers_onnx_initializer_399 = self.initializers.onnx_initializer_399
    encoder13_ffn_dense1_w = getattr(self, "encoder13/ffn/dense1/w")(
        encoder13_ln1, initializers_onnx_initializer_399
    )
    initializers_onnx_initializer_399 = None
    initializers_onnx_initializer_400 = self.initializers.onnx_initializer_400
    encoder13_ffn_dense1_b = getattr(self, "encoder13/ffn/dense1/b")(
        encoder13_ffn_dense1_w, initializers_onnx_initializer_400
    )
    encoder13_ffn_dense1_w = initializers_onnx_initializer_400 = None
    encoder13_ffn_dense1_sqrrelu_relu = getattr(
        self, "encoder13/ffn/dense1/sqrrelu/relu"
    )(encoder13_ffn_dense1_b)
    encoder13_ffn_dense1_b = None
    encoder13_ffn_dense1_sqrrelu_sqr = getattr(
        self, "encoder13/ffn/dense1/sqrrelu/sqr"
    )(encoder13_ffn_dense1_sqrrelu_relu, encoder13_ffn_dense1_sqrrelu_relu)
    encoder13_ffn_dense1_sqrrelu_relu = None
    initializers_onnx_initializer_401 = self.initializers.onnx_initializer_401
    encoder13_ffn_dense2_w = getattr(self, "encoder13/ffn/dense2/w")(
        encoder13_ffn_dense1_sqrrelu_sqr, initializers_onnx_initializer_401
    )
    encoder13_ffn_dense1_sqrrelu_sqr = initializers_onnx_initializer_401 = None
    initializers_onnx_initializer_402 = self.initializers.onnx_initializer_402
    encoder13_ffn_dense2_b = getattr(self, "encoder13/ffn/dense2/b")(
        encoder13_ffn_dense2_w, initializers_onnx_initializer_402
    )

    encoder13_ffn_dense2_b = self.mlp_output[13](
        encoder13_ffn_dense2_b.view(-1, 64, 768)
    ).view(-1, 768)

    encoder13_ffn_dense2_w = initializers_onnx_initializer_402 = None
    initializers_onnx_initializer_403 = self.initializers.onnx_initializer_403
    encoder13_alpha_out1 = getattr(self, "encoder13/alpha*out1")(
        encoder13_ln1, initializers_onnx_initializer_403
    )
    encoder13_ln1 = initializers_onnx_initializer_403 = None
    encoder13_ffn_skip = getattr(self, "encoder13/ffn/skip")(
        encoder13_ffn_dense2_b, encoder13_alpha_out1
    )
    encoder13_ffn_dense2_b = encoder13_alpha_out1 = None
    encoder13_ln2 = getattr(self, "encoder13/ln2")(encoder13_ffn_skip)
    encoder13_ffn_skip = None

    ########################################################################
    # Make residuals accessible with squares as a separate dim,
    # this is important for nnsight.
    ########################################################################
    # This is just an nn.Identity layer
    residual_reasonable_shape = self.post_mlp[13](encoder13_ln2.reshape(-1, 64, 768))
    encoder13_ln2 = residual_reasonable_shape.reshape(-1, 768)
    residual_reasonable_shape = None

    ########################################################################
    # Layer 14
    ########################################################################
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
    encoder14_smolgen_dense1_swish = getattr(self, "encoder14/smolgen/dense1/swish")(
        encoder14_smolgen_dense1_swish_sigmoid, encoder14_smolgen_dense1_b
    )
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
    encoder14_smolgen_dense2_swish = getattr(self, "encoder14/smolgen/dense2/swish")(
        encoder14_smolgen_dense2_swish_sigmoid, encoder14_smolgen_dense2_b
    )
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

    encoder14_mha_out_dense_b = self.attention_output[14](
        encoder14_mha_out_dense_b.view(-1, 64, 768)
    ).view(-1, 768)

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

    ########################################################################
    # Make residuals accessible with squares as a separate dim,
    # this is important for nnsight.
    ########################################################################
    # This is just an nn.Identity layer
    residual_reasonable_shape = self.post_attention[14](
        encoder14_ln1.reshape(-1, 64, 768)
    )
    encoder14_ln1 = residual_reasonable_shape.reshape(-1, 768)
    residual_reasonable_shape = None

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

    encoder14_ffn_dense2_b = self.mlp_output[14](
        encoder14_ffn_dense2_b.view(-1, 64, 768)
    ).view(-1, 768)

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

    ########################################################################
    # Make residuals accessible with squares as a separate dim,
    # this is important for nnsight.
    ########################################################################
    # This is just an nn.Identity layer
    residual_reasonable_shape = self.post_mlp[14](encoder14_ln2.reshape(-1, 64, 768))
    encoder14_ln2 = residual_reasonable_shape.reshape(-1, 768)
    residual_reasonable_shape = None

    ########################################################################
    # Policy head
    ########################################################################
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
    policy_matmul = getattr(self, "policy/matmul")(policy_q_reshape, policy_k_transpose)
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
        policy_promotion_reshape2, policy_promotion_reshape2, policy_promotion_reshape2
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
    policy_reshape = initializers_onnx_initializer_452 = None

    ########################################################################
    # Value head
    ########################################################################
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
    output_wdl = getattr(self, "output/wdl")(value_dense2_add)
    value_dense2_add = None
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
    mlh_embed_mish_softplus = getattr(self, "mlh/embed/mish/softplus")(mlh_embed_add)
    mlh_embed_mish_tanh = getattr(self, "mlh/embed/mish/tanh")(mlh_embed_mish_softplus)
    mlh_embed_mish_softplus = None
    mlh_embed_mish = getattr(self, "mlh/embed/mish")(mlh_embed_mish_tanh, mlh_embed_add)
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
    mlh_dense1_mish_softplus = getattr(self, "mlh/dense1/mish/softplus")(mlh_dense1_add)
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
    mlh_dense2_mish_softplus = getattr(self, "mlh/dense2/mish/softplus")(mlh_dense2_add)
    mlh_dense2_mish_tanh = getattr(self, "mlh/dense2/mish/tanh")(
        mlh_dense2_mish_softplus
    )
    mlh_dense2_mish_softplus = None
    mlh_dense2_mish = getattr(self, "mlh/dense2/mish")(
        mlh_dense2_mish_tanh, mlh_dense2_add
    )
    mlh_dense2_mish_tanh = mlh_dense2_add = None
    output_mlh = getattr(self, "output/mlh")(mlh_dense2_mish)
    mlh_dense2_mish = None
    return [output_policy, output_wdl, output_mlh]
