# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

from ..core import LLMBridge, register_model


@register_model("qwen3_next")
class Qwen3NextBridge(LLMBridge):
    _DIRECT_MAPPING = {
        "embedding.word_embeddings.weight": "model.embed_tokens.weight",
        "decoder.final_layernorm.weight": "model.norm.weight",
        "output_layer.weight": "lm_head.weight",
    }
    _ATTENTION_MAPPING = {
        # legacy
        "self_attention.linear_proj.weight": [
            "model.layers.{layer_number}.self_attn.o_proj.weight"
        ],
        "self_attention.linear_qkv.layer_norm_weight": [
            "model.layers.{layer_number}.input_layernorm.weight"
        ],
        "self_attention.q_layernorm.weight": [
            "model.layers.{layer_number}.self_attn.q_norm.weight"
        ],
        "self_attention.k_layernorm.weight": [
            "model.layers.{layer_number}.self_attn.k_norm.weight"
        ],
        "self_attention.linear_qkv.weight": [
            "model.layers.{layer_number}.self_attn.q_proj.weight",
            "model.layers.{layer_number}.self_attn.k_proj.weight",
            "model.layers.{layer_number}.self_attn.v_proj.weight",
        ],
        "self_attention.linear_qkv.bias": [
            "model.layers.{layer_number}.self_attn.q_proj.bias",
            "model.layers.{layer_number}.self_attn.k_proj.bias",
            "model.layers.{layer_number}.self_attn.v_proj.bias",
        ],
        # gated delta net
        "self_attention.A_log": [
            "model.layers.{layer_number}.linear_attn.A_log",
        ],
        "self_attention.dt_bias": [
            "model.layers.{layer_number}.linear_attn.dt_bias",
        ],
        "self_attention.conv1d.weight": [
            "model.layers.{layer_number}.linear_attn.conv1d.weight",
        ],
        "self_attention.in_proj_qkvz.weight": [
            "model.layers.{layer_number}.linear_attn.in_proj_qkvz.weight",
        ],
        "self_attention.in_proj_ba.weight": [
            "model.layers.{layer_number}.linear_attn.in_proj_ba.weight",
        ],
        "self_attention.out_proj.weight": [
            "model.layers.{layer_number}.linear_attn.out_proj.weight",
        ],
        "self_attention.norm.weight": [
            "model.layers.{layer_number}.linear_attn.norm.weight",
        ],
        # gated attention
        "self_attention.q_proj.weight": [
            "model.layers.{layer_number}.self_attn.q_proj.weight",
        ],
        "self_attention.k_proj.weight": [
            "model.layers.{layer_number}.self_attn.k_proj.weight",
        ],
        "self_attention.v_proj.weight": [
            "model.layers.{layer_number}.self_attn.v_proj.weight",
        ],
        "self_attention.q_norm.weight": [
            "model.layers.{layer_number}.self_attn.q_norm.weight",
        ],
        "self_attention.k_norm.weight": [
            "model.layers.{layer_number}.self_attn.k_norm.weight",
        ],
        "self_attention.o_proj.weight": [
            "model.layers.{layer_number}.self_attn.o_proj.weight",
        ],
    }
    _MLP_MAPPING = {
        "shared_experts.linear_fc1.weight": [
            "model.layers.{layer_number}.mlp.shared_expert.gate_proj.weight",
            "model.layers.{layer_number}.mlp.shared_expert.up_proj.weight",
        ],
        "pre_mlp_layernorm": [
            "model.layers.{layer_number}.post_attention_layernorm.weight"
        ],
        "shared_experts.linear_fc2.weight": [
            "model.layers.{layer_number}.mlp.shared_expert.down_proj.weight"
        ],
        "mlp.router.weight": ["model.layers.{layer_number}.mlp.gate.weight"],
        "shared_experts.gate_weight": [
            "model.layers.{layer_number}.mlp.shared_expert_gate.weight"
        ],
        "mlp.experts.linear_fc1": [
            "model.layers.{layer_number}.mlp.experts.{expert_id}.gate_proj.weight",
            "model.layers.{layer_number}.mlp.experts.{expert_id}.up_proj.weight",
        ],
        "mlp.experts.linear_fc2": [
            "model.layers.{layer_number}.mlp.experts.{expert_id}.down_proj.weight"
        ],
    }
    _OTHER_MAPPING = {
        # input layernorm
        "input_layernorm.weight": [
            "model.layers.{layer_number}.input_layernorm.weight",
        ],
    }

    def _build_config(self):
        tf_config = self._build_base_config(
            use_cpu_initialization=False,
            # MoE specific
            moe_ffn_hidden_size=self.hf_config.moe_intermediate_size,
            moe_router_bias_update_rate=0.001,
            moe_router_topk=self.hf_config.num_experts_per_tok,
            num_moe_experts=self.hf_config.num_experts,
            moe_aux_loss_coeff=self.hf_config.router_aux_loss_coef,
            # moe_router_load_balancing_type="aux_loss",
            moe_router_load_balancing_type="none",  # default None for RL
            moe_grouped_gemm=True,
            moe_router_score_function="softmax",
            # Other optimizations
            persist_layer_norm=True,
            bias_activation_fusion=True,
            bias_dropout_fusion=True,
            # Qwen specific
            moe_router_pre_softmax=False,
            qk_layernorm=True,
            # Qwen3Next specific
            is_hybrid_model=True,
            multi_latent_attention=True,
            layernorm_zero_centered_gamma=True,
            mtp_num_layers=1,
            moe_shared_expert_intermediate_size=self.hf_config.shared_expert_intermediate_size,
        )

        # if not hasattr(tf_config, "get_config_for_layer"):

        #     def _get_config_for_layer(self, layer_num: int):
        #         # For our use-case, per-layer dims don’t change, so return self.
        #         return self

        #     import types

        #     tf_config.get_config_for_layer = types.MethodType(
        #         _get_config_for_layer, tf_config
        #     )

        return tf_config

    def _weight_name_mapping_mlp(self, name: str) -> list[str]:
        layer_number = name.split(".")[2]
        convert_names = []
        for keyword, mapping_names in self._MLP_MAPPING.items():
            if keyword in name:
                if "{expert_id}" in mapping_names[0]:
                    expert_id = name.split("weight")[-1]
                    convert_names.extend(
                        [
                            x.format(layer_number=layer_number, expert_id=expert_id)
                            for x in mapping_names
                        ]
                    )
                else:
                    convert_names.extend(
                        [x.format(layer_number=layer_number) for x in mapping_names]
                    )
                break
        if len(convert_names) == 0:
            raise NotImplementedError(f"Unsupported parameter name: {name}")
        return convert_names
