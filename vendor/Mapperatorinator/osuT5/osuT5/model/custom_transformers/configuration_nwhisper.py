from transformers import WhisperConfig


class NWhisperConfig(WhisperConfig):
    model_type = "whisper"

    def __init__(
            self,
            vocab_size=51865,
            num_mel_bins=80,
            encoder_layers=4,
            encoder_attention_heads=6,
            decoder_layers=4,
            decoder_attention_heads=6,
            decoder_ffn_dim=1536,
            encoder_ffn_dim=1536,
            encoder_layerdrop=0.0,
            decoder_layerdrop=0.0,
            decoder_start_token_id=50257,
            use_cache=True,
            is_encoder_decoder=True,
            activation_function="gelu",
            d_model=384,
            dropout=0.0,
            attention_dropout=0.0,
            activation_dropout=0.0,
            init_std=0.02,
            scale_embedding=False,
            max_source_positions=1500,
            max_target_positions=448,
            pad_token_id=50256,
            bos_token_id=50256,
            eos_token_id=50256,
            suppress_tokens=None,
            begin_suppress_tokens=[220, 50256],
            use_weighted_layer_sum=False,
            classifier_proj_size=256,
            apply_spec_augment=False,
            mask_time_prob=0.05,
            mask_time_length=10,
            mask_time_min_masks=2,
            mask_feature_prob=0.0,
            mask_feature_length=10,
            mask_feature_min_masks=0,
            median_filter_width=7,
            attn_norm_qk=True,  # they say the query/key normalization is optional
            manual_norm_weights=False,
            num_hyperspheres=1,
            # below are all the scale related hyperparameters, for controlling effective relative learning rates throughout the network
            alpha_init: float | None = 1.,
            # this would set the alpha init for all residuals, but would be overridden by alpha_attn_init and alpha_ff_init if they are specified
            s_logit_init: float = 1.,
            s_logit_scale: float | None = None,
            encoder_alpha_pos_init: float | None = 0.1,
            encoder_alpha_pos_scale: float | None = None,
            encoder_alpha_attn_init: float | tuple[float, ...] | None = 0.05,
            encoder_alpha_attn_scale: float | tuple[float, ...] | None = None,
            encoder_alpha_ff_init: float | tuple[float, ...] | None = 0.05,
            encoder_alpha_ff_scale: float | tuple[float, ...] | None = None,
            encoder_s_qk_init: float | tuple[float, ...] = 1.,
            encoder_s_qk_scale: float | tuple[float, ...] | None = None,
            decoder_alpha_pos_init: float | None = 0.1,
            decoder_alpha_pos_scale: float | None = None,
            decoder_alpha_attn_init: float | tuple[float, ...] | None = 0.05,
            decoder_alpha_attn_scale: float | tuple[float, ...] | None = None,
            decoder_alpha_cross_attn_init: float | tuple[float, ...] | None = 0.05,
            decoder_alpha_cross_attn_scale: float | tuple[float, ...] | None = None,
            decoder_alpha_ff_init: float | tuple[float, ...] | None = 0.05,
            decoder_alpha_ff_scale: float | tuple[float, ...] | None = None,
            decoder_s_qk_init: float | tuple[float, ...] = 1.,
            decoder_s_qk_scale: float | tuple[float, ...] | None = None,
            decoder_cross_s_qk_init: float | tuple[float, ...] = 1.,
            decoder_cross_s_qk_scale: float | tuple[float, ...] | None = None,
            norm_eps=0.,  # greater than 0 allows the norm to be around (1. - norm_eps) to (1. + norm_eps)
            input_vocab_size=None,
            ** kwargs,
    ):
        self.attn_norm_qk = attn_norm_qk
        self.manual_norm_weights = manual_norm_weights
        self.num_hyperspheres = num_hyperspheres
        self.alpha_init = alpha_init
        self.s_logit_init = s_logit_init
        self.s_logit_scale = s_logit_scale
        self.encoder_alpha_pos_init = encoder_alpha_pos_init
        self.encoder_alpha_pos_scale = encoder_alpha_pos_scale
        self.encoder_alpha_attn_init = encoder_alpha_attn_init
        self.encoder_alpha_attn_scale = encoder_alpha_attn_scale
        self.encoder_alpha_ff_init = encoder_alpha_ff_init
        self.encoder_alpha_ff_scale = encoder_alpha_ff_scale
        self.encoder_s_qk_init = encoder_s_qk_init
        self.encoder_s_qk_scale = encoder_s_qk_scale
        self.decoder_alpha_pos_init = decoder_alpha_pos_init
        self.decoder_alpha_pos_scale = decoder_alpha_pos_scale
        self.decoder_alpha_attn_init = decoder_alpha_attn_init
        self.decoder_alpha_attn_scale = decoder_alpha_attn_scale
        self.decoder_alpha_cross_attn_init = decoder_alpha_cross_attn_init
        self.decoder_alpha_cross_attn_scale = decoder_alpha_cross_attn_scale
        self.decoder_alpha_ff_init = decoder_alpha_ff_init
        self.decoder_alpha_ff_scale = decoder_alpha_ff_scale
        self.decoder_s_qk_init = decoder_s_qk_init
        self.decoder_s_qk_scale = decoder_s_qk_scale
        self.decoder_cross_s_qk_init = decoder_cross_s_qk_init
        self.decoder_cross_s_qk_scale = decoder_cross_s_qk_scale
        self.norm_eps = norm_eps
        self.input_vocab_size = input_vocab_size

        super().__init__(
            vocab_size=vocab_size,
            num_mel_bins=num_mel_bins,
            d_model=d_model,
            encoder_layers=encoder_layers,
            encoder_attention_heads=encoder_attention_heads,
            decoder_layers=decoder_layers,
            decoder_attention_heads=decoder_attention_heads,
            decoder_ffn_dim=decoder_ffn_dim,
            encoder_ffn_dim=encoder_ffn_dim,
            dropout=dropout,
            attention_dropout=attention_dropout,
            activation_dropout=activation_dropout,
            activation_function=activation_function,
            init_std=init_std,
            encoder_layerdrop=encoder_layerdrop,
            decoder_layerdrop=decoder_layerdrop,
            use_cache=use_cache,
            scale_embedding=scale_embedding,
            max_source_positions=max_source_positions,
            max_target_positions=max_target_positions,
            classifier_proj_size=classifier_proj_size,
            use_weighted_layer_sum=use_weighted_layer_sum,
            apply_spec_augment=apply_spec_augment,
            mask_time_prob=mask_time_prob,
            mask_time_length=mask_time_length,
            mask_time_min_masks=mask_time_min_masks,
            mask_feature_prob=mask_feature_prob,
            mask_feature_length=mask_feature_length,
            mask_feature_min_masks=mask_feature_min_masks,
            median_filter_width=median_filter_width,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            is_encoder_decoder=is_encoder_decoder,
            decoder_start_token_id=decoder_start_token_id,
            suppress_tokens=suppress_tokens,
            begin_suppress_tokens=begin_suppress_tokens,
            **kwargs,
        )
