"""Supplementary train arguments in musubi-tuner f8a1b03, grouped by argparse type.

Cache-only CLI switches belong to their respective stages, never the trainer TOML.
Model-specific parameter surfaces in params.py take precedence over this list.
"""

EXTRA_OPTIONS = {
    ("common", "str"): (
        "compile_dynamic dynamo_mode huggingface_path_in_repo huggingface_repo_type "
        "huggingface_repo_visibility huggingface_token log_prefix log_tracker_config log_tracker_name "
        "metadata_arch metadata_reso wandb_api_key "
    ).split(),
    ("common", "int"): (
        "compile_cache_size_limit ddp_timeout "
    ).split(),
    ("common", "bool"): (
        "ddp_gradient_as_bucket_view ddp_static_graph dynamo_dynamic dynamo_fullgraph log_config "
        "log_grad_metrics resume_from_huggingface save_state_to_huggingface "
    ).split(),
    ("common", "str[]"): (
        "base_weights "
    ).split(),
    ("common", "float[]"): (
        "base_weights_multiplier "
    ).split(),
    ("hunyuan_video", "str"): (
        "text_encoder_dtype "
    ).split(),
    ("hunyuan_video_15", "bool"): (
        "vae_enable_patch_conv "
    ).split(),
    ("wan", "bool"): (
        "force_v2_1_time_embedding vae_cache_cpu "
    ).split(),
    ("qwen_image", "bool"): (
        "edit edit_plus remove_first_image_from_target "
    ).split(),
    ("hidream_o1", "float"): (
        "dino_loss_weight noise_clip_std noise_scale_end noise_scale_start "
    ).split(),
    ("hidream_o1", "bool"): (
        "dino_loss_use_gram dino_loss_use_norm skip_t2i_visual_dummy "
    ).split(),
    ("hidream_o1", "str"): (
        "dino_loss_backend dino_loss_feature_mode dino_loss_model_type "
    ).split(),
    ("hidream_o1", "int"): (
        "dino_loss_every_n_steps dino_loss_layer dino_loss_resize "
    ).split(),
    ("ideogram4", "str"): (
        "unconditional_dit "
    ).split(),
    ("ideogram4", "bool"): (
        "log_loss_stats use_unconditional_dit_for_lora_sampling validate_caption_structure "
        "warn_on_caption_issues "
    ).split(),
    ("ideogram4", "float"): (
        "ideogram4_timestep_mu ideogram4_timestep_std initial_sigma "
    ).split(),
    ("kandinsky5", "str"): (
        "nabla_method override_dit "
    ).split(),
    ("kandinsky5", "bool"): (
        "force_nabla_attention nabla_add_sta no_vae_load offload_dit_during_sampling "
    ).split(),
    ("kandinsky5", "float"): (
        "nabla_P scheduler_scale "
    ).split(),
    ("kandinsky5", "int"): (
        "nabla_wH nabla_wT nabla_wW "
    ).split(),
    ("minimax_h3", "bool"): (
        "convrot_int8 h3_allow_experimental_sample_duration h3_teacher_matching nvfp4_scaled_mm prune_adaln "
        "video_only "
    ).split(),
    ("minimax_h3", "float"): (
        "audio_loss_weight h3_audio_cond_clean h3_guidance_loss_scale h3_guidance_loss_scale_audio "
        "h3_guidance_loss_sigma_min h3_shift_audio h3_shift_video h3_teacher_condition_sigma_max "
        "h3_teacher_condition_sigma_min h3_teacher_loss_dc_weight h3_teacher_loss_mag_weight "
        "h3_teacher_preservation_weight h3_timestep_focus_max h3_timestep_focus_min "
        "h3_timestep_focus_prob h3_visual_cond_clean "
    ).split(),
    ("minimax_h3", "int"): (
        "text_encoder_blocks_to_swap "
    ).split(),
    ("minimax_h3", "str"): (
        "convrot_int8_bwd h3_guidance_loss_uncond_cache h3_teacher_conditions text_encoder_attn_mode "
    ).split(),
    ("framepack", "bool"): (
        "bulk_decode "
    ).split(),
}

EXTRA_CHOICES = {
    "convrot_int8_bwd": ("bf16", "int8"),
    "compile_dynamic": ('true', 'false', 'auto'),
    "dino_loss_backend": ('vit', 'convnext'),
    "dino_loss_feature_mode": ('all', 'patch', 'cls', 'both'),
    "dynamo_mode": ('default', 'reduce-overhead', 'max-autotune'),
    "text_encoder_attn_mode": ('sdpa', 'flash_attention_2', 'eager'),
}

# Range checks performed after parsing by the architecture-specific trainer.
EXTRA_RANGES = {
    "h3_shift_video": (0.01, 100.0), "h3_shift_audio": (0.01, 100.0),
    "h3_visual_cond_clean": (0.0, 1.0), "h3_audio_cond_clean": (0.0, 1.0),
    "h3_timestep_focus_prob": (0.0, 1.0), "h3_guidance_loss_sigma_min": (0.0, 1.0),
    "h3_guidance_loss_scale": (0.0, None), "h3_guidance_loss_scale_audio": (0.0, None),
    "h3_teacher_loss_dc_weight": (0.0, None), "h3_teacher_loss_mag_weight": (0.0, None),
    "h3_teacher_preservation_weight": (0.0, None), "dino_loss_every_n_steps": (1, None),
}
