"""Per-architecture musubi-tuner pipeline: caching scripts and their arguments.

Musubi cannot train without pre-computed caches, so every architecture runs
three stages in order: cache latents -> cache text-encoder outputs -> train.
Argument names come from each ``*_cache_*.py`` script's ``setup_parser`` and each
``*_train_network.py``'s ``*_setup_parser`` (see ``params.py`` for the train side).
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Pipeline:
    script: str
    latents: str
    text_encoder: str
    module: str
    # (CLI argument name, parameter key) for the latent cache stage.
    latents_args: tuple[tuple[str, str], ...] = ()
    # (CLI argument name, parameter key) for the text-encoder cache stage.
    te_args: tuple[tuple[str, str], ...] = ()
    # Parameter keys that must be passed on the training command line itself
    # (argparse ``required=True`` args cannot come from ``--config_file``).
    cli_args: tuple[str, ...] = ()
    # Parameter keys that must be non-empty before submission.
    required: tuple[str, ...] = ()


PIPELINES: dict[str, Pipeline] = {
    "hunyuan_video": Pipeline(
        script="hv_train_network.py",
        latents="cache_latents.py",
        text_encoder="cache_text_encoder_outputs.py",
        module="networks.lora",
        latents_args=(
            ("vae", "vae"),
            ("vae_tiling", "hv_vae_tiling"),
            ("vae_chunk_size", "hv_vae_chunk_size"),
            ("vae_spatial_tile_sample_min_size", "hv_vae_spatial_tile_sample_min_size"),
        ),
        te_args=(("text_encoder1", "hv_text_encoder1"), ("text_encoder2", "hv_text_encoder2"), ("fp8_llm", "hv_fp8_llm")),
        required=("vae", "hv_text_encoder1", "hv_text_encoder2"),
    ),
    "hunyuan_video_15": Pipeline(
        script="hv_1_5_train_network.py",
        latents="hv_1_5_cache_latents.py",
        text_encoder="hv_1_5_cache_text_encoder_outputs.py",
        module="networks.lora_hv_1_5",
        latents_args=(
            ("vae", "vae"),
            ("i2v", "hv15_i2v"),
            ("image_encoder", "hv15_image_encoder"),
            ("vae_sample_size", "hv15_vae_sample_size"),
        ),
        te_args=(("text_encoder", "hv15_text_encoder"), ("byt5", "hv15_byt5"), ("fp8_vl", "hv15_fp8_vl")),
        required=("vae", "hv15_text_encoder", "hv15_byt5"),
    ),
    "wan": Pipeline(
        script="wan_train_network.py",
        latents="wan_cache_latents.py",
        text_encoder="wan_cache_text_encoder_outputs.py",
        module="networks.lora_wan",
        latents_args=(
            ("vae", "vae"),
            ("i2v", "wan_i2v"),
            ("clip", "wan_clip"),
            ("vae_cache_cpu", "wan_vae_cache_cpu"),
        ),
        te_args=(("t5", "wan_t5"), ("fp8_t5", "wan_fp8_t5")),
        required=("vae", "wan_t5"),
    ),
    "flux2": Pipeline(
        script="flux_2_train_network.py",
        latents="flux_2_cache_latents.py",
        text_encoder="flux_2_cache_text_encoder_outputs.py",
        module="networks.lora_flux_2",
        latents_args=(("vae", "vae"), ("model_version", "flux2_model_version")),
        te_args=(
            ("text_encoder", "flux2_text_encoder"),
            ("fp8_text_encoder", "flux2_fp8_text_encoder"),
            ("model_version", "flux2_model_version"),
        ),
        required=("vae", "flux2_text_encoder"),
    ),
    "flux1_kontext": Pipeline(
        script="flux_kontext_train_network.py",
        latents="flux_kontext_cache_latents.py",
        text_encoder="flux_kontext_cache_text_encoder_outputs.py",
        module="networks.lora_flux",
        latents_args=(("vae", "vae"),),
        te_args=(
            ("text_encoder1", "kontext_text_encoder1"),
            ("text_encoder2", "kontext_text_encoder2"),
            ("fp8_t5", "kontext_fp8_t5"),
        ),
        required=("vae", "kontext_text_encoder1", "kontext_text_encoder2"),
    ),
    "qwen_image": Pipeline(
        script="qwen_image_train_network.py",
        latents="qwen_image_cache_latents.py",
        text_encoder="qwen_image_cache_text_encoder_outputs.py",
        module="networks.lora_qwen_image",
        latents_args=(("vae", "vae"), ("model_version", "qwen_model_version")),
        te_args=(
            ("text_encoder", "qwen_text_encoder"),
            ("fp8_vl", "qwen_fp8_vl"),
            ("model_version", "qwen_model_version"),
        ),
        required=("vae", "qwen_text_encoder"),
    ),
    "z_image": Pipeline(
        script="zimage_train_network.py",
        latents="zimage_cache_latents.py",
        text_encoder="zimage_cache_text_encoder_outputs.py",
        module="networks.lora",
        latents_args=(("vae", "vae"),),
        te_args=(("text_encoder", "zimage_text_encoder"), ("fp8_llm", "zimage_fp8_llm")),
        required=("vae", "zimage_text_encoder"),
    ),
    "hidream_o1": Pipeline(
        script="hidream_o1_train_network.py",
        latents="hidream_o1_cache_pixel.py",
        text_encoder="hidream_o1_cache_text_encoder_outputs.py",
        module="networks.lora_hidream_o1",
        latents_args=(),
        te_args=(("model_type", "hidream_model_type"),),
        required=(),
    ),
    "ideogram4": Pipeline(
        script="ideogram4_train_network.py",
        latents="ideogram4_cache_latents.py",
        text_encoder="ideogram4_cache_text_encoder_outputs.py",
        module="networks.lora_ideogram4",
        latents_args=(("vae", "vae"),),
        te_args=(("text_encoder", "ideogram_text_encoder"),),
        required=("vae", "ideogram_text_encoder"),
    ),
    "kandinsky5": Pipeline(
        script="kandinsky5_train_network.py",
        latents="kandinsky5_cache_latents.py",
        text_encoder="kandinsky5_cache_text_encoder_outputs.py",
        module="networks.lora_kandinsky",
        latents_args=(("vae", "vae"),),
        te_args=(
            ("text_encoder_qwen", "kandinsky_text_encoder_qwen"),
            ("text_encoder_clip", "kandinsky_text_encoder_clip"),
        ),
        cli_args=("kandinsky_task",),
        required=("vae", "kandinsky_text_encoder_qwen", "kandinsky_text_encoder_clip", "kandinsky_task"),
    ),
    "krea2": Pipeline(
        script="krea2_train_network.py",
        latents="krea2_cache_latents.py",
        text_encoder="krea2_cache_text_encoder_outputs.py",
        module="networks.lora_krea2",
        latents_args=(("vae", "vae"),),
        te_args=(("text_encoder", "krea2_text_encoder"),),
        required=("vae", "krea2_text_encoder"),
    ),
    "minimax_h3": Pipeline(
        script="minimax_h3_train_network.py",
        latents="minimax_h3_cache_latents.py",
        text_encoder="minimax_h3_cache_text_encoder_outputs.py",
        module="networks.lora_minimax_h3",
        latents_args=(
            ("task", "h3_task"),
            ("video_vae", "h3_video_vae"),
            ("audio_vae", "h3_audio_vae"),
            ("one_frame", "h3_one_frame"),
        ),
        te_args=(("task", "h3_task"), ("text_encoder", "h3_text_encoder"), ("one_frame", "h3_one_frame")),
        required=("h3_task", "h3_video_vae", "h3_audio_vae", "h3_text_encoder"),
    ),
    "framepack": Pipeline(
        script="fpack_train_network.py",
        latents="fpack_cache_latents.py",
        text_encoder="fpack_cache_text_encoder_outputs.py",
        module="networks.lora_framepack",
        latents_args=(
            ("vae", "vae"),
            ("image_encoder", "fpack_image_encoder"),
            ("f1", "fpack_f1"),
            ("one_frame", "fpack_one_frame"),
        ),
        te_args=(
            ("text_encoder1", "fpack_text_encoder1"),
            ("text_encoder2", "fpack_text_encoder2"),
            ("fp8_llm", "fpack_fp8_llm"),
        ),
        required=("vae", "fpack_image_encoder", "fpack_text_encoder1", "fpack_text_encoder2"),
    ),
}


def pipeline_for(architecture: str) -> Pipeline | None:
    return PIPELINES.get(architecture)
