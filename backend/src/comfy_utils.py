#!/usr/bin/env python3

import io
import gc
import os
import random
import uuid
import sys
import torch
import threading
import soundfile as sf
import numpy as np
import imageio.v3 as iio
from functools import wraps
from pathlib import Path
from typing import Any, Dict, Optional

from image_utils import tensor_to_pil
from model_map import MODEL_MAP

# Add ComfyUI paths and imports
comfy_dir = Path(os.environ.get("COMFYUI_DIR", "/comfyui"))
model_dir = comfy_dir / "models"
if os.getenv("COMFYUI_MODELS_DIR"):
    model_dir = Path(os.getenv("COMFYUI_MODELS_DIR"))

from comfy_script.runtime.real import *

extra_flags = []

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)

if DEVICE == "cpu":
    print("⚠️ Running in CPU mode, inference may be slow")
    extra_flags.append("--cpu")
if os.getenv("USE_SAGEATTENTION", "false").lower() == "true" and DEVICE == "cuda":
    print("✅ Using SageAttention for better performance")
    extra_flags.append("--use-sage-attention")
if os.getenv("TORCH_COMPILE", "false").lower() == "true":
    print("✅ Using torch.compile when available for model acceleration")

args = ComfyUIArgs(*extra_flags)

load(naked=True, no_server=True, args=args)
from comfy_script.runtime.real.nodes import *

sys.path.append(comfy_dir)
from comfy.model_management import unload_all_models, soft_empty_cache


def download_model(model_name: str) -> str:
    """
    Download a model from Hugging Face if it doesn't exist locally.
    """
    model_info = MODEL_MAP.get(model_name)
    if not model_info:
        raise ValueError(f"Model {model_name} not found in MODEL_MAP")

    model_path = model_dir / model_info[0]
    if not model_path.exists():
        print(f"Downloading {model_name} to {model_path}")
        os.makedirs(model_path.parent, exist_ok=True)
        download_url = model_info[1]
        torch.hub.download_url_to_file(download_url, str(model_path))
    return str(model_path)


class ModelCache:
    """
    A thread-safe singleton class for caching ML models.
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(ModelCache, cls).__new__(cls)
                cls._instance._initialize()
            return cls._instance

    def _initialize(self):
        """Initialize the cache storage."""
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._cache_lock = threading.Lock()

    def get(
        self, model_type: str, model_name: str, model_subtype: str = None
    ) -> Optional[Any]:
        """
        Retrieve a model from the cache.

        Args:
            model_type: Type of model (e.g., 'unet', 'vae', 'clip')
            model_name: Name/identifier of the specific model
            model_subtype: Optional subtype for models like CLIP

        Returns:
            The cached model or None if not found
        """
        cache_key = (
            f"{model_type}_{model_name}{f'_{model_subtype}' if model_subtype else ''}"
        )
        with self._cache_lock:
            return self._cache.get(cache_key, {}).get("model")

    def set(
        self,
        model_type: str,
        model_name: str,
        model: Any,
        model_subtype: str = None,
        **metadata,
    ) -> None:
        """
        Store a model in the cache with optional metadata.

        Args:
            model_type: Type of model
            model_name: Name/identifier of the specific model
            model: The model object to cache
            model_subtype: Optional subtype for models like CLIP
            **metadata: Additional metadata to store with the model
        """
        cache_key = (
            f"{model_type}_{model_name}{f'_{model_subtype}' if model_subtype else ''}"
        )
        with self._cache_lock:
            self._cache[cache_key] = {"model": model, "metadata": metadata}

    def clear(
        self, model_type: Optional[str] = None, model_name: Optional[str] = None
    ) -> None:
        """
        Clear specific models or entire cache.

        Args:
            model_type: Optional type of models to clear
            model_name: Optional specific model to clear
        """
        with self._cache_lock:
            if model_type and model_name:
                cache_key = f"{model_type}_{model_name}"
                self._cache.pop(cache_key, None)
            elif model_type:
                keys_to_remove = [
                    k for k in self._cache.keys() if k.startswith(f"{model_type}_")
                ]
                for key in keys_to_remove:
                    self._cache.pop(key, None)
            else:
                self._cache.clear()

            # Clean up GPU memory
            unload_all_models()
            soft_empty_cache()
            gc.collect()

    def exists(self, model_type: str, model_name: str) -> bool:
        """Check if a model exists in the cache."""
        cache_key = f"{model_type}_{model_name}"
        with self._cache_lock:
            return cache_key in self._cache


def clear_comfyui_model_cache():
    """
    Clear the ComfyUI model cache to free up GPU memory.
    This function is a wrapper around ModelCache.clear() to ensure
    it can be called from API endpoints.
    """
    ModelCache().clear()
    gc.collect()
    return {"success": True, "message": "Model cache cleared successfully"}


def cached_model(model_type: str, model_name_param: str, subtype_param: str = None):
    """
    Decorator for caching model loading functions.

    Args:
        model_type: Type of model being cached
        model_name_param: Parameter name that contains the model name in the decorated function
        subtype_param: Optional parameter name that contains the model subtype (e.g., 'type' for CLIP models)
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            cache = ModelCache()

            # Extract model name from args or kwargs
            model_name = None
            if model_name_param in kwargs:
                model_name = kwargs[model_name_param]
            else:
                # Try to find the parameter position in the function signature
                import inspect

                sig = inspect.signature(func)
                param_names = list(sig.parameters.keys())
                try:
                    param_idx = param_names.index(model_name_param)
                    if len(args) > param_idx:
                        model_name = args[param_idx]
                except ValueError:
                    raise ValueError(
                        f"Cannot find parameter {model_name_param} in function signature"
                    )

            if not model_name:
                raise ValueError(
                    f"Model name parameter {model_name_param} not provided"
                )

            # Get subtype if specified
            model_subtype = None
            if subtype_param:
                model_subtype = kwargs.get(subtype_param)
                if model_subtype is None and subtype_param in sig.parameters:
                    try:
                        param_idx = param_names.index(subtype_param)
                        if len(args) > param_idx:
                            model_subtype = args[param_idx]
                    except ValueError:
                        pass

            # Check cache first
            cached_model_ = cache.get(model_type, model_name, model_subtype)
            if cached_model_ is not None:
                if isinstance(cached_model_, tuple):
                    return cached_model_
                return (cached_model_,)

            # Load model if not in cache
            result = func(*args, **kwargs)

            # Cache the result
            if isinstance(result, tuple):
                cache.set(model_type, model_name, result, model_subtype)
            else:
                cache.set(model_type, model_name, (result,), model_subtype)

            return result

        return wrapper

    return decorator


@cached_model("unet", "unet_name")
def load_unet(unet_name, weight_dtype="default"):
    download_model(unet_name)
    unet_loader = UNETLoader()
    return unet_loader.load_unet(unet_name=unet_name, weight_dtype=weight_dtype)


@cached_model("dual_clip", "clip_name1", "type")
def load_dual_clip(clip_name1, clip_name2, type="flux"):
    download_model(clip_name1)
    download_model(clip_name2)
    dual_clip_loader = DualCLIPLoader()
    return dual_clip_loader.load_clip(
        clip_name1=clip_name1, clip_name2=clip_name2, type=type
    )


@cached_model("clip", "clip_name", "type")
def load_clip(clip_name, type="default"):
    download_model(clip_name)
    clip_loader = CLIPLoader()
    return clip_loader.load_clip(clip_name=clip_name, type=type, device=DEVICE)


@cached_model("vae", "vae_name")
def load_vae(vae_name):
    download_model(vae_name)
    vae_loader = VAELoader()
    return vae_loader.load_vae(vae_name=vae_name)


@cached_model("lora_stack", "lora_1_name")
def load_lora_stack(lora_1_name, lora_1_strength=1.0):
    download_model(lora_1_name)
    easy_lora_stack = EasyLoraStack()
    return easy_lora_stack.stack(
        toggle=True,
        mode="simple",
        num_loras=1,
        lora_1_name=lora_1_name,
        lora_1_strength=lora_1_strength,
    )


@cached_model("wan_video_model", "model_name")
def load_wan_video_model(
    model_name,
    base_precision="bf16",
    quantization="disabled",
    attention_mode="sageattn",
    block_swap_args=None,
    lora_1_name=None,
    lora_1_strength=1.0,
    vram_management_args=None,
    vace_model=None,
    fantasytalking_model=None,
    multitalk_model=None,
    compile_model=False,
    backend="inductor",
    fullgraph=False,
    mode="default",
    dynamic=False,
    dynamo_cache_size_limit=64,
    compile_transformer_blocks_only=True,
    dynamo_recompile_limit=128,
):
    download_model(model_name)
    if (
        os.getenv("USE_SAGEATTENTION", "false").lower() == "false"
        and attention_mode == "sageattn"
    ):
        attention_mode = "default"
    if compile_model or os.getenv("TORCH_COMPILE", "false").lower() == "true":
        wan_video_torch_compile_settings = WanVideoTorchCompileSettings()
        model_compile_args = wan_video_torch_compile_settings.set_args(
            backend=backend,
            fullgraph=fullgraph,
            mode=mode,
            dynamic=dynamic,
            dynamo_cache_size_limit=dynamo_cache_size_limit,
            compile_transformer_blocks_only=compile_transformer_blocks_only,
            dynamo_recompile_limit=dynamo_recompile_limit,
        )[0]
    else:
        model_compile_args = None

    lora = None

    wan_video_model_loader = WanVideoModelLoader()
    return wan_video_model_loader.loadmodel(
        model=model_name,
        base_precision=base_precision,
        load_device=DEVICE,
        quantization=quantization,
        compile_args=model_compile_args,
        attention_mode=attention_mode,
        block_swap_args=block_swap_args,
        lora=lora,
        vram_management_args=vram_management_args,
        vace_model=vace_model,
        fantasytalking_model=fantasytalking_model,
        multitalk_model=multitalk_model,
    )


@cached_model("wan_video_vae", "vae_name")
def load_wan_video_vae(vae_name, precision="bf16"):
    download_model(vae_name)
    wan_video_vae_loader = WanVideoVAELoader()
    return wan_video_vae_loader.loadmodel(
        model_name=vae_name, precision=precision, compile_args=None
    )


@cached_model("wan_video_t5", "model_name")
def load_wan_video_t5(model_name, precision="bf16", quantization="disabled"):
    download_model(model_name)
    wan_video_t5_text_encoder = LoadWanVideoT5TextEncoder()
    return wan_video_t5_text_encoder.loadmodel(
        model_name=model_name,
        precision=precision,
        load_device=DEVICE,
        quantization=quantization,
    )


@cached_model("llm_local", "model_name_or_path")
def llm_local_loader(model_name_or_path, dtype="auto", is_locked=True):
    """
    Load a local LLM model with optional device and dtype settings.
    """
    # model files are handled through huggingface, no need to explicitly download here
    llm_local_loader = LLMLocalLoader()
    return llm_local_loader.chatbot(
        model_name_or_path=model_name_or_path,
        device=DEVICE,
        dtype=dtype,
        is_locked=is_locked,
    )


def run_local_llm_inference(
    model_name_or_path,
    prompt,
    system_prompt=None,
    temperature=0.7,
    max_length=2048,
    dtype="auto",
    is_locked=True,
    model_type="LLM",  # "LLM", "LLM-GGUF", "VLM-GGUF", "VLM(llama-v)", "VLM(qwen-vl)", "VLM(deepseek-janus-pro)"
):
    """
    Run inference on a local LLM model.
    """
    with torch.inference_mode():
        model, tokenizer = llm_local_loader(
            model_name_or_path=model_name_or_path,
            dtype=dtype,
            is_locked=is_locked,
        )

        llm_local = LLMLocal()
        response, history, llm_tools_json, image_out = llm_local.chatbot(
            user_prompt=prompt,
            main_brain="enable",
            system_prompt=system_prompt,
            model_type=model_type,
            model=model,
            tokenizer=tokenizer,
            temperature=temperature,
            max_length=max_length,
            is_memory="disable",
            is_locked="disable",
            system_prompt_input="",
            user_prompt_input="",
            tools=None,
            file_content=None,
            image=None,
            conversation_rounds=100,
            historical_record="",
            is_enable=True,
            extra_parameters=None,
            user_history=None,
        )
        if "</think>" in response:
            reasoning_content = response.split("</think>", 1)[0].strip()
            response = response.split("</think>", 1)[1].strip()
        else:
            reasoning_content = ""
            response = response.strip()
        return (
            response,
            history,
            llm_tools_json,
            image_out,
            reasoning_content,
        )


def run_api_llm_inference(
    model_name,
    prompt,
    system_prompt=None,
    temperature=0.7,
    max_length=2048,
    model_base_url=None,
    model_api_key=None,
    is_ollama=False,
):
    """
    Run inference on an API-based LLM model.
    """
    with torch.inference_mode():
        llm_api_loader = LLMApiLoader()
        model = llm_api_loader.chatbot(
            model_name=model_name,
            base_url=model_base_url,
            api_key=model_api_key,
            is_ollama=is_ollama,
        )[0]

        llm = Llm()
        response, history, llm_tools_json, image_out, reasoning_content = llm.chatbot(
            user_prompt=prompt,
            main_brain="enable",
            system_prompt=system_prompt,
            model=model,
            temperature=temperature,
            is_memory="disable",
            is_tools_in_sys_prompt="disable",
            is_locked="disable",
            max_length=max_length,
            system_prompt_input="",
            user_prompt_input="",
            tools=None,
            file_content=None,
            images=None,
            imgbb_api_key=None,
            conversation_rounds=100,
            historical_record="",
            is_enable=True,
            extra_parameters=None,
            user_history=None,
            img_URL=None,
            stream=False,
        )
        return response, history, llm_tools_json, image_out, reasoning_content


def run_llm_inference(
    model_name_or_path,
    prompt,
    system_prompt=None,
    use_local=True,
    model_type="LLM",  # "LLM", "LLM-GGUF", "VLM-GGUF", "VLM(llama-v)", "VLM(qwen-vl)", "VLM(deepseek-janus-pro)"
    temperature=0.7,
    max_length=2048,
    model_base_url=None,
    model_api_key=None,
    is_ollama=False,
    dtype="auto",
    is_locked=True,
):
    """
    Run inference on a local or api-based LLM model.
    """
    if use_local:
        return run_local_llm_inference(
            model_name_or_path=model_name_or_path,
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_length=max_length,
            dtype=dtype,
            is_locked=is_locked,
            model_type=model_type,
        )
    else:
        return run_api_llm_inference(
            model_name=model_name_or_path,
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_length=max_length,
            model_base_url=model_base_url,
            model_api_key=model_api_key,
            is_ollama=is_ollama,
        )


def run_qwen_image_inference(
    prompt,
    negative_prompt="",
    steps=20,
    cfg=2.5,
    width=1328,
    height=1328,
    model_name="qwen_image_fp8_e4m3fn.safetensors",
    model_dtype="fp8_e4m3fn",
    text_encoder_name="qwen_2.5_vl_7b_fp8_scaled.safetensors",
    vae_name="qwen_image_vae.safetensors",
    lora_1_name="",
    lora_1_strength=1.0,
    seed=None,
    sampler_name="euler",
    scheduler_name="simple",
    denoise=1.0,
    shift=3.1,
    batch_size=1,
):
    with torch.inference_mode():
        if not seed or seed == -1:
            seed = random.randint(-1000000000000000, 1000000000000000)

        # ensure width/height are a multiples of 16
        width = (width // 16) * 16
        height = (height // 16) * 16

        model = load_unet(unet_name=model_name, weight_dtype=model_dtype)[0]

        model_sampling_aura_flow = ModelSamplingAuraFlow()
        model = model_sampling_aura_flow.patch_aura(model=model, shift=shift)[0]

        clip = load_clip(clip_name=text_encoder_name, type="qwen_image")[0]

        clip_text_encode = CLIPTextEncode()

        conditioning = clip_text_encode.encode(clip=clip, text=prompt)[0]
        negative_conditioning = clip_text_encode.encode(
            clip=clip, text=negative_prompt
        )[0]

        empty_sd3_latent_image = EmptySD3LatentImage()
        latent = empty_sd3_latent_image.generate(
            width=width, height=height, batch=batch_size
        )[0]

        ksampler = KSampler()
        samples = ksampler.sample(
            model=model,
            seed=seed,
            steps=steps,
            cfg=cfg,
            sampler_name=sampler_name,
            scheduler=scheduler_name,
            positive=conditioning,
            negative=negative_conditioning,
            latent_image=latent,
            denoise=denoise,
        )[0]

        vae = load_vae(vae_name=vae_name)[0]

        vae_decode = VAEDecode()
        images = vae_decode.decode(vae=vae, latent=samples)[0]

        return tensor_to_pil(images)


def run_chatterbox_tts(
    text,
    model_name="resembleai_default_voice",
    max_new_tokens=4096,  # 4096 tokens is the maximum for Chatterbox TTS
    flow_cfg_scale=0.7,
    exaggeration=0.5,
    temperature=0.8,
    cfg_weight=0.5,
    repetition_penalty=1.2,
    min_p=0.05,
    top_p=1.0,
    seed=None,
    use_watermark=False,
    audio_prompt=None,
):
    if max_new_tokens > 4096:
        raise ValueError("max_new_tokens cannot exceed 4096 for Chatterbox TTS")
    with torch.inference_mode():
        if not seed or seed == -1:
            seed = random.randint(-1000000000000000, 1000000000000000)
        chatterbox_tts = ChatterboxTTS()
        out = chatterbox_tts.synthesize(
            model_pack_name=model_name,
            text=text,
            max_new_tokens=max_new_tokens,
            flow_cfg_scale=flow_cfg_scale,
            exaggeration=exaggeration,
            temperature=temperature,
            cfg_weight=cfg_weight,
            repetition_penalty=repetition_penalty,
            min_p=min_p,
            top_p=top_p,
            seed=seed,
            use_watermark=use_watermark,
            audio_prompt=audio_prompt,
        )[0]
        waveform = out["waveform"]
        sr = out["sample_rate"]

        buf = io.BytesIO()
        waveform_cpu = waveform.cpu()[0]
        current_waveform = (
            torch.mean(waveform_cpu, dim=0)
            if waveform_cpu.shape[0] > 1
            else waveform_cpu.squeeze(0)
        )
        sf.write(buf, current_waveform.numpy().astype(np.float32), sr, format="WAV")
        buf.seek(0)
        audio_data = buf.read()
        buf.close()
    return audio_data


def run_wan_2_2_video_5B_inference(
    prompt,
    negative_prompt="Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards",
    steps=30,
    cfg=5.0,
    scheduler="flowmatch_pusa",
    input_image=None,
    compile_model=False,
    attention_mode="sageattn",
    max_dimension=1024,
    num_frames=121,
    width=768,
    height=768,
    seed=None,
    model_name="wan2.2_ti2v_5B_fp16.safetensors",
    vae_name="wan2.2_vae.safetensors",
    text_encoder_name="umt5-xxl-enc-bf16.safetensors",
    lora_1_name="",
    lora_1_strength=1.0,
    base_precision="bf16",
):
    with torch.inference_mode():
        if not seed or seed == -1:
            seed = random.randint(-1000000000000000, 1000000000000000)

        model = load_wan_video_model(
            model_name=model_name,
            base_precision=base_precision,
            attention_mode=attention_mode,
            compile_model=compile_model,
            lora_1_name=lora_1_name,
            lora_1_strength=lora_1_strength,
        )[0]

        vae = load_wan_video_vae(vae_name=vae_name, precision=base_precision)[0]

        text_encoder = load_wan_video_t5(
            model_name=text_encoder_name,
            precision=base_precision,
        )[0]

        if input_image is not None:
            image_resize_kjv2 = ImageResizeKJv2()
            image, width, height, _ = image_resize_kjv2.resize(
                image=input_image,
                width=max_dimension,
                height=max_dimension,
                keep_proportion="crop",
                upscale_method="lanczos",
                divisible_by=32,
                pad_color="0, 0, 0",
                crop_position="center",
                device="cpu",
                mask=None,
                unique_id=str(uuid.uuid4()),
            )

            wan_video_encode = WanVideoEncode()
            latent = wan_video_encode.encode(
                vae=vae,
                image=image,
                enable_vae_tiling=False,
                tile_x=272,
                tile_y=272,
                tile_stride_x=144,
                tile_stride_y=128,
                noise_aug_strength=0.0,
                latent_strength=1.0,
                mask=None,
            )[0]
        else:
            latent = None

        wan_video_empty_embeds = WanVideoEmptyEmbeds()
        image_embeds = wan_video_empty_embeds.process(
            num_frames=num_frames,
            width=width,
            height=height,
            control_embeds=None,
            extra_latents=latent,
        )[0]

        wan_video_text_encode = WanVideoTextEncode()
        text_embeds = wan_video_text_encode.process(
            positive_prompt=prompt,
            negative_prompt=negative_prompt,
            t5=text_encoder,
            force_offload=True,
            model_to_offload=model,
            use_disk_cache=False,
            device="gpu" if DEVICE == "cuda" else DEVICE,
        )[0]

        wan_video_easy_cache = WanVideoEasyCache()
        cache_args = wan_video_easy_cache.setargs(
            easycache_thresh=0.015,
            start_step=10,
            end_step=-1,
            cache_device="offload_device",
        )[0]

        wan_video_slg = WanVideoSLG()
        slg_args = wan_video_slg.process(
            blocks="7,8,9", start_percent=0.1, end_percent=0.7
        )[0]

        wan_video_experimental_args = WanVideoExperimentalArgs()
        exp_args = wan_video_experimental_args.process(
            video_attention_split_steps="",
            cfg_zero_star=True,
            use_zero_init=False,
            zero_star_steps=0,
            use_fresca=False,
            fresca_scale_low=1.0,
            fresca_scale_high=1.25,
            fresca_freq_cutoff=20,
            use_tcfg=True,
        )[0]

        wan_video_sampler = WanVideoSampler()
        samples, _ = wan_video_sampler.process(
            model=model,
            image_embeds=image_embeds,
            shift=8.0,
            steps=steps,
            cfg=cfg,
            seed=seed,
            scheduler=scheduler,
            riflex_freq_index=0,
            text_embeds=text_embeds,
            force_offload=True,
            samples=None,
            feta_args=None,
            denoise_strength=1.0,
            context_options=None,
            cache_args=cache_args,
            teacache_args=None,
            flowedit_args=None,
            batched_cfg=False,
            slg_args=slg_args,
            rope_function="comfy",
            loop_args=None,
            experimental_args=exp_args,
            sigmas=None,
            unianimate_poses=None,
            fantasytalking_embeds=None,
            uni3c_embeds=None,
            multitalk_embeds=None,
            freeinit_args=None,
            start_step=0,
            end_step=-1,
            add_noise_to_samples=False,
        )

        wan_video_decode = WanVideoDecode()
        images = wan_video_decode.decode(
            vae=vae,
            samples=samples,
            enable_vae_tiling=False,
            tile_x=272,
            tile_y=272,
            tile_stride_x=144,
            tile_stride_y=128,
            normalization="default",
        )[0]

        frames = tensor_to_pil(images)

        buf = io.BytesIO()
        iio.imwrite(
            buf,
            [np.asarray(im) for im in frames],
            fps=24,
            codec="libx264",
            extension=".mp4",
        )
        buf.seek(0)
        video_data = buf.read()
        buf.close()
    return video_data


def run_flux_inpaint(
    input_image,
    input_mask=None,
    input_seed=-1,
    steps=25,
    cfg=2.5,
    neg_cfg=4,
    std_flux_1_cfg=1,
    sampler_name="euler_ancestral",
    scheduler_name="simple",
    denoise=0.97,
    batch_size=1,
    unet_name="flux1-dev-kontext_fp8_scaled.safetensors",
    unet_dtype="fp8_e4m3fn",
    vae_name="ae.safetensors",
    clip_1_name="t5xxl_fp8_e4m3fn_scaled.safetensors",
    clip_2_name="clip_l.safetensors",
    lora_1_name="",
    lora_1_strength=1.0,
    prompt="",
    negative_prompt=None,
    start_negative=0,
    end_negative=0.1,
    resize_dimension=1152,
    do_resize=True,
    cleanup_models=False,
    skip_output=False,
):
    with torch.inference_mode():
        if not input_seed or input_seed == -1:
            seed = random.randint(-1000000000000000, 1000000000000000)
        else:
            seed = input_seed

        if isinstance(input_image, str):
            load_image = LoadImage()
            image, mask = load_image.load_image(
                image=input_image,
            )
            if input_mask:
                mask = (
                    load_image.load_image(image=input_mask)[0]
                    if isinstance(input_mask, str)
                    else input_mask
                )
        else:
            image = input_image
            mask = input_mask if input_mask is not None else torch.zeros_like(image)

        if do_resize:
            ad_image_resize = ADImageResize()
            image = ad_image_resize.execute(
                image=image,
                width=resize_dimension,
                height=resize_dimension,
                interpolation="lanczos",
                method="keep proportion",
                condition="downscale if bigger",
                multiple_of=1,
            )[0]

        get_image_size_and_count = GetImageSizeAndCount()
        image, width, height, count = get_image_size_and_count.getsize(image=image)[
            "result"
        ]

        resize_mask = ResizeMask()
        mask = resize_mask.resize(
            mask=mask,
            width=width,
            height=height,
            keep_proportions=False,
            upscale_method="bicubic",
            crop=False,
        )[0]

        width = (width // 8) * 8
        height = (height // 8) * 8

        image_crop = ImageCrop()
        image = image_crop.crop(image=image, width=width, height=height, x=0, y=0)[0]

        crop_mask = CropMask()
        mask = crop_mask.crop(
            mask=mask,
            width=width,
            height=height,
            x=0,
            y=0,
        )[0]

        grow_mask_with_blur = GrowMaskWithBlur()
        mask = grow_mask_with_blur.expand_mask(
            mask=mask,
            expand=2,
            incremental_expandrate=0,
            tapered_corners=True,
            flip_input=False,
            blur_radius=3.5,
            lerp_alpha=1,
            decay_factor=1,
            fill_holes=False,
        )[0]

        # if mask is all zeros, turn to ones
        if mask.sum() == 0:
            mask = torch.ones_like(mask)

        unet = load_unet(
            unet_name=unet_name,
            weight_dtype=unet_dtype,
        )[0]

        differential_diffusion = DifferentialDiffusion()
        unet = differential_diffusion.apply(
            model=unet,
        )[0]

        model_sampling_flux = ModelSamplingFlux()
        unet = model_sampling_flux.patch(
            model=unet,
            max_shift=1.15,
            base_shift=0.5,
            width=width,
            height=height,
        )[0]

        dual_clip = load_dual_clip(
            clip_name1=clip_1_name, clip_name2=clip_2_name, type="flux"
        )[0]

        if lora_1_name:
            lora_stack = load_lora_stack(
                lora_1_name=lora_1_name,
                lora_1_strength=lora_1_strength,
            )[0]
        else:
            lora_stack = None

        cr_apply_lora_stack = CRApplyLoRAStack()
        unet, clip, _ = cr_apply_lora_stack.apply_lora_stack(
            model=unet, clip=dual_clip, lora_stack=lora_stack
        )

        apply_fb_cache_on_model = ApplyFBCacheOnModel()
        unet = apply_fb_cache_on_model.patch(
            model=unet,
            object_to_patch="diffusion_model",
            residual_diff_threshold=0.14,
            start=0.1,
            end=1.0,
            max_consecutive_cache_hits=5,
        )[0]

        clip_text_encode = CLIPTextEncode()
        conditioning = clip_text_encode.encode(clip=clip, text=prompt)[0]

        flux_guidance = FluxGuidance()
        conditioning = flux_guidance.append(conditioning=conditioning, guidance=cfg)[0]

        vae = load_vae(vae_name=vae_name)[0]

        vae_encode = VAEEncode()
        latent = vae_encode.encode(
            pixels=image,
            vae=vae,
        )[0]

        reference_latent = ReferenceLatent()
        conditioning = reference_latent.append(
            conditioning=conditioning,
            latent=latent,
        )[0]

        negative_conditioning = clip_text_encode.encode(
            clip=clip,
            text=negative_prompt if negative_prompt else "",
        )[0]
        if negative_prompt:
            conditioning_set_timestep_range = ConditioningSetTimestepRange()
            negative_conditioning = conditioning_set_timestep_range.set_range(
                conditioning=negative_conditioning,
                start=start_negative,
                end=end_negative,
            )[0]

            support_empty_uncond = SupportEmptyUncond()
            unet = support_empty_uncond.patch(model=unet, method="from cond")[0]

            skimmed_cfg = SkimmedCFG()
            unet = skimmed_cfg.patch(
                model=unet,
                Skimming_CFG=3.8,
                full_skim_negative=False,
                disable_flipping_filter=False,
            )[0]

        latent2 = vae_encode.encode(
            pixels=image,
            vae=vae,
        )[0]

        set_latent_noise_mask = SetLatentNoiseMask()
        latent2 = set_latent_noise_mask.set_mask(
            samples=latent2,
            mask=mask,
        )[0]

        ksampler = KSampler()
        samples = ksampler.sample(
            model=unet,
            seed=seed,
            steps=steps,
            cfg=std_flux_1_cfg if not negative_prompt else neg_cfg,
            sampler_name=sampler_name,
            scheduler=scheduler_name,
            positive=conditioning,
            negative=negative_conditioning,
            latent_image=latent2,
            denoise=denoise,
        )[0]

        vae_decode = VAEDecode()
        gen_image = vae_decode.decode(vae=vae, samples=samples)[0]

        image_composite_from_mask_batch = ImageCompositeFromMaskBatch()
        composite_image = image_composite_from_mask_batch.execute(
            image_from=image,
            image_to=gen_image,
            mask=mask,
        )[0]

        if skip_output:
            return

        composite_image = tensor_to_pil(composite_image)[0]
        gen_image = tensor_to_pil(gen_image)[0]
        image = tensor_to_pil(image)[0]
        mask = tensor_to_pil(mask)[0]

        if cleanup_models:
            ModelCache().clear()

        soft_empty_cache()
        gc.collect()

        return {
            "composite_image": composite_image,
            "gen_image": gen_image,
            "input_image": image,
            "mask": mask,
        }
