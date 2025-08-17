MODEL_MAP = {
    # Flux Kontext
    "ae.safetensors": (
        "vae/ae.safetensors",
        "https://huggingface.co/Comfy-Org/Lumina_Image_2.0_Repackaged/resolve/main/split_files/vae/ae.safetensors",
    ),
    "t5xxl_fp8_e4m3fn_scaled.safetensors": (
        "text_encoders/t5xxl_fp8_e4m3fn_scaled.safetensors",
        "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp8_e4m3fn_scaled.safetensors",
    ),
    "clip_l.safetensors": (
        "text_encoders/clip_l.safetensors",
        "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors",
    ),
    "flux1-dev-kontext_fp8_scaled.safetensors": (
        "diffusion_models/flux1-dev-kontext_fp8_scaled.safetensors",
        "https://huggingface.co/Comfy-Org/flux1-kontext-dev_ComfyUI/resolve/main/split_files/diffusion_models/flux1-dev-kontext_fp8_scaled.safetensors",
    ),
    # Flux Kontext LoRAs
    "clothes_removal_lora_v1.safetensors": (
        "loras/clothes_removal_lora_v1.safetensors",
        "",
    ),
    # WAN 2.2 (5B)
    "umt5-xxl-enc-fp8_e4m3fn.safetensors": (
        "text_encoders/umt5-xxl-enc-fp8_e4m3fn.safetensors",
        "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/umt5-xxl-enc-fp8_e4m3fn.safetensors",
    ),
    "umt5-xxl-enc-bf16.safetensors": (
        "text_encoders/umt5-xxl-enc-bf16.safetensors",
        "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/umt5-xxl-enc-bf16.safetensors",
    ),
    "wan2.2_ti2v_5B_fp16.safetensors": (
        "diffusion_models/wan2.2_ti2v_5B_fp16.safetensors",
        "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/diffusion_models/wan2.2_ti2v_5B_fp16.safetensors",
    ),
    "wan2.2_vae.safetensors": (
        "vae/wan2.2_vae.safetensors",
        "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/vae/wan2.2_vae.safetensors",
    ),
    # Qwen-Image
    "qwen_2.5_vl_7b_fp8_scaled.safetensors": (
        "text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors",
        "https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors",
    ),
    "qwen_image_fp8_e4m3fn.safetensors": (
        "diffusion_models/qwen_image_fp8_e4m3fn.safetensors",
        "https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/diffusion_models/qwen_image_fp8_e4m3fn.safetensors",
    ),
    "qwen_2.5_vl_7b.safetensors": (
        "text_encoders/qwen_2.5_vl_7b.safetensors",
        "https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/text_encoders/qwen_2.5_vl_7b.safetensors",
    ),
    "qwen_image_bf16.safetensors": (
        "diffusion_models/qwen_image_bf16.safetensors",
        "https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/diffusion_models/qwen_image_bf16.safetensors",
    ),
    "qwen_image_vae.safetensors": (
        "vae/qwen_image_vae.safetensors",
        "https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/vae/qwen_image_vae.safetensors",
    ),
}
