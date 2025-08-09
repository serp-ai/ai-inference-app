#!/usr/bin/env python3
"""
FastAPI application for ComfyUI based scripts
"""

import os
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor

import psutil
import GPUtil
import uvicorn
from typing import Optional
from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response

# Local imports
from models import (
    InpaintingRequest,
    InpaintingClothesRemovalRequest,
    InpaintingResponse,
    VideoRequest,
    QwenImageRequest,
    QwenImageResponse,
    ErrorResponse,
    ClearCacheResponse,
    LogsResponse,
    SystemMetrics,
)
from auth import verify_api_key, is_auth_enabled
from logging_utils import setup_log_capture, get_logs
from comfy_utils import (
    run_flux_inpaint,
    clear_comfyui_model_cache,
    run_wan_2_2_video_5B_inference,
    run_chatterbox_tts,
    run_qwen_image_inference,
)
from image_utils import (
    base64_to_comfy_tensor,
    base64_to_mask_tensor,
    image_to_base64,
    get_mask_tensor,
)
from audio_utils import load_bytes

# Initialize FastAPI app
app = FastAPI(
    title="AI Inference API",
    description="Production-ready API for AI inferencing",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Configure CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup log capture
setup_log_capture()

# Create thread pool for heavy inference tasks
thread_pool = ThreadPoolExecutor(max_workers=1)  # Only 1 worker for GPU constraint


def run_video_inference_sync(
    prompt,
    negative_prompt,
    steps,
    cfg,
    input_image_tensor,
    max_dimension,
    num_frames,
    width,
    height,
    seed,
):
    """Synchronous wrapper for video inference to run in thread pool"""
    # Force stdout flush to ensure logs appear
    import sys

    print("🎬 Video inference starting...", flush=True)
    sys.stdout.flush()

    try:
        result = run_wan_2_2_video_5B_inference(
            prompt=prompt,
            negative_prompt=negative_prompt,
            steps=steps,
            cfg=cfg,
            input_image=input_image_tensor,
            max_dimension=max_dimension,
            num_frames=num_frames,
            width=width,
            height=height,
            seed=seed,
        )
        print("✅ Video inference completed!", flush=True)
        sys.stdout.flush()
        return result
    except Exception as e:
        print(f"❌ Video inference failed: {str(e)}", flush=True)
        sys.stderr.flush()
        raise


def run_image_inference_sync(
    input_image,
    input_mask,
    steps,
    cfg,
    denoise,
    input_seed,
    prompt=None,
    negative_prompt=None,
    lora_1_name=None,
    resize_dimension=1152,
    do_resize=True,
):
    """Synchronous wrapper for image inference to run in thread pool"""
    # Force stdout flush to ensure logs appear
    import sys

    print("🖼️ Image inference starting...", flush=True)
    sys.stdout.flush()

    try:
        result = run_flux_inpaint(
            input_image=input_image,
            input_mask=input_mask,
            steps=steps,
            cfg=cfg,
            denoise=denoise,
            input_seed=input_seed,
            prompt=prompt or "",
            negative_prompt=negative_prompt,
            lora_1_name=lora_1_name or "",
            resize_dimension=resize_dimension,
            do_resize=do_resize,
        )
        print("✅ Image inference completed!", flush=True)
        sys.stdout.flush()
        return result
    except Exception as e:
        print(f"❌ Image inference failed: {str(e)}", flush=True)
        sys.stderr.flush()
        raise


def run_tts_inference_sync(
    text,
    max_new_tokens,
    flow_cfg_scale,
    exaggeration,
    temperature,
    cfg_weight,
    repetition_penalty,
    min_p,
    top_p,
    seed,
    use_watermark,
    audio_prompt,
):
    """Synchronous wrapper for TTS inference to run in thread pool"""
    # Force stdout flush to ensure logs appear
    import sys

    print("🎤 TTS inference starting...", flush=True)
    sys.stdout.flush()

    try:
        result = run_chatterbox_tts(
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
        )
        print("✅ TTS inference completed!", flush=True)
        sys.stdout.flush()
        return result
    except Exception as e:
        print(f"❌ TTS inference failed: {str(e)}", flush=True)
        sys.stderr.flush()
        raise


def run_qwen_image_inference_sync(
    prompt,
    negative_prompt,
    steps,
    cfg,
    width,
    height,
    seed,
    sampler_name,
    scheduler_name,
    shift,
):
    """Synchronous wrapper for Qwen image inference to run in thread pool"""
    # Force stdout flush to ensure logs appear
    import sys

    print("🖼️ Qwen image inference starting...", flush=True)
    sys.stdout.flush()

    try:
        result = run_qwen_image_inference(
            prompt=prompt,
            negative_prompt=negative_prompt,
            steps=steps,
            cfg=cfg,
            width=width,
            height=height,
            seed=seed,
            sampler_name=sampler_name,
            scheduler_name=scheduler_name,
            shift=shift,
        )
        print("✅ Qwen image inference completed!", flush=True)
        sys.stdout.flush()
        return result
    except Exception as e:
        print(f"❌ Qwen image inference failed: {str(e)}", flush=True)
        sys.stderr.flush()
        raise


@app.get("/api/logs", response_model=LogsResponse)
async def get_terminal_logs(since: int = 0, _: bool = Depends(verify_api_key)):
    """Get captured terminal logs since specified index"""
    logs_data = get_logs(since)
    return LogsResponse(logs=logs_data["logs"], total=logs_data["total"])


@app.get("/api/system/metrics", response_model=SystemMetrics)
async def get_system_metrics(_: bool = Depends(verify_api_key)):
    """Get comprehensive system performance metrics"""
    try:
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=0.1)
        cpu_count = psutil.cpu_count()
        cpu_freq = psutil.cpu_freq()

        # Memory metrics
        memory = psutil.virtual_memory()

        # Disk metrics
        disk = psutil.disk_usage("/")

        # GPU metrics
        gpus = []
        try:
            gpu_list = GPUtil.getGPUs()
            for gpu in gpu_list:
                gpus.append(
                    {
                        "id": gpu.id,
                        "name": gpu.name,
                        "load": round(gpu.load * 100, 1),
                        "memory_used": round(gpu.memoryUsed, 1),
                        "memory_total": round(gpu.memoryTotal, 1),
                        "memory_percent": round(
                            (gpu.memoryUsed / gpu.memoryTotal) * 100, 1
                        ),
                        "temperature": gpu.temperature,
                    }
                )
        except Exception:
            # GPU metrics unavailable (CPU-only deployment)
            pass

        return SystemMetrics(
            cpu={
                "percent": round(cpu_percent, 1),
                "count": cpu_count,
                "frequency": round(cpu_freq.current, 1) if cpu_freq else None,
            },
            memory={
                "percent": round(memory.percent, 1),
                "used_gb": round(memory.used / (1024**3), 2),
                "total_gb": round(memory.total / (1024**3), 2),
                "available_gb": round(memory.available / (1024**3), 2),
            },
            disk={
                "percent": round((disk.used / disk.total) * 100, 1),
                "used_gb": round(disk.used / (1024**3), 2),
                "total_gb": round(disk.total / (1024**3), 2),
                "free_gb": round(disk.free / (1024**3), 2),
            },
            gpus=gpus,
            timestamp=time.time(),
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to retrieve system metrics: {str(e)}"
        )


@app.post(
    "/api/video/generate",
    responses={200: {"content": {"video/mp4": {}}}, 500: {"model": ErrorResponse}},
)
async def generate_video(request: VideoRequest, _: bool = Depends(verify_api_key)):
    """
    Generate video using WAN 2.2 Video 5B model

    This endpoint processes text prompts (with optional first frame input) to generate videos
    using the WAN 2.2 Video 5B model via ComfyUI backend.
    """
    try:
        print("🚀 Starting video generation...")
        print(
            f"⚙️ Parameters: Steps={request.steps}, CFG={request.cfg}, Frames={request.num_frames}"
        )
        print(
            f"📐 Dimensions: {request.width}x{request.height}, Max Dimension={request.max_dimension}"
        )

        # Process input image if provided
        input_image_tensor = None
        if request.input_image:
            print("🖼️ Processing input image...")
            input_image_tensor, _ = base64_to_comfy_tensor(request.input_image)

        print("🤖 Loading AI models and starting video inference...")
        loop = asyncio.get_event_loop()
        video_data = await loop.run_in_executor(
            thread_pool,
            run_video_inference_sync,
            request.prompt,
            request.negative_prompt,
            request.steps,
            request.cfg,
            input_image_tensor,
            request.max_dimension,
            request.num_frames,
            request.width,
            request.height,
            request.seed,
        )

        print("✅ Video generation completed successfully!")

        # Return video as binary response
        return Response(
            content=video_data,
            media_type="video/mp4",
            headers={"Content-Disposition": "attachment; filename=generated_video.mp4"},
        )

    except Exception as e:
        print(f"❌ Video generation failed: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Video inference failed: {str(e)}"
        ) from e


@app.post(
    "/api/inpainting/generate-clothes-removal",
    response_model=InpaintingResponse,
    responses={500: {"model": ErrorResponse}},
)
async def generate_clothes_removal_inpainting(
    request: InpaintingClothesRemovalRequest, _: bool = Depends(verify_api_key)
):
    """
    Generate inpainted image using ComfyUI Flux Kontext model

    This endpoint processes input images and masks to generate clothes-removed variants
    using state-of-the-art diffusion models via ComfyUI backend.
    """
    try:
        print("🚀 Starting image generation...")
        print(
            f"⚙️ Parameters: Steps={request.num_steps}, Guidance={request.guidance}, Strength={request.strength}"
        )

        # Convert base64 images to ComfyUI tensor format
        print("🖼️ Processing input image...")
        image_tensor, _ = base64_to_comfy_tensor(request.input_image)

        # Process mask if provided
        if request.mask_image:
            print("🎭 Processing mask image...")
            mask_tensor = base64_to_mask_tensor(request.mask_image)
        else:
            print("🎭 No mask provided, using default...")
            h, w = image_tensor.shape[1:3]  # Get height, width from image tensor
            mask_tensor = get_mask_tensor(h, w)

        print("🤖 Loading AI models and starting inference...")
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            thread_pool,
            run_image_inference_sync,
            image_tensor,
            mask_tensor,
            request.num_steps,
            request.guidance,
            request.strength,
            request.seed,
            "GETNAKED",
            None,
            "clothes_removal_lora_v1.safetensors",
            request.resize_dimension,
            request.do_resize,
        )

        # Convert PIL images back to base64
        print("🎨 Converting results to base64...")
        response = InpaintingResponse(
            composite_image=image_to_base64(result["composite_image"]),
            gen_image=image_to_base64(result["gen_image"]),
            input_image=image_to_base64(result["input_image"]),
            mask=image_to_base64(result["mask"]),
        )

        print("✅ Image generation completed successfully!")
        return response

    except Exception as e:
        print(f"❌ Generation failed: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Inference failed: {str(e)}"
        ) from e


@app.post(
    "/api/inpainting/generate",
    response_model=InpaintingResponse,
    responses={500: {"model": ErrorResponse}},
)
async def generate_inpainting(
    request: InpaintingRequest, _: bool = Depends(verify_api_key)
):
    """
    Generate inpainted image using ComfyUI Flux Kontext model

    This endpoint processes input images and masks to generate inpainted variants
    using state-of-the-art diffusion models via ComfyUI backend.
    """
    try:
        print("🚀 Starting image generation...")
        print(
            f"⚙️ Parameters: Steps={request.num_steps}, Guidance={request.guidance}, Strength={request.strength}"
        )

        # Convert base64 images to ComfyUI tensor format
        print("🖼️ Processing input image...")
        image_tensor, _ = base64_to_comfy_tensor(request.input_image)

        # Process mask if provided
        if request.mask_image:
            print("🎭 Processing mask image...")
            mask_tensor = base64_to_mask_tensor(request.mask_image)
        else:
            print("🎭 No mask provided, using default...")
            h, w = image_tensor.shape[1:3]  # Get height, width from image tensor
            mask_tensor = get_mask_tensor(h, w)

        print("🤖 Loading AI models and starting inference...")
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            thread_pool,
            run_image_inference_sync,
            image_tensor,
            mask_tensor,
            request.num_steps,
            request.guidance,
            request.strength,
            request.seed,
            request.prompt,
            request.negative_prompt,
            "",
            request.resize_dimension,
            request.do_resize,
        )

        # Convert PIL images back to base64
        print("🎨 Converting results to base64...")
        response = InpaintingResponse(
            composite_image=image_to_base64(result["composite_image"]),
            gen_image=image_to_base64(result["gen_image"]),
            input_image=image_to_base64(result["input_image"]),
            mask=image_to_base64(result["mask"]),
        )

        print("✅ Image generation completed successfully!")
        return response

    except Exception as e:
        print(f"❌ Generation failed: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Inference failed: {str(e)}"
        ) from e


@app.post(
    "/api/tts/generate",
    responses={200: {"content": {"audio/wav": {}}}, 500: {"model": ErrorResponse}},
)
async def generate_tts(
    text: str = Form(...),
    max_new_tokens: int = Form(4096),
    flow_cfg_scale: float = Form(0.7),
    exaggeration: float = Form(0.5),
    temperature: float = Form(0.8),
    cfg_weight: float = Form(0.5),
    repetition_penalty: float = Form(1.2),
    min_p: float = Form(0.05),
    top_p: float = Form(1.0),
    seed: int = Form(-1),
    use_watermark: bool = Form(False),
    prompt_wav: Optional[UploadFile] = File(None),
    _: bool = Depends(verify_api_key),
):
    """
    Generate speech from text using Chatterbox TTS model

    This endpoint processes text prompts (with optional audio prompt) to generate speech
    using the Chatterbox TTS model via ComfyUI backend.
    """
    try:
        print("🚀 Starting TTS generation...")
        print(f"📝 Text: {text[:100]}{'...' if len(text) > 100 else ''}")
        print(
            f"⚙️ Parameters: Temperature={temperature}, " f"Max Tokens={max_new_tokens}"
        )

        # Process audio prompt if provided
        audio_prompt = None
        if prompt_wav is not None:
            print("🎵 Processing audio prompt...")
            data = await prompt_wav.read()
            wav, sr = load_bytes(data)
            # Add batch dimension that Chatterbox expects: [C,T] -> [1,C,T]
            wav = wav.unsqueeze(0)
            audio_prompt = {"waveform": wav, "sample_rate": sr}

        print("🤖 Loading AI models and starting TTS inference...")
        loop = asyncio.get_event_loop()
        audio_data = await loop.run_in_executor(
            thread_pool,
            run_tts_inference_sync,
            text,
            max_new_tokens,
            flow_cfg_scale,
            exaggeration,
            temperature,
            cfg_weight,
            repetition_penalty,
            min_p,
            top_p,
            seed,
            use_watermark,
            audio_prompt,
        )

        print("✅ TTS generation completed successfully!")

        # Return audio as binary response
        return Response(
            content=audio_data,
            media_type="audio/wav",
            headers={
                "Content-Disposition": "attachment; filename=generated_speech.wav"
            },
        )

    except Exception as e:
        print(f"❌ TTS generation failed: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"TTS inference failed: {str(e)}"
        ) from e


@app.post(
    "/api/qwen-image/generate",
    response_model=QwenImageResponse,
    responses={500: {"model": ErrorResponse}},
)
async def generate_qwen_image(
    request: QwenImageRequest, _: bool = Depends(verify_api_key)
):
    """
    Generate image using Qwen image generation model

    This endpoint processes text prompts to generate images
    using the Qwen image generation model via ComfyUI backend.
    """
    try:
        print("🚀 Starting Qwen image generation...")
        print(
            f"📝 Prompt: {request.prompt[:100]}{'...' if len(request.prompt) > 100 else ''}"
        )
        print(
            f"⚙️ Parameters: Steps={request.steps}, CFG={request.cfg}, "
            f"Size={request.width}x{request.height}"
        )

        print("🤖 Loading AI models and starting Qwen image inference...")
        loop = asyncio.get_event_loop()
        result_images = await loop.run_in_executor(
            thread_pool,
            run_qwen_image_inference_sync,
            request.prompt,
            request.negative_prompt,
            request.steps,
            request.cfg,
            request.width,
            request.height,
            request.seed,
            request.sampler_name,
            request.scheduler_name,
            request.shift,
        )

        # Convert PIL images to base64
        print("🎨 Converting results to base64...")
        images_b64 = [image_to_base64(img) for img in result_images]

        response = QwenImageResponse(images=images_b64)

        print("✅ Qwen image generation completed successfully!")
        return response

    except Exception as e:
        print(f"❌ Qwen image generation failed: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Qwen image inference failed: {str(e)}"
        ) from e


@app.get("/health")
async def health_check():
    """Health check endpoint for load balancers and monitoring"""
    return {"status": "healthy", "message": "API is running"}


@app.post(
    "/api/models/clear-cache",
    response_model=ClearCacheResponse,
    responses={500: {"model": ErrorResponse}},
)
async def clear_model_cache(_: bool = Depends(verify_api_key)):
    """Clear ComfyUI model cache to free GPU memory"""
    try:
        clear_comfyui_model_cache()
        return ClearCacheResponse(
            success=True, message="Model cache cleared successfully"
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                success=False, error="Failed to clear cache", details=str(e)
            ).dict(),
        )


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up thread pool on shutdown"""
    thread_pool.shutdown(wait=True)


if __name__ == "__main__":
    # Configuration from environment
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))

    # Display authentication status
    if is_auth_enabled():
        print("🔒 API Key authentication enabled")
    else:
        print("🔓 API Key authentication disabled (set API_KEY env var to enable)")

    print(f"Starting NSFW Clothes Remover API on {host}:{port}")

    # Start server with production settings
    uvicorn.run(
        app,
        host=host,
        port=port,
        workers=1,  # Single worker for GPU constraint
        log_level="info",
        access_log=True,
        loop="asyncio",
    )
