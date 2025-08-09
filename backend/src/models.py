"""
Pydantic models for API request/response schemas
"""

from typing import Optional
from pydantic import BaseModel, Field


class InpaintingRequest(BaseModel):
    """Request model for image inpainting generation"""

    prompt: str = Field(..., description="Text prompt for image generation")
    negative_prompt: Optional[str] = Field(
        None, description="Negative prompt to avoid certain features"
    )
    input_image: str = Field(..., description="Base64 encoded input image")
    mask_image: Optional[str] = Field(None, description="Base64 encoded mask image")
    num_steps: int = Field(25, ge=1, le=100, description="Number of diffusion steps")
    guidance: float = Field(2.5, ge=1.0, le=10.0, description="Guidance scale")
    strength: float = Field(0.97, ge=0.1, le=1.0, description="Transformation strength")
    seed: Optional[int] = Field(-1, description="Random seed for reproducibility")
    resize_dimension: Optional[int] = Field(
        1152, ge=64, le=2048, description="Resize dimension"
    )
    do_resize: bool = Field(False, description="Whether to resize input image")


class InpaintingClothesRemovalRequest(BaseModel):
    """Request model for clothes removal image inpainting generation"""

    input_image: str = Field(..., description="Base64 encoded input image")
    mask_image: Optional[str] = Field(None, description="Base64 encoded mask image")
    num_steps: int = Field(25, ge=1, le=100, description="Number of diffusion steps")
    guidance: float = Field(2.5, ge=1.0, le=10.0, description="Guidance scale")
    strength: float = Field(0.97, ge=0.1, le=1.0, description="Transformation strength")
    seed: Optional[int] = Field(-1, description="Random seed for reproducibility")
    resize_dimension: Optional[int] = Field(
        1152, ge=64, le=2048, description="Resize dimension"
    )
    do_resize: bool = Field(False, description="Whether to resize input image")


class InpaintingResponse(BaseModel):
    """Response model for successful image generation"""

    composite_image: str = Field(..., description="Base64 encoded composite result")
    gen_image: str = Field(..., description="Base64 encoded generated image")
    input_image: str = Field(..., description="Base64 encoded input image")
    mask: str = Field(..., description="Base64 encoded mask used")


class ErrorResponse(BaseModel):
    """Response model for API errors"""

    success: bool = Field(False, description="Success status")
    error: str = Field(..., description="Error message")
    details: Optional[str] = Field(None, description="Additional error details")


class HealthResponse(BaseModel):
    """Response model for health check"""

    status: str = Field(..., description="Health status")
    message: str = Field(..., description="Status message")


class ClearCacheResponse(BaseModel):
    """Response model for cache clearing operation"""

    success: bool = Field(..., description="Success status")
    message: str = Field(..., description="Operation result message")


class LogEntry(BaseModel):
    """Model for individual log entries"""

    timestamp: str = Field(..., description="Log timestamp")
    level: str = Field(..., description="Log level")
    message: str = Field(..., description="Log message")


class LogsResponse(BaseModel):
    """Response model for logs endpoint"""

    logs: list[LogEntry] = Field(..., description="List of log entries")
    total: int = Field(..., description="Total number of logs available")


class SystemMetrics(BaseModel):
    """Response model for system metrics"""

    cpu: dict = Field(..., description="CPU metrics")
    memory: dict = Field(..., description="Memory metrics")
    disk: dict = Field(..., description="Disk metrics")
    gpus: list[dict] = Field(..., description="GPU metrics")
    timestamp: float = Field(..., description="Metrics timestamp")


class QwenImageRequest(BaseModel):
    """Request model for Qwen image generation"""

    prompt: str = Field(..., description="Text prompt for image generation")
    negative_prompt: Optional[str] = Field("", description="Negative prompt to avoid certain features")
    steps: int = Field(20, ge=1, le=100, description="Number of diffusion steps")
    cfg: float = Field(2.5, ge=1.0, le=10.0, description="CFG scale")
    width: int = Field(1328, ge=64, le=2048, description="Image width")
    height: int = Field(1328, ge=64, le=2048, description="Image height")
    seed: Optional[int] = Field(None, description="Random seed for reproducibility")
    sampler_name: str = Field("euler", description="Sampler method")
    scheduler_name: str = Field("simple", description="Scheduler method")
    shift: float = Field(3.1, description="Model sampling shift parameter")


class QwenImageResponse(BaseModel):
    """Response model for successful Qwen image generation"""

    images: list[str] = Field(..., description="List of base64 encoded generated images")


class VideoRequest(BaseModel):
    """Request model for text-to-video generation"""

    prompt: str = Field(..., description="Text prompt for video generation")
    negative_prompt: Optional[str] = Field(
        "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards",
        description="Negative prompt to avoid certain features",
    )
    steps: int = Field(30, ge=1, le=100, description="Number of diffusion steps")
    cfg: float = Field(5.0, ge=1.0, le=20.0, description="CFG scale")
    input_image: Optional[str] = Field(
        None, description="Base64 encoded input image for image-to-video"
    )
    max_dimension: int = Field(
        1024, ge=256, le=2048, description="Maximum dimension for resizing"
    )
    num_frames: int = Field(
        121, ge=1, le=121, description="Number of frames to generate"
    )
    width: int = Field(768, ge=64, le=2048, description="Video width")
    height: int = Field(768, ge=64, le=2048, description="Video height")
    seed: Optional[int] = Field(-1, description="Random seed for reproducibility")
