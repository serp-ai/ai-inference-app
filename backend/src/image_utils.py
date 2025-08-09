from typing import List
import base64
import io
from typing import Any, Dict, Optional
from torchvision.transforms.functional import to_pil_image
import torch
import numpy as np
from PIL import Image, ImageOps


def get_mask_tensor(h: int, w: int) -> torch.Tensor:
    """
    Create a default mask tensor with the same height and width as the input image.
    The mask is initialized to all zeros (no inpainting area).
    """
    return torch.zeros((1, h, w), dtype=torch.float32)


def tensor_to_pil(tensor: torch.Tensor) -> List[Image.Image]:
    """
    Convert a tensor or batch of tensors to a list of PIL Images.

    The input can be one of:
        • CHW
        • HWC
        • C H W (channels in middle)
        • NCHW
        • NHWC
        • N C H W (channels in 2nd dim)
    """
    t = tensor.detach().cpu()

    # Ensure we have a batch dimension → (N, …)
    if t.dim() in (2, 3):  # single image (CHW or HWC)
        t = t.unsqueeze(0)
    elif t.dim() not in (3, 4):  # 2-D or ≥5-D tensors aren’t supported
        raise ValueError(
            f"Expected 3- or 4-D tensor (with optional batch dim); got {t.shape}"
        )

    images = []
    for img_t in t:  # iterate over batch
        # Put channels first.
        if img_t.shape[0] in (1, 3, 4):  # already CHW
            pass
        elif img_t.shape[-1] in (1, 3, 4):  # HWC → CHW
            img_t = img_t.permute(2, 0, 1)
        elif img_t.dim() == 3 and img_t.shape[1] in (1, 3, 4):  # H C W → CHW
            img_t = img_t.permute(1, 0, 2)
        else:
            raise ValueError(
                f"Can't find a 1/3/4-channel axis in {img_t.shape}. "
                "Double-check the tensor layout."
            )

        images.append(to_pil_image(img_t))

    return images


def base64_to_image(base64_string: str) -> Image.Image:
    """Convert base64 string to PIL Image."""
    image_data = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_data))
    return image.convert("RGB")


def image_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 string, removing alpha channel."""
    if image.mode != "RGB":
        if image.mode == "RGBA":
            r, g, b, _ = image.split()
            image = Image.merge("RGB", (r, g, b))
        else:
            image = image.convert("RGB")

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


def base64_to_comfy_tensor(base64_string: str) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Convert base64 image to ComfyUI-compatible tensor format.
    Returns (image_tensor, mask_tensor) matching ComfyUI's load_image output.
    """
    # Remove data URL prefix if present (e.g., "data:image/png;base64,")
    if base64_string.startswith("data:"):
        base64_string = base64_string.split(",", 1)[1]

    # Add padding if missing
    missing_padding = len(base64_string) % 4
    if missing_padding:
        base64_string += "=" * (4 - missing_padding)

    # Decode base64 to PIL Image
    image_data = base64.b64decode(base64_string)
    img = Image.open(io.BytesIO(image_data))

    # Apply EXIF orientation
    img = ImageOps.exif_transpose(img)

    # Handle different image modes
    if img.mode == "I":
        img = img.point(lambda i: i * (1 / 255))

    if img.mode == "RGBA":
        # Split RGBA and recombine just RGB channels
        r, g, b, _ = img.split()
        image = Image.merge("RGB", (r, g, b))
    else:
        image = img.convert("RGB")

    # Convert to numpy array and normalize to 0-1 range
    image_array = np.array(image).astype(np.float32) / 255.0

    # Convert to torch tensor with shape [1, H, W, C] (batch dimension added)
    image_tensor = torch.from_numpy(image_array)[None,]

    # Handle alpha channel for mask
    if "A" in img.getbands():
        mask = np.array(img.getchannel("A")).astype(np.float32) / 255.0
        mask = 1.0 - torch.from_numpy(mask)  # Invert mask
    elif img.mode == "P" and "transparency" in img.info:
        mask = np.array(img.convert("RGBA").getchannel("A")).astype(np.float32) / 255.0
        mask = 1.0 - torch.from_numpy(mask)
    else:
        # Create default mask with same height/width as image
        h, w = image_array.shape[:2]
        mask = torch.zeros((h, w), dtype=torch.float32, device="cpu")

    # Add batch dimension to mask
    mask_tensor = mask.unsqueeze(0)

    return (image_tensor, mask_tensor)


def base64_to_mask_tensor(base64_string: str) -> torch.Tensor:
    """
    Convert base64 mask image to ComfyUI-compatible mask tensor.
    Expects a grayscale or RGB mask where white=1 (inpaint area).
    """
    # Remove data URL prefix if present
    if base64_string.startswith("data:"):
        base64_string = base64_string.split(",", 1)[1]

    # Add padding if missing
    missing_padding = len(base64_string) % 4
    if missing_padding:
        base64_string += "=" * (4 - missing_padding)

    # Decode base64 to PIL Image
    image_data = base64.b64decode(base64_string)
    img = Image.open(io.BytesIO(image_data))

    # Convert to grayscale and get mask values
    if img.mode != "L":
        img = img.convert("L")

    # Convert to numpy array and normalize to 0-1 range
    mask_array = np.array(img).astype(np.float32) / 255.0

    # Convert to torch tensor with batch dimension
    mask_tensor = torch.from_numpy(mask_array).unsqueeze(0)

    return mask_tensor


def save_image(image_tensor, filename: str) -> str:
    """Save ComfyUI image tensor to file and return filename."""
    # Convert tensor to PIL Image (adjust this based on your tensor format)
    if hasattr(image_tensor, "cpu"):
        # If tensor is on GPU, move to CPU
        image_array = image_tensor.cpu().numpy()
    else:
        image_array = image_tensor

    # Assuming image_array is in format [batch, height, width, channels]
    if len(image_array.shape) == 4:
        image_array = image_array[0]  # Take first image from batch

    # Convert to 0-255 range if needed
    if image_array.max() <= 1.0:
        image_array = (image_array * 255).astype("uint8")

    image = Image.fromarray(image_array)
    filepath = IMAGES_DIR / filename
    image.save(filepath, "PNG")
    return filename
