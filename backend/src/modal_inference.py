import io
import random
import time
from pathlib import Path
from typing import Optional

import modal

MINUTES = 60


app = modal.App("example-text-to-image")

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install(
        "curl",
        "ffmpeg",
        "ninja-build",
        "git",
        "git-lfs",
        "wget",
        "aria2",
        "vim",
        "libgl1",
        "libglib2.0-0",
        "build-essential",
        "gcc",
    )
    .run_commands(
        "git clone https://github.com/comfyanonymous/ComfyUI.git comfyui",
        "git -C /comfyui checkout 1e638a140b2f459595fafc73ade5ea5b4024d4b4",
        "pip install --no-cache-dir -r /comfyui/requirements.txt",
        "mkdir -p /comfyui/custom_nodes",
        # ComfyUI-Manager
        "git clone https://github.com/ltdrdata/ComfyUI-Manager.git /comfyui/custom_nodes/ComfyUI-Manager",
        "git -C /comfyui/custom_nodes/ComfyUI-Manager checkout ab684cdc993ffaa24a2a7f1abc22b9f7ba1a2998",
        "pip install --no-cache-dir -r /comfyui/custom_nodes/ComfyUI-Manager/requirements.txt",
        # KJNodes
        "git clone https://github.com/kijai/ComfyUI-KJNodes.git /comfyui/custom_nodes/ComfyUI-KJNodes",
        "git -C /comfyui/custom_nodes/ComfyUI-KJNodes checkout a6b867b63a29ca48ddb15c589e17a9f2d8530d57",
        "pip install --no-cache-dir -r /comfyui/custom_nodes/ComfyUI-KJNodes/requirements.txt",
        # TeaCache
        "git clone https://github.com/welltop-cn/ComfyUI-TeaCache.git /comfyui/custom_nodes/ComfyUI-TeaCache",
        "git -C /comfyui/custom_nodes/ComfyUI-TeaCache checkout 91dff8e31684ca70a5fda309611484402d8fa192",
        "pip install --no-cache-dir -r /comfyui/custom_nodes/ComfyUI-TeaCache/requirements.txt",
        # Essentials
        "git clone https://github.com/cubiq/ComfyUI_essentials.git /comfyui/custom_nodes/ComfyUI_essentials",
        "git -C /comfyui/custom_nodes/ComfyUI_essentials checkout 9d9f4bedfc9f0321c19faf71855e228c93bd0dc9",
        "pip install --no-cache-dir -r /comfyui/custom_nodes/ComfyUI_essentials/requirements.txt",
        # Easy Use
        "git clone https://github.com/yolain/ComfyUI-Easy-Use.git /comfyui/custom_nodes/ComfyUI-Easy-Use",
        "git -C /comfyui/custom_nodes/ComfyUI-Easy-Use checkout 717092a3ceb51c474b5b3f77fc188979f0db9d67",
        "pip install --no-cache-dir -r /comfyui/custom_nodes/ComfyUI-Easy-Use/requirements.txt",
        # ComfyUI-mxToolkit
        "git clone https://github.com/Smirnov75/ComfyUI-mxToolkit.git /comfyui/custom_nodes/ComfyUI-mxToolkit",
        "git -C /comfyui/custom_nodes/ComfyUI-mxToolkit checkout 7f7a0e584f12078a1c589645d866ae96bad0cc35",
        # Comfyroll
        "git clone https://github.com/Suzie1/ComfyUI_Comfyroll_CustomNodes.git /comfyui/custom_nodes/ComfyUI_Comfyroll_CustomNodes",
        "git -C /comfyui/custom_nodes/ComfyUI_Comfyroll_CustomNodes checkout d78b780ae43fcf8c6b7c6505e6ffb4584281ceca",
        # rgthree
        "git clone https://github.com/rgthree/rgthree-comfy.git /comfyui/custom_nodes/rgthree-comfy",
        "git -C /comfyui/custom_nodes/rgthree-comfy checkout 944d5353a1b0a668f40844018c3dc956b95a67d7",
        "pip install --no-cache-dir -r /comfyui/custom_nodes/rgthree-comfy/requirements.txt",
        # was node suite
        "git clone https://github.com/WASasquatch/was-node-suite-comfyui.git /comfyui/custom_nodes/was-node-suite-comfyui",
        "git -C /comfyui/custom_nodes/was-node-suite-comfyui checkout ea935d1044ae5a26efa54ebeb18fe9020af49a45",
        "pip install --no-cache-dir -r /comfyui/custom_nodes/was-node-suite-comfyui/requirements.txt",
        # Impact Pack
        "git clone https://github.com/ltdrdata/ComfyUI-Impact-Pack.git /comfyui/custom_nodes/ComfyUI-Impact-Pack",
        "git -C /comfyui/custom_nodes/ComfyUI-Impact-Pack checkout e22f68fb97115b24ddf35d3d8801cc99fdb8af8d",
        "pip install --no-cache-dir -r /comfyui/custom_nodes/ComfyUI-Impact-Pack/requirements.txt",
        # Inpaint Nodes
        "git clone https://github.com/Acly/comfyui-inpaint-nodes.git /comfyui/custom_nodes/comfyui-inpaint-nodes",
        "git -C /comfyui/custom_nodes/comfyui-inpaint-nodes checkout 726e16ff2742be285b3da78b73333ba6227ad234",
        # ComfyUI-Inpaint-CropAndStitch
        "git clone https://github.com/lquesada/ComfyUI-Inpaint-CropAndStitch.git /comfyui/custom_nodes/ComfyUI-Inpaint-CropAndStitch",
        "git -C /comfyui/custom_nodes/ComfyUI-Inpaint-CropAndStitch checkout b432b2411cbb7d3192d35953bd3aafea05a0e245",
        # Video Helper Suite
        "git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git /comfyui/custom_nodes/ComfyUI-VideoHelperSuite",
        "git -C /comfyui/custom_nodes/ComfyUI-VideoHelperSuite checkout 330bce6c3c0d47ebdedcc0348d9ab355707b7523",
        "pip install --no-cache-dir -r /comfyui/custom_nodes/ComfyUI-VideoHelperSuite/requirements.txt",
        # Wavespeed
        "git clone https://github.com/chengzeyi/Comfy-WaveSpeed.git /comfyui/custom_nodes/Comfy-WaveSpeed",
        "git -C /comfyui/custom_nodes/Comfy-WaveSpeed checkout 16ec6f344f8cecbbf006d374043f85af22b7a51d",
        # Efficiency Nodes
        "git clone https://github.com/jags111/efficiency-nodes-comfyui.git /comfyui/custom_nodes/efficiency-nodes-comfyui",
        "git -C /comfyui/custom_nodes/efficiency-nodes-comfyui checkout f0971b5553ead8f6e66bb99564431e2590cd3981",
        "pip install --no-cache-dir -r /comfyui/custom_nodes/efficiency-nodes-comfyui/requirements.txt",
        # ComfyUI-Lora-Manager
        "git clone https://github.com/willmiao/ComfyUI-Lora-Manager.git /comfyui/custom_nodes/ComfyUI-Lora-Manager",
        "git -C /comfyui/custom_nodes/ComfyUI-Lora-Manager checkout c0eff2bb5ef8967aab06d6952aa39bd73091585b",
        "pip install --no-cache-dir -r /comfyui/custom_nodes/ComfyUI-Lora-Manager/requirements.txt",
        # ComfyUI_ExtraModels
        "git clone https://github.com/city96/ComfyUI_ExtraModels.git /comfyui/custom_nodes/ComfyUI_ExtraModels",
        "git -C /comfyui/custom_nodes/ComfyUI_ExtraModels checkout 92f556ed4d3bec1a3f16117d2de10f195c36d68e",
        "pip install --no-cache-dir -r /comfyui/custom_nodes/ComfyUI_ExtraModels/requirements.txt",
        # ComfyUI-Image-Saver
        "git clone https://github.com/alexopus/ComfyUI-Image-Saver.git /comfyui/custom_nodes/ComfyUI-Image-Saver",
        "git -C /comfyui/custom_nodes/ComfyUI-Image-Saver checkout 8c5668322484df7f12b7474667e5f07aa5a1b463",
        "pip install --no-cache-dir -r /comfyui/custom_nodes/ComfyUI-Image-Saver/requirements.txt",
        # ComfyUI-Custom-Scripts
        "git clone https://github.com/pythongosssss/ComfyUI-Custom-Scripts.git /comfyui/custom_nodes/ComfyUI-Custom-Scripts",
        "git -C /comfyui/custom_nodes/ComfyUI-Custom-Scripts checkout aac13aa7ce35b07d43633c3bbe654a38c00d74f5",
        # ComfyUI-LogicUtils
        "git clone https://github.com/aria1th/ComfyUI-LogicUtils.git /comfyui/custom_nodes/ComfyUI-LogicUtils",
        "git -C /comfyui/custom_nodes/ComfyUI-LogicUtils checkout 5992e91930b9e00c5afd687eb406f6795b0d198f",
        # Skimmed_CFG
        "git clone https://github.com/Extraltodeus/Skimmed_CFG.git /comfyui/custom_nodes/Skimmed_CFG",
        "git -C /comfyui/custom_nodes/Skimmed_CFG checkout 2712803a8b721665d43d5aeb4430e5ac0e931091",
        # pre_cfg_comfy_nodes_for_ComfyUI
        "git clone https://github.com/Extraltodeus/pre_cfg_comfy_nodes_for_ComfyUI.git /comfyui/custom_nodes/pre_cfg_comfy_nodes_for_ComfyUI",
        "git -C /comfyui/custom_nodes/pre_cfg_comfy_nodes_for_ComfyUI checkout 967b1816462a3f9834887a6295e2daea838d0b62",
        # ComfyScript
        "git clone https://github.com/Chaoses-Ib/ComfyScript.git /comfyui/custom_nodes/ComfyScript",
        "git -C /comfyui/custom_nodes/ComfyScript checkout 6bfc76020a1a884a70aed353edd2f57395f4d045",
        "pip install --no-cache-dir -r /comfyui/custom_nodes/ComfyScript/requirements.txt",
        "pip install -e /comfyui/custom_nodes/ComfyScript",
        # Server/aux libs
        "pip install --no-cache-dir requests websocket-client onnxruntime-gpu triton",
    )
    .run_commands(
        "pip install fastapi[standard]",
    )
    .run_commands(
        "mkdir -p /comfyui/models/vae",
        "mkdir -p /comfyui/models/text_encoders",
        "mkdir -p /comfyui/models/diffusion_models",
        "mkdir -p /comfyui/models/loras",
        "wget -q -O /comfyui/models/vae/ae.safetensors https://huggingface.co/Comfy-Org/Lumina_Image_2.0_Repackaged/resolve/main/split_files/vae/ae.safetensors",
        "wget -q -O /comfyui/models/text_encoders/t5xxl_fp8_e4m3fn_scaled.safetensors https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp8_e4m3fn_scaled.safetensors",
        "wget -q -O /comfyui/models/text_encoders/clip_l.safetensors https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors",
        "wget -q -O /comfyui/models/diffusion_models/flux1-dev-kontext_fp8_scaled.safetensors https://huggingface.co/Comfy-Org/flux1-kontext-dev_ComfyUI/resolve/main/split_files/diffusion_models/flux1-dev-kontext_fp8_scaled.safetensors",
    )
    .env(
        {
            "PIP_PREFER_BINARY": "1",
            "PYTHONUNBUFFERED": "1",
            "CMAKE_BUILD_PARALLEL_LEVEL": "8",
        }
    )
    .add_local_file(
        local_path="E:/Python/ComfyUI/models/loras/clothes_removal_lora_v1.safetensors",
        remote_path="/comfyui/models/loras/clothes_removal_lora_v1.safetensors",
        copy=True,
    )
    .run_commands("pip install GPUtil")
    # ComfyUI-Addoor
    .run_commands(
        "git clone https://github.com/Eagle-CN/ComfyUI-Addoor /comfyui/custom_nodes/ComfyUI-Addoor",
        "git -C /comfyui/custom_nodes/ComfyUI-Addoor checkout d51659f66696e9a89c7f6adbf159c1f074742b7a",
        "pip install --no-cache-dir -r /comfyui/custom_nodes/ComfyUI-Addoor/requirements.txt",
    )
    .add_local_python_source("comfy_utils")
    .add_local_python_source("image_utils")
    .add_local_python_source("api_handler")
)


@app.cls(
    image=image,
    gpu="A10G",
    timeout=600,
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},
    max_containers=1,
)
class ComfyUIServer:
    @modal.enter(snap=True)
    def _warmup(self):
        from api_handler import (
            app as fastapi_app,
            run_flux_inpaint,
            base64_to_comfy_tensor,
        )
        import requests
        import base64

        self.app = fastapi_app

        # initialize models
        print("🌡️ Warming up ComfyUI server...")
        response = requests.get(
            "https://raw.githubusercontent.com/cat-milk/Anime-Girls-Holding-Programming-Books/master/Python/Davinci_Python_magic.png?raw=true"
        )
        img_bytes = response.content
        base64_string = base64.b64encode(img_bytes).decode("ascii")
        image_tensor, _ = base64_to_comfy_tensor(base64_string)
        run_flux_inpaint(
            input_image=image_tensor,
            input_mask=None,
            steps=1,
            skip_output=True,
        )

    @modal.asgi_app()
    def fastapi_app(self):
        return self.app
