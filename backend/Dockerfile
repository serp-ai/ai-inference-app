# Build arguments
ARG BASE_IMAGE="nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04"

FROM ${BASE_IMAGE}

ARG REQUIREMENTS_FILE="requirements-cuda.txt"
ARG INCLUDE_FLUX_KONTEXT="false"
ARG INCLUDE_WAN_2_2_5B="false"
ARG INCLUDE_WAN_2_2_5B_QUANTIZED_TEXT_ENCODER="false"
ARG INCLUDE_QWEN_IMAGE="false"
ARG USE_QUANTIZED_QWEN_IMAGE="true"
ARG INSTALL_SAGEATTENTION="false"
ARG PYTHON_VERSION="3.10"


RUN echo "Requirements file: ${REQUIREMENTS_FILE}" && \
    echo "Include Flux Kontext: ${INCLUDE_FLUX_KONTEXT}" && \
    echo "Include Wan 2.2 5B: ${INCLUDE_WAN_2_2_5B}" && \
    echo "Install SageAttention: ${INSTALL_SAGEATTENTION}"

# TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0;12.0"

ENV DEBIAN_FRONTEND=noninteractive \
   PIP_PREFER_BINARY=1 \
   PYTHONUNBUFFERED=1 \
   CMAKE_BUILD_PARALLEL_LEVEL=8 \
   TORCH_CUDA_ARCH_LIST="8.9"

WORKDIR /

# Install Python and make it the default
RUN apt-get update && \
    # Add deadsnakes PPA for newer Python versions
    if [ "${PYTHON_VERSION}" != "3.10" ]; then \
        apt-get install -y software-properties-common && \
        add-apt-repository ppa:deadsnakes/ppa && \
        apt-get update; \
    fi && \
    apt-get install -y --no-install-recommends \
       python${PYTHON_VERSION} python${PYTHON_VERSION}-dev python${PYTHON_VERSION}-venv \
       curl ffmpeg ninja-build git git-lfs wget aria2 vim libgl1 libglib2.0-0 build-essential gcc && \
    # Install distutils for older Python versions
    if [ "${PYTHON_VERSION}" = "3.10" ] || [ "${PYTHON_VERSION}" = "3.9" ]; then \
        apt-get install -y python${PYTHON_VERSION}-distutils; \
    fi && \
    # Remove system pip/setuptools to avoid conflicts
    apt-get remove -y python3-pip python3-setuptools || true && \
    ln -sf /usr/bin/python${PYTHON_VERSION} /usr/bin/python && \
    ln -sf /usr/bin/python${PYTHON_VERSION} /usr/bin/python3 && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python${PYTHON_VERSION} && \
    ln -sf /usr/local/bin/pip /usr/bin/pip && \
    ln -sf /usr/local/bin/pip /usr/bin/pip3 && \
    # Install clean setuptools and wheel via pip
    pip install --no-cache-dir --upgrade pip setuptools wheel && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/comfyanonymous/ComfyUI.git comfyui && \
    git -C /comfyui checkout 32691b16f4e1a897e461e77f9d6dceba2d6f0cd1

WORKDIR /comfyui
RUN pip install --no-cache-dir -r requirements.txt

# Custom nodes: ComfyUI-Manager
RUN mkdir -p custom_nodes && \
    git clone https://github.com/ltdrdata/ComfyUI-Manager.git custom_nodes/ComfyUI-Manager && \
    git -C custom_nodes/ComfyUI-Manager checkout ab684cdc993ffaa24a2a7f1abc22b9f7ba1a2998 && \
    pip install --no-cache-dir -r custom_nodes/ComfyUI-Manager/requirements.txt

# Custom nodes: KJNodes
RUN git clone https://github.com/kijai/ComfyUI-KJNodes.git custom_nodes/ComfyUI-KJNodes && \
    git -C custom_nodes/ComfyUI-KJNodes checkout a6b867b63a29ca48ddb15c589e17a9f2d8530d57 && \
    pip install --no-cache-dir -r custom_nodes/ComfyUI-KJNodes/requirements.txt

# Custom nodes: TeaCache
RUN git clone https://github.com/welltop-cn/ComfyUI-TeaCache.git custom_nodes/ComfyUI-TeaCache && \
    git -C custom_nodes/ComfyUI-TeaCache checkout 91dff8e31684ca70a5fda309611484402d8fa192 && \
    pip install --no-cache-dir -r custom_nodes/ComfyUI-TeaCache/requirements.txt

# Custom nodes: Essentials
RUN git clone https://github.com/cubiq/ComfyUI_essentials.git custom_nodes/ComfyUI_essentials && \
    git -C custom_nodes/ComfyUI_essentials checkout 9d9f4bedfc9f0321c19faf71855e228c93bd0dc9 && \
    pip install --no-cache-dir -r custom_nodes/ComfyUI_essentials/requirements.txt

# Custom nodes: Easy Use
RUN git clone https://github.com/yolain/ComfyUI-Easy-Use.git custom_nodes/ComfyUI-Easy-Use && \
    git -C custom_nodes/ComfyUI-Easy-Use checkout 717092a3ceb51c474b5b3f77fc188979f0db9d67 && \
    pip install --no-cache-dir -r custom_nodes/ComfyUI-Easy-Use/requirements.txt

# Custom nodes: ComfyUI-mxToolkit
RUN git clone https://github.com/Smirnov75/ComfyUI-mxToolkit.git custom_nodes/ComfyUI-mxToolkit && \
    git -C custom_nodes/ComfyUI-mxToolkit checkout 7f7a0e584f12078a1c589645d866ae96bad0cc35

# Custom nodes: Comfyroll
RUN git clone https://github.com/Suzie1/ComfyUI_Comfyroll_CustomNodes.git custom_nodes/ComfyUI_Comfyroll_CustomNodes && \
    git -C custom_nodes/ComfyUI_Comfyroll_CustomNodes checkout d78b780ae43fcf8c6b7c6505e6ffb4584281ceca

# Custom nodes: rgthree
RUN git clone https://github.com/rgthree/rgthree-comfy.git custom_nodes/rgthree-comfy && \
    git -C custom_nodes/rgthree-comfy checkout 944d5353a1b0a668f40844018c3dc956b95a67d7 && \
    pip install --no-cache-dir -r custom_nodes/rgthree-comfy/requirements.txt

# Custom nodes: was node suite
RUN git clone https://github.com/WASasquatch/was-node-suite-comfyui.git custom_nodes/was-node-suite-comfyui && \
    git -C custom_nodes/was-node-suite-comfyui checkout ea935d1044ae5a26efa54ebeb18fe9020af49a45 && \
    pip install --no-cache-dir -r custom_nodes/was-node-suite-comfyui/requirements.txt

# Custom nodes: Impact Pack
RUN git clone https://github.com/ltdrdata/ComfyUI-Impact-Pack.git custom_nodes/ComfyUI-Impact-Pack && \
    git -C custom_nodes/ComfyUI-Impact-Pack checkout e22f68fb97115b24ddf35d3d8801cc99fdb8af8d && \
    pip install --no-cache-dir -r custom_nodes/ComfyUI-Impact-Pack/requirements.txt

# Custom nodes: Inpaint Nodes
RUN git clone https://github.com/Acly/comfyui-inpaint-nodes.git custom_nodes/comfyui-inpaint-nodes && \
    git -C custom_nodes/comfyui-inpaint-nodes checkout 726e16ff2742be285b3da78b73333ba6227ad234

# Custom nodes: ComfyUI-Inpaint-CropAndStitch
RUN git clone https://github.com/lquesada/ComfyUI-Inpaint-CropAndStitch.git custom_nodes/ComfyUI-Inpaint-CropAndStitch && \
    git -C custom_nodes/ComfyUI-Inpaint-CropAndStitch checkout b432b2411cbb7d3192d35953bd3aafea05a0e245

# Custom nodes: Video Helper Suite
RUN git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git custom_nodes/ComfyUI-VideoHelperSuite && \
    git -C custom_nodes/ComfyUI-VideoHelperSuite checkout 330bce6c3c0d47ebdedcc0348d9ab355707b7523 && \
    pip install --no-cache-dir -r custom_nodes/ComfyUI-VideoHelperSuite/requirements.txt

# Custom nodes: Wavespeed
RUN git clone https://github.com/chengzeyi/Comfy-WaveSpeed.git custom_nodes/Comfy-WaveSpeed && \
    git -C custom_nodes/Comfy-WaveSpeed checkout 16ec6f344f8cecbbf006d374043f85af22b7a51d

# Custom nodes: Efficiency Nodes
RUN git clone https://github.com/jags111/efficiency-nodes-comfyui.git custom_nodes/efficiency-nodes-comfyui && \
    git -C custom_nodes/efficiency-nodes-comfyui checkout f0971b5553ead8f6e66bb99564431e2590cd3981 && \
    pip install --no-cache-dir -r custom_nodes/efficiency-nodes-comfyui/requirements.txt

# Custom nodes: ComfyUI-Lora-Manager
RUN git clone https://github.com/willmiao/ComfyUI-Lora-Manager.git custom_nodes/ComfyUI-Lora-Manager && \
    git -C custom_nodes/ComfyUI-Lora-Manager checkout e6b94c7b214525095045d2df694e5ecc25335f4e && \
    pip install --no-cache-dir -r custom_nodes/ComfyUI-Lora-Manager/requirements.txt

# Custom nodes: ComfyUI_ExtraModels
RUN git clone https://github.com/city96/ComfyUI_ExtraModels.git custom_nodes/ComfyUI_ExtraModels && \
    git -C custom_nodes/ComfyUI_ExtraModels checkout 92f556ed4d3bec1a3f16117d2de10f195c36d68e && \
    pip install --no-cache-dir -r custom_nodes/ComfyUI_ExtraModels/requirements.txt

# Custom nodes: ComfyUI-Image-Saver
RUN git clone https://github.com/alexopus/ComfyUI-Image-Saver.git custom_nodes/ComfyUI-Image-Saver && \
    git -C custom_nodes/ComfyUI-Image-Saver checkout 8c5668322484df7f12b7474667e5f07aa5a1b463 && \
    pip install --no-cache-dir -r custom_nodes/ComfyUI-Image-Saver/requirements.txt

# Custom nodes: ComfyUI-Custom-Scripts
RUN git clone https://github.com/pythongosssss/ComfyUI-Custom-Scripts.git custom_nodes/ComfyUI-Custom-Scripts && \
    git -C custom_nodes/ComfyUI-Custom-Scripts checkout aac13aa7ce35b07d43633c3bbe654a38c00d74f5

# Custom nodes: ComfyUI-LogicUtils
RUN git clone https://github.com/aria1th/ComfyUI-LogicUtils.git custom_nodes/ComfyUI-LogicUtils && \
    git -C custom_nodes/ComfyUI-LogicUtils checkout 5992e91930b9e00c5afd687eb406f6795b0d198f

# Custom nodes: ComfyUI-Skimmed_CFG
RUN git clone https://github.com/Extraltodeus/Skimmed_CFG.git custom_nodes/Skimmed_CFG && \
    git -C custom_nodes/Skimmed_CFG checkout 2712803a8b721665d43d5aeb4430e5ac0e931091

# Custom nodes: pre_cfg_comfy_nodes_for_ComfyUI
RUN git clone https://github.com/Extraltodeus/pre_cfg_comfy_nodes_for_ComfyUI.git custom_nodes/pre_cfg_comfy_nodes_for_ComfyUI && \
    git -C custom_nodes/pre_cfg_comfy_nodes_for_ComfyUI checkout 967b1816462a3f9834887a6295e2daea838d0b62

# Custom nodes: ComfyUI-Addoor
RUN git clone https://github.com/Eagle-CN/ComfyUI-Addoor /comfyui/custom_nodes/ComfyUI-Addoor && \
    git -C /comfyui/custom_nodes/ComfyUI-Addoor checkout d51659f66696e9a89c7f6adbf159c1f074742b7a && \
    pip install --no-cache-dir -r custom_nodes/ComfyUI-Addoor/requirements.txt

# Custom nodes: ComfyUI-WanVideoWrapper
RUN git clone https://github.com/kijai/ComfyUI-WanVideoWrapper.git custom_nodes/ComfyUI-WanVideoWrapper && \
    git -C custom_nodes/ComfyUI-WanVideoWrapper checkout bacb2a2053000b9bfa3bee89e067994405dc685e && \
    pip install --no-cache-dir -r custom_nodes/ComfyUI-WanVideoWrapper/requirements.txt

# Custom nodes: ComfyUI-Chatterbox
RUN git clone https://github.com/wildminder/ComfyUI-Chatterbox.git custom_nodes/ComfyUI-Chatterbox && \
    git -C custom_nodes/ComfyUI-Chatterbox checkout a64e0a14aad8d014e48eaf1d42cf0bbfe592f864 && \
    pip install --no-cache-dir -r custom_nodes/ComfyUI-Chatterbox/requirements.txt

# Custom nodes: ComfyScript
RUN git clone https://github.com/Chaoses-Ib/ComfyScript.git custom_nodes/ComfyScript && \
    git -C custom_nodes/ComfyScript checkout 6bfc76020a1a884a70aed353edd2f57395f4d045 && \
    pip install --no-cache-dir -r custom_nodes/ComfyScript/requirements.txt && \
    pip install -e custom_nodes/ComfyScript

# Copy and install hardware-specific requirements
COPY requirements-base.txt /tmp/requirements-base.txt
COPY requirements-cuda.txt /tmp/requirements-cuda.txt
COPY requirements-cuda-optimized.txt /tmp/requirements-cuda-optimized.txt
COPY requirements-amd.txt /tmp/requirements-amd.txt
COPY requirements-cpu.txt /tmp/requirements-cpu.txt
COPY requirements-osx.txt /tmp/requirements-osx.txt
# uninstall existing torch packages to avoid conflicts
RUN pip uninstall -y torch torchvision torchaudio
RUN pip install --no-cache-dir -r /tmp/requirements-base.txt
RUN if [ "${REQUIREMENTS_FILE}" = "requirements-cuda.txt" ]; then \
    pip install --no-cache-dir -r /tmp/requirements-cuda.txt; \
    elif [ "${REQUIREMENTS_FILE}" = "requirements-cuda-optimized.txt" ]; then \
    pip install --no-cache-dir -r /tmp/requirements-cuda-optimized.txt; \
    elif [ "${REQUIREMENTS_FILE}" = "requirements-amd.txt" ]; then \
    pip install --no-cache-dir -r /tmp/requirements-amd.txt; \
    elif [ "${REQUIREMENTS_FILE}" = "requirements-osx.txt" ]; then \
    pip install --no-cache-dir -r /tmp/requirements-osx.txt; \
    else \
    pip install --no-cache-dir -r /tmp/requirements-cpu.txt; \
    fi

RUN mkdir -p models/vae
RUN mkdir -p models/text_encoders
RUN mkdir -p models/diffusion_models
RUN mkdir -p models/loras

RUN if [ "${INCLUDE_FLUX_KONTEXT}" = "true" ]; then \
    wget -q -O models/vae/ae.safetensors https://huggingface.co/Comfy-Org/Lumina_Image_2.0_Repackaged/resolve/main/split_files/vae/ae.safetensors && \
    wget -q -O models/text_encoders/t5xxl_fp8_e4m3fn_scaled.safetensors https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp8_e4m3fn_scaled.safetensors && \
    wget -q -O models/text_encoders/clip_l.safetensors https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors && \
    wget -q -O models/diffusion_models/flux1-dev-kontext_fp8_scaled.safetensors https://huggingface.co/Comfy-Org/flux1-kontext-dev_ComfyUI/resolve/main/split_files/diffusion_models/flux1-dev-kontext_fp8_scaled.safetensors; \
    fi

RUN if [ "${INCLUDE_WAN_2_2_5B}" = "true" ]; then \
    if [ "${INCLUDE_WAN_2_2_5B_QUANTIZED_TEXT_ENCODER}" = "true" ]; then \
        wget -q -O models/text_encoders/umt5-xxl-enc-fp8_e4m3fn.safetensors https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/umt5-xxl-enc-fp8_e4m3fn.safetensors; \
    else \
        wget -q -O models/text_encoders/umt5-xxl-enc-bf16.safetensors https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/umt5-xxl-enc-bf16.safetensors; \
    fi && \
    wget -q -O models/diffusion_models/wan2.2_ti2v_5B_fp16.safetensors https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/diffusion_models/wan2.2_ti2v_5B_fp16.safetensors && \
    wget -q -O models/vae/wan2.2_vae.safetensors https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/vae/wan2.2_vae.safetensors; \
    fi

RUN if [ "${INCLUDE_QWEN_IMAGE}" = "true" ]; then \
    if [ "${USE_QUANTIZED_QWEN_IMAGE}" = "true" ]; then \
        wget -q -O models/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors && \
        wget -q -O models/diffusion_models/qwen_image_fp8_e4m3fn.safetensors https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/diffusion_models/qwen_image_fp8_e4m3fn.safetensors; \
    else \
        wget -q -O models/text_encoders/qwen_2.5_vl_7b.safetensors https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/text_encoders/qwen_2.5_vl_7b.safetensors && \
        wget -q -O models/diffusion_models/qwen_image_bf16.safetensors https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/diffusion_models/qwen_image_bf16.safetensors; \
    fi && \
    wget -q -O models/vae/qwen_image_vae.safetensors https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/vae/qwen_image_vae.safetensors; \
    fi

COPY models/loras/ models/loras/

WORKDIR /

ENV EXT_PARALLEL=1
ENV MAX_JOBS=1

# Optional SageAttention installation (For NVIDIA GPUs)
RUN if [ "${INSTALL_SAGEATTENTION}" = "true" ]; then \
    git clone https://github.com/thu-ml/SageAttention.git && \
    cd SageAttention && \
    sed -i '/compute_capabilities = set()/a\
env_cc = os.environ.get("TORCH_CUDA_ARCH_LIST", "")\n\
if not compute_capabilities and env_cc:\n\
    compute_capabilities = {cc.strip() for cc in env_cc.replace(",", ";").split(";") if cc.strip() and cc in SUPPORTED_ARCHS}' setup.py && \
    sed -i 's/--threads=8/--threads=2/' setup.py && \
    python -m pip install --no-build-isolation .; \
    fi

COPY src/ /src/

COPY start.sh /start.sh
RUN chmod +x /start.sh

EXPOSE 8000
CMD ["/start.sh"]