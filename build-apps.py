#!/usr/bin/env python3
"""
Build system for creating modular Docker images based on app configuration
"""

import os
import sys
import yaml
import subprocess
import re
import ast
from pathlib import Path
from typing import Dict, List, Any


def load_config() -> Dict[str, Any]:
    """Load the apps configuration"""
    with open("apps.yaml", "r") as f:
        return yaml.safe_load(f)


def extract_route_functions(api_handler_path: str) -> Dict[str, str]:
    """Extract route functions from existing api_handler.py using AST"""
    with open(api_handler_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Parse the Python file
    tree = ast.parse(content)
    routes = {}

    # Get source lines for extracting original code
    lines = content.split("\n")

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
            # Check if this function has FastAPI route decorators
            for decorator in node.decorator_list:
                if isinstance(decorator, ast.Call):
                    # Check if it's app.post, app.get, etc.
                    if (
                        isinstance(decorator.func, ast.Attribute)
                        and isinstance(decorator.func.value, ast.Name)
                        and decorator.func.value.id == "app"
                        and decorator.func.attr in ["post", "get", "put", "delete"]
                    ):

                        # Extract the route path from the first argument
                        if decorator.args and isinstance(
                            decorator.args[0], ast.Constant
                        ):
                            route_path = decorator.args[0].value

                            # Extract the original source code for this function
                            start_line = node.lineno - 1
                            end_line = (
                                node.end_lineno
                                if hasattr(node, "end_lineno")
                                else len(lines)
                            )

                            # Find decorator start - go back to find all decorators
                            decorator_start = start_line
                            # Go back and find the FIRST decorator for this function
                            while decorator_start > 0:
                                prev_line = lines[decorator_start - 1].strip()
                                if (
                                    prev_line.startswith("@")
                                    or prev_line == ""
                                    or prev_line.endswith((",", ")", "}"))
                                ):
                                    decorator_start -= 1
                                else:
                                    break

                            # For end_line, if we don't have it from AST, find it manually
                            if not hasattr(node, "end_lineno") or end_line >= len(
                                lines
                            ):
                                # Find the end by looking for the next function/class/decorator at same indentation
                                func_line = next(
                                    i
                                    for i in range(start_line, len(lines))
                                    if lines[i]
                                    .strip()
                                    .startswith(("def ", "async def "))
                                )
                                func_indent = len(lines[func_line]) - len(
                                    lines[func_line].lstrip()
                                )

                                end_line = len(lines)
                                for i in range(func_line + 1, len(lines)):
                                    line = lines[i]
                                    if line.strip() == "":
                                        continue
                                    current_indent = len(line) - len(line.lstrip())
                                    if current_indent <= func_indent and (
                                        line.strip().startswith(
                                            (
                                                "@",
                                                "def ",
                                                "async def ",
                                                "class ",
                                                "if __name__",
                                            )
                                        )
                                    ):
                                        end_line = i
                                        break

                            function_source = "\n".join(lines[decorator_start:end_line])
                            routes[route_path] = function_source
                            break

    print(f"🔍 Found {len(routes)} routes: {list(routes.keys())}")
    return routes


def extract_imports_and_setup(api_handler_path: str) -> tuple[str, str]:
    """Extract imports and app setup from existing api_handler.py"""
    with open(api_handler_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Extract everything before the first route definition
    first_route_match = re.search(r"@app\.(post|get|put|delete)", content)
    if first_route_match:
        setup_code = content[: first_route_match.start()].strip()
    else:
        setup_code = content

    # Extract main block if it exists
    main_match = re.search(r'if __name__ == "__main__":.*', content, re.DOTALL)
    main_code = (
        main_match.group(0)
        if main_match
        else """if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)"""
    )

    return setup_code, main_code


def generate_app_specific_handler(
    app_name: str, api_routes: List[str], existing_handler_path: str
) -> str:
    """Generate app-specific API handler by extracting only needed routes"""

    # Extract existing routes and setup
    all_routes = extract_route_functions(existing_handler_path)
    setup_code, main_code = extract_imports_and_setup(existing_handler_path)

    # Update the FastAPI app title in setup code
    setup_code = re.sub(
        r'title="[^"]*"', f'title="AI Inference API - {app_name.title()}"', setup_code
    )
    setup_code = re.sub(
        r'description="[^"]*"',
        f'description="AI inference API for {app_name}"',
        setup_code,
    )

    # Get only the routes we need for this app
    needed_routes = []
    for route_path in api_routes:
        if route_path in all_routes:
            needed_routes.append(all_routes[route_path])
        else:
            print(f"⚠️  Route {route_path} not found in {existing_handler_path}")

    # Combine everything
    handler_content = f"""{setup_code}

# Route handlers for {app_name}
{chr(10).join(needed_routes)}

{main_code}
"""

    return handler_content


def generate_app_dockerfile(app_name: str, base_dockerfile_path: str) -> str:
    """Generate app-specific Dockerfile that uses the custom handler"""

    with open(base_dockerfile_path, "r", encoding="utf-8") as f:
        base_content = f.read()

    # Modify the COPY src/ command to use the app-specific handler
    modified_content = base_content.replace(
        "COPY src/ /src/",
        f"""COPY src/ /src/
COPY src/api_handler_{app_name}.py /src/api_handler.py""",
    )

    return modified_content


def generate_frontend_dockerfile(
    app_name: str, pages: List[str], base_dockerfile_path: str
) -> str:
    """Generate frontend Dockerfile that only builds specified pages"""

    with open(base_dockerfile_path, "r", encoding="utf-8") as f:
        base_content = f.read()

    # Insert page filtering after COPY . .
    first_page = pages[0] if pages else None

    # Create explicit remove commands for pages not in the list
    remove_commands = []
    for page in pages:
        remove_commands.append(f'echo "Keeping {page}.vue"')

    filter_commands = f"""COPY . .

# Filter pages for {app_name} app - remove unwanted pages
RUN cd /app/pages && ls -la && \\
    {" && ".join([f'rm -f {page}.vue' for page in ['img2img', 'txt2img', 'video', 'tts', 'nsfw-clothes-remover'] if page not in pages])} && \\
    echo "Pages after filtering:" && ls -la"""

    if first_page:
        filter_commands += f'''

# Rename first page to index.vue
RUN cd /app/pages && \\
    mv "{first_page}.vue" "index.vue" && \\
    echo "Renamed {first_page}.vue to index.vue"'''

    filter_commands += """

RUN npm install"""

    modified_content = base_content.replace(
        "COPY . .\nRUN npm install", filter_commands
    )

    return modified_content


def build_app_images(
    app_name: str, config: Dict[str, Any], push: bool = False, platform: str = "cuda"
):
    """Build both backend and frontend images for an app"""
    print(f"🚀 Building {app_name} app images...")

    # Generate app-specific API handler
    existing_handler = "backend/src/api_handler.py"
    api_handler_content = generate_app_specific_handler(
        app_name, config["api_routes"], existing_handler
    )

    # Generate app-specific Dockerfiles
    backend_dockerfile_content = generate_app_dockerfile(app_name, "backend/Dockerfile")
    frontend_dockerfile_content = generate_frontend_dockerfile(
        app_name, config["pages"], "frontend/Dockerfile"
    )

    # Write temporary files
    backend_handler_path = f"backend/src/api_handler_{app_name}.py"
    backend_dockerfile_path = f"backend/Dockerfile.{app_name}"
    frontend_dockerfile_path = f"frontend/Dockerfile.{app_name}"

    with open(backend_handler_path, "w", encoding="utf-8") as f:
        f.write(api_handler_content)

    with open(backend_dockerfile_path, "w", encoding="utf-8") as f:
        f.write(backend_dockerfile_content)

    with open(frontend_dockerfile_path, "w", encoding="utf-8") as f:
        f.write(frontend_dockerfile_content)

    # Configure platform-specific build args
    platform_configs = {
        "cuda": {
            "base_image": "nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04",
            "requirements": "requirements-cuda.txt",
            "tag_suffix": "cuda",
            "platform": "linux/amd64",
        },
        "cuda128": {
            "base_image": "nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04",
            "requirements": "requirements-cuda-optimized.txt",
            "tag_suffix": "cuda128",
            "platform": "linux/amd64",
            "python_version": "3.12",
            "install_sageattention": "true",
            "use_torch_compile": "true",
            "compute_capability": ["8.0", "8.6", "8.9", "9.0", "12.0"],
        },
        "amd": {
            "base_image": "rocm/pytorch:rocm6.2.3_ubuntu22.04_py3.10_pytorch_release_2.3.0",
            "requirements": "requirements-amd.txt",
            "tag_suffix": "amd",
            "platform": "linux/amd64",
        },
        "osx": {
            "base_image": "ubuntu:22.04",
            "requirements": "requirements-osx.txt",
            "tag_suffix": "osx",
            "platform": "linux/arm64",
        },
        "cpu": {
            "base_image": "ubuntu:22.04",
            "requirements": "requirements-cpu.txt",
            "tag_suffix": "cpu",
            "platform": "linux/amd64",
        },
    }

    platform_config = platform_configs.get(platform, platform_configs["cuda"])

    # Build backend
    backend_tag = f"{config['image_tag']}-backend:{platform_config['tag_suffix']}"
    backend_build_args = [
        "docker",
        "buildx",
        "build",
        "-f",
        backend_dockerfile_path,
        "-t",
        backend_tag,
        "--platform",
        platform_config["platform"],
        "--build-arg",
        f"BASE_IMAGE={platform_config['base_image']}",
        "--build-arg",
        f"REQUIREMENTS_FILE={platform_config['requirements']}",
        "--build-arg",
        "INCLUDE_FLUX_KONTEXT=false",
        "--build-arg",
        "INCLUDE_WAN_2_2_5B=false",
        "--build-arg",
        "INCLUDE_QWEN_IMAGE=false",
    ]

    # Add optional platform-specific args
    if platform_config.get("python_version"):
        backend_build_args.extend(
            ["--build-arg", f"PYTHON_VERSION={platform_config['python_version']}"]
        )
    if platform_config.get("install_sageattention"):
        backend_build_args.extend(
            [
                "--build-arg",
                f"INSTALL_SAGEATTENTION={platform_config['install_sageattention']}",
            ]
        )
    if platform_config.get("use_torch_compile"):
        backend_build_args.extend(
            ["--build-arg", f"USE_TORCH_COMPILE={platform_config['use_torch_compile']}"]
        )

    if push:
        backend_build_args.append("--push")
    backend_build_args.append("backend/")

    # Build frontend
    frontend_tag = f"{config['image_tag']}-frontend:{platform_config['tag_suffix']}"
    frontend_build_args = [
        "docker",
        "buildx",
        "build",
        "-f",
        frontend_dockerfile_path,
        "-t",
        frontend_tag,
        "--platform",
        platform_config["platform"],
    ]

    if push:
        frontend_build_args.append("--push")
    frontend_build_args.append("frontend/")

    # Execute builds
    print(f"Building backend: {backend_tag}")
    if config.get("compute_capability"):
        for cc in config["compute_capability"]:
            backend_build_args_ = backend_build_args.copy()
            backend_build_args_.extend(["--build-arg", f"TORCH_CUDA_ARCH_LIST={cc}"])
            backend_tag_ = f"{backend_tag}-cc{cc.replace('.', '')}"
            print(f"Building backend with compute capability {cc}: {backend_tag_}")
            backend_build_args_[6] = backend_tag_  # Update tag in args
            backend_result = subprocess.run(backend_build_args_)
    else:
        backend_result = subprocess.run(backend_build_args)

    print(f"Building frontend: {frontend_tag}")
    frontend_result = subprocess.run(frontend_build_args)

    # # Cleanup temp files
    for temp_file in [
        backend_handler_path,
        backend_dockerfile_path,
        frontend_dockerfile_path,
    ]:
        if os.path.exists(temp_file):
            os.remove(temp_file)

    success = backend_result.returncode == 0 and frontend_result.returncode == 0

    if success:
        print(f"✅ Successfully built {app_name} images")
        return [backend_tag, frontend_tag]
    else:
        print(f"❌ Failed to build {app_name} images")
        return []


def main():
    """Main build function"""
    # Parse arguments
    push = "--push" in sys.argv
    if push:
        sys.argv.remove("--push")

    platform = "cuda"  # default
    if "--platform" in sys.argv:
        platform_idx = sys.argv.index("--platform")
        if platform_idx + 1 < len(sys.argv):
            platform = sys.argv[platform_idx + 1]
            # Remove platform args
            sys.argv.pop(platform_idx)  # remove --platform
            sys.argv.pop(platform_idx)  # remove platform value

    if len(sys.argv) > 1:
        target_apps = sys.argv[1:]
    else:
        config = load_config()
        target_apps = list(config["apps"].keys())

    config = load_config()
    all_built_tags = []

    for app_name in target_apps:
        if app_name not in config["apps"]:
            print(f"❌ App '{app_name}' not found in config")
            continue

        app_config = config["apps"][app_name]

        built_tags = build_app_images(app_name, app_config, push, platform)
        all_built_tags.extend(built_tags)

    print(f"\n✅ Build complete! Built {len(all_built_tags)} images:")
    for tag in all_built_tags:
        print(f"  - {tag}")


if __name__ == "__main__":
    main()
