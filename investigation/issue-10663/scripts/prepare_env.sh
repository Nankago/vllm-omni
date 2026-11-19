#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Repro recipe; use only this task's checkout, environment, caches and runtime.
set -euo pipefail
task_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
checkout="${1:-$task_root/checkout}"
venv="${2:-$task_root/.venv-main}"
export UV_CACHE_DIR=/tmp/vllm-omni-10663-uv
export UV_HTTP_TIMEOUT=180
export UV_CONCURRENT_DOWNLOADS=4
export VLLM_OMNI_TARGET_DEVICE=cuda
export CUDA_VISIBLE_DEVICES=''
if [[ ! -x "$venv/bin/python" ]]; then uv venv "$venv" --python /usr/bin/python3; fi
mkdir -p "$task_root/cache/gradio" "$task_root/evidence"
# This follows Dockerfile.ci's source-pinned Gradio/Pillow compatibility setup.
python3 "$checkout/tools/build_gradio_pillow13.py" --output-dir "$task_root/cache/gradio"
uv pip install --python "$venv/bin/python" --link-mode=copy --torch-backend=cu130 \
  -r "$checkout/requirements/cuda-runtime.txt" \
  --override "$checkout/requirements/cuda-telemetry-overrides.txt" \
  'vllm==0.34.0' 'nvidia-nccl-cu13==2.30.7' \
  "$task_root/cache/gradio/gradio-6.31.0+omni.pillow13-py3-none-any.whl" \
  -e "$checkout[dev]"
if [[ ! -f "$task_root/runtime/cudnn/nvidia/cudnn/lib/libcudnn.so.9" ]]; then
  uv pip install --python "$venv/bin/python" --target "$task_root/runtime/cudnn" \
    --no-deps --only-binary=:all: --require-hashes -r "$checkout/requirements/cudnn-runtime.txt"
fi
uv pip check --python "$venv/bin/python"
uv pip freeze --python "$venv/bin/python" > "$task_root/evidence/dependencies-main.txt"
