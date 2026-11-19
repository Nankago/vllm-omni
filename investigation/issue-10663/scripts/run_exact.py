#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Run the untouched full-model node with an external evidence observer."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

TASK = Path(__file__).resolve().parents[1]
MODEL = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
REVISION = "26291f793822fb6be9555850f06dfe95f2d7e695"
NODE = "tests/e2e/online_serving/test_qwen3_omni_expansion.py::test_text_audio_to_text_audio_002[batch_token_2048]"


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", type=Path, default=TASK / "checkout")
    parser.add_argument("--venv", type=Path, default=TASK / ".venv-main")
    parser.add_argument("--gpus", default="6,7")
    parser.add_argument("--label", default="main-exact")
    args = parser.parse_args()
    checkout, venv = args.checkout.resolve(), args.venv.resolve()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = TASK / "evidence" / f"{args.label}-{stamp}"
    out.mkdir(parents=True)
    python = venv / "bin/python"
    meta = dict(
        start_utc=stamp,
        checkout=str(checkout),
        venv=str(venv),
        gpu_indices=args.gpus,
        model=MODEL,
        model_revision=REVISION,
        node=NODE,
        status="preflight",
        inference_requests=0,
    )
    meta["code_sha"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=checkout, text=True
    ).strip()
    env = os.environ.copy()
    root = TASK / "sysroot/root"
    libdir = root / "usr/lib/x86_64-linux-gnu"
    env.update(
        CUDA_VISIBLE_DEVICES=args.gpus,
        VLLM_WORKER_MULTIPROC_METHOD="spawn",
        VLLM_TARGET_DEVICE="cuda",
        VLLM_OMNI_TARGET_DEVICE="cuda",
        HF_HOME=str(TASK / "cache/huggingface"),
        HF_HUB_CACHE=str(TASK / "cache/huggingface/hub"),
        HF_HUB_OFFLINE="1",
        XDG_CACHE_HOME=str(TASK / "cache/xdg"),
        VLLM_CACHE_ROOT=str(TASK / "cache/vllm" / meta["code_sha"]),
        TORCHINDUCTOR_CACHE_DIR=str(TASK / "cache/torchinductor" / meta["code_sha"]),
        TRITON_CACHE_DIR=str(TASK / "cache/triton"),
        TMPDIR=str(out / "tmp"),
        EVIDENCE_DIR=str(out / "requests"),
        PYTHONPATH=os.pathsep.join([str(TASK / "scripts"), str(checkout)]),
        ESPEAK_DATA_PATH=str(libdir),
        LD_LIBRARY_PATH=str(libdir)
        + (os.pathsep + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else ""),
        PATH=os.pathsep.join(
            [str(venv / "bin"), str(root / "usr/bin"), str(TASK / "bin"), env["PATH"]]
        ),
    )
    Path(env["TMPDIR"]).mkdir()
    for name in (
        "HF_HOME",
        "HF_HUB_CACHE",
        "XDG_CACHE_HOME",
        "VLLM_CACHE_ROOT",
        "TORCHINDUCTOR_CACHE_DIR",
        "TRITON_CACHE_DIR",
    ):
        Path(env[name]).mkdir(parents=True, exist_ok=True)
    errors = []
    cudnn = TASK / "runtime/cudnn/nvidia/cudnn/lib/libcudnn.so.9"
    if cudnn.is_file():
        env["LD_PRELOAD"] = str(cudnn) + (
            " " + env["LD_PRELOAD"] if env.get("LD_PRELOAD") else ""
        )
        env["LD_LIBRARY_PATH"] = str(cudnn.parent) + os.pathsep + env["LD_LIBRARY_PATH"]
    else:
        errors.append("Pinned cuDNN 9.25.1 runtime absent; run scripts/prepare_env.sh")
    deps = subprocess.run(
        [
            str(python),
            "-c",
            "import importlib.metadata as m,json; names=['vllm','torch','torchvision','torchaudio','transformers','huggingface-hub','numpy','soundfile','openai-whisper','pytest','pyttsx3','opencc','vllm-omni']; print(json.dumps({n:m.version(n) if m.distribution(n) else None for n in names}))",
        ],
        env=env,
        cwd=checkout,
        text=True,
        capture_output=True,
    )
    (out / "dependency-preflight.log").write_text(deps.stdout + deps.stderr)
    if deps.returncode:
        errors.append("Runtime packages missing; see dependency-preflight.log")
    else:
        meta["dependencies"] = json.loads(deps.stdout)
        if meta["dependencies"]["vllm"] != "0.34.0":
            errors.append("vLLM version is not 0.34.0")
    manifest = json.loads((TASK / "evidence/model-info.json").read_text())
    snapshot = (
        TASK
        / "cache/huggingface/hub/models--Qwen--Qwen3-Omni-30B-A3B-Instruct/snapshots"
        / REVISION
    )
    missing = [
        f["path"]
        for f in manifest["files"]
        if not (snapshot / f["path"]).is_file()
        or (snapshot / f["path"]).stat().st_size != f["size"]
    ]
    if missing:
        errors.append("Model snapshot incomplete: " + ", ".join(missing))
    gpu = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,uuid,memory.used,memory.free",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        capture_output=True,
    )
    (out / "gpus.csv").write_text(gpu.stdout + gpu.stderr)
    requested = args.gpus.split(",")
    if len(set(requested)) != 2:
        errors.append("Exactly two distinct physical GPUs are required")
    found = {}
    for line in gpu.stdout.splitlines():
        fields = [x.strip() for x in line.split(",")]
        if len(fields) == 5:
            found[fields[0]] = fields
    for index in requested:
        if (
            index not in found
            or "H100" not in found[index][1]
            or int(found[index][3]) > 1024
        ):
            errors.append(
                f"GPU {index} missing, not H100 or busy; do not stop other tasks"
            )
    meta["gpu_details"] = {i: found.get(i) for i in requested}
    if errors:
        meta.update(status="blocked-before-test", preflight_errors=errors)
        dump(out / "run.json", meta)
        print(json.dumps({"evidence": str(out), **meta}, ensure_ascii=False, indent=2))
        return 2
    # Bind the model name to this fixed revision only inside our private cache.
    refs = snapshot.parent.parent / "refs"
    refs.mkdir(exist_ok=True)
    (refs / "main").write_text(REVISION)
    helper = subprocess.run(
        [
            str(python),
            "-c",
            "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())",
        ],
        env=env,
        text=True,
        capture_output=True,
    )
    if helper.returncode == 0:
        (TASK / "bin").mkdir(exist_ok=True)
        target = TASK / "bin/ffmpeg"
        if not target.exists():
            target.symlink_to(helper.stdout.strip())
    runtime = subprocess.run(
        [
            str(python),
            "-c",
            "import torch; from vllm._version import __commit_id__, __version__; print(__version__, __commit_id__, torch.__version__, torch.backends.cudnn.version()); assert torch.backends.cudnn.version() == 92501; assert __version__ == '0.34.0'; assert __commit_id__ and len(__commit_id__.removeprefix('g')) >= 8; assert '62d4bc3c7e33b1b6c906e233e6af2d3506c482ca'.startswith(__commit_id__.removeprefix('g'))",
        ],
        env=env,
        cwd=checkout,
        text=True,
        capture_output=True,
    )
    (out / "runtime-identity.log").write_text(runtime.stdout + runtime.stderr)
    if runtime.returncode:
        meta.update(
            status="blocked-before-test",
            preflight_errors=["Runtime identity check failed"],
        )
        dump(out / "run.json", meta)
        print("Runtime identity check failed:", out)
        return 2
    freeze = subprocess.run(
        ["uv", "pip", "freeze", "--python", str(python)],
        env=env,
        text=True,
        capture_output=True,
    )
    (out / "dependencies.txt").write_text(freeze.stdout + freeze.stderr)
    command = [
        str(python),
        "-m",
        "pytest",
        "-o",
        "addopts=",
        "--run-level=full_model",
        "-s",
        "-p",
        "capture10663",
        f"--junitxml={out}/junit.xml",
        NODE,
    ]
    meta.update(
        command=command,
        status="running",
        environment={
            k: env[k]
            for k in (
                "CUDA_VISIBLE_DEVICES",
                "HF_HOME",
                "HF_HUB_CACHE",
                "HF_HUB_OFFLINE",
                "TMPDIR",
                "EVIDENCE_DIR",
                "VLLM_CACHE_ROOT",
                "TORCHINDUCTOR_CACHE_DIR",
                "TRITON_CACHE_DIR",
                "ESPEAK_DATA_PATH",
            )
        },
    )
    dump(out / "run.json", meta)
    print("Running:", command, flush=True)
    print("Evidence:", out, flush=True)
    with (out / "pytest.log").open("w") as log:
        completed = subprocess.run(
            command, cwd=checkout, env=env, stdout=log, stderr=subprocess.STDOUT
        )
    meta.update(
        status="finished",
        pytest_exit_code=completed.returncode,
        end_utc=datetime.now(timezone.utc).isoformat(),
        inference_requests=len(
            list((out / "requests").glob("request-*/response.json"))
        ),
    )
    dump(out / "run.json", meta)
    print(json.dumps(meta, indent=2))
    return completed.returncode


if __name__ == "__main__":
    sys.exit(main())
