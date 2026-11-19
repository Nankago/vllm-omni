#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Download one immutable HF snapshot into the task's own cache."""

import os
from pathlib import Path

TASK = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(TASK / "cache/huggingface"))
os.environ.setdefault("HF_XET_CACHE", str(TASK / "cache/xet"))
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
if __name__ == "__main__":
    from huggingface_hub import snapshot_download

    print(
        snapshot_download(
            "Qwen/Qwen3-Omni-30B-A3B-Instruct",
            revision="26291f793822fb6be9555850f06dfe95f2d7e695",
            cache_dir=TASK / "cache/huggingface/hub",
            token=False,
            max_workers=4,
        ),
        flush=True,
    )
