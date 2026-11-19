# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""External pytest observer: preserve exact payloads, responses and unchanged verdicts."""

import base64
import functools
import hashlib
import io
import itertools
import json
import os
from pathlib import Path
import threading
import time

ROOT = Path(os.environ.get("EVIDENCE_DIR", "/tmp/issue-10663-capture")).resolve()
LOCK = threading.Lock()
COUNTER = itertools.count(1)
COMPLETIONS = itertools.count(1)


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n")


def audio_stats(raw):
    import numpy as np
    import soundfile as sf

    data, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    block = max(1, sr // 10)
    rms = [
        float(np.sqrt(np.mean(mono[i : i + block] ** 2)))
        for i in range(0, len(mono), block)
    ]
    return dict(
        bytes=len(raw),
        sha256=hashlib.sha256(raw).hexdigest(),
        sample_rate=sr,
        channels=data.shape[1],
        frames=len(data),
        duration_seconds=len(data) / sr,
        finite=bool(np.isfinite(data).all()),
        peak=float(np.max(np.abs(data))) if len(data) else None,
        rms=float(np.sqrt(np.mean(data**2))) if len(data) else None,
        fraction_abs_below_1e_4=float(np.mean(np.abs(data) < 1e-4))
        if len(data)
        else None,
        fraction_clipped=float(np.mean(np.abs(data) >= 0.999)) if len(data) else None,
        rms_per_100ms=rms,
    )


def pytest_collection_finish(session):
    from openai.resources.chat.completions import Completions
    import tests.helpers.assertions as assertions
    import tests.helpers.client as client

    ROOT.mkdir(parents=True, exist_ok=True)
    create = Completions.create

    @functools.wraps(create)
    def observed_create(self, *args, **kwargs):
        with LOCK:
            index = next(COUNTER)
        dest = ROOT / f"request-{index:03d}"
        dest.mkdir()
        write_json(dest / "request.json", kwargs)
        record = dict(
            send_index=index,
            start_ns=time.time_ns(),
            stream=kwargs.get("stream", False),
        )
        try:
            result = create(self, *args, **kwargs)
        except BaseException as error:
            record.update(end_ns=time.time_ns(), error=repr(error))
            write_json(dest / "transport.json", record)
            raise
        with LOCK:
            record["completion_index"] = next(COMPLETIONS)
        record.update(end_ns=time.time_ns(), request_id=getattr(result, "id", None))
        write_json(dest / "transport.json", record)
        if not kwargs.get("stream", False):
            write_json(dest / "response.json", result.model_dump(mode="json"))
            for i, choice in enumerate(result.choices):
                message = choice.message
                (dest / f"text-{i}.txt").write_text(message.content or "")
                audio = getattr(message, "audio", None)
                if audio and audio.data:
                    raw = base64.b64decode(audio.data)
                    (dest / f"audio-{i}.wav").write_bytes(raw)
                    try:
                        stats = audio_stats(raw)
                    except Exception as error:
                        stats = {"decode_error": repr(error), "bytes": len(raw)}
                    write_json(dest / f"audio-{i}-stats.json", stats)
        return result

    Completions.create = observed_create
    resolve = assertions._resolve_audio_transcript

    @functools.wraps(resolve)
    def observed_resolve(response, config, run_level, **kwargs):
        transcript = resolve(response, config, run_level, **kwargs)
        result = dict(
            request_id=getattr(response, "request_id", None),
            transcript=transcript,
            text=getattr(response, "text_content", None),
            run_level=run_level,
            model_size=config.get("transcript_model", "small"),
            language=config.get("transcript_language"),
        )
        if transcript is not None and result["text"]:
            result["similarity"] = assertions.cosine_similarity_text(
                transcript.lower(), result["text"].lower()
            )
        name = str(result["request_id"]).replace("/", "_")
        write_json(ROOT / f"asr-{name}.json", result)
        return transcript

    assertions._resolve_audio_transcript = observed_resolve
    check = client.assert_omni_response

    @functools.wraps(check)
    def observed_check(response, config, run_level):
        verdict = dict(
            request_id=getattr(response, "request_id", None),
            threshold=config.get("similarity_threshold", 0.8),
        )
        try:
            result = check(response, config, run_level)
            verdict["passed"] = True
            return result
        except BaseException as error:
            verdict.update(passed=False, exception=repr(error))
            raise
        finally:
            name = str(verdict["request_id"]).replace("/", "_")
            write_json(ROOT / f"assertion-{name}.json", verdict)

    client.assert_omni_response = observed_check
