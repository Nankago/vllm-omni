#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Score every preserved WAV, including responses after pytest's first failure.

Use the matching environment and PYTHONPATH=checkout. No inference requests are sent.
A stronger ASR is a diagnostic comparison, never a replacement of the small-model verdict.
"""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--transcript-model", default="small")
    args = parser.parse_args()
    from tests.helpers.assertions import assert_omni_response
    from tests.helpers.media import (
        convert_audio_file_to_text,
        cosine_similarity_text,
        release_audio_transcriber,
    )

    results = []
    try:
        for folder in sorted((args.run / "requests").glob("request-*")):
            wav = folder / "audio-0.wav"
            if not wav.is_file():
                results.append({"request": folder.name, "error": "No preserved WAV"})
                continue
            text = (folder / "text-0.txt").read_text()
            transcript = convert_audio_file_to_text(
                str(wav.resolve()), model_size=args.transcript_model
            )
            response = SimpleNamespace(
                success=True,
                text_content=text,
                audio_bytes=wav.read_bytes(),
                audio_content=transcript,
                request_id=folder.name,
            )
            config = {
                "modalities": ["text", "audio"],
                "key_words": {"audio": ["test"]},
                "similarity_threshold": 0.8,
            }
            result = {
                "request": folder.name,
                "text": text,
                "transcript": transcript,
                "transcript_model": args.transcript_model,
                "similarity": cosine_similarity_text(
                    transcript.lower(), text.strip().lower()
                ),
            }
            try:
                assert_omni_response(response, config, run_level="full_model")
                result["passed"] = True
            except AssertionError as error:
                result.update(passed=False, assertion=str(error))
            results.append(result)
            (folder / f"evaluation-{args.transcript_model}.json").write_text(
                json.dumps(result, indent=2, ensure_ascii=False) + "\n"
            )
    finally:
        release_audio_transcriber()
    summary = {
        "transcript_model": args.transcript_model,
        "results": results,
        "failed": sum(r.get("passed") is False for r in results),
        "evaluated": sum("passed" in r for r in results),
        "missing_audio": sum("error" in r for r in results),
    }
    (args.run / f"evaluation-{args.transcript_model}.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
