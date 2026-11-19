# Reproduce / resume #10663

These are prepared commands, not successful inference evidence. Current result is preflight blocked, 0 requests. Work root:

```bash
cd /gemini-0/space/private/lmx/temp-restore/Vllm-Omni/issue-10663
```

Recheck draft PR #10725: it removes this test variant rather than fixing the production path. Keep the pinned original node for diagnosis. Check `HANDOFF.md` and `evidence/stopped-network-jobs.json` before starting; do not duplicate a downloader. Use the approved command executor if the default sandbox helper is still missing. Do not change proxy/TLS configuration or other task environments.

## Complete the independent environment

The first three bare vLLM installation attempts are preserved in `evidence/install-vllm-main*.log`. After the network is available, use the source checkout's CUDA pins and CI install recipe:

```bash
bash scripts/prepare_env.sh > evidence/prepare-main.log 2>&1
python3 scripts/download_model.py > evidence/download-model-resume.log 2>&1
```

`prepare_env.sh` registers this checkout editable and includes its dev requirements, the source-pinned Gradio/Pillow wheel and isolated cuDNN runtime. It may take substantial downloads; do not mix in the system development Torch or `.venv-9477`, `.venv-task12`, `issue-10642/.venv-main`. Its recipe does not rebuild unused LMCache native extensions as the entire CI image does; record any relevant runtime difference before claiming an exact CI environment match. `uv pip check` must pass. When model transport is complete, preserve its fixed revision and verify file sizes/checksums if using a different cache/source.

The system Python used for model download already has huggingface-hub1.28.0. It is only a transport helper, not the inference environment. HF snapshot is fixed to `26291f793822fb6be9555850f06dfe95f2d7e695`. Original CI weight revision is unknown. Local `sysroot/` audio dependencies are already downloaded and SHA256 verified; see `evidence/espeak-package-verification.log`.

## Run the unchanged main node

```bash
python3 scripts/run_exact.py --gpus 6,7 --label main-exact
```

This runs the independent `.venv-main/bin/python -m pytest` with:

```text
-o addopts= --run-level=full_model -s -p capture10663 --junitxml=...
tests/e2e/online_serving/test_qwen3_omni_expansion.py::test_text_audio_to_text_audio_002[batch_token_2048]
```

The launcher resolves all paths and creates `evidence/main-exact-<UTC>/`. Check `run.json`, `pytest.log`, `junit.xml`, stage logs under `tmp/`, and `requests/`. It checks the two specified GPUs and exits if busy; it never kills other jobs. The raw client request does not set a new seed or temperature. The resolved CI deploy printed in `pytest.log` contains actual stage defaults (Talker seed42 / temperature.9 / top_k50).

`requests/request-NNN/` has the complete input payload, raw OpenAI response, `text-0.txt`, `audio-0.wav`, waveform statistics and transport timing. NNN is collector entry order, not guaranteed network arrival order; use timestamps, server request IDs and stage logs. Completion order is separately recorded. ASR/actual pytest assertion records live at `requests/asr-*.json` and `assertion-*.json`.

If pytest stops at the first failed assertion, its remaining concurrent responses are still captured. Evaluate every preserved WAV with the matching environment/source (replace RUN_DIR):

```bash
export CUDA_VISIBLE_DEVICES=6,7
export XDG_CACHE_HOME="$PWD/cache/xdg"
export PYTHONPATH="$PWD/checkout"
export PATH="$PWD/bin:$PATH"
.venv-main/bin/python scripts/evaluate_captures.py RUN_DIR --transcript-model small
.venv-main/bin/python scripts/evaluate_captures.py RUN_DIR --transcript-model large-v3
```

Run only after checking those GPUs are still owned/idle. More powerful ASR is additional diagnostic evidence; preserve the small-model result and threshold. The offline entrypoint has only static validation so far. Inspect/listen to the original WAV before assigning an ASR or generation cause. No Whisper checkpoint has been downloaded in this task yet.

## Evidence-guided next experiments

1. Start with one exact main run (5 concurrent identical requests). Inspect waveform, duration/silence, transcripts and assertions. If it passes, repeat at most twice to estimate stability before expanding. Count both failing requests/checked requests and failing groups/total groups; keep unjudged responses separate.
2. For an observed failing payload, compare a fresh-service single request, five serial identical requests in one service, and the original five concurrent requests. Keep payload, stage config, seed and code/model versions unchanged. Repeat the small set after restarting the service. Same payloads have no meaningful input-order permutation; submission timing/concurrency is the variable.
3. If main does not reproduce, create the original-code comparison only then:

```bash
git -C checkout worktree add --detach ../original 1194d6ba2096459f8d96af06fbfb765b14e01cac
bash scripts/prepare_env.sh "$PWD/original" "$PWD/.venv-original" > evidence/prepare-original.log 2>&1
python3 scripts/run_exact.py --checkout original --venv .venv-original --gpus 6,7 --label original-exact
```

The original has its own editable source/environment and uses its own dependency files. Recheck versions and log any necessary recipe differences. Do not label the main environment as the original environment. Compare the same fixed local model bytes; acknowledge that the issue did not pin its original weight revision.
4. Only if evidence points to continuous/concurrent state should request caches, Talker conditioning queues, codec chunk assembly and EOS cleanup be instrumented. Only profile a narrowed suspect. A related PR's existence or a different seed passing proves neither cause nor repair.
5. Only after a deterministic defect is confirmed: minimal patch, regression that fails before/passes after, same-input before/after single/serial/concurrent audio tests, then an English PR draft. No upstream comment/PR is authorized.

## Git backup / migration

The task branch stores investigation documents/scripts only. Large logs, cache, model weights, audio, environments and credentials are outside the checkout and are not pushed. The final bundle is incremental against product main `9a2ae280e6bc3f92f57d22a2243bd064fe7a2faf`. To restore it, first obtain that upstream base, then:

```bash
git fetch /path/to/qwen3-omni-audio-mismatch.bundle refs/heads/codex/qwen3-omni-audio-mismatch
git switch -c codex/qwen3-omni-audio-mismatch FETCH_HEAD
```

Restore the backed-up `investigation/issue-10663` files into a fresh task root, or copy its `scripts/`, `HANDOFF.md`, `REPORT.md`, `REPRODUCE.md` to the parent of this checkout. Evidence snapshots are under its `source-evidence/`; copy `model-info.json` to task-root `evidence/` before using the launcher. Credentials and downloads must be supplied anew. `BACKUP_STATE.json` outside Git records exact commit, bundle and remote verification.
