# Qwen3-Omni issue #10663 investigation

Updated 2026-11-19 UTC. Goal: reproduce sequential request audio/text mismatch,
diagnose from preserved audio and ASR, and only fix a confirmed defect.
No upstream comments or PRs authorized. Independent task directory: issue-10663/.

## State
- GitHub issue read live: open, no comments, no assignees. PR search for 10663 empty.
- Original revision: 1194d6ba2096459f8d96af06fbfb765b14e01cac,
  release/v0.34.0, vLLM 0.34.0, 2 x H100 80GB.
- Original test: tests/e2e/online_serving/test_qwen3_omni_expansion.py::test_text_audio_to_text_audio_002[batch_token_2048].
- No inference, reproduction or root cause established yet.
- Read current upstream AGENTS, CONTRIBUTING, issue lessons, precheck intent gate.
  User explicitly requests minimal confirmed fix and matching-input regression;
  no contribution implementation before diagnosis. Human review not claimed.
- All eight H100s idle at first inspection; choose physical GPU 6/7 after recheck.
  Another independent task targets GPU 0. Do not modify or kill other processes.
- Shared disk free 848 TB; /tmp free 3.2 TB.
- Default command sandbox cannot spawn: codex-linux-sandbox target missing.
  Approved require_escalated shell works. Do not alter global sandbox config.
- Root directory is not a Git repo; upstream-vllm-omni contains unrelated #9477
  committed fix. Do not change it or root HANDOFF.md, .venv-task12, RL-Kernel-task12,
  issue-10642, or their dependencies.
- Nankago/vllm-omni fork accessible with push permission via GitHub connection.
  Existing gh login documented by Task 12; do not initiate a new authentication.

## Next executable action
Create independent checkout/branch codex/qwen3-omni-audio-mismatch,
fetch current main, inspect exact tests/config/dependencies, create own venv/cache.
Preserve commands, server logs, payloads, raw audio, transcripts and assertions.

## Checkout / interpretation update
- Independent checkout complete, branch codex/qwen3-omni-audio-mismatch.
  Base main SHA 9a2ae280e6bc3f92f57d22a2243bd064fe7a2faf.
- Original SHA fetched into this independent repository only.
  evidence/original-main-paths.diff: selected tests/helpers/input/ASR/locks
  identical; deploy differences in ROCm only. CUDA comparison still requires runs.
- IMPORTANT: send_omni_request(request_num=5) submits 5 concurrent identical
  requests via ThreadPoolExecutor and asserts in completion order. The fifth
  printed score is not evidence of sequential cross-request leakage.
- Request does not supply seed/temperature. Deploy stage defaults: thinker
  temperature 0; talker seed 42, temperature .9, top_k 50, repetition_penalty 1.05;
  code2wav temperature 0. Capture resolved deployment and all actual HTTP payloads.
- Own .venv-main installation launched with official vllm==0.34.0 --torch-backend=cu130.
  Install log evidence/install-vllm-main.log, exec session 6835.
- Public/non-gated model Qwen/Qwen3-Omni-30B-A3B-Instruct revision
  26291f793822fb6be9555850f06dfe95f2d7e695, 70,530,595,574 bytes all repo files.
  Download into cache/huggingface/hub, log evidence/download-model.log, session 50454.
- GitHub CLI existing authenticated user Nankago confirmed. No new login needed.
- Next: finish dependency install/download; add external pytest evidence plugin;
  run exact full_model node on GPUs6,7 with original concurrency, then small
  single/serial/concurrent repeated comparisons guided by observed failures.

## Current checkpoint (supersedes the earlier running-job entries)

2026-11-19 UTC: **blocked before inference**. No server, inference, audio or ASR
quality experiment was run; failures/total = 0/0, unexecuted (not "unreproduced").
No confirmed code defect, product patch, regression pass or fixing PR draft.

- User was asked asynchronously for an existing weight/wheel/internal-mirror path;
  no answer at checkpoint time. Continue with provided paths if a reply arrives.
- Three bare vLLM install variants were attempted in .venv-main: official CUDA13
  wheel route timed out on torchvision; configured PyPI mirror timed out; official
  PyPI resolved199 packages but downloads stalled/retried. See install-vllm-main*.log.
- Xet model transport then HTTP transport attempted. HTTP delivered only partial
  weights and repeatedly closed bodies after ~15-60MB. HF 1MB Range took20.05s.
  ModelScope official-source probe failed TLS EOF. No alternate weights used.
- Stopped only our verified uv/model download processes; exact PIDs and action are
  evidence/stopped-network-jobs.json. No other job killed. Partial caches retained.
- .venv-main is incomplete. No torch/vllm inference imports succeeded there.
  It is not a working, version-qualified environment. No original-revision venv.
- eSpeak-NG official .deb SHA256 checks passed; private sysroot voices query works.
  No system package installation. cuDNN9.25.1 private runtime still absent.
- Related merged #10593 (8eb85650) defers Code2Wav graph capture to reduce memory;
  it is in main but absent from original. It does not claim an #10663 quality fix.
  Selected test/helper/CUDA-lock paths are identical; production code is not.
- scripts/run_exact.py saves dependency/GPU/model preflight and launches the exact
  node only if prerequisites pass. scripts/capture10663.py saves all raw responses
  even if pytest stops on an early assertion; request NNN is collector entry order,
  completion index is separate, server IDs/timestamps must establish actual order.
- scripts/evaluate_captures.py can assess every raw WAV, with small and strong ASR
  recorded separately. No real capture/evaluation integration has been exercised.
- Validation performed: Python compile, Ruff, bash -n, --help; API stand-in check
  of five concurrent requests and unchanged assertion propagation. These are
  infrastructure checks, not model/ASR/waveform correctness or E2E passes.
- Final preflight evidence: evidence/final-preflight.log and its run.json. Status
  blocked-before-test, requests0, missing dependencies/weights/privatecuDNN.
- Goal/scope/diagnosis: REPORT.md. Resume commands and limited experiment matrix:
  REPRODUCE.md. No hypothesis about silence/ASR/state/model has been eliminated.
- Branch codex/qwen3-omni-audio-mismatch; product base remains main
  9a2ae280e6bc3f92f57d22a2243bd064fe7a2faf. Documentation checkpoint commit and
  portable incremental bundle are recorded in BACKUP_STATE.json after commit.

Next executable command after network/cache availability:
`bash scripts/prepare_env.sh > evidence/prepare-main.log 2>&1`
then `python3 scripts/download_model.py > evidence/download-model-resume.log 2>&1`
and `python3 scripts/run_exact.py --gpus 6,7 --label main-exact`.
Do not start a background inference service from an incomplete environment.

## Latest upstream refresh — supersedes the initial no-PR search

- Issue remains open, comments0, no assignees.
- Found draft/open/unmerged PR #10725 by psv666, created2026-11-19T08:27:22Z,
  head5c23a02dc3b0a3af45541ddbd6aa122c273499cd, base9a2ae280.
  Read body, actual diff and empty comments. The sole change removes the
  reduced_token_params variant from test_text_audio_to_text_audio_002.
- It reports external L20X original-setting reproduction2/20 five-request groups,
  and greedy Talker+code-predictor9/15 groups. No independent H100 replay here.
  This is test selection, not a diagnosed production fix. Do not duplicate it,
  apply it to hide the original node, or treat it as resolution of this task.
- Evidence: evidence/github-final-status.json. Recheck its latest head before
  resuming. No upstream comments/PR were sent by this task.
- First local docs/script checkpoint: d9596bc668931398b76fd5a3d84a1f44b69a8dc0.
  A follow-up records this upstream discovery; final HEAD is in BACKUP_STATE.json.
