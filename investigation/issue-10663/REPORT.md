# #10663 调查阶段报告

截至 2026-11-19 UTC，调查停在**环境准备阻塞**。没有启动模型服务、没有发送推理请求、没有音频或 ASR 质量结果。失败次数/总次数：**0/0，尚未执行**；这不等于“未复现”。没有确认代码缺陷，没有产品修复、回归通过或英文修复 PR 草稿。

## 已核实的事实

- 最新 [issue #10663](https://github.com/vllm-project/vllm-omni/issues/10663) 为 open，无评论、无指派人员。初次按编号搜索未发现关联 PR，收尾刷新发现 **draft PR #10725**（见下一项）。初次/最终快照：`evidence/github-status.json`、`github-final-status.json`。
- 已阅读 [PR #10725](https://github.com/vllm-project/vllm-omni/pull/10725) 最新正文、diff、评论（0条）。作者 psv666，open/draft/unmerged，head `5c23a02dc3b0a3af45541ddbd6aa122c273499cd`。它**只删除**长音频测试的 `batch_token_2048` 参数，属于 CI 范围调整，没有生产代码修复。作者报告在 L20X 上原设置失败2/20组，Talker及code predictor均greedy时失败9/15组；这些是外部报告，不是本次H100验证或根因证据。不要重复该补丁，也不要把移除用例当作问题解决。
- 本任务独立 checkout：`checkout/`，分支 `codex/qwen3-omni-audio-mismatch`。产品代码基线为 main `9a2ae280e6bc3f92f57d22a2243bd064fe7a2faf`。未修改其他任务源码、依赖、进程或根目录的旧 HANDOFF。
- 原 issue SHA `1194d6ba2096459f8d96af06fbfb765b14e01cac` 已获取用于对照；未安装或运行原 revision 环境。其测试、输入/ASR/断言 helper 及 CUDA 依赖锁与当前 main 一致。Qwen3-Omni 生产代码存在变化，不能把两版本视为等同。
- **原测试是 5 个并发、相同 payload 的请求**，并非 5 个串行请求。`send_omni_request()` 通过 `ThreadPoolExecutor(max_workers=5)` 发出，按 `as_completed()` 做 ASR 和断言。“最后一个相似度低”表示最后完成/被检查的响应，不能证明状态随请求次数累积退化。给这 5 个相同输入重新编号或排序不能改变输入序列。
- 原场景为 120 秒、48kHz、单声道合成的重复“test”音频。Thinker/Talker 的 `max_num_batched_tokens` 均设为 2048，CI overlay 各 stage 的 `max_num_seqs=5`。客户端不传 seed/temperature；部署中 Talker 已有 seed=42、temperature=.9、top_k=50、repetition_penalty=1.05。没有更换 seed 或降低 .8 阈值。
- main 已合入 [PR #10593](https://github.com/vllm-project/vllm-omni/pull/10593)，将 Code2Wav graph 捕获移到加载后，以降低显存占用。PR 没有验证 #10663 的音文质量，不能当成已修复本 issue。原始信息：`evidence/pr-10593.json`。
- 可见 8×H100 80GB；选定物理 GPU6/7（各空闲 81,079 MiB）用于后续。它们未被本任务占用。另一项调查计划用 GPU0；没有干扰它或 RL-Kernel。快照：`evidence/preflight/`。
- 公共非门控模型固定 revision 为 `26291f793822fb6be9555850f06dfe95f2d7e695`，完整仓库约70.53GB。原 CI 使用的模型 revision 没有在 issue 中记录，不能声称精确重建了原 CI 的权重字节。元数据：`evidence/model-info.json`。

## 阻塞与尝试

独立 Python3.12 环境 `.venv-main/` 已建立，未复用系统开发版 Torch 或其他任务 venv。官方 vLLM0.34.0/CUDA13 安装遇到：

1. `torchvision==0.29.0+cu130` 官方 PyTorch 下载连续超时：`evidence/install-vllm-main.log`。
2. 配置的 Huawei PyPI 镜像连接超时：`evidence/install-vllm-main-retry.log`。
3. 改用官方 PyPI、保留原代理和 TLS 后解析了199个包，但下载仍反复停顿/重试，未完成安装：`evidence/install-vllm-main-pypi.log`。

HF Xet 与官方普通 HTTP 下载均尝试过。HTTP 1MB Range 读取约20秒；大文件在约15–60MB后多次报告 `peer closed connection without sending complete message body`。所有15个权重分片仍不完整。ModelScope 官方发布源的少量字节探测发生 TLS EOF，未取得或使用其权重。日志：`evidence/download-model*.log`。没有关闭代理或 TLS 校验。

已通过官方 APT 包 SHA256 校验，在私有 `sysroot/` 解压 eSpeak-NG，voices 查询成功；系统包未改变。其余前提还包括私有 cuDNN9.25.1 runtime 和完整 CI 依赖。完整准备配方见 `scripts/prepare_env.sh`，尚未完整执行成功。

网络重试已停止，仅停止了本任务明确验证的安装/下载 PID；缓存保留。详情 `evidence/stopped-network-jobs.json`。最终 `evidence/final-preflight.log` 及其 `run.json` 明确记录 `blocked-before-test`、缺依赖/模型/私有cuDNN、请求数0。

## 可用交付与验证范围

- `scripts/run_exact.py`：检查版本、模型完整性、GPU空闲情况，以原节点及 `--run-level=full_model` 运行；保存真实命令、依赖、GPU UUID、日志和 JUnit。该入口已实际运行，得到上述前置阻塞结果。
- `scripts/capture10663.py`：外部 pytest 插件，保存每个 payload、响应、文本、原始 WAV、采样率/时长/RMS/静音和削顶统计、调用时间/完成顺序、Whisper 转写/相似度及原始断言。不修改模型采样与测试门槛。
- `scripts/evaluate_captures.py`：离线补评 pytest 在首个失败后未继续判定的其余 WAV；Whisper-small 与更强 ASR 的结果分开保存。尚无真实音频可评估。
- Python 编译、Ruff、shell 语法检查通过。采集器用 API stand-in 检查了5并发 payload/字节保留及原断言异常透传；**没有验证真实模型、ASR、音频解码或在线插桩集成**。记录见 `evidence/capture-observer-check.log`、`scripts-lint.log`。

没有排除静音/截断、ASR错误、Talker随机退化、批调度/状态/分块缺陷中的任何一项。唯一纠正的是把原测试解释成串行请求。下一步应先修复下载条件或提供已存在的固定版本缓存，然后运行 main 原测试；根据真实 WAV 和小量重复结果决定对照实验，见 `REPRODUCE.md`。不应先修改生产代码或配置。
