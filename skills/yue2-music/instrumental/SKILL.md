---
name: yue2-instrumental
description: 用 YuE2 生成纯音乐或制作纯音乐 cover。默认由 YuE2 写谱，自动将 Vocal 音符移到 Ins 后生成；只有用户明确要求 agent 作曲时才由 agent 写谱。支持文字描述、ABC 或音频参考，交付试听、真实 prompt 和转谱检查。
---

# YuE2 纯音乐与器乐 cover

用户只需描述音乐或提供参考。你负责环境、转谱、运行和交付；用户不需要懂声部格式或推理参数。

推荐使用 GPT-6 Astra 作为执行本 skill 的 agent；默认作曲与声音生成仍由 YuE2 完成。

## 选择来源

- **文字生成，默认模式**：调用 YuE2 的符号规划器写谱，保存原谱，将 Vocal 旋律移到 Ins 后重新生成音频。不要自己写 events/ABC 替代这一步，也不要套用示例旋律。
- **用户明确要求 agent 作曲**：才由你写谱，使用 `--composer agent`。复杂谱可以写 [events.json](references/event-schema.md)。
- **用户给了 ABC**：以用户曲谱为准，直接转为纯音乐条件，不重新作曲。只有不规范输入才读 [修谱规则](references/abc-repair.md)；保留原件与改动说明。
- **纯音乐 cover**：读 [cover 流程](references/cover.md)。音频先经 SheetSage2 转谱，再做相同的 Vocal → Ins 转换。默认保留旋律、让 YuE2 重新配伴奏；要求保留原和声时加 `--keep-harmony`。这不是直接去人声或伴奏分离。

模型规划失败或曲谱无法解析时保留原始输出，定位问题后至多重试一次。不要自动切换成 agent 作曲；音乐性修改必须符合用户授权。严格时长和音频逐音跟谱不是模型保证。

## 一条命令

首次运行读 [安装与迁移](references/setup.md)。下面路径相对本 skill 目录，输出使用新目录。

```bash
.venv/bin/python scripts/instrumental.py run \
  --style "Instrumental, warm solo piano, lyrical and gentle, no vocals" \
  --output work/song
```

默认 `--composer model`。脚本先保存 `planning/` 中的 YuE2 原始计划，再转谱和生成。最终歌词为空或只有曲式标签，不含歌词正文。`--plan-mode melody` 可让模型只规划旋律；默认规划旋律与和弦。

已有 ABC 可用 `run --abc work/source.abc ...`。明确让 agent 作曲时用 `run --composer agent --events work/events.json ...`。`--prepare-only` 在生成音频前停止；新写模型计划本身仍需要 GPU，已有 ABC/events 的准备只需 CPU。已有离线模型加 `--models-root models --offline`。

## 转谱规则

转换的是音符事件，不是轨名字符串：

- Vocal 的全部音高、起音和时值移到 Ins；Vocal 只留休止和和弦。
- 保留原 Ins 中不与 Vocal 重叠的前奏、间奏和填充。
- Ins 只能是单旋律。两轨重叠时默认 Vocal 优先：保留全部 Vocal 音符，仅保留原 Ins 音符未重叠的部分，逐项记录。`--overlap error` 可拒绝任何这类缩编。
- 校验连音、变音、速度、拍号和保留的和弦时刻；失败不继续生成。原谱、转换后谱和 `conversion/transfer.json` 都保留。

修改过的曲谱作为新的固定 ABC 输入，不能直接继续旧计划的 token 前缀。脚本检查实际生成前缀和保存的 ABC 与转换后输入一致。

## 输出

- `listen/index.html`：可播放音频、实际完整 prompt、标准谱。
- `listen/listening.mp3` 或 `listen/audio.flac`：直接交付的试听。
- `planning/` 或 `transcription/`：真实来源；`conversion/`：转谱与保留检查；`prepared/` 和 `generation/`：完整本地记录。

直接展示音频与实际风格 prompt，链接完整 `prompt.txt`，并说明曲谱由 YuE2、用户、SheetSage2 或明确授权的 agent 提供。报告实际时长。文件检查不等于已确认无人声或听觉旋律保真；没有听音能力时不编造听感结论。

试听用 **m-a-p/YuE2-Vae**；保持默认采样、BF16 AR/NAR、FP32 VAE 和 32 步，每张 GPU 顺序运行。环境不足时解决资源问题，不暗中改歌长或降设置。

发布 skill 只用核验后的发布包。`listen/` 不带原始会话、机器配置和日志，但仍包含用户作品与 prompt，只交给本人或授权对象。见 [分发与隐私](PRIVACY.md)。
