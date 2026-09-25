# 纯音乐 cover

把参考旋律重新演奏成指定风格。它不保留原录音波形或歌手身份，也不是分离出原伴奏。

## ABC 参考

```bash
.venv/bin/python scripts/instrumental.py cover \
  --abc reference.abc --style "Lyrical solo cello with warm piano accompaniment" \
  --output work/cello-cover
```

默认移除原和弦，使用 melody 条件，让 YuE2 重配伴奏；需要保留和声时加 `--keep-harmony`。输入需为 native Vocal/Ins 两轨格式；不规范 ABC 先依 [修谱规则](abc-repair.md) 修复。不要另写旋律替代参考。

## 音频参考

SheetSage2 使用单独的 Python 环境。固定版本见 `assets/cover-release.json`；它包含公开 SheetSage2 和 MERT-v2-FullSong 仓库、revision 与权重哈希。按固定 SheetSage2 快照的 requirements 安装，避免修改 YuE2 环境。

```bash
# 在独立转谱环境下载固定快照；也可让 agent 根据 cover-release.json 调用 snapshot_download。
huggingface-cli download m-a-p/SheetSage2 \
  --revision eab522a8168e8b8b8c4856bf8609cd86198f01fe --local-dir models/SheetSage2
huggingface-cli download m-a-p/MERT-v2-FullSong \
  --revision d8ba1c745e733b3908ce6ad16ebeb17ac7600a42 --local-dir models/MERT-v2-FullSong
```

在具备 GPU 的节点用 YuE2 环境调用统一入口：

```bash
.venv/bin/python scripts/instrumental.py cover \
  --audio reference.wav \
  --transcriber-python .venv-transcribe/bin/python \
  --transcriber-model models/SheetSage2 --base-model models/MERT-v2-FullSong \
  --style "Instrumental, expressive acoustic guitar, intimate and gentle" \
  --models-root models --offline --output work/guitar-cover
```

脚本先在单独的解释器中转谱，进程退出释放 GPU 后再加载 YuE2。无需把原音频复制到发布包。默认使用 `melody_full` 和 `melody_only=True`，覆盖人声主旋律与器乐段落；`--keep-harmony` 改为完整转谱。没有进行音频裁剪。

也可独立转谱排错：

```bash
.venv-transcribe/bin/python scripts/transcribe_cover.py reference.wav \
  --task melody-full --output work/transcription
```

审阅 `transcription/score.abc` 和 warnings 后，再走 ABC cover。转谱准确率是独立误差来源；转轨检查只证明保留了转写谱上的音符，不证明转写完全忠于录音。

## 检查

`conversion/transfer.json` 记录移动的 Vocal 音符数、保留的原 Ins 音符，以及所有发生重叠缩编的片段。输出 Vocal 必须没有发声音符，Vocal 的原音高、起音和时值须全部出现在输出 Ins 中。

只想检查转谱而暂不生成音频时，`cover --abc ... --prepare-only` 可在 CPU 上完成；音频转写仍需要转谱环境和 GPU。未经用户明确要求，不把失败的转谱换成 agent 新作。
