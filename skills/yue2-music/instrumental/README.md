# YuE2 纯音乐 skill

**说一句想要什么音乐，agent 负责生成。** 不需要懂曲谱，也不用手填模型参数。

推荐使用 **GPT-6 Astra** 作为执行 skill 的 agent。默认仍由 YuE2 写谱和生成音频，agent 负责调用、转轨和检查。

把这个文件夹交给能运行终端的 agent，说：

> 先读 yue2-instrumental/SKILL.md。帮我生成一段约一分钟的温柔钢琴纯音乐，适合夜晚阅读，不要人声。把音频和完整 prompt 给我。

也可以贴 ABC：

> 把下面的曲谱做成电影弦乐纯音乐，保留旋律。修谱和参数你处理，直接给我试听和 prompt。

默认由 YuE2 写谱，agent 自动把 Vocal 旋律移到 Ins，再让 YuE2 生成纯音乐。只有你明确说“让 agent 作曲”时，agent 才写谱。它会准备环境、生成音频，最后给出可播放的 MP3/FLAC、完整 prompt、曲谱和本地试听页。音频实际时长可能与谱面时长不同；是否有人声泄漏、是否准确跟谱仍需试听确认。

## 首次使用

需要 Linux、Python 3.10+ 和支持 BF16 的 NVIDIA 24 GB 级 GPU。模型文件约 7.8 GB，依赖另占空间。没有 GPU 时可以先修复和校验已有曲谱；默认的模型写谱、音频转谱和声音生成需要 GPU。

将本文件夹放到 agent 支持的 skill 目录即可安装；也可以直接让 agent 读取 `SKILL.md`。环境准备细节见 [安装与迁移](references/setup.md)，交给 agent 执行即可。

已有环境时，agent 用一条命令完成准备、生成和试听导出：

```bash
.venv/bin/python scripts/instrumental.py run \
  --style "Instrumental, warm solo piano, gentle and lyrical" \
  --output work/song
```

默认 `--composer model` 调用 YuE2 的真实符号规划器。输出打开 `work/song/listen/index.html`。原始模型计划、转轨后曲谱和保留检查分别保存在工作目录中。

需要纯音乐 cover 时，说“把这段录音（或 ABC）翻成钢琴纯音乐，保留旋律”，agent 会走 [音频或 ABC cover 流程](references/cover.md)。

## 发布与测试

**1.2.0** 使用专门编写的测试谱和公开模型标识。发布包不含用户会话、音频、权重或机器配置；试听导出采用固定文件清单，详见 [分发与隐私](PRIVACY.md)。

```bash
python3 -B scripts/verify_bundle.py
python3 -B scripts/test_tools.py
```

验收范围和结果见 [validation.json](assets/validation.json)。脚本和格式测试不能代替听音。

已知限制：音频长度不保证等于谱面长度，纯音乐条件也不保证绝无人声。cover 的转谱准确率需要另行核对；不会用裁剪、归一化或 agent 另写旋律来掩盖失败。

本包原创说明和辅助脚本为 MIT；依赖、官方推理代码与模型遵循各自许可，模型权重为 CC BY-NC 4.0。本包不包含或重新许可权重。
