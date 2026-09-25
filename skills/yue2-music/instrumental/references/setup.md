# 安装、迁移和验收

修谱/准备工具：Python 3.10+ 标准库即可。推理：Linux、Python 3.10+（本包验证用 3.12）、支持 BF16 的 NVIDIA 24 GB 级 GPU，至少 24 GB 可用主机内存。FFmpeg 可选，用于导出方便播放的 MP3；没有时仍交付无损 FLAC。GPU 与驱动由集群提供。

本包不携带模型、PyTorch、推理 wheel 或凭据。推理 wheel 来自公开官方仓库，固定 **yue2-infer 0.1.5**；模型的精确 revision、必需文件及哈希写在 `assets/release.json`。默认两个权重文件合计约 7.8 GB；环境依赖另占空间。此文件是唯一版本来源，不用 latest 替换它。

## 新集群

在解压后的 skill 目录执行；其他目录下使用脚本绝对路径。

```bash
# 先检查包，尚不安装依赖。
python3 scripts/verify_bundle.py
python3 scripts/test_tools.py

# 使用目标集群可用的 Python 3.10+，创建独立环境。
python3 scripts/setup_runtime.py install --venv .venv

# 只下载推理必需模型文件，不下载 demo 音频和整份开发仓库。
.venv/bin/python scripts/setup_runtime.py models --output models

# 此步必须在真正的 GPU allocation 中执行。
.venv/bin/python scripts/setup_runtime.py doctor --require-gpu --models-root models
```

官方 wheel 的 SHA-256 被固定到 requirements URL 中。wheel 自带核心依赖的版本要求，doctor 会核对这些版本。安装失败先查 Python/CUDA/网络及具体报错，不安装同名但来源不明的 PyPI 包。若已存在专用环境，运行 doctor 验证后复用，不破坏别人的环境。

模型下载支持既有 Hugging Face 身份和缓存；不要把 token 写进 prompt、脚本、报告或聊天。使用网络代理/镜像时保持 revision 与哈希核验，不把失败的下载标成成功。

默认试听模型：`m-a-p/YuE2-3B` + `m-a-p/YuE2-Vae`。`YuE2-Vae-legacy` 是指标评测版，不在本纯音乐试听配方中下载。

## 本机新生成验收

在目标机器上实际生成一次，可直接使用用户本次的新作作为首跑验收。下面提供固定示例供安装排错使用，不要求每次创作前都额外生成。读取旧预测不能代替本机测试。

```bash
.venv/bin/python scripts/instrumental.py run \
  --abc assets/example-ins.abc --source assets/example-original.abc \
  --style-file assets/example-style.txt --repairs assets/example-repairs.json \
  --models-root models --offline --output work/smoke-song
python3 scripts/instrumental.py verify work/smoke-song/generation
```

验收以验证器为准：输入有发声 Ins、Vocal 无音符、无歌词正文、逐小节对齐；实际保存的 prefix/ABC 与请求一致；新音频有限且非全零；未截断；token、latent、音频和配置存在且哈希匹配。生成后打开 `work/smoke-song/listen/index.html` 试听。此测试是接口、格式与迁移可用性验证，不是音乐质量基准或无人声检测。

`run --prepare-only` 在生成音频前停止；提供 ABC 时只需 CPU，默认的新作品仍需 GPU 让 YuE2 写谱。之后用 `generate --prepared work/song/prepared --output work/song/generation` 继续生成，再用 `share work/song/generation --output work/song/listen` 导出试听；这些子命令均通过 `scripts/instrumental.py` 调用，生成时沿用模型目录及 offline 参数。

不同 GPU/库实现可能产生不同采样音频，因此不要求跨设备逐字节匹配参考音频。若要判断音质或人声泄漏，应听音，或另做明确的音频评估；这里不编造评分门槛。

## 已有权重和离线使用

`--models-root /path/to/models` 指向包含 `YuE2-3B/` 与 `YuE2-Vae/` 的目录。用 doctor 校验文件哈希，允许权重在共享存储，不必复制。`--offline` 禁止生成时联网取模型。

若目标 GPU 节点不能联网，可在联网节点运行安装/下载，使用共享的模型目录；环境应按目标节点 ABI/CUDA 配置安装。不要把不可搬迁的整个 venv 当作通用软件包。下载命令的 `--offline` 仅用于已存在 Hugging Face 缓存，并不能凭空得到模型。

## 调度和故障

先查 `nvidia-smi`、可用解释器、集群现有调度说明。在 Slurm 上通过正常 sbatch/srun 申请一张卡；账号、分区、QOS 按用户集群设置，不能复制开发机账号或借用他人已占 GPU。谱面准备可以在登录节点完成。

GPU doctor 失败说明还未到适用 GPU 环境。OOM 时保留失败记录，更换足够资源，不缩短曲目或降低推理默认值。网络、安装或资源问题最多做有证据的修复重试，不无限循环下载/提交作业。生成中断不伪装为完成；保留失败目录，以新目录重新运行。

脚本退出码：0=通过/完成；1=生成产物需复核（如截断）；2=参数、格式、依赖、资源或完整性错误。输出目录拒绝覆盖。

官方接口依据：[YuE2-3B](https://huggingface.co/m-a-p/YuE2-3B)、[试听 VAE](https://huggingface.co/m-a-p/YuE2-Vae)。发布配方固定于 `assets/release.json`，而不是随网页自动升级。
