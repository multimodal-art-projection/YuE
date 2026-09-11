# Benchmark results and evaluation scope

The reported results describe generation with symbolic planning under the recorded automatic-evaluation protocols. They support **frontier quality: YuE2 is competitive with Suno v5/v6**. They do not establish universal metric superiority or human preference.

The [interactive demo results](https://map-yue2.github.io/#model-overview), [full WSB results CSV](benchmark-results.csv), and [WildSongBench dataset](https://huggingface.co/datasets/m-a-p/WildSongBench) provide the public result and benchmark resources.

## WildSongBench

The September 12, 2026 comparison covers **192 prompts and 17 settings**: eight public comparison systems, seven proprietary systems, YuE2, and YuE2 (best-of-8). All displayed aggregate metrics cover the same 192 selected outputs per setting.

| Setting | SongBench Avg ↑ | SB Musicality ↑ | SongEval Avg ↑ | SE Musicality ↑ | AudioBox PQ ↑ | MuLan ↑ | AllMusicCaps ↑ | Q3O ↑ | PER ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **YuE2 (best-of-8)** | 6.9632 | 6.2666 | 4.2960 | 4.2506 | 8.2714 | 0.5051 | 0.3980 | 4.7009 | 9.79% |
| Mureka 9 | 6.9377 | 6.0488 | 4.4111 | 4.3619 | 8.0226 | 0.4394 | 0.4102 | 4.6368 | 11.69% |
| Suno v5 | 6.8721 | 5.9918 | 4.3579 | 4.3051 | 8.1698 | 0.5428 | 0.4353 | 4.5907 | 8.10% |
| **YuE2** | 6.7316 | 5.9075 | 4.2625 | 4.2151 | 8.2598 | 0.5068 | 0.4054 | 4.6819 | 8.44% |
| Suno v5.5 | 6.7150 | 5.8087 | 4.2152 | 4.1497 | 8.1955 | 0.5089 | 0.3917 | 4.5914 | 5.96% |
| Suno v4.5 | 6.6995 | 5.8317 | 4.3666 | 4.3198 | 8.2541 | 0.5022 | 0.3873 | 4.4149 | 5.80% |
| Suno v6 | 6.5562 | 5.6558 | 4.3086 | 4.2635 | 8.1296 | 0.4916 | 0.4305 | 4.6258 | 7.58% |
| Suno v6 Wild | 6.4195 | 5.5644 | 4.2199 | 4.1716 | 8.1785 | 0.4999 | 0.4316 | 4.5898 | 7.45% |
| LeVo 2 | 6.3247 | 5.4590 | 4.0234 | 3.9819 | 8.3966 | 0.3542 | 0.2680 | 3.9458 | 26.12% |
| MiniMax Music 2.6 | 6.3222 | 5.4437 | 4.0985 | 4.0560 | 8.1711 | 0.4251 | 0.3670 | 4.5688 | 24.55% |
| MiniMax Music 3 | 6.2830 | 5.3482 | 4.0994 | 4.0431 | 8.2825 | 0.3928 | 0.3609 | 4.4362 | 6.27% |
| HeartMuLa | 6.2483 | 5.4963 | 4.5519 | 4.5329 | 8.2933 | 0.3823 | 0.2786 | 3.4907 | 10.71% |
| Muse | 6.0349 | 5.1692 | 3.7887 | 3.7362 | 8.0517 | 0.3937 | 0.3466 | 4.4038 | 33.42% |
| ACE-Step 1.5 | 6.0118 | 5.1588 | 3.8465 | 3.8051 | 8.0518 | 0.4372 | 0.3869 | 4.5809 | 7.46% |
| DiffRhythm 2 | 5.2428 | 4.4775 | 3.5245 | 3.4711 | 7.9782 | 0.3782 | 0.3255 | 4.0870 | 18.41% |
| YuE 1 | 4.9165 | 4.0847 | 3.2150 | 3.1524 | 7.8683 | 0.2623 | 0.2882 | 3.7301 | 36.38% |
| SongBloom | 4.2350 | 3.4493 | 3.2051 | 3.2048 | 8.1539 | 0.2697 | 0.1926 | 3.0287 | 19.19% |

SongBench Avg is the mean of seven dimensions. SongEval Avg is the mean of five dimensions from a separate automatic quality evaluator. SB Musicality and SE Musicality report the respective musicality dimensions. AudioBox PQ measures production quality; MuLan and AllMusicCaps assess text/audio alignment; prompt control uses Q3O on a 0–5 scale. PER is phoneme error rate, for which lower is better.

**Candidate selection.** Standard YuE2 selects the lower-PER candidate from two generations. Best-of-8 selects by SongBench Musicality, then prompt control, then PER. Each candidate's PER uses the lowest PER from four ASR passes. This selection uses a SongBench dimension and is not equivalent to one unselected pipeline call. Prompt-control weights differ on 10 of 192 prompts between the two selected YuE2 settings.

Public baselines, Suno v6, and Suno v6 Wild use two candidates and four ASR passes per candidate, followed by lower-PER selection; earlier proprietary systems retain their delivered-candidate protocols. MiniMax Music 3 uses its official caption rewriter and is classified as a public model because its weights are released and the evaluated campaign uses them locally. SongBloom uses a fixed audio prompt. These are documented system comparisons, not matched-compute experiments.

Both YuE2 settings use melody-and-chord planning and the verified evaluation decoder distributed as **[YuE2-Vae-legacy](https://huggingface.co/m-a-p/YuE2-Vae-legacy)**. Default listening uses **[YuE2-Vae](https://huggingface.co/m-a-p/YuE2-Vae)**. Keep their audio separate; use the same cached acoustic latents for decoder comparisons. The release names determine these roles, not the everyday meaning of “legacy.”

**What the lead means.** YuE2 (best-of-8) has the highest observed SongBench Avg, 6.9632; the highest proprietary mean is Mureka 9 at 6.9377, while Suno v5 scores 6.8721, Suno v6 scores 6.5562, and Suno v6 Wild scores 6.4195. The small gaps are descriptive, not claims of statistical significance. Different systems lead other metrics: Suno v6 has higher SongEval scores than both YuE2 settings, and both v6 variants have lower PER and higher AllMusicCaps scores. The unqualified YuE2 setting scores 6.7316 and is already within the proprietary quality range.

The overview figure combines SongBench and SongEval into a normalized song-quality index and MuLan, AllMusicCaps, and prompt control into a normalized text-alignment index. These axes are comparison indices, not percentages of correct output. Bubble area indicates AudioBox PQ; outlined points are Pareto-optimal on the two plotted axes. The 15 external baselines define the z-score reference; quality weights SongBench Avg and SongEval Avg at 2:1, while alignment weights its three metrics equally. Each axis maps the observed extrema across all 17 settings to 10 and 90. [Figure data and normalization](frontier-composites.json) · [Full-precision CSV](benchmark-results.csv) · [JSON](benchmark-results.json).

## Zero-shot cover generation

The cover evaluation uses **948 SHS100K works**, two requested styles and two seeds per work: **3,792 outputs per method**, without candidate selection. All YuE2 conditions use the general song-generation checkpoint and the benchmark decoder. The generator received no original–cover paired supervision or cover-specific fine-tuning; exclusion of the evaluated works from generator training was confirmed by the authors. This claim does not assert training-data exclusion for external analyzers, evaluators, or comparison models.

| Method | CLEWS mAP ↑ | CLEWS Hit@1 ↑ | Discogs-VINet mAP ↑ | MuLan ↑ | SongBench Musicality ↑ |
|---|---:|---:|---:|---:|---:|
| SongEcho | 0.419 | 48.4% | 0.122 | 0.366 | 3.286 |
| ACE-Step 1.5 | 0.024 | 2.4% | 0.006 | 0.166 | 3.689 |
| YuE2 (full score) | 0.647 | 71.3% | 0.288 | 0.382 | 5.104 |
| YuE2 (without chords) | 0.598 | 67.3% | 0.179 | 0.417 | 5.490 |
| YuE2 (without score) | 0.006 | 0.3% | 0.004 | 0.474 | 5.691 |

CLEWS and Discogs-VINet assess preserved work identity against a source-excluded retrieval gallery of 10,545 recordings per query. MuLan assesses alignment with the requested target style; SongBench measures musicality. Every metric displayed here covers all 3,792 outputs per method. Incomplete Q3O results are omitted.

The full score best preserves work identity in this comparison. Relaxing the supplied score improves target-style alignment and quality in the current cover configuration. These are distinct outcomes: a fixed transcription from a source performance can constrain adaptation to a contrasting style. This does not show that generating a fresh symbolic plan from the current prompt reduces quality. Product guidance recommends melody-only covers to leave accompaniment freer to adapt.

## Editing evidence

A separate paired editing study uses ten original works, two seeds, and 380 full-song recordings. Local changed-note melody attainment increases from 0.0083 to 0.9375; changed-duration harmony attainment increases from 0 to 0.8313. Their units differ and should not be combined into one score. Content outside melody/harmony edits remains close to unedited regeneration on the automatic measurements.

This is evidence for selective control through the score, not identical waveform preservation. The measurements cover ten works and were developed on that cohort. The public agentic demo illustrates a multi-step workflow; it is not a separate statistically controlled human-preference study.
