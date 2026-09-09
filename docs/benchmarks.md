# Benchmark results and evaluation scope

The reported results describe generation with symbolic planning under the recorded automatic-evaluation protocols. They support **frontier quality: competitiveness with the evaluated proprietary song-generation systems**. They do not establish universal metric superiority or human preference.

The [interactive demo results](https://map-yue2.github.io/#model-overview), [full WSB results CSV](benchmark-results.csv), and [WildSongBench dataset](https://huggingface.co/datasets/m-a-p/WildSongBench) provide the public result and benchmark resources.

## WildSongBench

The September 5, 2026 comparison covers **192 prompts and 15 settings**: eight public comparison systems, five proprietary systems, YuE2, and YuE2 (best-of-8). All displayed aggregate metrics cover the same 192 selected outputs per setting.

| Setting | SongBench Avg ↑ | SongEval Avg ↑ | AudioBox PQ ↑ | MuLan ↑ | AllMusicCaps ↑ | Prompt control ↑ | PER ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| YuE2 | 6.7316 | 4.2625 | 8.2598 | 0.5068 | 0.4054 | 4.6819 | 8.44% |
| YuE2 (best-of-8) | 6.9632 | 4.2960 | 8.2714 | 0.5051 | 0.3980 | 4.7009 | 9.79% |

SongBench Avg is the mean of seven dimensions. SongEval is a separate automatic quality evaluator. AudioBox PQ measures production quality; MuLan and AllMusicCaps assess text/audio alignment; prompt control uses Q3O on a 0–5 scale. PER is phoneme error rate, for which lower is better.

**Candidate selection.** Standard YuE2 selects the lower-PER candidate from two generations. Best-of-8 selects by SongBench Musicality, then prompt control, then PER. Each candidate's PER uses the lowest PER from four ASR passes. This selection uses a SongBench dimension and is not equivalent to one unselected pipeline call. Prompt-control weights differ on 10 of 192 prompts between the two selected YuE2 settings.

Public baselines use two candidates and four ASR passes; proprietary systems retain their delivered-candidate protocols. MiniMax Music 3 uses its official caption rewriter and is classified as a public model because its weights are released and the evaluated campaign uses them locally. SongBloom uses a fixed audio prompt. These are documented system comparisons, not matched-compute experiments.

Both YuE2 settings use melody-and-chord planning and the verified evaluation decoder distributed as **[YuE2-Vae-legacy](https://huggingface.co/m-a-p/YuE2-Vae-legacy)**. Default listening uses **[YuE2-Vae](https://huggingface.co/m-a-p/YuE2-Vae)**. Keep their audio separate; use the same cached acoustic latents for decoder comparisons. The release names determine these roles, not the everyday meaning of “legacy.”

**What the lead means.** YuE2 (best-of-8) has the highest observed SongBench Avg, 6.9632; the highest proprietary mean is Mureka 9 at 6.9377, while Suno v5 scores 6.8721. The small gaps are descriptive, not claims of statistical significance. Different systems lead other metrics. The unqualified YuE2 setting scores 6.7316 and is already within the proprietary quality range.

The overview figure combines SongBench and SongEval into a normalized song-quality index and MuLan, AllMusicCaps, and prompt control into a normalized text-alignment index. These axes are comparison indices, not percentages of correct output. Bubble area indicates AudioBox PQ; outlined points are Pareto-optimal on the two plotted axes.

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
