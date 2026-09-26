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

The cover evaluation uses **948 SHS100K test works**, two requested styles and two seeds per work: **3,792 outputs per method**, without candidate selection. All YuE2 conditions use the general song-generation checkpoint and the benchmark decoder. The generator received no original–cover paired supervision or cover-specific fine-tuning; exclusion of the evaluated works, including alternate performances, from generator training was verified using CLEWS and Discogs-VINet. This claim concerns YuE2's generator training; it does not assert training-data exclusion for external analyzers, evaluators, or comparison models.

One recording supplies each source score and its automatically transcribed lyrics. Two other performances of the same work supply contrasting target-style descriptions. YuE2 receives the fixed score, lyrics, and target style; SongEcho and ACE-Step 1.5 receive source audio, lyrics, and target style through their native cover interfaces. YuE2's three conditions share the generator and decoder. Removing chords also changes score serialization.

**CLEWS work-identity retrieval**

| Method | mAP ↑ | MRR ↑ | Hit@1 (%) ↑ | Hit@5 (%) ↑ |
|---|---:|---:|---:|---:|
| SongEcho | 0.419 | 0.536 | 48.4 | 58.8 |
| ACE-Step 1.5 | 0.024 | 0.036 | 2.4 | 4.1 |
| YuE2 (full score) | 0.647 | 0.748 | 71.3 | 78.6 |
| YuE2 (without chords) | 0.598 | 0.715 | 67.3 | 76.1 |
| YuE2 (without score) | 0.006 | 0.008 | 0.3 | 0.9 |

**Discogs-VINet work-identity retrieval**

| Method | mAP ↑ | MRR ↑ | Hit@1 (%) ↑ | Hit@5 (%) ↑ |
|---|---:|---:|---:|---:|
| SongEcho | 0.122 | 0.227 | 16.6 | 28.4 |
| ACE-Step 1.5 | 0.006 | 0.014 | 0.6 | 1.4 |
| YuE2 (full score) | 0.288 | 0.438 | 37.5 | 50.3 |
| YuE2 (without chords) | 0.179 | 0.304 | 23.6 | 36.6 |
| YuE2 (without score) | 0.004 | 0.008 | 0.2 | 0.8 |

Both encoders rank the same gallery of 10,546 recordings from 1,692 works. Each generated query excludes its exact source recording, leaving **10,545 candidates**; other recordings of the source work are relevant matches. mAP measures average precision, MRR measures the reciprocal rank of the first relevant result, and Hit@k is the percentage of queries with a relevant result in the first k ranks. All eight retrieval metrics cover all 3,792 outputs per method. **Full-score YuE2 leads both evaluated cover systems on all eight retrieval measures.**

**Target-style alignment and audio quality**

| Method | MuLan ↑ | Q3O ↑ | AudioBox PQ ↑ | SongBench Musicality ↑ |
|---|---:|---:|---:|---:|
| SongEcho | 0.366 | 4.474 | 6.862 | 3.286 |
| ACE-Step 1.5 | 0.166 | 4.190 | 6.918 | 3.689 |
| YuE2 (full score) | 0.382 | 4.273 | 8.044 | 5.104 |
| YuE2 (without chords) | 0.417 | 4.482 | 8.117 | 5.490 |
| YuE2 (without score) | 0.474 | 4.837 | 8.186 | 5.691 |

MuLan measures alignment with the requested target-style text, excluding lyrics. AudioBox PQ measures production quality, and SongBench measures Musicality; these three columns cover all **3,792 outputs per method**. Q3O is a 0–5 target-style score on the same **3,286 outputs per method (86.7%)**, matched across methods by source, style, and seed. This available-case subset covers 933 works: 710 contribute four outputs and 223 contribute two; 506 outputs per method are unscored. It is not the full-cohort Q3O mean. Qwen3-Omni supplies both target-style annotations and Q3O judgments. The downloadable results also retain the supplementary AllMusicCaps scores.

Full-score YuE2 leads the two external systems in production quality and Musicality as well as retrieval. Within YuE2, retaining source harmony gives the strongest work-identity retrieval, while removing chords allows greater target-style adaptation and raises quality scores. Removing the entire score further raises alignment and quality while losing work identity. These findings concern rendering a fixed source composition in a contrasting style. The [cover guide](covers.md) explains the full-score and melody-only settings.

[Cover results CSV](cover-results.csv) · [Scores, units, and metric coverage JSON](cover-results.json)

## Editing evidence

The controlled score-editing evaluation uses the first generated score for each of **192 WildSongBench prompts**, with two seeds per applicable condition and **3,844 full-song recordings**, including unedited controls. All outputs are retained, including 32 at the six-minute cap. Lyrics, style, model, benchmark decoder, and sampling settings stay fixed. Scores are averaged over seeds and edit strengths within each source, then over sources with equal weight.

**Edit adherence**

| Edit | Adherence metric | Score (0–100) ↑ | Sources | Recordings |
|---|---:|---:|---:|---:|
| Melody | Target-pitch accuracy | 84.17 | 190 | 380 |
| Harmony | Target-chord agreement | 79.54 | 181 | 362 |
| Rhythm | Relative onset accuracy | 73.43 | 107 | 214 |
| Key | Weighted key score | 90.58 | 191 | 1528 |
| Tempo | Acc2 | 95.68 | 191 | 764 |

Melody and rhythm edits affect up to four bars of the first chorus. Melody edits raise vocal pitches by two scale steps; rhythm edits exchange adjacent quarter- and eighth-note durations while preserving pitches and each pair's total duration. Harmony edits change chord roots and bass notes while preserving chord quality and timing, and extend substitutions to matching phrases. Key edits transpose the complete score by −5, −2, +2, or +5 semitones. Tempo edits multiply BPM by 0.8 or 1.2, with the target rounded to an integer. Eligibility depends on the applicable score content: 190 melody, 181 harmony, 107 rhythm, and 191 key/tempo sources.

Harmony adherence and all content-preservation scores use SheetSage2-AR; melody, rhythm, and key adherence use SheetSage2-Prober. The public AR analysis uses 300-second windows, 200-second overlap, and 100-second lookahead. Tempo is estimated directly from audio using madmom; the target score does not enter beat tracking.

- **Melody:** exact target-pitch accuracy on edited notes.
- **Harmony:** duration-weighted root-and-triad agreement in the edited chorus window. Unchanged vocal notes locate the window.
- **Rhythm:** relative onset intervals after normalizing local offset and tempo; an interval must be within 1/8 beat of the edited target and closer to it than to the original timing. Unresolved correspondences count as misses. Correspondence coverage is **80.36%** across 210 source note pairs.
- **Key:** agreement with target tonic and mode, with partial credit for related keys.
- **Tempo:** Acc2 accepts BPM estimates within 4% of the rounded target times any of {1/3, 1/2, 1, 2, 3}. The stricter target-beat-level Acc1 is **75.92%** on the same 764 recordings.

Missing events and unresolved contexts score zero for melody/harmony adherence. The rows measure different aspects of control and are not combined into an overall adherence score.

**Content preservation**

| Edit | Melody agreement (%) ↑ | Harmony agreement (%) ↑ |
|---|---:|---:|
| Melody | 93.10 | 94.05 |
| Harmony | 93.43 | 90.16 |
| Rhythm | 94.34 | 93.08 |

These scores compare unedited melody and harmony with the source composition. Melody and rhythm edits are evaluated outside the edited content; harmony edits retain the full vocal melody and compare unchanged chords outside all edited spans. Undefined preservation scores are excluded. Melody-edit preservation covers **368 recordings from 184 sources**; rhythm-edit preservation covers **207 from 104**. For harmony edits, melody agreement covers **362 from 181**, while unchanged-chord agreement covers **340 from 170**. Each metric averages available seeds within its evaluable sources, then weights sources equally.

**Song quality and lyric retention**

| Condition | SongBench Avg ↑ | PER ↓ | CER ↓ |
|---|---:|---:|---:|
| Unedited | 6.704 | 0.184 | 0.181 |
| Melody | 6.717 | 0.191 | 0.181 |
| Harmony | 6.674 | 0.179 | 0.175 |
| Key | 6.639 | 0.190 | 0.188 |
| Unedited (rhythm) | 6.728 | 0.214 | 0.197 |
| Rhythm | 6.731 | 0.185 | 0.175 |

SongBench Avg averages seven quality dimensions on a 0–10 scale. PER and CER are phoneme and character error rates, shown as ratios; lower is better. They use greedy audio-only Qwen3-ASR transcripts. The rhythm rows use their own matched source scores and generation seeds. Source and recording counts for every metric are included in the downloadable results.

The results support selective control of musical content through the score. Editing regenerates the entire recording; preserved score content does not imply an identical waveform, voice, or timbre. These tables report point estimates, without a claim of statistical significance or human preference. The [agentic editing demo](https://map-yue2.github.io/#agentic-music-editing) illustrates a multi-step workflow; it is distinct from this controlled score-editing evaluation. See the [editing guide](editing.md) for usage.

[Editing results CSV](editing-results.csv) · [Scores, effective denominators, and metric definitions JSON](editing-results.json)
