# 事件编译器

复杂 ABC 可以先写成明确事件，再编译，避免手算休止、连音和分组。所有时间以**整首连续的四分音符**为单位，不是秒、L 单位或小节内偏移。

```json
{
  "bpm": 81,
  "key": "Bm",
  "bars": [
    {"meter": "4/4", "section": "verse"},
    {},
    {"meter": "5/8", "section": "outro"}
  ],
  "notes": [[0, 1, 71], [1, "1/2", 74], ["3/2", "1/2", 78], [2, 3, 74], [5, 1, 73], [6, 2, 71], [8, "5/2", 71]],
  "chords": [[0, "Bm"], [4, "F#"], [8, "Bm"]]
}
```

- notes 每项是 `[起音, 时长, MIDI音高]`。同一长音一项，跨小节自动连音；同音重新起音则另写事件。音高已经应用原调号和临时变音。
- chords 每项是 `[起始时间, 和弦名]`，持续到下个事件；同一时间只能一个和弦。不必每小节重写。
- bars 每项一个小节，meter/key/section 沿用前项。第一小节默认 4/4，初调取顶层 key。上例总长10.5个四分音符。
- 时间只用整数或分数字符串（如 `"3/2"`），不用浮点数。事件间空隙自动补休止。
- 只能单旋律，不能有重叠或把 pitch 写成叠音数组；先由 agent 缩编。支持 MIDI 0–127。
- 编译器选至少 L:1/32 的网格；非2的幂分数如1/3需由 agent 显式量化并记录，脚本不会偷偷舍入。
- 跨小节改调时保持同一实际 MIDI 音高，编译器重拼写，两轨同时变更 M/K。

```bash
python3 scripts/compile_score.py work/events.json --output work/compiled
python3 scripts/abc_tools.py inspect work/compiled/score.abc
```

完整准备可直接用 `instrumental.py prepare --events ...`。它和 `--abc` 二选一。超出总长、重叠音、非法音高、冲突和弦与不可表达时值均报错，须修复后再生成。
