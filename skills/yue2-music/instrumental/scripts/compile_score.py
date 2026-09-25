#!/usr/bin/env python3
"""Compile explicit beat-domain events to YuE2's instrumental ABC dialect.

Standard library only. This is an event compiler, not a general ABC importer.
The agent interprets/repairs the source; this tool checks and serializes its result.
All event times are quarter-note counts, using integers or fraction strings.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import json
from math import lcm
from pathlib import Path
import sys

from abc_tools import DURATIONS, NATURAL, key_accidentals, meter_value, parse_abc, report
from common import write_json

SECTIONS = {'intro', 'verse', 'pre-chorus', 'chorus', 'bridge', 'interlude', 'outro', 'instrumental'}


def fraction(value, context):
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ValueError(f'{context}: use integer or fraction string, not a float')
    try:
        return Fraction(value)
    except (ValueError, ZeroDivisionError) as exc:
        raise ValueError(f'{context}: invalid fraction {value!r}') from exc


def lengths(value):
    if value.denominator != 1 or value <= 0:
        raise ValueError(f'Unrepresentable duration in ABC units: {value}')
    left = int(value)
    result = []
    for part in sorted(DURATIONS, reverse=True):
        while left >= part:
            result.append(part)
            left -= part
    return result


def spelling(pitch, key, active):
    signature = key_accidentals(key)
    prefer_sharp = sum(signature.values()) >= 0
    candidates = []
    for letter, pc in NATURAL.items():
        for alteration in (-1, 0, 1):
            base = pitch - alteration - 60 - pc
            if base % 12:
                continue
            octave = base // 12
            score = (alteration != signature[letter], abs(alteration),
                     (alteration < 0) if prefer_sharp else (alteration > 0))
            candidates.append((score, letter, alteration, octave))
    _, letter, alteration, octave = min(candidates)
    accidental = ''
    if alteration != active.get(letter, signature[letter]):
        accidental = {-1: '_', 0: '=', 1: '^'}[alteration]
        active[letter] = alteration
    name = letter + ',' * (-octave) if octave < 0 else (
        letter if octave == 0 else letter.lower() + "'" * (octave - 1))
    return accidental + name


def compress(bars):
    result, index = [], 0
    while index < len(bars):
        if bars[index] != 'Z':
            result.append(bars[index] + '|')
            index += 1
            continue
        end = index + 1
        while end < len(bars) and bars[end] == 'Z':
            end += 1
        count = end - index
        result.append('Z' + (str(count) if count > 1 else '') + '|')
        index = end
    return ''.join(result)


def compile_events(data):
    if not isinstance(data, dict) or set(data) - {'bpm', 'key', 'bars', 'notes', 'chords'}:
        raise ValueError('Score requires bpm, key, bars, notes and optional chords; unknown fields are rejected')
    bpm = data.get('bpm')
    if type(bpm) is not int or bpm <= 0:
        raise ValueError('bpm must be a positive integer quarter-note tempo')
    key = data.get('key', 'C')
    key_accidentals(key)
    if not isinstance(data.get('bars'), list) or not data['bars']:
        raise ValueError('bars must be a nonempty list')
    bars, time, meter, section = [], Fraction(0), '4/4', None
    unit_denominator = 32
    for i, item in enumerate(data['bars']):
        if not isinstance(item, dict) or set(item) - {'meter', 'key', 'section'}:
            raise ValueError(f'bar {i + 1}: only meter, key and section are supported')
        meter, key = item.get('meter', meter), item.get('key', key)
        n, d = meter_value(meter)
        key_accidentals(key)
        unit_denominator = lcm(unit_denominator, 4 * d)
        incoming = item.get('section', section)
        if incoming is not None and incoming not in SECTIONS:
            raise ValueError(f'bar {i + 1}: section must be one of {sorted(SECTIONS)}')
        section = incoming
        end = time + Fraction(4 * n, d)
        bars.append(dict(start=time, end=end, meter=meter, key=key, section=section))
        time = end
    notes = []
    for i, item in enumerate(data.get('notes', [])):
        if not isinstance(item, list) or len(item) != 3:
            raise ValueError(f'note {i + 1}: expected [onset, duration, MIDI pitch]')
        start, duration = fraction(item[0], f'note {i + 1} onset'), fraction(item[1], f'note {i + 1} duration')
        pitch = item[2]
        if type(pitch) is not int or not 0 <= pitch <= 127:
            raise ValueError(f'note {i + 1}: pitch must be one MIDI integer; reduce polyphony first')
        if start < 0 or duration <= 0 or start + duration > time:
            raise ValueError(f'note {i + 1}: onset/duration is outside the score')
        notes.append((start, pitch, duration))
        unit_denominator = lcm(unit_denominator, (start / 4).denominator, (duration / 4).denominator)
    notes.sort()
    if not notes:
        raise ValueError('Instrumental score has no sounding notes')
    for a, b in zip(notes, notes[1:]):
        if a[0] + a[2] > b[0]:
            raise ValueError(f'Overlapping melody notes at quarter {b[0]}; choose one lead line before compilation')
    from abc_tools import CHORD
    chords = []
    for i, item in enumerate(data.get('chords', [])):
        if not isinstance(item, list) or len(item) != 2:
            raise ValueError(f'chord {i + 1}: expected [onset, symbol]')
        when = fraction(item[0], f'chord {i + 1} onset')
        symbol = item[1]
        if when < 0 or when >= time or not isinstance(symbol, str) or not CHORD.fullmatch(symbol):
            raise ValueError(f'chord {i + 1}: unsupported symbol or onset')
        chords.append((when, symbol))
        unit_denominator = lcm(unit_denominator, (when / 4).denominator)
    chords.sort()
    if any(a[0] == b[0] for a, b in zip(chords, chords[1:])):
        raise ValueError('Conflicting/duplicate chord events at the same onset; choose one harmony timeline')
    if unit_denominator > 1024 or unit_denominator & (unit_denominator - 1):
        raise ValueError('Rhythm is outside the power-of-two training grid; explicitly quantize tuplets and record the changes')
    unit_quarters = Fraction(4, unit_denominator)
    ins, vocal = [], []
    for bar in bars:
        start, end = bar['start'], bar['end']
        events = [x for x in notes if x[0] < end and x[0] + x[2] > start]
        pieces, cursor, active = [], start, {}
        for onset, pitch, duration in events:
            a, b = max(start, onset), min(end, onset + duration)
            if a > cursor:
                pieces.extend('z' + (str(n) if n != 1 else '') for n in lengths((a - cursor) / unit_quarters))
            parts = lengths((b - a) / unit_quarters)
            for i, count in enumerate(parts):
                tie = '-' if i < len(parts) - 1 or b < onset + duration else ''
                pieces.append(spelling(pitch, bar['key'], active) + (str(count) if count != 1 else '') + tie)
            cursor = b
        if cursor < end:
            pieces.extend('z' + (str(n) if n != 1 else '') for n in lengths((end - cursor) / unit_quarters))
        ins.append(''.join(pieces) if events else 'Z')
        current = next((symbol for when, symbol in reversed(chords) if when <= start), None)
        changes = [(start, current)] + [(when, symbol) for when, symbol in chords if start < when < end]
        chunks = []
        for i, (when, symbol) in enumerate(changes):
            stop = changes[i + 1][0] if i + 1 < len(changes) else end
            chunks.append(('"' + symbol + '"') if symbol else '')
            chunks.extend('z' + (str(n) if n != 1 else '') for n in lengths((stop - when) / unit_quarters))
        vocal.append(''.join(chunks) if any(symbol for _, symbol in changes) else 'Z')
    header = ['X:1', 'T:', 'M:' + bars[0]['meter'], f'L:1/{unit_denominator}', f'Q:1/4={bpm}',
        'V: Vocal clef=treble name="Vocal Melody" snm="Vocal"',
        'V: Ins clef=treble name="Ins Melody" snm="Inst."', 'K:' + bars[0]['key']]
    lines, index = header, 0
    while index < len(bars):
        bar = bars[index]
        previous = bars[index - 1] if index else bar
        end = index + 1
        while end < len(bars) and end - index < 4 and all(bars[end][k] == bar[k] for k in ('meter', 'key', 'section')):
            end += 1
        if bar['section'] is not None and (index == 0 or bar['section'] != previous['section']):
            lines.append('% ' + bar['section'])
        for voice, rendered in (('Vocal', vocal), ('Ins', ins)):
            lines.append('V: ' + voice)
            if bar['meter'] != previous['meter']:
                lines.append('M:' + bar['meter'])
            if bar['key'] != previous['key']:
                lines.append('K:' + bar['key'])
            lines.append(compress(rendered[index:end]))
        index = end
    text = '\n'.join(lines) + '\n'
    parsed = parse_abc(text)
    if [tuple(x) for x in parsed.voices['Ins'].notes] != notes or parsed.voices['Vocal'].notes:
        raise ValueError('Internal note roundtrip failed')
    # Compare active harmony; repeating the active symbol at bar starts is native.
    def dedup(events):
        result = []
        for item in events:
            if not result or result[-1][1] != item[1]:
                result.append(tuple(item))
        return result
    if dedup(parsed.voices['Vocal'].chords) != dedup(chords):
        raise ValueError('Internal harmony roundtrip failed')
    check = report(parsed)
    check['event_roundtrip'] = True
    check['scope'] = 'Portable compiler and independent ABC event parser; not the original training exporter'
    return text, check


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('events', type=Path)
    cli.add_argument('--output', type=Path, required=True)
    args = cli.parse_args()
    try:
        text, check = compile_events(json.loads(args.events.read_text(encoding='utf-8')))
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output / 'score.abc').write_text(text, encoding='utf-8')
        write_json(args.output / 'score-check.json', check)
        print(args.output / 'score.abc')
        return 0
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f'Score compilation failed: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
