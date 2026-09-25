#!/usr/bin/env python3
"""Move a native YuE2/SheetSage2 vocal melody to the instrumental voice."""
from __future__ import annotations
import argparse
from fractions import Fraction
from pathlib import Path
import re
import sys

from abc_tools import parse_abc
from common import fresh_directory, write_json
from compile_score import SECTIONS, compile_events


def section_starts(text, score):
    """Map native section comments to their absolute bar-start times."""
    cursor, pending = 0, None
    result = {}
    for index, line in enumerate(text.splitlines()):
        if line.startswith('% '):
            pending = line[2:]
            if pending not in SECTIONS:
                raise ValueError(f'Unknown section label: {pending}')
        if score.music_lines.get(index) != 'Vocal':
            continue
        if pending is not None:
            result[score.voices['Vocal'].bars[cursor][0]] = pending
            pending = None
        for bar in line[:-1].split('|'):
            rest = re.fullmatch(r'Z([2-4])?', bar.strip())
            cursor += int(rest.group(1) or 1) if rest else 1
    return result


def subtract_intervals(start, duration, occupied):
    pieces = [(start, start + duration)]
    for left, right in occupied:
        remaining = []
        for a, b in pieces:
            if right <= a or left >= b:
                remaining.append((a, b))
            else:
                if a < left:
                    remaining.append((a, left))
                if right < b:
                    remaining.append((right, b))
        pieces = remaining
    return pieces


def convert_score(text, *, overlap='vocal', keep_chords=True):
    if overlap not in ('vocal', 'error'):
        raise ValueError('overlap must be vocal or error')
    if not isinstance(text, str) or not text.strip():
        raise ValueError('Need a nonempty native score')
    source = parse_abc(text)
    vocal, ins = source.voices['Vocal'], source.voices['Ins']
    if not vocal.notes and not ins.notes:
        raise ValueError('The score contains no sounding notes')
    occupied = [(t, t+d) for t, _, d in vocal.notes]
    merged = [list(n) for n in vocal.notes]
    affected = []
    unchanged = 0
    for index, (start, pitch, duration) in enumerate(ins.notes):
        pieces = subtract_intervals(start, duration, occupied)
        if pieces == [(start, start+duration)]:
            unchanged += 1
        else:
            if overlap == 'error':
                raise ValueError('Vocal and Ins overlap; choose vocal priority or edit the arrangement explicitly')
            affected.append(dict(ins_note=index, onset=str(start), pitch=pitch, duration=str(duration),
                retained_segments=[[str(a), str(b-a)] for a,b in pieces]))
        merged.extend([a, pitch, b-a] for a,b in pieces)
    merged.sort()
    sections = section_starts(text, source)
    starts = {start for start, _, _ in vocal.bars}
    if any(when not in starts for when, _ in vocal.keys):
        raise ValueError('An inline key change occurs inside a bar; normalize its spelling without changing pitches first')
    bars = []
    for start, _, (n,d) in vocal.bars:
        key = next(key for t,key in reversed(vocal.keys) if t <= start)
        bar = dict(meter=f'{n}/{d}',key=key)
        if start in sections:
            bar['section'] = sections[start]
        bars.append(bar)
    chords = []
    if keep_chords:
        for when,symbol in vocal.chords:
            if chords and chords[-1][0] == str(when):
                if chords[-1][1] != symbol:
                    raise ValueError('Conflicting chord labels at the same time')
                continue
            chords.append([str(when),symbol])
    events = dict(bpm=source.bpm,key=vocal.keys[0][1],bars=bars,
        notes=[[str(t),str(d),p] for t,p,d in merged],chords=chords)
    converted, _ = compile_events(events)
    target = parse_abc(converted)
    if target.voices['Ins'].notes != merged or target.voices['Vocal'].notes:
        raise ValueError('Voice transfer changed the intended note events')
    if target.voices['Ins'].bars != ins.bars or target.bpm != source.bpm:
        raise ValueError('Voice transfer changed meter, timing or tempo')
    for note in vocal.notes:
        if note not in target.voices['Ins'].notes:
            raise ValueError('A vocal note was not preserved exactly')
    def harmony(chord_list):
        result=[]
        for event in chord_list:
            if not result or result[-1][1] != event[1]: result.append(event)
        return result
    if keep_chords and harmony(target.voices['Vocal'].chords) != harmony(vocal.chords):
        raise ValueError('Voice transfer changed harmony')
    check = dict(vocal_notes_before=len(vocal.notes),vocal_notes_after=0,
        original_ins_notes=len(ins.notes),unaltered_ins_notes=unchanged,
        output_ins_notes=len(merged),vocal_pitch_onset_duration_preserved=True,
        meter_and_tempo_preserved=True,chords='preserved' if keep_chords else 'removed for melody cover',
        overlap_policy=overlap,affected_ins_notes=affected,
        nominal_seconds=float(ins.time*60/source.bpm))
    return converted, check


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('abc',type=Path)
    cli.add_argument('--output',type=Path,required=True)
    cli.add_argument('--overlap',choices=('vocal','error'),default='vocal')
    cli.add_argument('--drop-chords',action='store_true')
    args=cli.parse_args()
    try:
        text,check=convert_score(args.abc.read_text(encoding='utf-8'),overlap=args.overlap,keep_chords=not args.drop_chords)
        out=fresh_directory(args.output)
        (out/'score.abc').write_text(text,encoding='utf-8')
        write_json(out/'transfer.json',check)
        return 0
    except (ValueError,KeyError,TypeError,OSError) as exc:
        print(f'{type(exc).__name__}: {exc}',file=sys.stderr)
        return 2


if __name__=='__main__':
    raise SystemExit(main())
