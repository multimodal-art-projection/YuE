#!/usr/bin/env python3
"""CPU acceptance tests for symbolic meaning and failure behavior; no model required."""
from argparse import Namespace
from fractions import Fraction as F
import json
import random
from pathlib import Path
import tempfile
import unittest
import subprocess
import sys
from unittest.mock import patch

from abc_tools import parse_abc
from compile_score import compile_events
from instrumental import (hashes, load_prepared, prepare, prompt_text, share,
                          validate_request, validate_score, verify, verify_hashes, write_player)
from common import write_json
from verify_bundle import verify as verify_bundle
from instrumentalize import convert_score
import instrumental

ROOT = Path(__file__).resolve().parents[1]


def example_events():
    source = parse_abc((ROOT / 'assets/example-ins.abc').read_text(encoding='utf-8'))
    events = dict(bpm=source.bpm, key='G', bars=[],
        notes=[[str(t), str(d), pitch] for t, pitch, d in source.voices['Ins'].notes],
        chords=[[str(t), name] for t, name in source.voices['Vocal'].chords])
    for i, (_, _, (n, d)) in enumerate(source.voices['Ins'].bars):
        bar = {'meter': f'{n}/{d}'}
        if i in (0, 4):
            bar['section'] = 'verse' if i == 0 else 'outro'
        events['bars'].append(bar)
    return source, events


class SymbolicTests(unittest.TestCase):
    def test_synthetic_reference_melody_and_meter(self):
        source, events = example_events()
        text, check = compile_events(events)
        actual = validate_score(text)
        self.assertEqual(source.voices['Ins'].notes, actual.voices['Ins'].notes)
        self.assertEqual(source.voices['Ins'].bars, actual.voices['Ins'].bars)
        self.assertEqual(len(actual.voices['Ins'].notes), 19)
        self.assertEqual(actual.voices['Ins'].time, 21)
        self.assertEqual(check['nominal_duration_seconds'], 13.125)
        self.assertFalse(actual.voices['Vocal'].notes)
        # Manually authored synthetic fixture: preserve broken rhythm, repeat and tie.
        self.assertEqual(actual.voices['Ins'].notes[:7], [
            [F(0), 74, F(1)], [F(1), 76, F(3, 4)], [F(7, 4), 74, F(1, 4)],
            [F(2), 71, F(1)], [F(4), 66, F(1)], [F(5), 66, F(1)], [F(6), 65, F(1)]])
        self.assertEqual(actual.voices['Ins'].notes[-3:], [
            [F(18), 71, F(1)], [F(19), 67, F(1)], [F(20), 62, F(1)]])
        for point in [F(n, 8) for n in range(21 * 8)]:
            def active(score):
                return next((c for t, c in reversed(score.voices['Vocal'].chords) if t <= point), None)
            self.assertEqual(active(source), active(actual))

    def test_rearticulation_is_not_merged(self):
        text, _ = compile_events({'bpm': 90, 'key': 'C', 'bars': [{}], 'notes': [[0, 2, 60], [2, 2, 60]]})
        self.assertEqual(parse_abc(text).voices['Ins'].notes, [[F(0), 60, F(2)], [F(2), 60, F(2)]])

    def test_long_note_across_meter_and_key_changes(self):
        text, _ = compile_events({'bpm': 90, 'key': 'C', 'bars': [{}, {'meter': '5/8', 'key': 'F#'}],
                                  'notes': [[3, '7/2', 66]]})
        score = parse_abc(text)
        self.assertEqual(score.voices['Ins'].notes, [[F(3), 66, F(7, 2)]])
        self.assertEqual(score.voices['Vocal'].keys, score.voices['Ins'].keys)

    def test_accidentals_across_octaves_and_all_midi_pitches(self):
        for key in ('C', 'Bm', 'Db', 'C#', 'Cb', 'F#m'):
            text, _ = compile_events({'bpm': 80, 'key': key, 'bars': [{}] * 32,
                                      'notes': [[i, 1, i] for i in range(128)]})
            self.assertEqual([p for _, p, _ in parse_abc(text).voices['Ins'].notes], list(range(128)))

    def test_compressed_rests_and_group_boundaries(self):
        text, _ = compile_events({'bpm': 81, 'key': 'C', 'bars': [{}] * 5, 'notes': [[16, 4, 60]]})
        self.assertIn('Z4|', text)
        self.assertEqual(len(parse_abc(text).voices['Ins'].bars), 5)

    def test_nonstandard_integer_duration_is_tied(self):
        text, _ = compile_events({'bpm': 80, 'key': 'C', 'bars': [{}], 'notes': [[0, '5/4', 60]]})
        self.assertNotIn('C10', text)
        self.assertEqual(parse_abc(text).voices['Ins'].notes, [[F(0), 60, F(5, 4)]])

    def test_invalid_event_inputs_fail(self):
        base = {'bpm': 90, 'key': 'C', 'bars': [{}], 'notes': [[0, 4, 60]]}
        cases = [dict(notes=[[0, 5, 60]]), dict(notes=[[-1, 1, 60]]), dict(notes=[[0, 0, 60]]),
                 dict(notes=[[0, 3, 60], [2, 1, 62]]), dict(notes=[[0, 4, [60, 64]]]),
                 dict(notes=[[0, '1/3', 60]]), dict(notes=[[0.5, 1, 60]]), dict(notes=[[0, 4, 128]]),
                 dict(chords=[[0, 'C13']]), dict(chords=[[0, 'C'], [0, 'G']]),
                 dict(chords=[[4, 'C']]), dict(bpm=0), dict(bars=[])]
        for change in cases:
            with self.subTest(change=change), self.assertRaises(ValueError):
                compile_events(dict(base, **change))

    def test_vocal_note_cannot_pass_instrumental_gate(self):
        text = (ROOT / 'assets/example-ins.abc').read_text()
        with self.assertRaises(ValueError):
            validate_score(text.replace('"G"z32', '"G"C32', 1))

    def test_empty_or_tag_only_lyrics(self):
        text, _ = compile_events({'bpm': 90, 'key': 'C', 'bars': [{'section': 'intro'}], 'notes': [[0, 4, 60]]})
        request = dict(id='test', style='Instrumental', lyrics='[Intro]\n', abc=text, cot='melody', seed=1)
        validate_request(request)
        validate_request(dict(request, lyrics=''))
        for changed in (dict(lyrics='la la'), dict(cot='off'), dict(abc=None), dict(seed=True)):
            with self.subTest(change=changed), self.assertRaises((ValueError, TypeError, AttributeError)):
                validate_request(dict(request, **changed))

    def test_preparation_and_tampered_missing_files(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            out = root / 'prepared'
            args = Namespace(abc=ROOT / 'assets/example-ins.abc', events=None,
                source=ROOT / 'assets/example-original.abc', repairs=ROOT / 'assets/example-repairs.json',
                style_file=ROOT / 'assets/example-style.txt', style=None, empty_lyrics=False,
                id='test', seed=831001, output=out)
            self.assertEqual(prepare(args), 0)
            request = load_prepared(out)
            self.assertEqual(request['lyrics'], '[Verse]\n\n[Outro]\n')
            self.assertEqual(request['style'].count('no singing'), 1)
            with self.assertRaises(FileExistsError):
                prepare(args)
            saved = (out / 'score.abc').read_bytes()
            (out / 'score.abc').write_bytes(saved + b' ')
            with self.assertRaises(ValueError):
                load_prepared(out)
            (out / 'score.abc').write_bytes(saved)
            (out / 'lyrics.txt').unlink()
            with self.assertRaises(ValueError):
                load_prepared(out)

    def test_manifest_path_cannot_escape(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ValueError):
                verify_hashes(Path(folder), {'../outside': {'bytes': 0, 'sha256': ''}})

    def test_seeded_roundtrips_across_keys_meters_and_rests(self):
        rng = random.Random(3107)
        for case in range(64):
            bars, notes, chords, cursor = [], [], [], F(0)
            for bar in range(8):
                n, d = rng.choice([(4, 4), (3, 4), (5, 8), (7, 8)])
                bars.append({'meter': f'{n}/{d}', 'key': rng.choice(['C', 'Dm', 'F#', 'Bb'])})
                end = cursor + F(4*n, d)
                chords.append([str(cursor), rng.choice(['C', 'Dm', 'G7', 'Bbmaj7'])])
                while cursor < end:
                    duration = min(rng.choice([F(1, 4), F(1, 2), F(1), F(3, 2)]), end-cursor)
                    if rng.random() > 0.2:
                        notes.append([str(cursor), str(duration), rng.randrange(48, 85)])
                    cursor += duration
            with self.subTest(case=case):
                text, _ = compile_events(dict(bpm=96, key='C', bars=bars, notes=notes, chords=chords))
                actual = validate_score(text)
                expected = [[F(t), p, F(d)] for t,d,p in notes]
                self.assertEqual(actual.voices['Ins'].notes, expected)
                self.assertEqual(actual.voices['Ins'].time, cursor)
                self.assertFalse(actual.voices['Vocal'].notes)


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='instrumental tests ')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def prepared(self, **changes):
        args = dict(abc=ROOT / 'assets/example-ins.abc', events=None,
            source=ROOT / 'assets/example-original.abc', repairs=None,
            style_file=None, style='Warm piano', empty_lyrics=False,
            id='test', seed=831001, output=self.root / 'prepared')
        args.update(changes)
        prepare(Namespace(**args))
        return args['output'], load_prepared(args['output'])

    def call_cli(self, *args):
        return subprocess.run([sys.executable, '-B', str(ROOT / 'scripts/instrumental.py'), *map(str, args)],
            cwd=self.root, capture_output=True, text=True, timeout=30)

    def test_one_command_preparation_from_foreign_directory_with_spaces(self):
        out = self.root / 'new song'
        result = self.call_cli('run', '--abc', ROOT / 'assets/example-ins.abc',
            '--style', 'Warm piano', '--output', out, '--prepare-only')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(load_prepared(out / 'prepared')['cot'], 'full')
        self.assertFalse((out / 'generation').exists())
        self.assertIn('"audio_generated": false', result.stdout)

    def test_event_entry_and_chord_free_empty_lyrics(self):
        events = self.root / 'events.json'
        write_json(events, dict(bpm=80, key='C', bars=[{'section': 'intro'}], notes=[[0, 4, 60]]))
        out = self.root / 'melody'
        result = self.call_cli('run', '--composer', 'agent', '--events', events, '--empty-lyrics', '--output', out, '--prepare-only')
        self.assertEqual(result.returncode, 0, result.stderr)
        request = load_prepared(out / 'prepared')
        self.assertEqual((request['cot'], request['lyrics']), ('melody', ''))
        self.assertEqual(json.loads((out / 'prepared/events.json').read_text()), json.loads(events.read_text()))

    def test_existing_output_is_preserved(self):
        out = self.root / 'existing'
        out.mkdir()
        (out / 'keep.txt').write_text('original')
        result = self.call_cli('run', '--abc', ROOT / 'assets/example-ins.abc', '--output', out, '--prepare-only')
        self.assertEqual(result.returncode, 2)
        self.assertEqual([p.name for p in out.iterdir()], ['keep.txt'])
        self.assertEqual((out / 'keep.txt').read_text(), 'original')

    def test_invalid_json_cli_fails_without_traceback_or_output(self):
        events = self.root / 'invalid.json'
        events.write_text('{invalid')
        out = self.root / 'rejected'
        result = self.call_cli('run', '--composer', 'agent', '--events', events, '--output', out, '--prepare-only')
        self.assertEqual(result.returncode, 2)
        self.assertNotIn('Traceback', result.stderr)
        self.assertFalse(out.exists())

    def test_no_hidden_template_when_score_missing(self):
        result = self.call_cli('run', '--composer', 'agent', '--style', 'Solo piano', '--output', self.root / 'no-score', '--prepare-only')
        self.assertEqual(result.returncode, 2)
        self.assertFalse((self.root / 'no-score').exists())

    def test_unresolved_repair_does_not_create_output(self):
        repairs = self.root / 'repairs.json'
        write_json(repairs, {'unresolved': ['Which melody voice?']})
        with self.assertRaises(ValueError):
            self.prepared(repairs=repairs)
        self.assertFalse((self.root / 'prepared').exists())

    def test_rehashed_style_or_lyric_mismatch_is_rejected(self):
        for name in ('style.txt', 'lyrics.txt'):
            with self.subTest(name=name):
                out, _ = self.prepared(output=self.root / name)
                (out / name).write_text('Different text')
                manifest = json.loads((out / 'prepared-manifest.json').read_text())
                manifest['files'].update(hashes(out, [name]))
                write_json(out / 'prepared-manifest.json', manifest)
                with self.assertRaises(ValueError):
                    load_prepared(out)

    def test_chord_mode_and_instrumental_constraints(self):
        _, request = self.prepared()
        for changed in (dict(cot='melody'), dict(lyrics='[Verse]\nla la'), dict(id='../song'),
                        dict(seed=-1), dict(seed=2**63), dict(abc=None), dict(abc=''), dict(style=' ')):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                validate_request(request | changed)

    def test_prompt_matches_request_and_conditions_are_not_duplicated(self):
        _, request = self.prepared(style='Instrumental, piano, no vocals, no singing, no choir, no spoken words.')
        for condition in ('no vocals', 'no singing', 'no choir', 'no spoken words'):
            self.assertEqual(request['style'].count(condition), 1)
        self.assertIn('[Tags]\n' + request['style'] + '\n[Lyrics]\n' + request['lyrics'], prompt_text(request))

    def test_style_change_preserves_score(self):
        _, piano = self.prepared(style='Solo piano', output=self.root / 'piano')
        _, strings = self.prepared(style='Cinematic strings', output=self.root / 'strings')
        self.assertEqual(piano['abc'], strings['abc'])
        self.assertNotEqual(piano['style'], strings['style'])

    def test_html_does_not_execute_prompt_markup(self):
        out, request = self.prepared(style='Piano <script>alert(1)</script> & strings')
        summary = dict(audio_seconds=1.0, decoder='public decoder', status='complete')
        write_player(out, request, summary, 'audio.flac', shared=True)
        html = (out / 'index.html').read_text()
        self.assertNotIn('<script>', html)
        self.assertIn('&lt;script&gt;', html)
        self.assertNotIn('native/', html)
        self.assertNotIn('repair-report.json', html)

    def generated_fixture(self, mp3=False):
        # Synthetic storage fixture only; real audio generation is tested separately.
        out, request = self.prepared()
        native = out / 'native'
        native.mkdir()
        for name in ('audio.flac', 'prefix.npy', 'semantic.npy', 'latent.npy'):
            (native / name).write_bytes(b'storage test fixture')
        (native / 'score.abc').write_text(request['abc'])
        write_json(native / 'request.json', request | {'cfg_scale': None})
        write_json(native / 'config.json', {'machine_path': '/private/example/model'})
        truncated = {'abc': False, 'semantic': False}
        write_json(native / 'result.json', dict(status='complete', audio_seconds=1.0,
            truncated=truncated, artifacts=hashes(native, [p.name for p in native.iterdir()])))
        write_json(out / 'summary.json', dict(status='complete', audio_seconds=1.0,
            sample_rate=48000, decoder='m-a-p/YuE2-Vae (listening)', seed=1, cot='full',
            package_version='0.1.5', truncated=truncated, audio_adherence_checked=False,
            absence_of_vocals_checked=False, gpu='private machine', weights={'path': '/private/example/model'}))
        (out / 'private.log').write_text('local-only')
        if mp3:
            (out / 'listening.mp3').write_bytes(b'mp3 storage test fixture')
        (out / 'prepared-manifest.json').unlink()
        self.rehash_delivery(out)
        return out

    def rehash_delivery(self, out):
        write_json(out / 'delivery-manifest.json', {'files': hashes(out,
            [p.relative_to(out).as_posix() for p in out.rglob('*') if p.is_file() and p.name != 'delivery-manifest.json'])})

    def test_listening_export_uses_allowlist_and_flac_fallback(self):
        source = self.generated_fixture()
        target = self.root / 'listen'
        share(Namespace(directory=source, output=target))
        self.assertEqual({p.name for p in target.iterdir()},
            {'audio.flac', 'score.abc', 'prompt.txt', 'summary.json', 'index.html', 'share-manifest.json'})
        self.assertEqual((target / 'audio.flac').read_bytes(), (source / 'native/audio.flac').read_bytes())
        self.assertNotIn('gpu', json.loads((target / 'summary.json').read_text()))
        self.assertNotIn('weights', json.loads((target / 'summary.json').read_text()))

    def test_listening_export_mp3_and_refusal_to_overwrite(self):
        source = self.generated_fixture(mp3=True)
        target = self.root / 'listen'
        share(Namespace(directory=source, output=target))
        self.assertTrue((target / 'listening.mp3').is_file())
        with self.assertRaises(FileExistsError):
            share(Namespace(directory=source, output=target))

    def test_rehashed_wrong_generated_prompt_is_rejected(self):
        source = self.generated_fixture()
        (source / 'prompt.txt').write_text('wrong prompt')
        self.rehash_delivery(source)
        with self.assertRaises(ValueError):
            verify(Namespace(directory=source))

    def test_truncated_output_is_not_exported_as_complete(self):
        source = self.generated_fixture()
        summary = json.loads((source / 'summary.json').read_text())
        summary['status'] = 'needs_review'
        write_json(source / 'summary.json', summary)
        self.rehash_delivery(source)
        target = self.root / 'listen'
        with self.assertRaises(ValueError):
            share(Namespace(directory=source, output=target))
        self.assertFalse(target.exists())

    def test_symlink_outside_manifest_root_is_rejected(self):
        outside = self.root / 'outside.txt'
        outside.write_text('outside')
        folder = self.root / 'inside'
        folder.mkdir()
        (folder / 'link').symlink_to(outside)
        with self.assertRaises(ValueError):
            verify_hashes(folder, hashes(folder, ['link']))

    def test_strict_bundle_rejects_extra_files(self):
        # Copy exactly the manifest payload, then simulate an accidentally included log.
        manifest = json.loads((ROOT / 'bundle-manifest.json').read_text())
        folder = self.root / 'bundle'
        for name in list(manifest['files']) + ['bundle-manifest.json']:
            target = folder / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((ROOT / name).read_bytes())
        verify_bundle(folder, strict=True)
        (folder / 'private.log').write_text('not for distribution')
        with self.assertRaises(ValueError):
            verify_bundle(folder, strict=True)


class VoiceTransferTests(unittest.TestCase):
    def native(self, vocal, ins, *, meter='4/4', key='C', extra=''):
        return ('X:1\nT:\nM:'+meter+'\nL:1/32\nQ:1/4=96\n'
            'V: Vocal clef=treble name="Vocal Melody" snm="Vocal"\n'
            'V: Ins clef=treble name="Ins Melody" snm="Inst."\nK:'+key+'\n'
            '% verse\nV: Vocal\n'+vocal+'\nV: Ins\n'+ins+'\n'+extra)

    def test_vocal_moves_without_changing_pitch_onset_or_duration(self):
        text=self.native('"C"C8D8E16|', 'Z|')
        out,check=convert_score(text)
        before,after=parse_abc(text),validate_score(out)
        self.assertEqual(after.voices['Ins'].notes,before.voices['Vocal'].notes)
        self.assertFalse(after.voices['Vocal'].notes)
        self.assertEqual(check['vocal_notes_before'],3)
        self.assertEqual(check['affected_ins_notes'],[])

    def test_instrumental_intro_and_fills_are_retained(self):
        text=self.native('"C"z32|C16z16|', 'E32|z16G16|')
        out,check=convert_score(text)
        self.assertEqual(parse_abc(out).voices['Ins'].notes,
            [[F(0),64,F(4)],[F(4),60,F(2)],[F(6),67,F(2)]])
        self.assertEqual(check['unaltered_ins_notes'],2)

    def test_overlap_keeps_all_vocal_notes_and_logs_ins_reduction(self):
        text=self.native('z8C16z8|', 'G32|')
        out,check=convert_score(text)
        self.assertEqual(parse_abc(out).voices['Ins'].notes,
            [[F(0),67,F(1)],[F(1),60,F(2)],[F(3),67,F(1)]])
        self.assertEqual(check['affected_ins_notes'][0]['retained_segments'],[['0','1'],['3','1']])

    def test_strict_overlap_refuses_lossy_merge(self):
        with self.assertRaisesRegex(ValueError,'overlap'):
            convert_score(self.native('C32|','E32|'),overlap='error')

    def test_fully_overlapped_ins_is_recorded(self):
        out,check=convert_score(self.native('C32|','E32|'))
        self.assertEqual(parse_abc(out).voices['Ins'].notes,[[F(0),60,F(4)]])
        self.assertEqual(check['affected_ins_notes'][0]['retained_segments'],[])

    def test_ties_across_meter_and_key_change_remain_one_note(self):
        text=self.native('"C"z24^F8-|','Z|',extra=
            'V: Vocal\nM:5/8\nK:F#\n"F#"F16-F4|\nV: Ins\nM:5/8\nK:F#\nZ|\n')
        out,_=convert_score(text)
        self.assertEqual(parse_abc(out).voices['Ins'].notes,[[F(3),66,F(7,2)]])
        self.assertEqual(parse_abc(out).voices['Ins'].keys,parse_abc(text).voices['Ins'].keys)

    def test_chord_timing_is_retained(self):
        text=self.native('"C"C8"G7"D8"Am"E16|','Z|')
        out,_=convert_score(text)
        self.assertEqual(parse_abc(out).voices['Vocal'].chords,parse_abc(text).voices['Vocal'].chords)

    def test_cover_removes_chords_without_changing_melody(self):
        text=self.native('"C"C8"G7"D8"Am"E16|','Z|')
        keep,_=convert_score(text)
        free,check=convert_score(text,keep_chords=False)
        self.assertEqual(parse_abc(keep).voices['Ins'].notes,parse_abc(free).voices['Ins'].notes)
        self.assertFalse(parse_abc(free).voices['Vocal'].chords)
        self.assertEqual(check['chords'],'removed for melody cover')

    def test_already_instrumental_score_is_musically_unchanged(self):
        text=(ROOT/'assets/example-ins.abc').read_text()
        out,check=convert_score(text)
        self.assertEqual(parse_abc(text).voices['Ins'].notes,parse_abc(out).voices['Ins'].notes)
        self.assertEqual(check['vocal_notes_before'],0)

    def test_empty_score_is_rejected(self):
        with self.assertRaises(ValueError): convert_score(self.native('Z|','Z|'))

    def test_invalid_overlap_policy_is_rejected(self):
        with self.assertRaises(ValueError): convert_score(self.native('C32|','Z|'),overlap='guess')

    def test_two_sections_at_different_meter_keep_their_locations(self):
        text=self.native('C32|','Z|',extra=
            '% outro\nV: Vocal\nM:3/4\nD24|\nV: Ins\nM:3/4\nZ|\n')
        out,_=convert_score(text)
        self.assertEqual(instrumental.lyric_tags(out),'[Verse]\n\n[Outro]\n')
        self.assertEqual(parse_abc(out).voices['Ins'].bars,parse_abc(text).voices['Ins'].bars)

    def test_inline_key_change_is_not_silently_discarded(self):
        text=self.native('C8[K:G]F24|','z8[K:G]z24|')
        with self.assertRaisesRegex(ValueError,'inline key'):
            convert_score(text)


class ComposerRoutingTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.source=self.root/'model-score.abc'
        self.source.write_text(VoiceTransferTests().native('"C"C8D8E16|','Z|'))

    def call(self, *options):
        with patch.object(sys,'argv',['instrumental.py',*map(str,options)]):
            return instrumental.main()

    def test_default_calls_model_then_transfers_vocal(self):
        out=self.root/'default'
        with patch.object(instrumental,'plan_with_model',return_value=self.source) as planner:
            self.assertEqual(self.call('run','--style','Piano','--output',out,'--prepare-only'),0)
            planner.assert_called_once()
        request=load_prepared(out/'prepared')
        self.assertFalse(parse_abc(request['abc']).voices['Vocal'].notes)
        self.assertEqual(json.loads((out/'workflow.json').read_text())['score_origin'],'YuE2')

    def test_agent_events_require_explicit_opt_in(self):
        events=self.root/'events.json'
        write_json(events,dict(bpm=80,key='C',bars=[{}],notes=[[0,4,60]]))
        with patch.object(instrumental,'plan_with_model') as planner:
            self.assertEqual(self.call('run','--events',events,'--output',self.root/'reject','--prepare-only'),2)
            planner.assert_not_called()
            self.assertEqual(self.call('run','--composer','agent','--events',events,'--output',self.root/'agent','--prepare-only'),0)
            planner.assert_not_called()
        self.assertTrue(json.loads((self.root/'agent/workflow.json').read_text())['agent_composed'])

    def test_supplied_score_bypasses_planning(self):
        with patch.object(instrumental,'plan_with_model') as planner:
            self.assertEqual(self.call('run','--abc',self.source,'--output',self.root/'score','--prepare-only'),0)
            planner.assert_not_called()

    def test_failed_planner_does_not_fall_back_to_agent(self):
        with patch.object(instrumental,'plan_with_model',side_effect=ValueError('truncated plan')):
            self.assertEqual(self.call('run','--output',self.root/'failed','--prepare-only'),2)
        self.assertFalse((self.root/'failed/prepared').exists())

    def test_cover_defaults_to_melody_and_empty_lyric_content(self):
        out=self.root/'cover'
        with patch.object(instrumental,'plan_with_model') as planner:
            self.assertEqual(self.call('cover','--abc',self.source,'--output',out,'--prepare-only'),0)
            planner.assert_not_called()
        req=load_prepared(out/'prepared')
        self.assertEqual(req['cot'],'melody')
        self.assertEqual(req['lyrics'],'[Verse]\n')
        self.assertFalse(parse_abc(req['abc']).voices['Vocal'].notes)

    def test_cover_can_keep_source_harmony(self):
        out=self.root/'keep'
        self.assertEqual(self.call('cover','--abc',self.source,'--keep-harmony','--output',out,'--prepare-only'),0)
        self.assertEqual(load_prepared(out/'prepared')['cot'],'full')

    def test_cover_without_reference_is_rejected(self):
        with patch.object(instrumental,'plan_with_model') as planner:
            self.assertEqual(self.call('cover','--output',self.root/'missing','--prepare-only'),2)
            planner.assert_not_called()

    def test_failed_transcription_cannot_fall_back_to_composition(self):
        with patch.object(instrumental,'transcribe_cover',side_effect=RuntimeError('transcription failed')), \
             patch.object(instrumental,'plan_with_model') as planner:
            self.assertEqual(self.call('cover','--audio',self.root/'ref.wav','--output',self.root/'failed','--prepare-only'),2)
            planner.assert_not_called()

    def test_audio_cover_passes_transcription_to_the_same_transfer(self):
        out=self.root/'audio-cover'
        with patch.object(instrumental,'transcribe_cover',return_value=self.source) as transcriber:
            self.assertEqual(self.call('cover','--audio',self.root/'ref.wav','--output',out,'--prepare-only'),0)
            transcriber.assert_called_once()
        self.assertEqual(json.loads((out/'workflow.json').read_text())['score_origin'],'SheetSage2 transcription')
        self.assertEqual(json.loads((out/'conversion/transfer.json').read_text())['vocal_notes_before'],3)


if __name__ == '__main__':
    unittest.main(verbosity=2)
