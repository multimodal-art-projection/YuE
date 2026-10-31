import importlib.util
import json
import pathlib
import sys
import unittest
from pathlib import Path
import pytest
import hashlib
import subprocess


ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('abc_tools', ROOT / 'skills/yue2-music/scripts/abc_tools.py')
a = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = a
SPEC.loader.exec_module(a)

def score(vocal, ins='Z|', meter='4/4', key='C', extra=''):
    return ('X:1\nT:\nM:' + meter + '\nL:1/32\nQ:1/4=88\n'
            'V: Vocal clef=treble name="Vocal Melody" snm="Vocal"\n'
            'V: Ins clef=treble name="Ins Melody" snm="Inst."\n'
            'K:' + key + '\n% verse\nV: Vocal\n' + vocal + '\nV: Ins\n' + ins + '\n' + extra)

class ABCChecks(unittest.TestCase):
    def test_key_relative_pitch(self):
        self.assertEqual(a.parse(score('F8C8=F8F8|', key='D')).voices['Vocal'].notes,
            [[0,66,1],[1,61,1],[2,65,1],[3,65,1]])
    def test_octave_accidental_propagation(self):
        self.assertEqual([x[1] for x in a.parse(score('^F8f8=F8f8|')).voices['Vocal'].notes], [66,78,65,77])
    def test_bar_accidental_reset(self):
        self.assertEqual([x[1] for x in a.parse(score('^F32|F32|', 'Z2|')).voices['Vocal'].notes],[66,65])
    def test_crossbar_tie(self):
        notes=a.parse(score('^F32-|F8F24|','Z2|')).voices['Vocal'].notes
        self.assertEqual(notes, [[0,66,5],[5,65,3]])
    def test_chord_split_tie(self):
        p=a.parse(score('"C"C16-"Am7"C16|'))
        self.assertEqual(p.voices['Vocal'].notes, [[0,60,4]])
        self.assertEqual(p.voices['Vocal'].chords, [(0,'C'),(2,'Am7')])
    def test_double_accidentals(self):
        self.assertEqual([x[1] for x in a.parse(score('^^F16__B16|')).voices['Vocal'].notes], [67,69])
    def test_minor_key(self):
        self.assertEqual(a.parse(score('F32|',key='F#m')).voices['Vocal'].notes[0][1],66)
    def test_inline_key_changes(self):
        p=a.parse(score('F16[K:D]F16|','z16[K:D]z16|'))
        self.assertEqual([x[1] for x in p.voices['Vocal'].notes],[65,66])
    def test_compound_meter(self):
        self.assertEqual(a.parse(score('C8D8E8|',meter='6/8')).voices['Vocal'].time,3)
    def test_meter_change(self):
        tail='V: Vocal\nM:3/4\nD24|\nV: Ins\nM:3/4\nZ|\n'
        self.assertEqual(a.parse(score('C32|',extra=tail)).voices['Vocal'].time,7)
    def test_compressed_rests(self):
        self.assertEqual(len(a.parse(score('Z4|','Z4|')).voices['Vocal'].bars),4)
    def test_strip_header_preserved(self):
        src=score('"Cmaj7/G"C16"Dbaug"C16|','E32|')
        dest=a.strip_chords(src)
        self.assertIn('name="Vocal Melody"',dest)
        self.assertIn('name="Ins Melody"',dest)
        self.assertTrue(a.compare(a.parse(src),a.parse(dest))['match'])
        self.assertEqual(a.parse(dest).voices['Vocal'].chords,[])
    def test_voice_selection(self):
        src=score('"C"C32|','E32|')
        dest=a.parse(a.strip_chords(src,'Ins'))
        self.assertEqual(dest.voices['Vocal'].notes,[])
        self.assertEqual(dest.voices['Ins'].notes,[[0,64,4]])
    def test_equal_sound_different_ties(self):
        self.assertTrue(a.compare(a.parse(score('C32|')),a.parse(score('C16-C16|')))['match'])
    def test_rearticulation_changes(self):
        self.assertFalse(a.compare(a.parse(score('C32|')),a.parse(score('C16C16|')))['match'])
    def test_onset_changes(self):
        self.assertFalse(a.compare(a.parse(score('C16z16|')),a.parse(score('z16C16|')))['match'])
    def test_tempo_change_explicit(self):
        p=a.parse(score('C32|')); q=a.parse(score('C32|').replace('=88','=80'))
        self.assertFalse(a.compare(p,q)['match'])
        self.assertTrue(a.compare(p,q,allow_tempo_change=True)['match'])
    def test_bad_input_cases(self):
        examples=[score('C16|'), score('C48|'), score('C32-|'), score('C32-|D32|','Z2|'),
          score('C32-|z32|','Z2|'), score('C32-|=C32|','Z2|',key='D').replace('C32-|','^C32-|'),
          score('z32-|'), score('^z32|'), score('C10C22|'), score('C/2C31|'),
          score('(3C8D8E8|'), score('[CEG]32|'), score('C16>C16|'), score('{C}D32|'),
          score('C32||'), score('C32:|'), score('Z1|'), score('Z5|'),
          score('"C13"C32|'), score('"A7alt"C32|'), score('C32"C"|'),
          score('C32|','"C"Z|'),score('C16[K:D]C16|'),
          score('C32|').replace('L:1/32','L:1/3'),
          score('C32|').replace('K:C','K:Cmix'),score('C32|').replace('% verse','%%MIDI program 1'),
          score('C32|').replace('C32|','C32|C32|C32|C32|C32|'),
          score('C32|','Z2|')]
        for i,text in enumerate(examples):
            with self.subTest(case=i):
                with self.assertRaises(a.AbcError): a.parse(text)

    def test_city_lights_sections_and_seven_onsets(self):
        request = json.loads((ROOT / 'examples/song.json').read_text(encoding='utf-8'))
        abc = (ROOT / 'examples/score.abc').read_text(encoding='utf-8')
        lines = [
            'Neon fades along the lane',
            'Footsteps keep the time of rain',
            'Fold the night and leave it here',
            'Morning has a sky to clear',
            'Let the day come into view',
            'Every road begins with you',
            'Hold a little room for light',
            'We will sing beyond the night',
        ]
        self.assertEqual([a.estimate_syllables(line) for line in lines], [7] * 8)
        report = a.align_lyrics(request['lyrics'], abc, require_line_per_bar=True,
                                require_onset_syllables=True)
        self.assertTrue(report['match'], report['differences'])
        self.assertEqual([pair['lyric_name'] for pair in report['pairs']], ['verse', 'chorus'])
        self.assertEqual([pair['score_measures'] for pair in report['pairs']], [4, 4])
        self.assertEqual(report['pairs'][0]['onsets_per_bar'], [7, 7, 7, 7])

    def test_section_name_mismatch_and_line_count(self):
        lyrics = '[Chorus]\nOne line only\n'
        report = a.align_lyrics(lyrics, score('"C"C32|'))
        self.assertFalse(report['match'])
        self.assertTrue(any('chorus' in item and 'verse' in item for item in report['differences']))
        strict = a.align_lyrics('[Verse]\nOne line\nTwo lines\n[Chorus]\nOnly one\n',
                                (ROOT / 'examples/score.abc').read_text(encoding='utf-8'),
                                require_line_per_bar=True)
        self.assertFalse(strict['match'])
        self.assertTrue(any('lyric lines vs' in item for item in strict['differences']))

    def test_mandarin_ideographs_count_as_syllables(self):
        self.assertEqual(a.estimate_syllables('晚风轻轻吹过窗前'), 8)
        self.assertEqual(a.normalize_section('Verse 1'), 'verse 1')
        self.assertEqual(a.normalize_section('PRE-CHORUS'), 'pre-chorus')

if __name__=='__main__': unittest.main(verbosity=2)


SCRIPT = Path(__file__).resolve().parents[1]/'skills/yue2-music/scripts/abc_tools.py'
SCORE = ('X:1\r\nT:\r\nM:4/4\r\nL:1/32\r\nQ:1/4=88\r\n'
    'V: Vocal clef=treble name="Vocal Melody" snm="Vocal"\r\n'
    'V: Ins clef=treble name="Ins Melody" snm="Inst."\r\nK:C\r\n'
    '% verse\r\nV: Vocal\r\n"C"C32|\r\nV: Ins\r\nZ|\r\n')

def test_inspect_hashes_original_crlf_bytes(tmp_path):
    source = tmp_path/'source.abc'; source.write_bytes(SCORE.encode())
    result = subprocess.run([sys.executable,str(SCRIPT),'inspect',str(source)],capture_output=True,text=True,check=True)
    assert json.loads(result.stdout)['sha256'] == hashlib.sha256(source.read_bytes()).hexdigest()

def test_strip_chords_preserves_original_line_endings(tmp_path):
    source = tmp_path/'source.abc'; source.write_bytes(SCORE.encode())
    destination = tmp_path/'stripped.abc'
    subprocess.run([sys.executable,str(SCRIPT),'strip-chords',str(source),str(destination)],check=True,capture_output=True)
    assert destination.read_bytes() == SCORE.replace('"C"','').encode()
    result = subprocess.run([sys.executable,str(SCRIPT),'compare',str(source),str(destination)],check=True,capture_output=True,text=True)
    assert json.loads(result.stdout)['match'] is True

def test_strip_chords_disables_windows_output_newline_translation(tmp_path, monkeypatch):
    import importlib.util
    spec = importlib.util.spec_from_file_location('abc_windows', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    source = tmp_path/'source.abc'; source.write_bytes(SCORE.encode())
    destination = tmp_path/'stripped.abc'
    original_open = Path.open
    def windows_open(path, mode='r', *args, **kwargs):
        # Exercise Python's actual text translation, including the Windows
        # default for omitted newline; explicit newline='' bypasses it.
        if 'b' not in mode and kwargs.get('newline') is None:
            kwargs['newline'] = '\r\n'
        return original_open(path, mode, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', windows_open)
    monkeypatch.setattr(sys, 'argv', [str(SCRIPT),'strip-chords',str(source),str(destination)])
    assert module.main() == 0
    assert destination.read_bytes() == SCORE.replace('"C"','').encode()
