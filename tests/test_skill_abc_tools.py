import importlib.util
import json
import pathlib
import sys
import unittest

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

if __name__=='__main__': unittest.main(verbosity=2)
