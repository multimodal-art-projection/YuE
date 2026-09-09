import hashlib
import importlib.util
import json
import pathlib
import tempfile
import unittest
import wave
from html.parser import HTMLParser

ROOT=pathlib.Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location('listen',ROOT/'skills/yue2-music/scripts/listen.py')
m=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(m)

class Tags(HTMLParser):
    def __init__(self):super().__init__();self.tags=[];self.sources=[]
    def handle_starttag(self,tag,attrs):
        self.tags.append(tag)
        if tag=='source': self.sources.append(dict(attrs)['src'])

def fixture(root, injected=False):
    root.mkdir()
    with wave.open(str(root/'audio.wav'),'wb') as handle:
        handle.setnchannels(1);handle.setsampwidth(2);handle.setframerate(8000);handle.writeframes(b'\0\0'*800)
    request={'id':'Synthetic test metadata <img src=x onerror=alert(1)>','cot':'off','seed':17,
             'style':'<script>alert("style")</script>','lyrics':'test & <iframe src="https://test.invalid">line</iframe>'} if injected else {'id':'synthetic fixture','cot':'off','style':'Test prompt','lyrics':'Test words'}
    (root/'request.json').write_text(json.dumps(request))
    (root/'config.json').write_text(json.dumps({'decoder_release':'synthetic-test-only'}))
    artifacts={name:m.digest(root/name) for name in ['audio.wav','request.json','config.json']}
    (root/'result.json').write_text(json.dumps({'status':'complete','truncated':{'abc':False,'semantic':False},'artifacts':artifacts,'weights':{'vae':{'test':'synthetic fixture only'}}}))
    (root/'latent.npy').write_text('DO NOT COPY')
    (root/'model.safetensors').write_text('DO NOT COPY')

class ListenChecks(unittest.TestCase):
    def test_html_escaping_and_exact_artifacts(self):
        with tempfile.TemporaryDirectory() as d:
            root=pathlib.Path(d);source=root/'song';fixture(source,True)
            result=m.build([source],root/'page')
            page=(root/'page/index.html').read_text();tags=Tags();tags.feed(page)
            self.assertNotIn('script',tags.tags);self.assertNotIn('iframe',tags.tags);self.assertNotIn('img',tags.tags)
            self.assertIn('&lt;script&gt;',page);self.assertEqual(tags.sources,['case-001/audio.wav'])
            self.assertEqual(result['cases'][0]['status'],'complete')
            self.assertEqual((root/'page/case-001/request.json').read_bytes(),(source/'request.json').read_bytes())
            self.assertFalse((root/'page/case-001/latent.npy').exists())
            self.assertFalse((root/'page/case-001/model.safetensors').exists())
            for path,wanted in result['files'].items():self.assertEqual(m.digest(root/'page'/path),wanted)
            with self.assertRaises(ValueError):m.build([source],root/'page')
    def test_failures_are_visible(self):
        with tempfile.TemporaryDirectory() as d:
            root=pathlib.Path(d);source=root/'failed';source.mkdir()
            (source/'failure.json').write_text(json.dumps({'status':'failed','error':'Synthetic failure for testing'}))
            (source/'input.json').write_text(json.dumps({'lyrics':'Still show intended lyrics','style':'Failed test request'}))
            result=m.build([source,root/'missing'],root/'page')
            self.assertEqual(len(result['cases']),2)
            self.assertTrue(all(x['status']=='needs_review' and x['audio'] is None for x in result['cases']))
            self.assertIn('Still show intended lyrics',(root/'page/index.html').read_text())
            self.assertIn('Synthetic failure',(root/'page/index.html').read_text())
    def test_mismatch_withholds_player(self):
        with tempfile.TemporaryDirectory() as d:
            root=pathlib.Path(d);source=root/'song';fixture(source)
            with (source/'audio.wav').open('ab') as f:f.write(b'mutated')
            result=m.build([source],root/'page');case=result['cases'][0]
            self.assertIsNone(case['audio']);self.assertEqual(case['native_artifact_checks']['audio.wav'],'FAILED')
    def test_secret_metadata_withheld(self):
        with tempfile.TemporaryDirectory() as d:
            root=pathlib.Path(d);source=root/'song';fixture(source)
            (source/'invocation.json').write_text(json.dumps({'token':'SYNTHETIC_TEST_SECRET_ONLY'}))
            (source/'credentials.json').write_text('SYNTHETIC_TEST_SECRET_ONLY')
            result=m.build([source],root/'page')
            self.assertFalse((root/'page/case-001/invocation.json').exists())
            self.assertFalse((root/'page/case-001/credentials.json').exists())
            for path in (root/'page').rglob('*'):
                if path.is_file() and path.suffix in {'.json','.html'}: self.assertNotIn('SYNTHETIC_TEST_SECRET_ONLY',path.read_text())

if __name__=='__main__':unittest.main(verbosity=2)
