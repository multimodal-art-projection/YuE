import base64
from pathlib import Path
import pytest
from yue2.tokenization_yue2 import YuE2TextTokenizer

@pytest.fixture(scope='module')
def native_merge_file(tmp_path_factory):
    path=tmp_path_factory.mktemp('native-bpe')/'qwen.tiktoken'
    tokens=[bytes([i]) for i in range(256)]+[b'\xfffixture'+i.to_bytes(4,'big') for i in range(151643-256)]
    path.write_bytes(b''.join(base64.b64encode(token)+b' '+str(rank).encode()+b'\n' for rank,token in enumerate(tokens)))
    return path

@pytest.mark.parametrize('last_rank', [0,151644,-1])
def test_bpe_rank_table_rejects_duplicate_missing_or_negative_ranks(native_merge_file,tmp_path,last_rank):
    lines=native_merge_file.read_bytes().splitlines()
    token,_=lines[-1].rsplit(b' ',1)
    lines[-1]=token+b' '+str(last_rank).encode()
    path=tmp_path/'qwen.tiktoken'; path.write_bytes(b'\n'.join(lines)+b'\n')
    with pytest.raises(ValueError,match='rank'):
        YuE2TextTokenizer(path)

def test_native_bpe_encode_decode_and_export_roundtrip(native_merge_file,tmp_path):
    tokenizer=YuE2TextTokenizer(native_merge_file)
    expected='Cafe\u0301 and a melody.'
    tokens=tokenizer.encode(expected)
    assert tokenizer.decode(tokens)=='Caf\u00e9 and a melody.'
    exported=tmp_path/'export'
    tokenizer.save_pretrained(exported)
    assert (exported/'qwen.tiktoken').read_bytes()==native_merge_file.read_bytes()
    assert YuE2TextTokenizer.from_pretrained(exported).encode(expected)==tokens
