"""CPU contract checks for progress integration, without loading song weights."""
import json
from types import SimpleNamespace

import pytest
import torch

from yue2 import cli, fast, nar, pipeline
from yue2.pipeline import YuE2Pipeline
from yue2.progress import Progress
from yue2.protocol import ABC_END, CODEC_OFFSET, MUSIC_END, GenerationConfig, SongRequest


class Tokenizer:
    def encode(self, text):
        return [ord(char) for char in text]

    def decode(self, ids):
        return ''.join(chr(token) for token in ids)


def bare_pipe(enabled=True, backend='torch-eager'):
    pipe = object.__new__(YuE2Pipeline)
    pipe.progress, pipe.backend = enabled, backend
    pipe.tokenizer = Tokenizer()
    pipe.generation_config = GenerationConfig()
    pipe._load_model = lambda **kwargs: object()
    return pipe


@pytest.mark.parametrize('backend', ['torch-eager', 'vllm'])
@pytest.mark.parametrize('phase', ['abc', 'semantic'])
def test_callback_coexists_with_progress_exactly_once_and_quiet_is_inert(backend, phase, monkeypatch, capsys):
    emitted = [65, 66, ABC_END] if phase == 'abc' else [CODEC_OFFSET, CODEC_OFFSET + 1, MUSIC_END]
    calls = []

    def generate(subject, prefix, sampling, seed, actual_phase, on_token=None, **kwargs):
        calls.append((prefix, sampling, seed, actual_phase, kwargs))
        for token in emitted:
            if on_token is not None:
                on_token(actual_phase, token)
        return emitted[:-1], {'output_tokens': len(emitted)}, False

    monkeypatch.setattr(pipeline, 'generate_tokens', generate)
    monkeypatch.setattr(fast, 'generate_vllm', generate)
    results = []
    rng_state = torch.random.get_rng_state().clone()
    for enabled in [True, False]:
        pipe = bare_pipe(enabled, backend)
        external = []
        results.append(pipe._generate([1] * 120, pipe.generation_config.semantic, 42, phase,
                       negative=[2] * 80, cfg_scale=1.2,
                       on_token=lambda name, token: external.append((name, token))))
        assert external == [(phase, token) for token in emitted]
        captured = capsys.readouterr()
        assert captured.out == ''
        if enabled:
            label = 'Planning score' if phase == 'abc' else 'Generating song'
            assert f'Completed {label}: 3 tokens' in captured.err
            assert '120 tokens' not in captured.err and '80 tokens' not in captured.err
        else:
            assert captured.err == ''
    assert results[0] == results[1] and calls[0] == calls[1]
    assert torch.equal(torch.random.get_rng_state(), rng_state)


@pytest.mark.parametrize('cot,abc', [('off', None), ('full', 'X:1\nK:C\nCDEF|'), ('melody', 'X:1\nK:C\nCDEF|')])
def test_provided_and_disabled_scores_emit_no_planner_tokens(cot, abc, capsys):
    pipe = bare_pipe()
    pipe._generate = lambda *args, **kwargs: pytest.fail('Planner ran for off/external score')
    tokens = []
    plan = pipe.plan('piano', 'original lyric', cot=cot, abc=abc,
                     on_token=lambda phase, token: tokens.append((phase, token)))
    assert tokens == []
    captured = capsys.readouterr()
    assert captured.out == '' and 'Planning score' not in captured.err and 'tokens/s' not in captured.err
    if abc is not None:
        assert plan.timing['output_tokens'] == 0
        assert plan.timing['external_prefix_tokens'] == len(plan.abc_ids)
        assert 'Using provided score' in captured.err
    else:
        assert plan.abc_ids == [] and captured.err == ''


@pytest.mark.parametrize('failure,expected', [(RuntimeError('backend failed'), 'Failed'),
                                            (InterruptedError('cancelled'), 'Cancelled'),
                                            (None, 'Finished (generation limit reached)')])
def test_generation_reports_failure_cancellation_and_truncation(failure, expected, monkeypatch, capsys):
    def generate(*args, on_token=None, **kwargs):
        on_token('semantic', CODEC_OFFSET)
        if failure is not None:
            raise failure
        return [CODEC_OFFSET], {'output_tokens': 1}, True

    monkeypatch.setattr(pipeline, 'generate_tokens', generate)
    pipe = bare_pipe()
    if failure is not None:
        with pytest.raises(type(failure), match=str(failure)):
            pipe._generate([1], pipe.generation_config.semantic, 42, 'semantic')
    else:
        assert pipe._generate([1], pipe.generation_config.semantic, 42, 'semantic')[2] is True
    captured = capsys.readouterr()
    assert captured.out == ''
    assert f'{expected} Generating song: 1 tokens' in captured.err
    assert 'Completed Generating song' not in captured.err


def test_user_token_callback_exception_propagates_with_failed_progress(monkeypatch, capsys):
    def generate(*args, on_token=None, **kwargs):
        on_token('semantic', CODEC_OFFSET)
        pytest.fail('Generation continued after callback exception')

    def on_token(phase, token):
        raise RuntimeError('user callback failed')

    monkeypatch.setattr(pipeline, 'generate_tokens', generate)
    pipe = bare_pipe()
    with pytest.raises(RuntimeError, match='user callback failed'):
        pipe._generate([1], pipe.generation_config.semantic, 42, 'semantic', on_token=on_token)
    assert 'Failed Generating song: 1 tokens' in capsys.readouterr().err


@pytest.mark.parametrize('enabled', [True, False])
def test_pipeline_passes_acoustic_callbacks_only_when_enabled(enabled, monkeypatch, capsys):
    pipe = bare_pipe(enabled)
    pipe.quantization, pipe.offload_ar, pipe.vae_core_frames = 'none', False, 1
    pipe.device, pipe._model = torch.device('cpu'), None
    expected = torch.arange(128, dtype=torch.float32).reshape(2, 64)
    seen = []

    def synthesize(model, prefix, tokens, seed, *, steps, context, offload_ar, cancelled, on_progress):
        seen.append(on_progress is not None)
        assert steps == pipe.generation_config.ode_steps and context == pipe.generation_config.context
        assert tokens == [1, 2] and seed == 42 and offload_ar is False
        if on_progress is not None:
            on_progress(1, steps)
            on_progress(steps, steps)
        return expected

    class Decoder:
        def to(self, device):
            return self

        def decode_tiled(self, latent, *, core_frames, halo_frames, output_device, on_progress):
            seen.append(on_progress is not None)
            assert torch.equal(latent, expected.T.unsqueeze(0))
            assert (core_frames, halo_frames, output_device) == (1, 16, 'cpu')
            if on_progress is not None:
                on_progress(1, 2)
                on_progress(2, 2)
            return torch.zeros(1, 2, 16)

    monkeypatch.setattr(nar, 'synthesize', synthesize)
    pipe._vae = Decoder()
    plan = pipe.plan('piano', 'original lyric', cot='off', seed=42)
    semantic = pipeline.SemanticResult(plan, [1, 2], {}, False)
    latent = pipe.synthesize(semantic)
    audio = pipe.decode(latent)
    assert torch.equal(torch.from_numpy(latent), expected) and audio.shape == (16, 2)
    assert seen == [enabled, enabled]
    captured = capsys.readouterr()
    assert captured.out == ''
    if enabled:
        assert 'Completed Synthesizing audio: 32/32 steps' in captured.err
        assert 'Completed Decoding audio: 2/2 chunks' in captured.err
    else:
        assert captured.err == ''


@pytest.mark.parametrize('command', ['generate', 'batch'])
@pytest.mark.parametrize('quiet_flag', [None, '--quiet', '--no-progress'])
def test_cli_quiet_propagates_and_does_not_pollute_result_stdout(command, quiet_flag, monkeypatch, tmp_path, capsys):
    class FakePipe:
        weights = {}
        closed = False

        def _request(self, **kwargs):
            return SongRequest(**kwargs)

        def effective_config(self, *args):
            return {}

        def __call__(self, **kwargs):
            with Progress(enabled=self.progress).stage('Generating song', unit='tokens') as stage:
                stage.advance()
            return SimpleNamespace(truncated={'abc': False, 'semantic': False}, timing={'e2e_seconds': 1.0},
                                   save_artifacts=lambda directory: {'identity': 'saved'})

        def close(self):
            self.closed = True

    fake = FakePipe()

    def from_pretrained(*args, **kwargs):
        fake.progress = kwargs['progress']
        return fake

    monkeypatch.setattr(YuE2Pipeline, 'from_pretrained', from_pretrained)
    argv = [command, '--output', str(tmp_path / 'output')]
    if command == 'batch':
        source = tmp_path / 'requests.jsonl'
        source.write_text(''.join(json.dumps({'id': f'song{index}', 'style': 'piano', 'lyrics': 'original lyric'}) + '\n'
                                  for index in range(2)))
        argv += ['--input', str(source)]
    if quiet_flag is not None:
        argv.append(quiet_flag)
    assert cli.main(argv) == 0 and fake.closed
    assert fake.progress is (quiet_flag is None)
    captured = capsys.readouterr()
    if command == 'generate':
        assert json.loads(captured.out)['status'] == 'complete'
    else:
        assert captured.out == ''
        assert json.loads((tmp_path / 'output/batch.json').read_text())['complete']
    if quiet_flag is not None:
        assert captured.err == ''
    else:
        assert 'Completed Generating song: 1 tokens' in captured.err
        if command == 'batch':
            assert 'Song 1/2: song0' in captured.err and 'Song 2/2: song1' in captured.err
