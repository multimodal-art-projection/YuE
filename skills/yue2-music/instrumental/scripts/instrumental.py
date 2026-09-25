#!/usr/bin/env python3
"""Prepare, generate and verify a YuE2 instrumental. Run --help for commands."""
from __future__ import annotations

import argparse
from argparse import Namespace
import html
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

from abc_tools import TOKEN, parse_abc, report
from common import fresh_directory, read_json, sha256, write_json
from compile_score import SECTIONS, compile_events
from instrumentalize import convert_score

ROOT = Path(__file__).resolve().parents[1]
INSTRUCTIONS = {
    'full': 'Generate a chord-annotated ABC transcription, then generate music with codec tokens from the given conditions.',
    'melody': 'Generate a melody-only ABC transcription without chord symbols, then generate music with codec tokens from the given conditions.',
}


def validate_score(text):
    if not isinstance(text, str) or not text.strip():
        raise ValueError('ABC must be nonempty text; let the agent write or repair the score first')
    score = parse_abc(text)
    if score.voices['Vocal'].notes:
        raise ValueError('Vocal contains sounding notes; move the intended lead to Ins and replace Vocal notes with rests')
    if not score.voices['Ins'].notes:
        raise ValueError('Ins has no sounding notes')
    for _, _, (_, d) in score.voices['Ins'].bars:
        if score.unit.denominator % (4 * d):
            raise ValueError('L is too coarse for the four-subbeat grid; scale ALL durations with a finer L')
    previous = None
    for line in text.splitlines():
        if line.startswith('% '):
            label = line[2:]
            if label not in SECTIONS or label == previous:
                raise ValueError('Use native section labels, without consecutive duplicate section comments')
            previous = label
    for index in score.music_lines:
        line = text.splitlines()[index]
        if re.search(r'Z[234]?\|Z[234]?(?:\||$)', line):
            raise ValueError('Compress adjacent full-bar rests to Z2/Z3/Z4 inside each group')
        for bar in line[:-1].split('|'):
            if re.fullmatch(r'(?:z\d*)+', bar):
                raise ValueError('Use Z for an unannotated full-bar rest')
            for match in TOKEN.finditer(bar):
                if match.group('duration') == '1':
                    raise ValueError('Omit explicit duration 1 in the native notation')
    return score


def lyric_tags(text):
    labels = [line[2:] for line in text.splitlines() if line.startswith('% ')]
    return '\n\n'.join('[' + x.title() + ']' for x in labels) + ('\n' if labels else '')


def validate_request(request):
    fields = {'id', 'style', 'lyrics', 'abc', 'cot', 'seed'}
    if not isinstance(request, dict) or set(request) != fields:
        raise ValueError('Request must contain exactly id, style, lyrics, abc, cot and seed')
    if not isinstance(request['id'], str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,100}', request['id']):
        raise ValueError('id must be a safe filename')
    if type(request['seed']) is not int or not 0 <= request['seed'] < 2**63:
        raise ValueError('seed must be an integer from 0 to 2**63-1')
    if not isinstance(request['style'], str) or not request['style'].strip():
        raise ValueError('Style must be nonempty')
    score = validate_score(request['abc'])
    if request['lyrics'] not in ('', lyric_tags(request['abc'])):
        raise ValueError('Instrumental lyrics must be empty or exactly the score section tags; no sung words')
    expected = 'full' if score.voices['Vocal'].chords else 'melody'
    if request['cot'] != expected:
        raise ValueError(f'Use cot={expected} for this score')
    return score


def prompt_text(request):
    return INSTRUCTIONS[request['cot']] + '\n[Tags]\n' + request['style'] + '\n[Lyrics]\n' + request['lyrics'] + '\n'


def hashes(directory, names):
    return {name: {'sha256': sha256(directory / name), 'bytes': (directory / name).stat().st_size} for name in names}


def verify_hashes(directory, records):
    root = Path(directory).resolve()
    for name, wanted in records.items():
        path = root / name
        if Path(name).is_absolute() or not path.resolve().is_relative_to(root):
            raise ValueError('Artifact path escapes its directory')
        if not path.is_file() or path.stat().st_size != wanted['bytes'] or sha256(path) != wanted['sha256']:
            raise ValueError(f'Missing or corrupt file: {name}')


def prepare(args):
    repairs = read_json(args.repairs) if args.repairs else {'changes': [], 'assumptions': []}
    if not isinstance(repairs, dict):
        raise ValueError('Repair record must be a JSON object')
    if repairs.get('unresolved'):
        raise ValueError('Resolve the material score ambiguities recorded in unresolved before generation')
    event_check = None
    if args.events:
        text, event_check = compile_events(read_json(args.events))
    else:
        text = args.abc.read_text(encoding='utf-8')
    score = validate_score(text)
    style = args.style_file.read_text(encoding='utf-8').strip() if args.style_file else args.style
    if not style:
        style = 'Expressive instrumental music'
    style = style.strip().rstrip('.,')
    if not re.match(r'^instrumental\b', style, re.I):
        style = 'Instrumental, ' + style
    for condition in ('no vocals', 'no singing', 'no choir', 'no spoken words'):
        if condition not in style.lower():
            style += ', ' + condition
    style += '.'
    request = dict(id=args.id, style=style, lyrics='' if args.empty_lyrics else lyric_tags(text),
                   abc=text, cot='full' if score.voices['Vocal'].chords else 'melody', seed=args.seed)
    validate_request(request)
    out = fresh_directory(args.output)
    (out / 'score.abc').write_text(text, encoding='utf-8')
    (out / 'prompt.txt').write_text(prompt_text(request), encoding='utf-8')
    (out / 'lyrics.txt').write_text(request['lyrics'], encoding='utf-8')
    (out / 'style.txt').write_text(style + '\n', encoding='utf-8')
    write_json(out / 'request.json', request)
    write_json(out / 'repair-report.json', repairs)
    check = report(score)
    check.update(passed=True, vocal_sounding_notes=0, lyric_words=0)
    if event_check:
        check['compiler'] = event_check
        shutil.copyfile(args.events, out / 'events.json')
    write_json(out / 'score-check.json', check)
    if args.source:
        shutil.copyfile(args.source, out / 'original.abc')
    names = [x.name for x in out.iterdir() if x.is_file()]
    write_json(out / 'prepared-manifest.json', {'files': hashes(out, names)})
    print(json.dumps({'prepared': str(out), 'mode': request['cot'], 'nominal_seconds': check['nominal_duration_seconds'],
                      'ins_notes': len(score.voices['Ins'].notes), 'vocal_notes': 0}, ensure_ascii=False))
    return 0


def load_prepared(path):
    receipt = read_json(path / 'prepared-manifest.json')
    required = {'request.json', 'score.abc', 'prompt.txt', 'lyrics.txt', 'style.txt', 'score-check.json', 'repair-report.json'}
    if not required <= set(receipt['files']):
        raise ValueError('Incomplete prepared manifest')
    verify_hashes(path, receipt['files'])
    request = read_json(path / 'request.json')
    validate_request(request)
    if (path / 'score.abc').read_text(encoding='utf-8') != request['abc'] or (path / 'prompt.txt').read_text(encoding='utf-8') != prompt_text(request):
        raise ValueError('Prompt/ABC disagree with the prepared request')
    if (path / 'lyrics.txt').read_text(encoding='utf-8') != request['lyrics'] or (path / 'style.txt').read_text(encoding='utf-8') != request['style'] + '\n':
        raise ValueError('Style/lyrics disagree with the prepared request')
    return request


def write_player(out, request, summary, audio_name, shared=False):
    esc = html.escape
    lossless = 'audio.flac' if shared else 'native/audio.flac'
    record_links = '<a href="prompt.txt">完整 prompt</a>' if shared else '<a href="summary.json">生成记录</a> · <a href="repair-report.json">修谱说明</a>'
    content = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>YuE2 纯音乐试听</title>
<style>body{{max-width:900px;margin:40px auto;padding:0 20px;font:17px/1.6 system-ui}}audio{{width:100%}}pre{{white-space:pre-wrap;overflow-wrap:anywhere}}details{{margin:20px 0}}</style>
<h1>YuE2 纯音乐试听</h1><audio controls preload="metadata" src="{esc(audio_name)}"></audio>
<p>{summary['audio_seconds']:.2f} 秒 · 48 kHz 立体声 · {esc(summary['decoder'])}</p>
<p>生成状态：{esc(summary['status'])}。谱面和文件校验不等于已确认音频无人声或逐音跟谱。</p>
<p><a href="{esc(audio_name)}" download>下载试听</a> · <a href="{lossless}" download>无损 FLAC</a> · <a href="score.abc">ABC</a></p>
<details open><summary>实际文本 prompt</summary><pre>{esc(prompt_text(request))}</pre></details>
<details><summary>实际 ABC</summary><pre>{esc(request['abc'])}</pre></details>
<p>{record_links}</p></html>'''
    (out / 'index.html').write_text(content, encoding='utf-8')


def open_pipeline(args):
    release = read_json(ROOT / 'assets/release.json')
    version = importlib.metadata.version('yue2-infer')
    if version != release['package_version']:
        raise ValueError(f"Use the pinned yue2-infer {release['package_version']}; run setup_runtime.py install")
    import torch
    from yue2 import YuE2Pipeline
    if not torch.cuda.is_available():
        raise RuntimeError('No CUDA device; run this command inside a GPU allocation, keeping CPU preparation intact')
    device = torch.device(args.device)
    if device.type != 'cuda':
        raise ValueError('This recipe requires a BF16-capable NVIDIA GPU')
    if torch.cuda.get_device_capability(device)[0] < 8:
        raise RuntimeError('This recipe requires a BF16-capable GPU (compute capability >= 8)')
    model_spec, vae_spec = release['models']['YuE2-3B'], release['models']['YuE2-Vae']
    if args.models_root:
        model, vae = args.models_root / 'YuE2-3B', args.models_root / 'YuE2-Vae'
        if not model.is_dir() or not vae.is_dir():
            raise ValueError('models-root must contain YuE2-3B and YuE2-Vae directories')
    else:
        model, vae = model_spec['repo'], vae_spec['repo']
    torch.set_num_threads(8)
    pipe = YuE2Pipeline.from_pretrained(model, vae=vae, device=args.device,
        revision=model_spec['revision'], vae_revision=vae_spec['revision'],
        local_files_only=args.offline, memory_budget_gib=24)
    try:
        from setup_runtime import check_model
        check_model(pipe.model_dir, model_spec)
        check_model(pipe.vae_dir, vae_spec)
        for name, role in (('YuE2-3B', 'mot'), ('YuE2-Vae', 'vae')):
            if pipe.weights[role]['files']['model.safetensors']['sha256'] != release['models'][name]['weights_sha256']:
                raise ValueError(f'{name} weights differ from the frozen recipe')
    except Exception:
        pipe.close()
        raise
    return pipe


def generate(args):
    request = load_prepared(args.prepared)
    version = importlib.metadata.version('yue2-infer')
    import numpy as np
    import torch
    from yue2.protocol import SongRequest, token_prefixes
    from yue2.storage import verify_result
    req = SongRequest(**request)
    if req.text() != prompt_text(request):
        raise ValueError('Installed runtime prompt template differs from the pinned protocol')
    out = fresh_directory(args.output)
    for name in ('score.abc', 'prompt.txt', 'lyrics.txt', 'style.txt', 'repair-report.json', 'score-check.json'):
        shutil.copyfile(args.prepared / name, out / name)
    if (args.prepared / 'original.abc').is_file():
        shutil.copyfile(args.prepared / 'original.abc', out / 'original.abc')
    if (args.prepared / 'events.json').is_file():
        shutil.copyfile(args.prepared / 'events.json', out / 'events.json')
    write_json(out / 'input-request.json', request)
    try:
        with open_pipeline(args) as pipe:
            if len(pipe.tokenizer.encode(request['abc'])) > 4096:
                raise ValueError('ABC exceeds the normal 4096-token planning budget; compact notation without dropping music')
            expected_prefix = token_prefixes(req, pipe.tokenizer)
            song = pipe(**request)
            song.save_artifacts(out / 'native')
        receipt = verify_result(out / 'native')
        if not np.array_equal(np.load(out / 'native/prefix.npy', allow_pickle=False), expected_prefix):
            raise ValueError('Saved prefix does not match the exact text and ABC input')
        if (out / 'native/score.abc').read_text(encoding='utf-8') != request['abc']:
            raise ValueError('Saved ABC differs from the intended input')
        if not np.isfinite(song.audio).all() or not np.any(song.audio):
            raise ValueError('Audio is nonfinite or completely silent')
        status = 'needs_review' if any(receipt['truncated'].values()) else 'complete'
        summary = dict(status=status, audio_seconds=receipt['audio_seconds'], sample_rate=48000,
            truncated=receipt['truncated'], exact_prefix_verified=True, exact_abc_verified=True,
            seed=request['seed'], cot=request['cot'], package_version=version,
            gpu=torch.cuda.get_device_name(args.device), peak=float(np.max(np.abs(song.audio))),
            rms=float(np.sqrt(np.mean(song.audio.astype(np.float64) ** 2))),
            decoder='m-a-p/YuE2-Vae (listening)', weights=receipt['weights'],
            audio_adherence_checked=False, absence_of_vocals_checked=False)
        write_json(out / 'summary.json', summary)
        audio_name = 'native/audio.flac'
        ffmpeg = shutil.which('ffmpeg')
        if ffmpeg:
            conversion = subprocess.run([ffmpeg, '-hide_banner', '-loglevel', 'error', '-n', '-i',
                str(out / 'native/audio.flac'), '-c:a', 'libmp3lame', '-b:a', '192k', str(out / 'listening.mp3')],
                capture_output=True, text=True)
            if conversion.returncode == 0:
                audio_name = 'listening.mp3'
            else:
                (out / 'mp3-conversion.log').write_text(conversion.stderr, encoding='utf-8')
        write_player(out, request, summary, audio_name)
        names = [x.relative_to(out).as_posix() for x in out.rglob('*') if x.is_file()]
        write_json(out / 'delivery-manifest.json', {'files': hashes(out, names)})
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0 if status == 'complete' else 1
    except Exception as exc:
        write_json(out / 'failure.json', {'status': 'failed', 'type': type(exc).__name__, 'error': str(exc)})
        raise


def verify(args):
    path = args.directory
    if (path / 'prepared-manifest.json').is_file():
        request = load_prepared(path)
        print(json.dumps({'verified': True, 'kind': 'prepared', 'id': request['id']}))
        return 0
    receipt = read_json(path / 'delivery-manifest.json')
    required = {'summary.json', 'native/result.json', 'native/request.json', 'native/audio.flac',
        'native/score.abc', 'native/semantic.npy', 'native/latent.npy', 'native/prefix.npy',
        'native/config.json', 'score.abc', 'prompt.txt', 'repair-report.json'}
    if not required <= set(receipt['files']):
        raise ValueError('Incomplete delivery manifest')
    verify_hashes(path, receipt['files'])
    native = read_json(path / 'native/result.json')
    native_required = {'audio.flac', 'prefix.npy', 'semantic.npy', 'latent.npy', 'request.json', 'config.json', 'score.abc'}
    if native.get('status') != 'complete' or not native_required <= set(native['artifacts']):
        raise ValueError('Native result is incomplete')
    verify_hashes(path / 'native', native['artifacts'])
    request = read_json(path / 'native/request.json')
    # The native protocol also saves cfg_scale:null.
    request.pop('cfg_scale', None)
    validate_request(request)
    if (path / 'score.abc').read_text(encoding='utf-8') != request['abc']:
        raise ValueError('Delivered ABC differs from native request')
    if (path / 'native/score.abc').read_text(encoding='utf-8') != request['abc']:
        raise ValueError('Native ABC differs from native request')
    if (path / 'prompt.txt').read_text(encoding='utf-8') != prompt_text(request):
        raise ValueError('Delivered prompt differs from native request')
    summary = read_json(path / 'summary.json')
    if any(native['truncated'].values()) or summary['status'] != 'complete':
        raise ValueError('Generation needs review: truncated or incomplete')
    print(json.dumps({'verified': True, 'kind': 'generated', 'seconds': native['audio_seconds'],
                      'scope': 'File integrity, complete generation and symbolic instrumental input; not a listening assessment'}))
    return 0


def share(args):
    """Export only the requested listening assets; keep machine records local."""
    verify(Namespace(directory=args.directory))
    source = args.directory
    request = read_json(source / 'native/request.json')
    summary = read_json(source / 'summary.json')
    public = {key: summary[key] for key in ('status', 'audio_seconds', 'sample_rate',
        'decoder', 'seed', 'cot', 'package_version', 'truncated',
        'audio_adherence_checked', 'absence_of_vocals_checked')}
    out = fresh_directory(args.output)
    for name in ('score.abc', 'prompt.txt'):
        shutil.copyfile(source / name, out / name)
    shutil.copyfile(source / 'native/audio.flac', out / 'audio.flac')
    audio_name = 'audio.flac'
    if (source / 'listening.mp3').is_file():
        shutil.copyfile(source / 'listening.mp3', out / 'listening.mp3')
        audio_name = 'listening.mp3'
    write_json(out / 'summary.json', public)
    write_player(out, request, public, audio_name, shared=True)
    write_json(out / 'share-manifest.json', {'files': hashes(out, [p.name for p in out.iterdir()])})
    print(json.dumps({'listening_page': str(out / 'index.html'), 'audio': str(out / audio_name),
                      'prompt': str(out / 'prompt.txt'), 'note': 'Local export; nothing uploaded'}))
    return 0


def plan_with_model(args, destination):
    """Save the untouched model plan, then return it for voice transfer."""
    style = args.style_file.read_text(encoding='utf-8').strip() if args.style_file else args.style
    style = style or 'Instrumental, warm lyrical acoustic piano, no vocals, no singing, no choir'
    lyrics = (args.planning_lyrics_file.read_text(encoding='utf-8') if args.planning_lyrics_file
              else '[Intro]\n\n[Verse]\n\n[Chorus]\n\n[Outro]\n')
    out = fresh_directory(destination)
    from yue2.protocol import SongRequest
    from yue2 import SymbolicPlan
    request = SongRequest(id=args.id, style=style, lyrics=lyrics, cot=args.plan_mode, seed=args.seed)
    write_json(out / 'request.json', request.to_dict())
    (out / 'prompt.txt').write_text(request.text(), encoding='utf-8')
    try:
        with open_pipeline(args) as pipe:
            plan = pipe.plan(request=request)
            plan.save(out)
            write_json(out / 'provenance.json', {'composer': 'YuE2', 'weights': pipe.weights,
                'config': pipe.effective_config(request)})
        restored = SymbolicPlan.load(out)
        if restored.truncated or not restored.abc:
            raise ValueError('Model score is empty or truncated; keep the failed plan and retry explicitly')
        return out / 'score.abc'
    except Exception as exc:
        write_json(out / 'failure.json', {'type': type(exc).__name__, 'error': str(exc)})
        raise


def transcribe_cover(args, destination):
    """Run SheetSage2 in its own interpreter before loading YuE2."""
    if not args.transcriber_python:
        raise ValueError('Audio cover needs --transcriber-python from a separate SheetSage2 environment')
    if not args.audio.is_file():
        raise FileNotFoundError(args.audio)
    command = [str(args.transcriber_python), str(ROOT / 'scripts/transcribe_cover.py'), str(args.audio),
        '--output', str(destination), '--task', 'full' if args.keep_harmony else 'melody-full',
        '--device', args.device]
    if args.transcriber_model:
        command += ['--model', args.transcriber_model]
    if args.transcriber_revision:
        command += ['--revision', args.transcriber_revision]
    if args.base_model:
        command += ['--base-model', args.base_model]
    if args.offline:
        command += ['--offline']
    environment = os.environ.copy()
    environment.pop('PYTHONPATH', None)
    environment['PYTHONNOUSERSITE'] = '1'
    result = subprocess.run(command, check=False, env=environment)
    if result.returncode:
        raise RuntimeError('SheetSage2 transcription failed; inspect its retained output before retrying')
    score = destination / 'score.abc'
    if not score.is_file():
        raise ValueError('Transcription did not produce a score')
    return score


def run(args):
    """Default: YuE2 plans, notes move to Ins, YuE2 renders the revised score."""
    if args.output.exists():
        raise FileExistsError('Choose a new output directory; previous songs are preserved')
    if args.events and args.composer != 'agent':
        raise ValueError('Agent-written events require explicit --composer agent')
    if args.composer == 'agent' and not (args.events or args.abc):
        raise ValueError('--composer agent needs a score written explicitly by the agent')
    if args.command == 'cover' and not (args.abc or args.audio):
        raise ValueError('Cover requires --audio or --abc; never replace a reference with a new composition')
    abc = args.abc
    origin = 'agent' if args.composer == 'agent' else 'provided score' if abc else 'YuE2'
    if args.audio:
        abc = transcribe_cover(args, args.output / 'transcription')
        origin = 'SheetSage2 transcription'
    elif not abc and not args.events:
        abc = plan_with_model(args, args.output / 'planning')
    prepared_args = vars(args).copy()
    if abc:
        converted, transfer = convert_score(abc.read_text(encoding='utf-8'), overlap=args.overlap,
            keep_chords=args.keep_harmony if args.command == 'cover' or args.audio else True)
        transfer.update(score_origin=origin)
        folder = fresh_directory(args.output / 'conversion')
        (folder / 'score.abc').write_text(converted, encoding='utf-8')
        write_json(folder / 'transfer.json', transfer)
        repairs = read_json(args.repairs) if args.repairs else {'changes': [], 'assumptions': []}
        repairs['instrumental_transfer'] = transfer
        write_json(folder / 'repairs.json', repairs)
        prepared_args.update(abc=folder / 'score.abc', source=args.source or abc,
                            repairs=folder / 'repairs.json')
    prepared = args.output / 'prepared'
    prepare(Namespace(**(prepared_args | {'output': prepared})))
    write_json(args.output / 'workflow.json', {'score_origin': origin,
        'composer_setting': args.composer, 'operation': 'cover' if args.audio or args.command == 'cover' else 'generate',
        'voice_transfer': bool(abc), 'agent_composed': origin == 'agent'})
    load_prepared(prepared)
    if args.prepare_only:
        print(json.dumps({'status': 'prepared', 'audio_generated': False, 'prepared': str(prepared)}))
        return 0
    generation = args.output / 'generation'
    result = generate(Namespace(prepared=prepared, output=generation,
        models_root=args.models_root, offline=args.offline, device=args.device))
    if result:
        return result
    return share(Namespace(directory=generation, output=args.output / 'listen'))


def add_score_arguments(p, required=True):
    source = p.add_mutually_exclusive_group(required=required)
    source.add_argument('--abc', type=Path)
    source.add_argument('--events', type=Path)
    if not required:
        source.add_argument('--audio', type=Path, help='Reference recording for an instrumental cover')
    style = p.add_mutually_exclusive_group()
    style.add_argument('--style')
    style.add_argument('--style-file', type=Path)
    p.add_argument('--source', type=Path, help='Untouched original ABC to retain')
    p.add_argument('--repairs', type=Path, help='Agent-authored repair decisions and assumptions, JSON')
    p.add_argument('--empty-lyrics', action='store_true')
    p.add_argument('--id', default='instrumental')
    p.add_argument('--seed', type=int, default=831001)
    p.add_argument('--output', type=Path, required=True)


def add_runtime_arguments(p):
    p.add_argument('--models-root', type=Path)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--offline', action='store_true')


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    commands = cli.add_subparsers(dest='command', required=True)
    for name in ('run', 'cover'):
        r = commands.add_parser(name, help='Model-planned instrumental' if name == 'run' else 'Instrumental cover of an audio or ABC reference')
        add_score_arguments(r, required=False)
        add_runtime_arguments(r)
        r.add_argument('--composer', choices=('model', 'agent'), default='model')
        r.add_argument('--plan-mode', choices=('full', 'melody'), default='full')
        r.add_argument('--planning-lyrics-file', type=Path, help='Optional planner-only lyrics/section tags; never sung in the final request')
        r.add_argument('--overlap', choices=('vocal', 'error'), default='vocal')
        r.add_argument('--keep-harmony', action='store_true', help='For covers, retain source chords instead of arranging new accompaniment')
        r.add_argument('--transcriber-python', type=Path, help='Separate SheetSage2 environment interpreter')
        r.add_argument('--transcriber-model')
        r.add_argument('--transcriber-revision')
        r.add_argument('--base-model', help='Local MERT-v2-FullSong snapshot for offline transcription')
        r.add_argument('--prepare-only', action='store_true', help='Stop before audio generation; a new model plan still requires a GPU')
    p = commands.add_parser('prepare', help='Check the agent-repaired score and create a portable request')
    add_score_arguments(p)
    g = commands.add_parser('generate', help='Generate once with the exact prepared ABC; save audio and native artifacts')
    g.add_argument('--prepared', type=Path, required=True)
    g.add_argument('--output', type=Path, required=True)
    add_runtime_arguments(g)
    v = commands.add_parser('verify', help='Fail on absent, corrupt, truncated or inconsistent handoff artifacts')
    v.add_argument('directory', type=Path)
    s = commands.add_parser('share', help='Export audio, prompt and score without original inputs or machine logs')
    s.add_argument('directory', type=Path)
    s.add_argument('--output', type=Path, required=True)
    args = cli.parse_args()
    try:
        return {'run': run, 'cover': run, 'prepare': prepare, 'generate': generate, 'verify': verify, 'share': share}[args.command](args)
    except (ValueError, RuntimeError, OSError, KeyError, TypeError, ImportError, importlib.metadata.PackageNotFoundError) as exc:
        print(f'{type(exc).__name__}: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
