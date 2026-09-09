#!/usr/bin/env python3
"""Build a local, self-contained listening comparison from saved YuE2 song directories.

Usage: python scripts/listen.py outputs/pop outputs/jazz --output outputs/comparison
Exit 0: all cases complete; 1: page created with cases needing review; 2: command error.
Only an explicit allowlist is copied. No models, latents, environment, logs or network.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import shutil
import sys
from pathlib import Path


METADATA = ("request.json", "config.json", "result.json", "invocation.json",
            "input.json", "failure.json", "run.json", "abc_check.json")
SECRET_KEYS = {"token", "access_token", "refresh_token", "api_key", "apikey", "hf_token",
               "password", "passwd", "secret", "client_secret", "authorization", "cookie",
               "aws_secret_access_key", "aws_session_token", "credentials", "credential"}
SECRET_TEXT = re.compile(
    r"(?:https?://[^\s/:]+:[^\s/@]+@|\bBearer\s+[A-Za-z0-9._~+/=-]{8,}|"
    r"\bhf_[A-Za-z0-9]{20,}\b|\bsk-[A-Za-z0-9_-]{20,}\b)", re.I)


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return {"sha256": value.hexdigest(), "bytes": path.stat().st_size}


def has_credentials(value):
    if isinstance(value, dict):
        return any((str(k).lower().replace("-", "_") in SECRET_KEYS and bool(v))
                   or has_credentials(v) for k, v in value.items())
    if isinstance(value, list):
        return any(has_credentials(v) for v in value)
    return isinstance(value, str) and SECRET_TEXT.search(value) is not None


def json_text(value):
    return json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)


def text(value):
    return html.escape(value if isinstance(value, str) else json_text(value), quote=True)


def json_object(path):
    if path.stat().st_size > 8 * 1024 * 1024:
        raise ValueError("metadata exceeds 8 MiB")
    value = json.loads(path.read_text(encoding="utf-8"),
                       parse_constant=lambda _: (_ for _ in ()).throw(ValueError("non-finite JSON value")))
    if not isinstance(value, dict):
        raise ValueError("expected a JSON object")
    if has_credentials(value):
        raise ValueError("credential-like content; file withheld")
    return value


def add_issue(case, message):
    case["issues"].append(message)


def metadata(source, case):
    objects = {}
    for name in METADATA:
        path = source / name
        if not path.is_file():
            continue
        try:
            objects[name] = json_object(path)
        except (OSError, UnicodeError, ValueError, RecursionError) as exc:
            # Exception details from a JSON parser can contain input fragments.
            reason = "credential-like content; file withheld" if "credential-like" in str(exc) else "unreadable or invalid JSON; file withheld"
            add_issue(case, f"{name}: {reason}")
    return objects


def copy_case(source, destination, number):
    label = source.name or f"case {number}"
    case = {"label": label, "directory": f"case-{number:03d}", "issues": [],
            "files": {}, "native_artifact_checks": {}, "audio": None}
    case_dir = destination / case["directory"]
    case_dir.mkdir()
    if not source.is_dir():
        add_issue(case, "Input directory is missing or is not a directory")
    objects = metadata(source, case) if source.is_dir() else {}
    request = objects.get("request.json", objects.get("input.json", {}))
    config = objects.get("config.json", {})
    result = objects.get("result.json", {})
    invocation = objects.get("invocation.json", {})
    fail_record = objects.get("failure.json", {})
    case["request"] = request
    case["decoder_release"] = config.get("decoder_release", "unknown")
    case["weights"] = result.get("weights", {})
    case["identity"] = result.get("identity", "unknown")
    case["loader"] = invocation.get("loader", {})
    case["native_status"] = result.get("status", fail_record.get("status", "unknown"))
    case["truncated"] = result.get("truncated", "unknown")
    case["audio_seconds"] = result.get("audio_seconds")
    case["sample_rate"] = result.get("sample_rate")
    case["validation_status"] = config.get("validation_status", "unknown")
    case["failure"] = fail_record
    case["abc_check"] = objects.get("abc_check.json", {})
    case["recorded_warnings"] = {name: obj["warnings"] for name, obj in objects.items() if obj.get("warnings")}
    if not request:
        add_issue(case, "Exact request is unavailable")
    if not config:
        add_issue(case, "Native configuration is unavailable; decoder identity is incomplete")
    if not result:
        add_issue(case, "Native result manifest is unavailable")
    if fail_record:
        add_issue(case, "A failure.json record is present; review the recorded failure")
    if result and result.get("status") != "complete":
        add_issue(case, "Native result does not report complete status")
    truncated = result.get("truncated")
    if isinstance(truncated, dict) and any(truncated.values()):
        add_issue(case, "Native result reports generation truncation")
    if case["abc_check"].get("status") == "failed":
        add_issue(case, "Saved ABC structural check failed")
    artifacts = result.get("artifacts", {})
    if not isinstance(artifacts, dict):
        artifacts = {}
        add_issue(case, "Native artifact manifest has an invalid shape")

    def copy_file(name):
        path = source / name
        copied = case_dir / name
        try:
            shutil.copyfile(path, copied)
            observed = digest(copied)
            case["files"][name] = observed
            wanted = artifacts.get(name)
            if wanted is None:
                case["native_artifact_checks"][name] = "not listed"
            elif not isinstance(wanted, dict) or not all(observed[k] == wanted.get(k) for k in ("sha256", "bytes")):
                case["native_artifact_checks"][name] = "FAILED"
                add_issue(case, f"{name}: copied bytes do not match native artifact hash/size")
            else:
                case["native_artifact_checks"][name] = "passed"
            return copied
        except OSError:
            copied.unlink(missing_ok=True)
            add_issue(case, f"{name}: copy failed")
            return None

    for name in objects:
        copy_file(name)
    score_path = source / "score.abc"
    if score_path.is_file():
        try:
            fail = score_path.stat().st_size > 8 * 1024 * 1024
            abc = score_path.read_text(encoding="utf-8") if not fail else ""
            if fail or has_credentials(abc):
                add_issue(case, "score.abc: oversized or credential-like content; file withheld")
            else:
                copy_file("score.abc")
                case["abc"] = abc
        except (OSError, UnicodeError):
            add_issue(case, "score.abc: unreadable UTF-8; file withheld")

    candidates = [name for name in ("audio.flac", "audio.wav") if (source / name).is_file()]
    if not candidates:
        add_issue(case, "Missing audio.flac/audio.wav; no player available")
    else:
        # Prefer the artifact explicitly identified by the native receipt.
        name = next((n for n in candidates if n in artifacts), candidates[0])
        copied = copy_file(name)
        if copied:
            with copied.open("rb") as stream:
                signature = stream.read(12)
            signature_ok = (name.endswith(".flac") and signature.startswith(b"fLaC")) or (
                name.endswith(".wav") and signature[:4] in (b"RIFF", b"RF64") and signature[8:12] == b"WAVE")
            if not signature_ok:
                add_issue(case, f"{name}: container signature is not recognized; player withheld")
            elif "FAILED" in case["native_artifact_checks"].values():
                add_issue(case, "Player withheld because copied artifacts disagree with the native receipt")
            else:
                case["audio"] = f"{case['directory']}/{name}"
                case["audio_type"] = "audio/flac" if name.endswith(".flac") else "audio/wav"
                if case["native_artifact_checks"].get(name) != "passed":
                    add_issue(case, "Audio is playable but has no native hash/size verification")
    case["status"] = "needs_review" if case["issues"] else "complete"
    return case


def detail(title, value):
    return f"<details><summary>{text(title)}</summary><pre>{text(value)}</pre></details>"


def case_html(case):
    req = case["request"]
    label = req.get("id", case["label"])
    parts = [f'<article><h2>{text(label)}</h2><p class="status">{text(case["status"])}</p>']
    if case["audio"]:
        parts.append(f'<audio controls preload="none"><source src="{text(case["audio"])}" type="{case["audio_type"]}">'
                     'Your browser does not support this audio format.</audio>')
        parts.append(f'<p><a href="{text(case["audio"])}" download>Download original audio</a></p>')
    else:
        parts.append('<p class="warning">No verified player is available for this case.</p>')
    if case["issues"]:
        parts.append('<ul class="warning">' + ''.join(f'<li>{text(x)}</li>' for x in case["issues"]) + '</ul>')
    fields = {"Mode": req.get("cot", "unknown"), "Seed": req.get("seed", "unknown"),
              "Decoder release": case["decoder_release"], "Native status": case["native_status"],
              "Truncated": case["truncated"], "Recorded duration (seconds)": case["audio_seconds"],
              "Sample rate": case["sample_rate"], "Runtime validation status": case["validation_status"],
              "Result identity": case["identity"]}
    parts.append('<dl>' + ''.join(f'<dt>{text(k)}</dt><dd>{text(v)}</dd>' for k,v in fields.items()) + '</dl>')
    parts.extend([f'<h3>Style prompt</h3><pre>{text(req.get("style", req.get("tags", "Unavailable")))}</pre>',
                  f'<h3>Lyrics</h3><pre>{text(req.get("lyrics", "Unavailable"))}</pre>'])
    for title, data in (("Model and decoder weight signatures",case["weights"]),
                        ("Requested model identifiers / revisions",case["loader"]),
                        ("Artifact integrity checks",case["native_artifact_checks"]),
                        ("Recorded warnings",case["recorded_warnings"]),
                        ("Failure record",case["failure"]), ("ABC structural check",case["abc_check"]),
                        ("Full exact request",req), ("Score ABC",case.get("abc"))):
        if data:
            parts.append(detail(title,data))
    links = ''.join(f'<li><a href="{case["directory"]}/{name}" download>{text(name)}</a></li>' for name in case["files"])
    parts.append(f'<details><summary>Copied artifacts</summary><ul>{links}</ul></details></article>')
    return '\n'.join(parts)


def page(cases):
    return '''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; media-src 'self' file:; base-uri 'none'; form-action 'none'">
<title>YuE2 listening comparison</title><style>
body{font:16px/1.5 system-ui,sans-serif;max-width:1100px;margin:36px auto;padding:0 20px;background:#f6f7f9;color:#17212b}
article{background:white;border:1px solid #d7dee7;border-radius:12px;padding:24px;margin:24px 0}
h1,h2,h3{line-height:1.2}h2{margin-top:0}audio{width:100%}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:14px/1.55 ui-monospace,monospace}
dt{font-weight:650}dd{margin:0 0 10px;overflow-wrap:anywhere}details{border-top:1px solid #d7dee7;padding:12px 0}summary{cursor:pointer;font-weight:600}
a{color:#174faf}.warning{color:#842b19}.status{font-weight:700}.note{color:#475569}
</style></head><body><h1>YuE2 listening comparison</h1>
<p class="note">Local listening bundle. Prompts, lyrics and available score/metadata are copied below.
Model and decoder signatures identify the recorded generation; no model weights or acoustic latents are included.
Integrity checks concern copied artifacts only, not musical quality, codec decoding, or a complete regeneration audit.
Use the download link if your browser does not play the original audio format.</p>
''' + '\n'.join(case_html(case) for case in cases) + '\n<p><a href="manifest.json">Bundle manifest and SHA-256 hashes</a></p></body></html>\n'


def build(sources, output):
    output = Path(output)
    if output.exists():
        raise ValueError("Output already exists; choose a fresh comparison directory")
    resolved = output.resolve()
    for source in sources:
        if resolved == source.resolve() or source.resolve() in resolved.parents:
            raise ValueError("Keep the comparison directory outside each source song directory")
    output.mkdir(parents=True, exist_ok=False)
    cases = [copy_case(Path(source), output, i) for i,source in enumerate(sources,1)]
    (output / "index.html").write_text(page(cases),encoding="utf-8")
    files = {str(path.relative_to(output)): digest(path) for path in sorted(output.rglob("*")) if path.is_file()}
    manifest = {"schema": "yue2-listening-bundle-v1", "cases": cases, "files": files,
                "hash_scope": "All copied files and index.html; manifest.json excludes its own recursive hash",
                "network_access": False, "weights_or_latents_copied": False}
    (output / "manifest.json").write_text(json_text(manifest)+'\n',encoding="utf-8")
    return manifest


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("sources",nargs="+",type=Path,help="Native saved YuE2 song directories")
    cli.add_argument("--output",required=True,type=Path,help="Fresh destination outside source song directories")
    args = cli.parse_args()
    try:
        manifest = build(args.sources,args.output)
    except (OSError,ValueError) as error:
        print(f"Listening bundle failed: {error}",file=sys.stderr)
        return 2
    print(json_text({"page":str(args.output/'index.html'),"cases":len(manifest['cases']),
                     "needs_review":sum(c['status']!='complete' for c in manifest['cases'])}))
    return int(any(c['status']!='complete' for c in manifest['cases']))


if __name__ == "__main__":
    raise SystemExit(main())
