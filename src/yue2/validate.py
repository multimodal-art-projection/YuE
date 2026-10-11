"""Preflight song requests without loading model weights."""
from __future__ import annotations
from pathlib import Path
import json
import re

from dataclasses import asdict

from .protocol import GenerationConfig, SongRequest, resolve_sampling
from .storage import write_json

REQUEST_FIELDS = ("id", "style", "cot", "seed", "cfg_scale")
SONG_FIELDS = {"style", "lyrics", "cot", "seed", "abc", "cfg_scale", "id"}


def request_kwargs(data, base=Path.cwd()):
    allowed = {"style", "tags", "lyrics", "cot", "seed", "abc", "cfg_scale", "id", "abc_sampling", "semantic_sampling"}
    metadata = {"lang", "eval_index", "clip_id", "prompt"}
    unknown = set(data) - allowed - metadata - {"abc_path"}
    if unknown:
        raise ValueError(f"Unknown request fields: {sorted(unknown)}")
    result = {k: v for k, v in data.items() if k in allowed}
    if "abc_path" in data:
        if data.get("abc") is not None:
            raise ValueError("Pass abc or abc_path, not both")
        result["abc"] = (base / data["abc_path"]).read_bytes().decode("utf-8")
    if data.get("prompt") is not None:
        request = SongRequest(style=result.get("style", result.get("tags")), lyrics=result["lyrics"],
                              cot=result.get("cot", "full"))
        if data["prompt"] != request.text():
            raise ValueError("Historical literal prompt does not match native instruction/style/lyrics")
    return result


def apply_overrides(data, overrides):
    payload = dict(data)
    for key, value in overrides.items():
        if value is not None:
            payload[key] = value
    if overrides.get("abc") is not None:
        payload.pop("abc_path", None)
    return payload


def build_request(data, base=Path.cwd()):
    kwargs = request_kwargs(data, base)
    style = kwargs.get("style")
    tags = kwargs.get("tags")
    if style is not None and tags is not None and style != tags:
        raise ValueError("style and tags are aliases and cannot disagree")
    style = tags if style is None else style
    lyrics = kwargs.get("lyrics")
    if style is None or lyrics is None:
        raise ValueError("Provide style and lyrics")
    request = SongRequest(style=style, lyrics=lyrics,
                          **{k: v for k, v in kwargs.items() if k in SONG_FIELDS - {"style", "lyrics"}})
    defaults = GenerationConfig()
    sampling = {}
    for key, default in (("abc_sampling", defaults.abc), ("semantic_sampling", defaults.semantic)):
        if kwargs.get(key) is not None:
            sampling[key] = asdict(resolve_sampling(kwargs[key], default))
    return request, sampling


def request_warnings(request):
    warnings = []
    if not request.style.strip():
        warnings.append("style is empty")
    if not request.lyrics.strip():
        warnings.append("lyrics are empty")
    if not re.search(r"^\[[^\]\n]+\]", request.lyrics, flags=re.M):
        warnings.append("lyrics have no [Section] tags")
    return warnings


def check_request(data, base=Path.cwd(), source="request"):
    try:
        request, sampling = build_request(data, base)
    except (TypeError, ValueError, OSError, KeyError) as exc:
        return {"ok": False, "source": source, "id": data.get("id") if isinstance(data, dict) else None,
                "error": str(exc), "type": type(exc).__name__}
    row = {"ok": True, "source": source, "id": request.id, "request": request.to_dict(),
           "warnings": request_warnings(request)}
    if sampling:
        row["sampling"] = sampling
    return row


def load_jsonl(path):
    path = Path(path)
    rows = []
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON on line {index}: {exc.msg}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"Batch line {index} must be a JSON object")
        rows.append(row)
    return rows


def check_rows(rows, base=Path.cwd(), source="input"):
    ids = [row.get("id") for row in rows]
    if any(item is None for item in ids) or len(ids) != len(set(ids)):
        raise ValueError("Every batch request must have a unique id")
    return [check_request(row, base, source=f"{source}:{row.get('id')}") for row in rows]


def summarize(results, *, strict=False):
    failed = 0
    for row in results:
        if not row.get("ok"):
            failed += 1
        elif strict and row.get("warnings"):
            row["ok"] = False
            row["error"] = "; ".join(row["warnings"])
            row["type"] = "StrictWarning"
            failed += 1
    return {"ok": failed == 0, "checked": len(results), "failed": failed, "results": results}


def collect_inputs(args):
    overrides = {key: getattr(args, key, None) for key in REQUEST_FIELDS}
    if getattr(args, "lyrics_file", None):
        overrides["lyrics"] = Path(args.lyrics_file).read_bytes().decode("utf-8")
    elif getattr(args, "lyrics", None) is not None:
        overrides["lyrics"] = args.lyrics
    if getattr(args, "abc_file", None):
        overrides["abc"] = Path(args.abc_file).read_bytes().decode("utf-8")
    if args.input:
        if args.request:
            raise ValueError("Pass --request or --input, not both")
        path = Path(args.input)
        rows = [apply_overrides(row, overrides) for row in load_jsonl(path)]
        if not rows:
            raise ValueError("Batch input has no request objects")
        return check_rows(rows, path.parent, source=str(path))
    if args.request:
        path = Path(args.request)
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("Request must be a JSON object")
        return [check_request(apply_overrides(data, overrides), path.parent, source=str(path))]
    data = {key: value for key, value in overrides.items() if value is not None}
    if "style" not in data or "lyrics" not in data:
        raise ValueError("Provide --request, --input, or both --style and --lyrics")
    return [check_request(data, Path.cwd(), source="flags")]


def validate_command(args):
    try:
        report = summarize(collect_inputs(args), strict=getattr(args, "strict", False))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        report = {"ok": False, "checked": 0, "failed": 0, "error": str(exc), "type": type(exc).__name__}
        print(json.dumps(report, indent=2, ensure_ascii=False))
        if args.output:
            write_json(args.output, report)
        return 2
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.output:
        write_json(args.output, report)
    return 0 if report["ok"] else 1
