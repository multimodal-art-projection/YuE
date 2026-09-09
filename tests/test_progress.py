import asyncio
import io
import os
import random
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

import yue2.progress as progress_module
from yue2.progress import Progress


class Terminal(io.StringIO):
    def __init__(self):
        super().__init__()
        self.refreshed = threading.Event()
        self.writes = 0

    def isatty(self):
        return True

    def write(self, value):
        result = super().write(value)
        if value.startswith("\r"):
            self.writes += 1
            if self.writes >= 2:
                self.refreshed.set()
        return result


def test_quiet_never_touches_stream_or_starts_thread(monkeypatch):
    class ForbiddenStream:
        def __getattr__(self, name):
            raise AssertionError(f"Quiet progress touched stream.{name}")

    def forbidden(*args, **kwargs):
        raise AssertionError("Quiet progress started a thread")

    monkeypatch.setattr(progress_module.threading, "Thread", forbidden)
    with Progress(enabled=False, stream=ForbiddenStream()) as reporter:
        with reporter.stage("Generating tokens", unit="tokens") as stage:
            stage.token("semantic", 123)
            stage.update(2)
        reporter.complete(1.5, 0.2)
    assert stage.completed == 2


def test_loading_heartbeat_refreshes_without_callbacks_and_stops():
    stream = Terminal()
    reporter = Progress(stream=stream, refresh_interval=0.01)
    with reporter.stage("Loading model"):
        assert stream.refreshed.wait(2.0), "No heartbeat during blocking model load"
        thread = reporter._thread
        assert thread.is_alive()
    assert not thread.is_alive()
    assert reporter._thread is None
    assert stream.getvalue().endswith("\n")
    assert "Completed Loading model" in stream.getvalue()


def test_non_tty_is_throttled_and_counts_only_callback_outputs(monkeypatch):
    now = [100.0]
    monkeypatch.setattr(progress_module, "_clock", lambda: now[0])
    stream = io.StringIO()
    with Progress(stream=stream) as reporter:
        with reporter.stage("Generating tokens", unit="tokens") as stage:
            for _ in range(100):
                stage.token("semantic", object())
            assert len(stream.getvalue().splitlines()) == 1
            now[0] += 5.0
            stage.update(100)
            stage.update(100)
            assert len(stream.getvalue().splitlines()) == 2
    output = stream.getvalue()
    assert "100 tokens | 20.0 tokens/s | elapsed 5.0s" in output
    assert "%" not in output
    assert "\r" not in output
    assert len(output.splitlines()) == 3


def test_real_total_can_arrive_after_stage_starts(monkeypatch):
    now = [0.0]
    monkeypatch.setattr(progress_module, "_clock", lambda: now[0])
    stream = io.StringIO()
    with Progress(stream=stream).stage("Synthesizing", unit="steps") as stage:
        now[0] = 5.0
        stage.update(8, total=32)
        stage.set_total(64)
        now[0] = 10.0
        stage.update(16)
        stage.advance(16)
    output = stream.getvalue()
    assert "8/32 steps (25%)" in output
    assert "16/64 steps (25%)" in output
    assert "32/64 steps (50%)" in output
    assert stage.completed == 32  # Exit must not invent completion of all 64 steps.


@pytest.mark.parametrize("exception,status", [
    (RuntimeError("model failed"), "Failed"),
    (InterruptedError("request cancelled"), "Cancelled"),
    (KeyboardInterrupt(), "Cancelled"),
    (asyncio.CancelledError(), "Cancelled"),
])
def test_exceptions_propagate_with_correct_status_and_no_thread_leak(exception, status):
    stream = io.StringIO()
    reporter = Progress(stream=stream)
    with pytest.raises(type(exception)):
        with reporter.stage("Working"):
            thread = reporter._thread
            raise exception
    assert status in stream.getvalue()
    assert "Completed" not in stream.getvalue()
    assert not thread.is_alive()


def test_truncated_stage_and_summary_do_not_claim_completion():
    stream = io.StringIO()
    with Progress(stream=stream) as reporter:
        with reporter.stage("Generating tokens", unit="tokens") as stage:
            stage.advance(200)
            thread = reporter._thread
            stage.finish(status="truncated")
            assert not thread.is_alive()
        reporter.complete(8.0, 1.5, truncated=True)
    output = stream.getvalue()
    assert output.count("Finished (generation limit reached)") == 2
    assert "Completed" not in output
    assert "8.0s audio in 1.5s" in output


def test_complete_uses_stderr_and_needs_no_thread(monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        raise AssertionError("Completion summary started a thread")

    monkeypatch.setattr(progress_module.threading, "Thread", forbidden)
    reporter = Progress()
    reporter.complete(123.45, 6.78)
    reporter.complete(123.45, 6.78)
    capture = capsys.readouterr()
    assert capture.out == ""
    assert capture.err == "[YuE2] Completed: 123.5s audio in 6.8s\n"


def test_concurrent_callbacks_are_counted_exactly_and_do_not_change_rng():
    before = random.getstate()
    stream = io.StringIO()
    with Progress(stream=stream).stage("Generating tokens", unit="tokens") as stage:
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(lambda _: stage.token("semantic", None), range(4000)))
    assert stage.completed == 4000
    assert "4000 tokens" in stream.getvalue()
    assert random.getstate() == before


def test_module_import_and_quiet_usage_require_no_torch_or_rng_modules():
    module_file = Path(progress_module.__file__).resolve()
    code = f"""
import importlib.util, sys
spec = importlib.util.spec_from_file_location('standalone_progress', {str(module_file)!r})
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
with module.Progress(enabled=False).stage('CPU test') as stage:
    stage.advance()
assert not {{'torch', 'numpy', 'random'}}.intersection(sys.modules)
"""
    subprocess.run([sys.executable, "-I", "-S", "-c", code], check=True, timeout=10)


def test_closed_output_stream_does_not_fail_work():
    stream = io.StringIO()
    stream.close()
    with Progress(stream=stream).stage("Loading model") as stage:
        stage.advance()
    assert stage.completed == 1


def test_outer_context_cleans_up_an_unfinished_stage_on_failure():
    stream = io.StringIO()
    with pytest.raises(RuntimeError, match="load failed"):
        with Progress(stream=stream) as reporter:
            stage = reporter.stage("Loading model").__enter__()
            thread = reporter._thread
            raise RuntimeError("load failed")
    assert not thread.is_alive()
    assert stage.status == "failed"
    assert "Completed" not in stream.getvalue()


def test_invalid_values_fail_without_spawning_threads():
    for interval in [0, -1, float("nan"), float("inf")]:
        with pytest.raises(ValueError):
            Progress(refresh_interval=interval)
    reporter = Progress(enabled=False)
    for total in [-1, 1.5, True]:
        with pytest.raises(ValueError):
            reporter.stage("Invalid", total=total)
    with reporter.stage("Valid", total=0) as stage:
        stage.set_total(None)
        stage.advance(2)
        with pytest.raises(ValueError):
            stage.update(1)
        with pytest.raises(ValueError):
            stage.finish(status="invented")
    assert stage.status == "completed"


def test_tty_output_is_ascii_single_line_and_refresh_is_throttled(monkeypatch):
    now = [0.0]
    monkeypatch.setattr(progress_module, "_clock", lambda: now[0])
    stream = Terminal()
    with Progress(stream=stream, refresh_interval=0.25).stage("Generating\nASCII \x1b text", unit="tokens") as stage:
        for _ in range(50):
            stage.advance()
        assert stream.writes == 1
        now[0] += 0.25
        stage.advance()
        assert stream.writes == 2
    output = stream.getvalue()
    assert output.isascii()
    assert "\x1b" not in output
    assert output.count("\n") == 1
    assert "51 tokens" in output


@pytest.mark.skipif(os.name != "posix", reason="PTY regression requires POSIX")
def test_zero_width_pty_preserves_stage_label_and_token_rate(monkeypatch):
    import fcntl
    import struct
    import termios

    monkeypatch.delenv("COLUMNS", raising=False)
    now = [0.0]
    monkeypatch.setattr(progress_module, "_clock", lambda: now[0])
    master, slave = os.openpty()
    try:
        fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", 0, 0, 0, 0))
        assert os.get_terminal_size(slave).columns == 0
        with os.fdopen(os.dup(slave), "w", buffering=1) as stream:
            with Progress(stream=stream).stage("Generating song", unit="tokens") as stage:
                now[0] = 10.0
                stage.update(1500)
        output = os.read(master, 8192).decode("ascii")
    finally:
        os.close(slave)
        os.close(master)
    assert "Completed Generating song: 1500 tokens | 150.0 tokens/s | 10.0s" in output
    assert "[YuE2] | :" not in output


@pytest.mark.parametrize("columns", [20, 59])
def test_narrow_terminal_uses_throttled_unclipped_lines(monkeypatch, columns):
    monkeypatch.delenv("COLUMNS", raising=False)
    monkeypatch.setattr(Terminal, "fileno", lambda self: 2)
    monkeypatch.setattr(progress_module.os, "get_terminal_size", lambda fd: os.terminal_size((columns, 24)))
    now = [0.0]
    monkeypatch.setattr(progress_module, "_clock", lambda: now[0])
    stream = Terminal()
    with Progress(stream=stream).stage("Generating song", unit="tokens") as stage:
        for _ in range(100):
            stage.advance()
        assert len(stream.getvalue().splitlines()) == 1
        now[0] = 5.0
        stage.update(100)
    output = stream.getvalue()
    assert "\r" not in output
    assert len(output.splitlines()) == 3
    assert "Completed Generating song: 100 tokens | 20.0 tokens/s | elapsed 5.0s" in output


@pytest.mark.parametrize("override,expected", [("120", 120), ("0", 80), ("-2", 80), ("invalid", 80)])
def test_columns_override_requires_positive_integer(monkeypatch, override, expected):
    monkeypatch.setenv("COLUMNS", override)
    monkeypatch.setattr(Terminal, "fileno", lambda self: 2)
    monkeypatch.setattr(progress_module.os, "get_terminal_size", lambda fd: os.terminal_size((0, 0)))
    assert Progress(stream=Terminal())._columns == expected
