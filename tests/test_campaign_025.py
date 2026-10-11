from pathlib import Path
from types import SimpleNamespace
import os
import subprocess
import sys
import pytest
from yue2 import fast

@pytest.mark.parametrize('exit_after_result', [True,False])
def test_worker_drains_short_pipe_reads_before_treating_child_exit_as_failure(monkeypatch, exit_after_result):
    popen = subprocess.Popen
    processes = []
    code = """
import json, sys
json.loads(sys.stdin.readline())
for line in sys.stdin:
    json.loads(line)
    print('YUE2_FAST\t' + json.dumps({'event':'result','ids':list(range(1000)),
                                     'timing':{'fixture':True},'truncated':False}),flush=True)
    if sys.argv[1] == 'exit': break
"""
    def launch(command, **kwargs):
        process = popen([sys.executable,'-c',code,'exit' if exit_after_result else 'persist'],**kwargs)
        processes.append(process)
        return process
    monkeypatch.setattr(fast.subprocess,'Popen',launch)
    native_read = os.read
    def short_read(fd,size):
        if not processes or fd != processes[0].stdout.fileno(): return native_read(fd,size)
        value = native_read(fd,min(size,100))
        if exit_after_result: processes[0].wait(timeout=5)
        return value
    monkeypatch.setattr(fast.os,'read',short_read)
    pipe = SimpleNamespace(model_dir=Path('.'),device='cpu',memory_budget_gib=24)
    worker = fast._Worker(pipe)
    try:
        actual = worker.request({'phase':'abc'})
        assert actual == (list(range(1000)),{'fixture':True},False)
    finally: worker.close()
    assert processes[0].poll() is not None
