"""Execution-based grader: runs the model's submitted code against a task's hidden
tests in a sandboxed subprocess and parses the structured result.

Sandboxing (portable, standard library only; POSIX = Linux/macOS):
  - isolated temp working directory, removed after each run (file writes contained)
  - stripped environment + isolated interpreter (python -I): no PYTHONPATH/user site
  - OS resource limits: CPU time, address space (Linux), file size, no core dumps,
    capped process count (anti fork-bomb)
  - best-effort network kill injected into the submission (indeval tasks are all
    offline, so a real solution never needs the network)
  - the child runs in its own session; on timeout the whole process group is killed,
    so any spawned children die too

Honest limit: this is hardened, not a hard security boundary. For an open public
benchmark grading fully untrusted/adversarial models, run it additionally inside a
container or nsjail/gVisor with no network namespace. The hooks here make that a
drop-in outer layer.
"""
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile

from indeval.schema import Task, GradeResult

_POSIX = os.name == "posix"
try:
    import resource  # POSIX only
except ImportError:  # pragma: no cover (Windows)
    resource = None

# Injected before every submission. indeval tasks are offline by design.
_PREAMBLE = (
    "import socket as _socket\n"
    "def _no_net(*a, **k):\n"
    "    raise OSError('network disabled in sandbox')\n"
    "_socket.socket = _no_net\n"
    "_socket.create_connection = _no_net\n"
    "try:\n"
    "    import urllib.request as _u\n"
    "    _u.urlopen = _no_net\n"
    "except Exception:\n"
    "    pass\n"
)


def _make_limiter(timeout, mem_bytes):
    def _apply():
        limits = [
            (resource.RLIMIT_CPU, timeout + 1),
            (resource.RLIMIT_FSIZE, 10 * 1024 * 1024),  # 10 MB max write
            (resource.RLIMIT_CORE, 0),                  # no core dumps
            (resource.RLIMIT_NPROC, 256),               # anti fork-bomb
        ]
        # RLIMIT_AS is reliable on Linux; on macOS it can break interpreter startup.
        if sys.platform == "linux":
            limits.append((resource.RLIMIT_AS, mem_bytes))
        for res, val in limits:
            try:
                resource.setrlimit(res, (val, val))
            except Exception:
                pass
    return _apply


def grade(task: Task, submission_code: str, timeout: int = 15,
          mem_bytes: int = 1024 * 1024 * 1024) -> GradeResult:
    program = _PREAMBLE + submission_code + "\n" + task.test_program
    workdir = tempfile.mkdtemp(prefix="indeval_")
    path = os.path.join(workdir, "submission.py")
    with open(path, "w", encoding="utf-8") as f:
        f.write(program)

    popen_kwargs = dict(
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        cwd=workdir,
        env={"PATH": os.environ.get("PATH", "")},  # stripped environment
    )
    if _POSIX and resource is not None:
        popen_kwargs["preexec_fn"] = _make_limiter(timeout, mem_bytes)
        popen_kwargs["start_new_session"] = True  # own process group

    try:
        proc = subprocess.Popen([sys.executable, "-I", path], **popen_kwargs)
        try:
            out, err = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            if _POSIX:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)  # kill the group
                except Exception:
                    proc.kill()
            else:
                proc.kill()
            proc.communicate()
            return GradeResult(task.task_id, 0, 0, error="timeout")
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    marker = "__RESULT__"
    line = next((l for l in out.splitlines() if l.startswith(marker)), None)
    if line is None:
        # submission crashed before tests could report (syntax error, missing fn, OOM, killed)
        msg = (err or out or "no output").strip().splitlines()
        return GradeResult(task.task_id, 0, 0, error=(msg[-1] if msg else "unknown error"))
    data = json.loads(line[len(marker):])
    return GradeResult(task.task_id, data["passed"], data["total"], failures=data["failures"])
