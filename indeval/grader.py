"""Execution-based grader: runs the model's submitted code against a task's hidden
tests in an isolated subprocess and parses the structured result.

NOTE: subprocess + timeout is adequate for a proof-of-concept. For an open public
benchmark you would harden this (nsjail/firejail/container, no network, rlimits)."""
import json
import subprocess
import sys
import tempfile
import os
from indeval.schema import Task, GradeResult


def grade(task: Task, submission_code: str, timeout: int = 15) -> GradeResult:
    program = submission_code + "\n" + task.test_program
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, dir="/tmp") as f:
        f.write(program)
        path = f.name
    try:
        proc = subprocess.run(
            [sys.executable, path],
            capture_output=True, text=True, timeout=timeout,
            env={"PATH": os.environ.get("PATH", "")},  # minimal env
        )
    except subprocess.TimeoutExpired:
        os.unlink(path)
        return GradeResult(task.task_id, 0, 0, error="timeout")
    os.unlink(path)

    marker = "__RESULT__"
    line = next((l for l in proc.stdout.splitlines() if l.startswith(marker)), None)
    if line is None:
        # The submission crashed before tests could report (syntax error, missing fn, etc.)
        err = (proc.stderr or proc.stdout or "no output").strip().splitlines()
        return GradeResult(task.task_id, 0, 0, error=(err[-1] if err else "unknown error"))
    data = json.loads(line[len(marker):])
    return GradeResult(
        task.task_id, data["passed"], data["total"], failures=data["failures"]
    )
