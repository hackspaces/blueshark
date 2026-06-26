"""Core data structures for the India-context agentic eval harness."""
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class Task:
    task_id: str
    domain: str            # "upi" | "gst" | "aadhaar"
    title: str
    prompt: str            # what the model-under-test sees
    entry_point: str       # the function name the model must define
    test_program: str      # python appended after the submission; emits JSON result
    reference_solution: str # known-correct impl (used to self-test the harness)
    naive_solution: str     # plausible-but-wrong impl (format-only, no Indian rule)


@dataclass
class GradeResult:
    task_id: str
    passed: int
    total: int
    failures: list = field(default_factory=list)
    error: str = ""

    @property
    def score(self) -> float:
        return self.passed / self.total if self.total else 0.0

    @property
    def solved(self) -> bool:
        return self.total > 0 and self.passed == self.total
