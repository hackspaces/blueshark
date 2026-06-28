"""The teacher: turns a task into a worked read->reason->act->verify trajectory.

Pluggable backends so you are never locked in:
  * ClaudeCodeTeacher  -> shells out to `claude -p` (uses your Max-plan login)
  * AnthropicAPITeacher-> uses ANTHROPIC_API_KEY (separate API credits)
  * MockTeacher        -> no calls; lets you test the whole pipeline offline

NOTE ON TERMS: automating a Claude subscription / using model outputs to train another
model has licensing implications. Confirm Anthropic's usage policy before relying on
this at scale; the backend is swappable (an open model like Qwen can be the teacher too).
"""
import json
import logging
import os
import re
import subprocess
import time
import urllib.request

log = logging.getLogger("datagen.teacher")

TEACHER_PROMPT = """You are an expert Indian software engineer writing a high-quality worked \
example that will teach a smaller model how to do this task correctly.

TASK:
{prompt}

Produce a single worked solution as an explicit agentic reasoning trace with four labelled \
phases, then the final code. Be correct and self-contained.

<reasoning>
READ: Restate the requirement and name the exact India-specific rule that governs it \
(cite the real source: the NPCI UPI Linking Specification, the GSTIN Luhn-mod-36 check \
digit, or the Aadhaar Verhoeff checksum). State precisely what "correct" means here.
REASON: Plan the implementation. Call out the edge cases a naive solution would miss \
(e.g. amount formatting, currency must be INR, the checksum step, first-digit rules).
ACT: Write the function.
VERIFY: Walk one normal input and one tricky/invalid input through your function and \
confirm the India-specific rule is actually enforced — not just the surface format.
</reasoning>

Then give the final solution in ONE ```python code block defining a function named \
exactly `{entry_point}`. No prose after the code block."""


def render_prompt(prompt: str, entry_point: str) -> str:
    return TEACHER_PROMPT.format(prompt=prompt, entry_point=entry_point)


def extract_code(text: str) -> str:
    m = re.search(r"```(?:python)?\s*(.*?)```", text, re.DOTALL)
    return (m.group(1) if m else "").strip()


class MockTeacher:
    """Returns a canned response so the pipeline can be tested with no API/auth.
    `quality` toggles whether the mock 'knows the rule' (passes) or not (filtered out)."""
    def __init__(self, solutions: dict, quality="good"):
        self.solutions = solutions
        self.quality = quality

    def __call__(self, prompt, entry_point, domain):
        code = self.solutions[domain][self.quality]
        reasoning = (f"<reasoning>\nREAD: {domain} task; the governing rule is enforced below.\n"
                     "REASON: handle the India-specific check, not just the format.\n"
                     "ACT: see code.\nVERIFY: checked a valid and an invalid input.\n</reasoning>")
        return f"{reasoning}\n\n```python\n{code}\n```"


class ClaudeCodeTeacher:
    """Drives Claude Code headlessly: `claude -p <prompt>`. Uses your Max-plan auth."""
    def __init__(self, model=None, bin="claude"):
        self.model, self.bin = model, bin

    def __call__(self, prompt, entry_point, domain):
        full = render_prompt(prompt, entry_point)
        cmd = [self.bin, "-p", full]
        if self.model:
            cmd += ["--model", self.model]
        log.info("[claude-code] -> domain=%s model=%s entry=%s prompt=%d chars",
                 domain, self.model or "default", entry_point, len(full))
        log.info("[claude-code] >>> PROMPT >>>\n%s\n<<< end prompt", full)
        t0 = time.time()
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        except Exception as e:
            log.error("[claude-code] domain=%s subprocess FAILED: %s", domain, e)
            raise
        dt = time.time() - t0
        log.info("[claude-code] <- domain=%s done in %.1fs rc=%d stdout=%d chars stderr=%d chars",
                 domain, dt, out.returncode, len(out.stdout), len(out.stderr))
        if out.stderr.strip():
            log.warning("[claude-code] domain=%s stderr: %s", domain, out.stderr.strip()[:800])
        log.info("[claude-code] <<< RESPONSE <<<\n%s\n<<< end response", out.stdout.strip()[:6000])
        return out.stdout


class AnthropicAPITeacher:
    """Uses the Anthropic API (ANTHROPIC_API_KEY). Separate from a Max subscription."""
    def __init__(self, model="claude-opus-4-8", api_key=None, max_tokens=2000):
        self.model = model
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.max_tokens = max_tokens

    def __call__(self, prompt, entry_point, domain):
        full = render_prompt(prompt, entry_point)
        body = json.dumps({
            "model": self.model, "max_tokens": self.max_tokens,
            "messages": [{"role": "user", "content": full}],
        }).encode()
        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages", data=body,
            headers={"content-type": "application/json",
                     "x-api-key": self.api_key, "anthropic-version": "2023-06-01"})
        with urllib.request.urlopen(req, timeout=120) as r:
            data = json.loads(r.read())
        return "".join(b.get("text", "") for b in data["content"])
