"""Agent runner for the model-under-test.

Talks to ANY OpenAI-compatible /v1/chat/completions endpoint (vLLM or SGLang serving
Gemma / Qwen / Ornith locally, or a hosted API). Uses only the stdlib so it runs on a
16 GB laptop with nothing installed.

It is lightly *agentic*: the model gets one optional self-debug round — if its first
solution fails, the failing test output is fed back and it may revise. Swapping this
for native tool-calling (a run_python tool) is a drop-in change.
"""
import json
import re
import urllib.request
from indeval.schema import Task

_SYSTEM = (
    "You are a coding agent for Indian software tasks. Solve the task by writing a single "
    "Python function with the exact name requested. Reply with ONLY a ```python code block "
    "containing the function and any imports it needs. No prose."
)


def _extract_code(text: str) -> str:
    m = re.search(r"```(?:python)?\s*(.*?)```", text, re.DOTALL)
    return (m.group(1) if m else text).strip()


def _chat(base_url, model, messages, api_key=None, temperature=0.2, max_tokens=1500):
    url = base_url.rstrip("/") + "/chat/completions"
    body = json.dumps({
        "model": model, "messages": messages,
        "temperature": temperature, "max_tokens": max_tokens,
    }).encode()
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = "Bearer " + api_key
    req = urllib.request.Request(url, data=body, headers=headers)
    with urllib.request.urlopen(req, timeout=120) as r:
        data = json.loads(r.read())
    return data["choices"][0]["message"]["content"]


def solve(task: Task, base_url, model, api_key=None, max_rounds=2, grader=None):
    """Returns the model's final submitted code string. If `grader` is provided and a
    first attempt fails, gives the model one feedback round (the agentic loop)."""
    messages = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": task.prompt},
    ]
    code = _extract_code(_chat(base_url, model, messages, api_key))
    for _ in range(max_rounds - 1):
        if grader is None:
            break
        res = grader(task, code)
        if res.solved:
            break
        feedback = "Your solution failed these checks:\n" + "\n".join(res.failures or [res.error])
        feedback += "\n\nReturn a corrected ```python code block."
        messages += [{"role": "assistant", "content": "```python\n" + code + "\n```"},
                     {"role": "user", "content": feedback}]
        code = _extract_code(_chat(base_url, model, messages, api_key))
    return code
