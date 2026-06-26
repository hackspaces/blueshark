"""Console + JSON reporting for an eval run."""
import json


def render(label, results):
    rows = []
    width = max((len(r.task_id) for r in results), default=10)
    total_p = sum(r.passed for r in results)
    total_t = sum(r.total for r in results)
    solved = sum(1 for r in results if r.solved)
    lines = []
    lines.append("")
    lines.append(f"  India-Context Agentic Eval  —  {label}")
    lines.append("  " + "-" * (width + 34))
    lines.append(f"  {'task'.ljust(width)}   tests   score   status")
    lines.append("  " + "-" * (width + 34))
    for r in results:
        status = "SOLVED" if r.solved else ("ERROR" if r.error else "FAILED")
        bar = f"{r.passed}/{r.total}".rjust(5)
        score = f"{r.score*100:5.0f}%"
        lines.append(f"  {r.task_id.ljust(width)}   {bar}   {score}   {status}")
        if not r.solved:
            for fail in (r.failures or ([r.error] if r.error else [])):
                lines.append(f"  {' '.ljust(width)}      -> {fail}")
    lines.append("  " + "-" * (width + 34))
    pct = (total_p / total_t * 100) if total_t else 0
    lines.append(f"  TOTAL: {solved}/{len(results)} tasks solved | "
                 f"{total_p}/{total_t} checks ({pct:.0f}%)")
    lines.append("")
    return "\n".join(lines)


def to_json(label, results):
    return json.dumps({
        "label": label,
        "tasks_solved": sum(1 for r in results if r.solved),
        "n_tasks": len(results),
        "checks_passed": sum(r.passed for r in results),
        "checks_total": sum(r.total for r in results),
        "results": [
            {"task_id": r.task_id, "passed": r.passed, "total": r.total,
             "solved": r.solved, "failures": r.failures, "error": r.error}
            for r in results
        ],
    }, indent=2)
