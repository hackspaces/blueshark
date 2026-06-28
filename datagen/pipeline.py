"""The data engine loop: generate -> teach -> VERIFY (indeval grader) -> keep -> write.

Only trajectories whose final code passes the real Indian-rule check are written. The
kept assistant message includes the teacher's read->reason->act->verify CoT, so the
student learns the reasoning, while the filter guarantees the answer was correct.
"""
import json
import random
from indeval.schema import Task
from indeval.grader import grade
from datagen.gen import GENERATORS
from datagen.teacher import extract_code

_SYSTEM = ("You are a careful Indian software engineer. Reason step by step "
           "(read, reason, act, verify), then give a single correct Python function.")


def build_dataset(teacher, n_per_domain=5, domains=None, seed=0, out_path="train.jsonl"):
    rng = random.Random(seed)
    domains = domains or list(GENERATORS)
    kept, attempted, rows = 0, 0, []
    stats = {d: {"kept": 0, "attempted": 0} for d in domains}

    for domain in domains:
        for _ in range(n_per_domain):
            attempted += 1
            stats[domain]["attempted"] += 1
            prompt, entry_point, test_program = GENERATORS[domain](rng)

            response = teacher(prompt, entry_point, domain)      # frontier teacher
            code = extract_code(response)
            if not code:
                continue

            # VERIFY against the real rule using a freshly-generated test program
            task = Task(task_id=f"{domain}_train", domain=domain, title="",
                        prompt=prompt, entry_point=entry_point,
                        test_program=test_program, reference_solution="", naive_solution="")
            result = grade(task, code)
            if not result.solved:
                continue   # teacher got it wrong on this instance -> drop it

            kept += 1
            stats[domain]["kept"] += 1
            rows.append({
                "domain": domain,
                "messages": [
                    {"role": "system", "content": _SYSTEM},
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": response.strip()},
                ],
                "verified": True,
                "checks": f"{result.passed}/{result.total}",
            })

    with open(out_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    return {"kept": kept, "attempted": attempted,
            "keep_rate": (kept / attempted if attempted else 0),
            "by_domain": stats, "out_path": out_path}
