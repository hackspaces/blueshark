"""SELF-TEST / DEMO  — proves the harness measures real India-specific correctness,
without needing any model or GPU.

For every task it grades TWO canned solutions:
  * the reference (correct) solution           -> must SOLVE all checks
  * a 'naive' format-only solution             -> must FAIL the India-rule checks
    (no UPI cu=INR/amount rigor, no GSTIN mod-36, no Aadhaar Verhoeff)

If the reference passes and the naive one fails on exactly the rule-bearing checks,
the harness is doing its job: it can tell a model that *knows the Indian rule* from
one that only produced something plausible.
"""
from indeval.tasks import TASKS
from indeval.grader import grade
from indeval import report

ref_results = [grade(t, t.reference_solution) for t in TASKS]
naive_results = [grade(t, t.naive_solution) for t in TASKS]

print(report.render("REFERENCE solution (a model that knows the rules)", ref_results))
print(report.render("NAIVE solution (plausible, but ignorant of the Indian rule)", naive_results))

# Assert the harness discriminates, the way a real eval must.
print("  Harness self-check:")
ok = True
for ref, naive, task in zip(ref_results, naive_results, TASKS):
    discriminates = ref.solved and not naive.solved
    ok = ok and discriminates
    verdict = "discriminates" if discriminates else "DOES NOT discriminate"
    print(f"    {task.task_id:18s} ref={ref.passed}/{ref.total}  "
          f"naive={naive.passed}/{naive.total}  -> {verdict}")
print()
print("  RESULT:", "PASS — harness separates rule-knowers from guessers"
      if ok else "FAIL — harness is not discriminating")
