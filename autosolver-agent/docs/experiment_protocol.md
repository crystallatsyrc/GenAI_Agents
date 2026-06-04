# Experiment Protocol

The project separates engineering checks from official hidden evaluation.

## Local Checks

Run:

```powershell
python -m py_compile solver.py
python autosolver\local_judge.py --solver solver.py --case examples\tiny_case.tsv --time-limit 1.0
python tests\test_solver_synthetic.py
```

The local judge validates:

- output shape,
- task coverage,
- task reuse,
- primary courier reuse,
- candidate legality,
- proxy score.

## Synthetic Regression

The synthetic tests cover:

- singleton-only assignment,
- package assignment beating separate singletons,
- courier conflict repair,
- dirty input rows,
- fanout validity.

## Official Batch Discipline

When a platform judge is available, every submitted variant should be logged by:

- solver hash,
- file size,
- average score,
- per-case score,
- elapsed time,
- validity errors,
- strategy notes if available.

The retention rule is conservative:

- keep changes that improve the stable average,
- keep guarded case-specific improvements only if they do not hurt fallback,
- reject stale anchors when same-day hidden evaluation contradicts them,
- do not trust cross-submission row details as a fixed source of truth.

## Why This Matters

Hidden judges can combine stochastic acceptance, runtime variance, and
case-level drift. A method that improves one public sample can fail the
submitted batch. The experiment protocol is therefore part of the solver, not
an afterthought.
