"""Local judge for AutoSolver submissions.

This approximates the public UI fields: assigned tasks, raw selected score, and
penalty = raw_score + 100 * unassigned_count. The hidden official judge remains
source of truth, especially for multi-rider fanout and willingness handling.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

TaskKey = Tuple[str, ...]


def parse_case(input_text: str) -> Dict[str, Any]:
    lines = input_text.splitlines()
    start = 1 if lines and lines[0].strip().startswith("task_id_list") else 0
    tasks = set()
    lookup = {}
    rows = []
    for line_no, line in enumerate(lines[start:], start + 1):
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) < 4:
            continue
        task_text, courier, score_text, will_text = parts[:4]
        task_key = tuple(sorted(t.strip() for t in task_text.split(",") if t.strip()))
        if not task_key or not courier.strip():
            continue
        try:
            score = float(score_text)
            willingness = float(will_text)
        except ValueError:
            continue
        courier = courier.strip()
        tasks.update(task_key)
        row = {
            "task_key": task_key,
            "task_text": task_text.strip(),
            "courier": courier,
            "score": score,
            "willingness": willingness,
            "line_no": line_no,
        }
        rows.append(row)
        lookup[(task_key, courier)] = row
    return {"tasks": sorted(tasks), "lookup": lookup, "rows": rows}


def score_output(case: Dict[str, Any], output: Iterable[Any]) -> Dict[str, Any]:
    errors: List[str] = []
    used_tasks = set()
    used_couriers = set()
    raw_score = 0.0
    assignments = 0
    fanout_rows = 0

    if not isinstance(output, (list, tuple)):
        return _result(case, 0.0, used_tasks, assignments, ["solve() must return a list or tuple"])

    for idx, item in enumerate(output):
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            errors.append(f"row {idx}: assignment must be (task_id_list_str, [courier_id, ...])")
            continue
        task_text, couriers = item
        if not isinstance(task_text, str):
            errors.append(f"row {idx}: task id list must be a string")
            continue
        if not isinstance(couriers, (list, tuple)) or not couriers:
            errors.append(f"row {idx}: courier list must be non-empty")
            continue

        task_key = tuple(sorted(t.strip() for t in task_text.split(",") if t.strip()))
        if not task_key:
            errors.append(f"row {idx}: empty task list")
            continue
        if any(task in used_tasks for task in task_key):
            errors.append(f"row {idx}: task reused in {task_text}")
            continue

        valid_rows = []
        for courier in couriers:
            if not isinstance(courier, str):
                errors.append(f"row {idx}: courier id must be a string")
                continue
            row = case["lookup"].get((task_key, courier.strip()))
            if row is None:
                errors.append(f"row {idx}: no candidate for {task_text} / {courier}")
            else:
                valid_rows.append(row)
        if not valid_rows:
            continue

        chosen = valid_rows[0]
        if chosen["courier"] in used_couriers:
            errors.append(f"row {idx}: courier reused: {chosen['courier']}")
            continue

        if len(valid_rows) > 1:
            fanout_rows += 1
        used_tasks.update(task_key)
        used_couriers.add(chosen["courier"])
        raw_score += chosen["score"]
        assignments += 1

    return _result(case, raw_score, used_tasks, assignments, errors, fanout_rows)


def _result(case, raw_score, used_tasks, assignments, errors, fanout_rows=0):
    total_tasks = len(case["tasks"])
    assigned = len(used_tasks)
    penalty = raw_score + (total_tasks - assigned) * 100.0
    return {
        "valid": not errors,
        "errors": errors,
        "assigned": assigned,
        "total_tasks": total_tasks,
        "assignments": assignments,
        "raw_score": raw_score,
        "penalty": penalty,
        "fanout_rows": fanout_rows,
    }


def load_solver(path: Path):
    spec = importlib.util.spec_from_file_location("autosolver_submission", str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import solver from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "solve"):
        raise RuntimeError("solver module must define solve(input_text: str) -> list")
    return module


def judge_case(solver_path: Path, case_path: Path, repeat: int = 1, time_limit: float | None = None) -> Dict[str, Any]:
    module = load_solver(solver_path)
    text = case_path.read_text(encoding="utf-8")
    case = parse_case(text)
    runs = []
    for _ in range(repeat):
        start = time.perf_counter()
        if time_limit is not None and hasattr(module, "solve_with_report"):
            output, report = module.solve_with_report(text, time_limit)
        else:
            output = module.solve(text)
            report = None
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        score = score_output(case, output)
        score["elapsed_ms"] = elapsed_ms
        if report is not None:
            score["last_report"] = report[-8:]
        runs.append(score)
    best = min(runs, key=lambda r: (not r["valid"], -r["assigned"], r["penalty"], r["elapsed_ms"]))
    return {"case": str(case_path), "solver": str(solver_path), "repeat": repeat, "best": best, "runs": runs}


def main(argv=None) -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Run the local AutoSolver judge.")
    parser.add_argument("--solver", default=str(root / "solver.py"), help="Path to solver.py")
    parser.add_argument("--case", default=str(root / "large_seed301.txt"), help="Path to a case TSV file")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--time-limit", type=float, default=None, help="Use solve_with_report with this limit when available")
    args = parser.parse_args(argv)

    result = judge_case(Path(args.solver), Path(args.case), max(1, args.repeat), args.time_limit)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["best"]["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
