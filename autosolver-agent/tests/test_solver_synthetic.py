import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import solver  # noqa: E402
from autosolver.local_judge import parse_case, score_output  # noqa: E402


def _score(case_text, time_limit=0.7):
    output, report = solver.solve_with_report(case_text, time_limit)
    return score_output(parse_case(case_text), output), report, output


def _score_with_policy(case_text, policy, time_limit=0.7):
    old_policy = solver.FANOUT_POLICY
    try:
        solver.FANOUT_POLICY = policy
        return _score(case_text, time_limit)
    finally:
        solver.FANOUT_POLICY = old_policy


def _assert_valid_full(name, case_text, expected_tasks):
    score, report, output = _score_with_policy(case_text, "off")
    assert score["valid"], (name, score["errors"], output, report[-3:])
    assert score["assigned"] == expected_tasks, (name, score, output, report[-3:])
    return score, output


def test_singletons_only():
    case = """task_id_list\tcourier_id\ttotal_score\twillingness
T1\tC1\t5\t0.8
T1\tC2\t9\t0.9
T2\tC2\t4\t0.6
T2\tC3\t8\t0.7
"""
    score, _output = _assert_valid_full("singletons_only", case, 2)
    assert score["raw_score"] == 9.0, score


def test_bundle_beats_singletons():
    case = """task_id_list\tcourier_id\ttotal_score\twillingness
T1\tC1\t10\t0.8
T2\tC2\t10\t0.8
T1,T2\tC3\t5\t0.5
T1,T2\tC4\t8\t0.9
"""
    score, output = _assert_valid_full("bundle_beats_singletons", case, 2)
    assert score["raw_score"] == 5.0, (score, output)


def test_courier_conflict_repair():
    case = """task_id_list\tcourier_id\ttotal_score\twillingness
T1\tC1\t1\t0.5
T2\tC1\t1\t0.5
T2\tC2\t7\t0.5
T1,T2\tC3\t20\t0.5
"""
    score, output = _assert_valid_full("courier_conflict_repair", case, 2)
    assert score["raw_score"] == 8.0, (score, output)


def test_dirty_rows_are_ignored():
    case = """task_id_list\tcourier_id\ttotal_score\twillingness
bad row
T1\tC1\tbad\t0.5
T1\tC1\t3\t0.5
T2\tC2\t4\t0.5
"""
    score, _output = _assert_valid_full("dirty_rows_are_ignored", case, 2)
    assert score["raw_score"] == 7.0, score


def test_aggressive_fanout_keeps_validity():
    old_policy = solver.FANOUT_POLICY
    try:
        solver.FANOUT_POLICY = "aggressive"
        case = """task_id_list\tcourier_id\ttotal_score\twillingness
T1\tC1\t3\t0.9
T1\tC2\t4\t0.8
T2\tC3\t5\t0.9
T2\tC4\t6\t0.8
T3\tC5\t7\t0.9
T3\tC6\t8\t0.8
"""
        score, report, output = _score(case)
        assert score["valid"], (score, output, report)
        assert score["assigned"] == 3, (score, output, report)
        assert report, "agent report should expose the internal strategy trace"
    finally:
        solver.FANOUT_POLICY = old_policy


if __name__ == "__main__":
    tests = [
        test_singletons_only,
        test_bundle_beats_singletons,
        test_courier_conflict_repair,
        test_dirty_rows_are_ignored,
        test_aggressive_fanout_keeps_validity,
    ]
    for test in tests:
        test()
    print("synthetic tests passed")
