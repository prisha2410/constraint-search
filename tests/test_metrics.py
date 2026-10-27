import math
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import pytest
from metrics import ndcg_at_k, constraint_violation_rate, paired_bootstrap


def test_ndcg_hand_example():
    # Partial, Exact, Irrelevant; matches sklearn.metrics.ndcg_score in notebook 01
    assert ndcg_at_k(["a", "b", "c"], {"a": 1, "b": 2, "c": 0}, k=3) == pytest.approx(0.8597, abs=1e-4)


OUTCOME = {"a": "match", "b": "mismatch", "c": "ambiguous", "d": "unknown", "e": "mismatch"}


def test_violation_rate_among_checkable():
    # top-5: 1 match, 2 mismatch, 2 unresolved -> 2 / 3
    assert constraint_violation_rate(list("abcde"), OUTCOME.get, k=5) == pytest.approx(2 / 3)
    assert constraint_violation_rate(list("abcde"), OUTCOME.get, k=2) == pytest.approx(1 / 2)


def test_violation_rate_per_k():
    assert constraint_violation_rate(list("abcde"), OUTCOME.get, k=5, among="k") == pytest.approx(2 / 5)


def test_violation_rate_nothing_checkable_is_nan():
    assert math.isnan(constraint_violation_rate(["c", "d"], OUTCOME.get, k=10))


def test_violation_rate_bad_mode():
    with pytest.raises(ValueError):
        constraint_violation_rate(["a"], OUTCOME.get, among="all")


def test_bootstrap_constant_shift():
    mean, (low, high), n = paired_bootstrap([0.5, 0.7, 0.2], [0.4, 0.6, 0.1])
    assert mean == pytest.approx(0.1) and low == pytest.approx(0.1) and high == pytest.approx(0.1) and n == 3


def test_bootstrap_drops_nan_pairs():
    mean, _, n = paired_bootstrap([0.5, float("nan"), 0.2], [0.1, 0.3, float("nan")])
    assert n == 1 and mean == pytest.approx(0.4)


def test_bootstrap_interval_contains_mean():
    mean, (low, high), _ = paired_bootstrap([0.1, 0.9, 0.4, 0.6], [0.3, 0.2, 0.5, 0.1])
    assert low <= mean <= high
