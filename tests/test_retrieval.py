import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from retrieval import rrf


def test_rrf_rewards_agreement():
    # "b" is 2nd in both lists, so it beats "a" and "c", which are 1st in only one list each
    assert rrf([["a", "b", "x"], ["c", "b", "y"]])[0] == "b"


def test_rrf_top_cut():
    assert len(rrf([list("abcdef")], top=3)) == 3
