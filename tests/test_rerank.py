import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from rerank import field_violates, field_matches, demote, hard_filter

BLUE = frozenset({"blue"})


def test_field_violates_only_on_a_named_wrong_color():
    assert field_violates(BLUE, "Black") is True
    assert field_violates(BLUE, "Navy") is False            # lenient: navy folds into blue
    assert field_violates(BLUE, "Navy", lenient=False) is True
    assert field_violates(BLUE, "") is False                # missing color is never penalized
    assert field_violates(BLUE, "Multicolor Black") is False
    assert field_violates(frozenset(), "Black") is False    # query asks for no color


def test_field_violates_v2_trusts_a_name_that_states_the_color():
    assert field_violates(BLUE, "Brown", name="blue throw pillow") is False
    assert field_violates(BLUE, "Brown", name="throw pillow") is True
    assert field_violates(BLUE, "Brown") is True            # v1 behaviour when no name is given


def test_field_matches():
    assert field_matches(BLUE, "Dark Blue") is True
    assert field_matches(BLUE, "") is False


def test_demote_moves_violators_down_by_penalty():
    bad = {"b"}.__contains__
    assert demote(list("abcd"), bad, 0) == list("abcd")
    assert demote(list("abcd"), bad, 1) == list("acbd")     # b at 1 -> 2; c (non-violating) wins the tie
    assert demote(list("abcd"), bad, 10) == list("acdb")
    assert demote(list("abcd"), bad, float("inf")) == list("acdb")


def test_hard_filter():
    assert hard_filter(list("abcd"), {"a", "c"}.__contains__) == ["a", "c"]
