import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from color_checker import query_colors, color_check


def chk(q, pc, title="", lenient=False):
    return color_check(query_colors(q, lenient), pc, title, lenient)


def test_not_color_requests():
    for q in ["cica cream", "dove rose petals", "aqua di gio profumo", "green bay packers hat",
              "round blue light glasses", "women white gold wedding bands", "frosted glass white board"]:
        assert chk(q, "Black") is None, q


def test_matches_and_mismatches():
    assert chk("womans pink robe", "Fuchsia With Hood") == "match"
    assert chk("1.63 black drain without overflow", "Matte Balck With Overflow") == "match"
    assert chk("gold link bracelets", "Style-2", "Gold Chain Bracelet") == "match"
    assert chk("white high waisted pants", "Black") == "mismatch"
    assert chk("green advertine crystal", "Multicolor") == "ambiguous"
    assert chk("womens pink capri leggings", "Camouflage", "Yoga Legging") == "unknown"


def test_lenient_folds_shades():
    assert chk("blue sweater", "Navy", lenient=False) == "mismatch"
    assert chk("blue sweater", "Navy", lenient=True) == "match"