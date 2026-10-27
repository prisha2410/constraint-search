"""color_checker.py  (v2.1)  Color-constraint checker shared by all notebooks.
Usage:  from color_checker import query_colors, parse_colors, color_check
"""
import re
from functools import lru_cache

# Different names for the SAME color (merged in both modes)
SYNONYM = {"grey": "gray", "violet": "purple", "fuchsia": "pink", "aqua": "teal", "turquoise": "teal"}
# Shades folded into a base color (lenient mode only)
SHADE = {"navy": "blue", "burgundy": "red", "maroon": "red", "charcoal": "gray", "khaki": "beige",
         "tan": "beige", "taupe": "beige", "cream": "white", "ivory": "white", "lavender": "purple",
         "magenta": "pink", "rose": "pink"}
BASE = {"black", "white", "gray", "silver", "blue", "red", "green", "yellow", "orange", "pink",
        "purple", "brown", "beige", "gold", "teal", "copper", "bronze"}
KNOWN = BASE | set(SYNONYM) | set(SHADE)
FUZZY_TARGETS = [k for k in KNOWN if len(k) >= 5]

# Phrases that contain a color word but are NOT a color request
PHRASES = ["white gold", "rose gold", "yellow gold", "blue light", "white board", "whiteboard",
           "white noise", "red wine", "white wine", "green tea", "black tea", "black friday", "red bull",
           # v2.1: names / entities that contain a color word
           "black lives matter", "green bay", "white sox", "red sox", "black panther", "black decker",
           "black+decker", "black&decker", "red dot", "thin blue line", "snow white", "big blue",
           "blue heaven", "princess in black", "black light", "white christmas", "jane green",
           "green hill", "golf green", "purple mattress"]
QUERY_DROP = {"cream", "rose", "aqua"}      # v2.1: product nouns, not color requests
MULTI = {"multi", "multicolor", "multicolored", "multicolour", "assorted", "mixed", "rainbow", "colorful"}

def _canon(w, lenient):
    if w not in KNOWN:
        return None
    w = SYNONYM.get(w, w)
    return SHADE.get(w, w) if lenient else w

def _near(a, b):
    """True if a and b differ by exactly one letter, or one adjacent swap (typo)."""
    if len(a) != len(b) or a == b:
        return False
    diff = [i for i in range(len(a)) if a[i] != b[i]]
    if len(diff) == 1:
        return True
    return (len(diff) == 2 and diff[1] == diff[0] + 1
            and a[diff[0]] == b[diff[1]] and a[diff[1]] == b[diff[0]])

@lru_cache(maxsize=None)
def parse_colors(text, lenient, fuzzy):
    """Return (set of canonical colors found, multicolor flag)."""
    cols, multi = set(), False
    for t in re.findall(r"[a-z]+", str(text).lower()):
        if t in MULTI:
            multi = True
            continue
        c = _canon(t, lenient)
        if c is None and fuzzy and len(t) >= 5:
            for k in FUZZY_TARGETS:
                if _near(t, k):
                    c = _canon(k, lenient)
                    break
        if c:
            cols.add(c)
    return frozenset(cols), multi

@lru_cache(maxsize=None)
def query_colors(q, lenient=False):
    s = str(q).lower()
    for p in PHRASES:
        s = s.replace(p, " ")
    s = " ".join(w for w in re.findall(r"[a-z]+", s) if w not in QUERY_DROP)
    return parse_colors(s, lenient, False)[0]

def color_check(qcols, pcolor, title, lenient=False):
    """None = query has no color request. Otherwise match / mismatch / ambiguous / unknown."""
    if not qcols:
        return None
    pcols, multi = parse_colors(pcolor, lenient, True)          # 1) trust the color field
    if not pcols and not multi:
        pcols, _ = parse_colors(title, lenient, False)          # 2) fall back to the title
        if not pcols:
            return "unknown"
    if qcols & pcols:
        return "match"
    return "ambiguous" if multi else "mismatch"