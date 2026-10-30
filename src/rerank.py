"""Constraint-aware reranking: push down products whose metadata contradicts the query's constraint.

The reranker's signal is deliberately simpler than the grader (color_checker.color_check):
it reads only the structured color field, with no title fallback and no typo tolerance,
and never penalizes a product whose color is missing or multicolor.
"""
from color_checker import parse_colors


def field_violates(qcols, field, lenient=True, name=""):
    """True only when the product's color field names a color and none of them is in the query.

    v2: pass the product name to spare products whose name states a query color,
    for fields that contradict their own name ("blue throw pillow" with field "brown").
    """
    if not qcols:
        return False
    pcols, multi = parse_colors(field, lenient, False)
    if not pcols or multi or (qcols & pcols):
        return False
    return not (qcols & parse_colors(name, lenient, False)[0])


def field_or_text_violates(qcols, field, description, lenient=True):
    """v3: the color field decides when it has a value; otherwise the start of the description does.

    Never reads the product name, so a name-based grader stays independent of this signal.
    """
    if field:
        return field_violates(qcols, field, lenient)
    return field_violates(qcols, description[:300], lenient)


def field_matches(qcols, field, lenient=True):
    """True when the product's color field names one of the query's colors."""
    pcols, _ = parse_colors(field, lenient, False)
    return bool(qcols & pcols)


def demote(ranked_ids, violates, penalty):
    """Move each product with violates(p) True `penalty` positions down.

    penalty = 0 leaves the list unchanged; float("inf") moves violators to the end.
    At equal positions a non-violating product goes first, otherwise the original order holds.
    """
    flag = {p: violates(p) for p in ranked_ids}
    pos = {p: i + (penalty if flag[p] else 0) for i, p in enumerate(ranked_ids)}
    return sorted(ranked_ids, key=lambda p: (pos[p], flag[p]))


def hard_filter(ranked_ids, keep):
    """Keep only products with keep(p) True, like a "filter by color" facet."""
    return [p for p in ranked_ids if keep(p)]
