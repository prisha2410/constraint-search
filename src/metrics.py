"""Shared ranking metrics: relevance (NDCG) and constraint violation, kept separate on purpose."""
import numpy as np


def dcg(gains):
    return sum(g / np.log2(i + 2) for i, g in enumerate(gains))


def ndcg_at_k(ranked_ids, judged, k=10):
    """ranked_ids: product ids in ranked order. judged: {product_id: gain}. Unlabeled products count as 0."""
    gains = [judged.get(p, 0) for p in ranked_ids[:k]]
    ideal = sorted(judged.values(), reverse=True)[:k]
    return dcg(gains) / dcg(ideal) if dcg(ideal) > 0 else 0.0


def constraint_violation_rate(ranked_ids, check, k=10, among="checkable"):
    """Share of the top-k results that violate the query's constraint.

    ranked_ids: product ids in ranked order.
    check: function product_id -> "match" / "mismatch" / "ambiguous" / "unknown"
           (or None when the query has no constraint).
    among: "checkable" -> mismatches / (match + mismatch); NaN if nothing in the top-k is checkable.
           "k"         -> mismatches / k; ambiguous and unknown results count as not violating.
    """
    if among not in ("checkable", "k"):
        raise ValueError(f"among must be 'checkable' or 'k', got {among!r}")
    out = [check(p) for p in ranked_ids[:k]]
    m, mm = out.count("match"), out.count("mismatch")
    if among == "k":
        return mm / k
    return mm / (m + mm) if (m + mm) else float("nan")


def paired_bootstrap(a, b, n=5000, seed=0):
    """Mean of a - b over paired per-query scores and its 95% bootstrap interval.

    a, b: per-query scores in the same query order. Pairs where either is NaN are dropped.
    Returns (mean_diff, (low, high), n_queries_used).
    """
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    d = d[~np.isnan(d)]
    if len(d) == 0:
        raise ValueError("no query has a score for both systems")
    rng = np.random.default_rng(seed)
    means = [rng.choice(d, len(d)).mean() for _ in range(n)]
    low, high = np.percentile(means, [2.5, 97.5])
    return d.mean(), (low, high), len(d)
