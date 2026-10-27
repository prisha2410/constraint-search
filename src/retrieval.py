"""Shared retrieval helpers."""


def rrf(rank_lists, k=60, top=100):
    """Reciprocal Rank Fusion: score(p) = sum over lists of 1 / (k + rank), rank starting at 1."""
    score = {}
    for rl in rank_lists:
        for r, p in enumerate(rl):
            score[p] = score.get(p, 0) + 1 / (k + r + 1)
    return [p for p, _ in sorted(score.items(), key=lambda x: -x[1])[:top]]
