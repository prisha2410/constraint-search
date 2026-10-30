# Constraint-Preserving E-Commerce Search (IT594)

Tests whether product search systems find relevant products **without breaking
requirements the shopper stated** (here: color). Relevance (NDCG@10) and constraint
violation are measured and reported **separately**.

## Layout
- `src/` – shared code: `metrics.py` (NDCG, violation rate, paired bootstrap),
  `color_checker.py` (v2.1 color checker), `retrieval.py` (RRF fusion)
- `notebooks/` – `01_ndcg_basics`, `02_esci_exploration`, `03_wands_baselines`, `eval`
- `tests/` – unit tests for everything in `src/`
- `annotations/` – checker validation sample; `color_validation_v1_filled_AI.csv`
  holds **AI-generated** labels
- `data/` – datasets, not committed (see below)

## Setup
```
python -m venv venv
venv\Scripts\activate
python -m pip install -r requirements.txt
python -m pytest -q tests
```

## Data
```
cd data
git clone https://github.com/wayfair/WANDS.git
git lfs install
git clone https://github.com/amazon-science/esci-data.git
cd esci-data
git lfs pull
```
Expected: `data/WANDS/dataset/{product,query,label}.csv` and
`data/esci-data/shopping_queries_dataset/` (the products parquet is ~1.1 GB;
if it is ~130 bytes it is a Git LFS pointer, so run `git lfs pull`).

## Running
Open the notebooks from inside `notebooks/` (they use `../src`, `../data` and `../annotations`).
`eval.ipynb` produces the final relevance and constraint tables and the bootstrap.

## Results (WANDS, checker v2.1, lenient mode)

**Relevance**

| System | NDCG@10 (480 queries) |
|---|---|
| BM25 | 0.687 |
| Dense (all-MiniLM-L6-v2) | 0.697 |
| Hybrid (RRF, k=60) | 0.748 |

**Color-constraint violation (44 color queries, top-10)**

| System | Violation rate (checkable) | Violating share of top-10 |
|---|---|---|
| BM25 | 0.316 | 0.195 |
| Dense | 0.592 | 0.323 |
| Hybrid | 0.416 | 0.220 |

**Paired bootstrap on color queries (95% CI)**

| Comparison | NDCG@10 difference | Violation-rate difference |
|---|---|---|
| Dense − BM25 | +0.027 [−0.046, +0.105] | +0.259 [+0.180, +0.342] |
| Hybrid − BM25 | +0.111 [+0.069, +0.158] | +0.077 [+0.010, +0.141] |
| Hybrid − Dense | +0.085 [+0.029, +0.139] | −0.180 [−0.242, −0.121] |

Dense retrieval violates the queried color far more often than BM25 with no detectable
difference in NDCG. Hybrid has the best NDCG and removes most of dense's violations.

## Checker validation
200 stratified ESCI pairs, labeled by two **AI** annotators (κ = 0.96 between runs).
In lenient mode, a flagged mismatch is a confirmed violation 74% of the time (94% excluding
"unclear"); a reported match is a violation 3% of the time. Precision on WANDS may differ.

## Limitations
- Only 44 color queries, one main dataset, one dense model.
- The checker mishandles multi-color and part-color requests: "white vanity black hardware"
  counts as a match if either color appears (under-counting), while "gold legs" can be flagged
  as a mismatch when the color field describes the seat (over-counting). Colors outside its
  list (e.g. "peach") are ignored.
- Unjudged products count as 0, so NDCG is a lower bound

