# Constraint-Preserving E-Commerce Search (IT594)

Tests whether product search systems find relevant products **without breaking
requirements the shopper stated** (here: color). Relevance (NDCG@10) and constraint
violation are measured and reported **separately**.

**Research question:** can a search system improve semantic retrieval quality without
increasing the rate at which explicit requirements are violated?

**Hypothesis:** on queries that name a color, dense retrieval has a higher top-10
color-violation rate than BM25, even when its NDCG@10 is equal or higher.

## Main findings (WANDS, 44 color queries)
1. **Dense retrieval breaks the requested color far more often than BM25**, at no
   detectable NDCG difference: +0.259 for MiniLM, +0.192 for MPNet.
2. **The relevance labels reward these errors:** 80 of dense retrieval's 142 top-10 color
   violations are labeled Exact or Partial, so NDCG alone cannot detect the failure.
3. **Missing color data is not the whole story:** adding the color to the product text lowers
   violations for every system, but the dense − BM25 gap stays at +0.212.
4. **The hypothesis fails on 7 of 36 comparable queries**, mostly where BM25 fails too
   (few products in the requested color, or few with a recorded color).
5. **Constraint-aware reranking:** hybrid + reranker v3 beats BM25 on NDCG@10 by
   +0.090 [+0.050, +0.132] with no detectable increase in violations, measured by a grader
   the reranker never reads. A hard color filter removes violations but halves relevance.

## Layout
- `src/` – shared code
  - `metrics.py` – NDCG, violation rate, paired bootstrap
  - `color_checker.py` – color checker v2.1
  - `retrieval.py` – reciprocal rank fusion (RRF)
  - `rerank.py` – constraint-aware reranking (v1 field, v2 name exception, v3 field else description) and hard filter
- `notebooks/`
  - `01_ndcg_basics` – data exploration, NDCG, constraint selection
  - `02_esci_exploration` – ESCI, color checker development and validation
  - `03_wands_baselines`, `eval` – BM25, dense, hybrid; relevance and violation tables, bootstrap
  - `04_constraint_rerank` – reranker v1/v2, penalty sweep, cross-validation, hard filter
  - `05_ablation_and_failures` – color-in-text ablation, where the hypothesis fails, second dense model
  - `06_independent_check` – name-only grader, blind sample, reranker v3
- `tests/` – unit tests for everything in `src/`
- `annotations/` – checker validation sample (`color_validation_v1_filled_AI.csv`: LLM-assisted labels)
- `results/` – spot checks and the blind sample (`blind_check_key.csv` maps items to systems)
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
Open the notebooks from inside `notebooks/` (they use `../src`, `../data`, `../annotations`
and `../results`). Embeddings are cached as `notebooks/*.npy` on the first run (not committed).
Notebook 05 section 6 encodes the catalog with MPNet, which is slow on a CPU the first time.

## Results (WANDS, checker v2.1, lenient mode, top 10)

### Baselines

**Relevance**

| System | NDCG@10 (480 queries) | NDCG@10 (44 color queries) |
|---|---|---|
| BM25 | 0.687 | 0.533 |
| Dense (all-MiniLM-L6-v2) | 0.697 | 0.560 |
| Dense (all-mpnet-base-v2) | 0.684 | 0.563 |
| Hybrid (RRF, MiniLM) | 0.748 | 0.644 |
| Hybrid (RRF, MPNet) | 0.752 | 0.670 |

**Color-constraint violation (44 color queries)**

| System | Violation rate (checkable) | Violating share of top-10 |
|---|---|---|
| BM25 | 0.316 | 0.195 |
| Dense (MiniLM) | 0.592 | 0.323 |
| Dense (MPNet) | 0.528 | 0.330 |
| Hybrid (MiniLM) | 0.416 | 0.220 |
| Hybrid (MPNet) | 0.390 | 0.234 |

**Paired bootstrap on color queries (95% CI)**

| Comparison | NDCG@10 difference | Violation-rate difference |
|---|---|---|
| Dense MiniLM − BM25 | +0.027 [−0.046, +0.105] | +0.259 [+0.180, +0.342] |
| Dense MPNet − BM25 | +0.030 [−0.046, +0.112] | +0.192 [+0.101, +0.284] |
| Hybrid − BM25 | +0.111 [+0.069, +0.158] | +0.077 [+0.010, +0.141] |
| Hybrid − Dense MiniLM | +0.085 [+0.029, +0.139] | −0.180 [−0.242, −0.121] |

Strict mode (shades kept apart) changes BM25's violation rate only from 0.316 to 0.330.

### Mechanism: adding the color to the product text (notebook 05)

| System | Violation rate, base text | + `Color: <value>.` | Change (95% CI) |
|---|---|---|---|
| BM25 | 0.316 | 0.208 | −0.108 [−0.178, −0.041] |
| Dense (MiniLM) | 0.592 | 0.453 | −0.154 [−0.204, −0.106] |
| Hybrid | 0.416 | 0.282 | −0.134 [−0.201, −0.069] |

NDCG does not change detectably. The dense − BM25 gap is +0.259 on the base text and still
+0.212 [+0.102, +0.311] with the color in the text.

### Constraint-aware reranking (notebooks 04 and 06)

The reranker moves products whose color evidence contradicts the query to the end of the
top 100 (penalty chosen by 2-fold cross-validation; ∞ in 5 of 6 cases). v1 reads the color
field; v3 reads the color field and, when it is empty, the description. The main grader also
reads the color field, so reranking is additionally evaluated with a **name-only grader**
that never reads the field or the description.

**Relevance (NDCG@10, color queries)**

| System | Base | + v1 | + v3 | + hard filter |
|---|---|---|---|---|
| BM25 | 0.533 | 0.538 | 0.535 | 0.299 |
| Dense (MiniLM) | 0.560 | 0.566 | 0.549 | 0.308 |
| Hybrid | 0.644 | 0.630 | 0.623 | 0.332 |

**Violation rate, name-only grader (independent of the reranker)**

| System | Base | + v1 | + v3 |
|---|---|---|---|
| BM25 | 0.156 | 0.090 | 0.073 |
| Dense (MiniLM) | 0.299 | 0.214 | 0.194 |
| Hybrid | 0.195 | 0.171 | 0.150 |

**Bootstrap (95% CI)**

| Comparison | NDCG@10 difference | Violation difference (name-only grader) |
|---|---|---|
| BM25 + v3 − BM25 | +0.002 [−0.024, +0.029] | −0.063 [−0.110, −0.022] |
| Dense + v3 − Dense | −0.011 [−0.036, +0.013] | −0.109 [−0.216, −0.015] |
| Hybrid + v3 − Hybrid | −0.022 [−0.042, −0.004] | −0.028 [−0.083, +0.026] |
| Hybrid + v3 − BM25 | +0.090 [+0.050, +0.132] | −0.016 [−0.096, +0.063] |

The hard filter leaves up to 22 color queries with fewer than 10 results. A blind sample
(80 products, system and color field hidden) gives 0.471 vs 0.433 for hybrid vs hybrid + v1.
v2 (trust a color stated in the product name) did not improve the results.

## Checker validation
200 ESCI pairs, stratified by checker outcome, with LLM-assisted labels. In lenient mode a
flagged mismatch is a confirmed violation 74% of the time (94% excluding "unclear"); a reported
match is a violation 3% of the time. The sample was drawn from ESCI, so precision on WANDS may differ.

## Limitations
- 44 color queries on one main dataset; several effects are in the expected direction but not detectable.
- Color is the only constraint studied.
- The checker mishandles multi-color and part-color requests: "white vanity black hardware"
  counts as a match if either color appears (under-counting), while "gold legs" can be flagged
  as a mismatch when the color field describes the seat (over-counting). Colors outside its
  list (e.g. "peach") are ignored.
- 61% of products have no recorded color, which limits every field-based check and fix.
- Unjudged products count as 0, so NDCG is a lower bound.
