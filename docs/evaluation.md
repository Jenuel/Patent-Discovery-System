# Retrieval Evaluation

How the retrieval stack is measured, what the numbers say, and why the shipped
defaults (`RETRIEVAL_ARM=hybrid`, reranking off) are what they are.

**Corpus:** 6,000 HUPD G06 (computing/AI) patents · 55,702 claim chunks
(`patents_hybrid` / `claims_hybrid` Qdrant collections). Built by
`apps/api/scripts/build_hupd_corpus.py`.

**Harness:** `apps/api/evaluation/retrieval_eval.py`, scoring
`HierarchicalRetriever` against the ground truth in
`apps/api/evaluation/fixtures/queries.jsonl` (20 queries, 19 graded) and
`queries_pooled.jsonl` (20 queries, pooled re-judgement — see
[Ground truth](#ground-truth) below). Metrics in `evaluation/metrics.py`.
Every arm/weight/rerank combination below runs through the ordinary
production code path — see the harness's own docstring for the exact flags.

**Standing caveat:** most tables here have **n=19 or n=20**. A 95% bootstrap
CI on MRR is roughly **±0.15**; treat any difference smaller than that as
noise rather than signal. Where a table's ordering matters more than any
single gap, the text says so explicitly.

**`evaluation/results/*.csv` is gitignored** — the CSVs a run produces don't
survive a clean clone. This document is the durable record. Update it
whenever the retrieval stack, weights, or defaults change.

---

## Ground truth

**Method: known-item + CPC family, hand-curated.**

1. Group the corpus by `main_cpc_label`.
2. Keep groups of 4–9 members, manually inspected to exclude off-topic
   entries (CPC groups are frequently not topically coherent — e.g. `G16H5030`
   mixes genetic counseling with sales solicitations).
3. Write a conceptual query per family, deliberately avoiding the source
   patents' title wording, so BM25 can't win on title-doubling alone.
4. Grade `2` (squarely on topic) or `1` (related/partial).

Candidates were sourced by scanning `data/hupd_g06/patents.jsonl` directly —
Qdrant was never queried while building ground truth, so the eval cannot be
an echo of what the retriever already returns.

### The title/abstract bug, and what fixing it changed

Four of the original twenty queries scored exactly **0.000**. Diagnosis (retrieve
top 200, then check where each ground-truth patent actually landed) showed
three of the four were **grading defects, not retrieval failures**: relevance
had been judged from patent titles, while the retriever embeds
`title + abstract + claims`. Titles are drafted for legal breadth, not
description, so the two disagree in both directions — some "relevant" titles
described unrelated content, and some genuinely relevant patents sat outside
the CPC anchor entirely because relevance doesn't respect CPC boundaries.

Re-grading those four queries from abstracts (documented per-query in the
`notes` field of `queries.jsonl`, prefixed `ABSTRACT-VERIFIED`) moved:

| Metric | Before | After | Δ |
|---|---|---|---|
| MRR (patent) | 0.3733 | 0.4905 | +31% |
| recall@10 (patent) | 0.3382 | 0.4066 | +20% |
| hit_rate@10 (patent) | 0.7895 | **1.0000** | every query |
| MRR (claim) | 0.3119 | 0.4029 | +29% |

**Lessons that generalize:** grade relevance from the same text the retriever
sees; patent titles are an unreliable relevance signal; a zero score is a
hypothesis, not a verdict — always check where the expected document actually
ranked; CPC codes are a useful candidate anchor but a dangerous ground-truth
definition.

**What these numbers are and are not:** a **lower bound** — only those four
queries were abstract-verified; the other fifteen retain title-based grading
and likely carry the same class of error, in the same direction (recall is
the most suppressed metric). **Useful for relative comparison** — the
fixtures are stable, so A/B changes to the retrieval stack can be compared
against each other with confidence. **Not a benchmark score** — don't publish
`recall@10 = 0.41` as this system's retrieval quality; the true figure is
higher by an unknown margin.

### Pooled re-judgement

The standard IR fix for the CPC-anchor bias above is TREC-style pooling: pull
top-k from multiple independent systems (dense arm, BM25 arm, hybrid), merge
into one candidate pool, and judge every candidate by abstract, blind to
which system produced it — so relevance isn't bounded by an initial CPC
guess. `queries_pooled.jsonl` is the result, and is what the current weighted-fusion
numbers (Ablation, Test 6 below) are measured against.

---

## Baseline (unweighted hybrid, the shipping default at the time)

**Config:** `text-embedding-3-small` (1536d) + `fastembed` BM25, RRF fused
server-side, `HierarchicalConfig` defaults.

**Patent level (Stage 1 — hybrid dense + BM25)**

| k | recall@k | precision@k | hit_rate@k | nDCG@k |
|---|---|---|---|---|
| 3 | 0.1900 | 0.3158 | 0.6842 | 0.3034 |
| 5 | 0.2564 | 0.2632 | 0.7895 | 0.2869 |
| 10 | **0.4066** | 0.2158 | **1.0000** | 0.3562 |

MRR = **0.4905**

**Claim level (Stage 2 — dense only, scoped to Stage-1 patents)**

| k | recall@k | precision@k | hit_rate@k | nDCG@k |
|---|---|---|---|---|
| 5 | 0.0754 | 0.1684 | 0.5263 | 0.1785 |
| 10 | 0.1383 | 0.1474 | 0.7368 | 0.1823 |
| 20 | **0.1877** | 0.1000 | **1.0000** | 0.2066 |

MRR = **0.4029**

**Assessment.** `hit_rate@10 = 1.0` — every graded query surfaces at least one
relevant patent in its top 10; retrieval never outright fails. MRR 0.49 puts
the first relevant result around rank 2 on average. `recall@10 = 0.41` is the
weakest number: with ~5.5 relevant patents per query, the system finds
roughly 2 of 5.5 — the metric that matters most for a "find the prior art you
didn't know about" tool. Claim level trails patent level throughout (MRR 0.40
vs 0.49) — see [Per-query patterns](#per-query-patterns), Pattern 3.

Two filter controls pass: `q19` (= `q04` + `cpc_prefix` filter) scores
bit-identically to `q04`, and `q20` (= `q02` + year filter) narrows the
relevant set without zeroing it — direct evidence the filter path isn't
silently dropping valid hits.

---

## Per-query patterns

Drawn from the 19-query baseline run (raw per-query CSV is regenerable via
the harness; not reproduced here).

**Distinctive vocabulary drives ranking; generic vocabulary sinks it.** The
five queries scoring MRR = 1.000 all contain terms rare in a computing
corpus and near-unique to their target ("garment", "spam", "point-in-time
copy"). The weakest scores come from queries built from words that appear
across thousands of patents ("restoring", "data", "storage", "failure") — on
a topically concentrated corpus, query specificity drives the score more
than retrieval tuning does.

**MRR and recall are decoupled.** Some queries nail the single best hit but
miss the rest of the relevant family (high MRR, low recall@10); others bury
the best hit while finding half the family (low MRR, high recall@10). Which
of these matters depends on whether the product is "find me the one right
patent" or "find me all the prior art."

**Stage 2 (claims) is the weak link.** Patent-level MRR beats claim-level in
11 of 19 queries; claim-level wins in only 4. Likely causes, in order of
impact: `claims_hybrid` has no BM25 arm (created with
`sparse_vectors_config={}`, so Stage 2 is dense-only and loses every lexical
signal); claim text is lexically homogeneous legal boilerplate, which
compresses embedding distances between unrelated claims; and Stage 2 can
only search within whatever patents Stage 1 already selected, so it inherits
Stage 1's misses. One query where the right *claim* ranked first despite its
*patent* ranking poorly (patent MRR 0.14, claim MRR 1.00) is evidence claim-level
dense retrieval adds real signal when Stage 1 gives it the chance —
strengthening the case for adding a BM25 arm at claim level (see
[Known gaps](#known-gaps-and-possible-next-steps)).

**Abstract-verified queries didn't just get easier.** Two of the four
re-graded queries (see [Ground truth](#ground-truth)) remained among the
weakest in the set after correction. Had the re-grading been self-serving,
all four would read near 1.0 — instead the corrections fixed identifiable
grading defects rather than manufacturing a better score.

---

## Ablation — is the BM25 arm earning its keep?

**Scope: patent level only** (`claims_hybrid` has no sparse arm to ablate).

**Headline: keep the hybrid.** Naively, dense-only beats hybrid on every
metric across the 19-query conceptual set — but that result is an artefact
of how the eval queries were deliberately written (avoiding lexical overlap
with titles, precisely so semantic retrieval could be tested), not a property
of the system. The two arms are good at different things, and the fixture
set only exercised one of them.

**Test 1 — conceptual queries (n=19), same as baseline above**

| Metric | hybrid | dense-only | bm25-only |
|---|---|---|---|
| MRR | 0.4905 | **0.6077** | 0.3558 |
| recall@10 | 0.4066 | **0.4891** | 0.1979 |
| nDCG@10 | 0.3557 | **0.4489** | 0.1863 |
| hit_rate@10 | 1.0000 | 1.0000 | 0.7368 |

Hybrid is strictly best in 0 of 19 queries. Read alone, this says "delete the
sparse arm" — but:

**Test 2 — lexical queries (known-item, n=40; query = a patent's own title)**

| Arm | MRR | found in top 20 |
|---|---|---|
| **hybrid** | **0.9315** | **40 / 40** |
| bm25-only | 0.9019 | 39 / 40 |
| dense-only | 0.8893 | 38 / 40 |

Hybrid wins here, and is the only arm with perfect recall — BM25 recovers
two patents dense-only misses entirely. *(Test 1 and Test 2 scores aren't
comparable to each other — known-item retrieval is a much easier task; only
within-test comparisons hold.)*

**Why hybrid loses on conceptual queries but wins on lexical ones:** RRF
fuses by rank, not score quality — a #1 BM25 result contributes exactly what
a #1 dense result contributes. When one arm is much weaker for a given
query, its top (still-weak) hits get promoted to parity and displace better
results from the other arm. On conceptual queries, that's BM25 dragging dense
down; on lexical queries, it's dense's blind spots getting rescued by BM25.

**Test 3 (prefetch sweep) and Test 4 (weighted RRF)** confirmed tuning prefetch
depth or arm weights on the conceptual set alone never closes the gap to
dense-only — the best static weighting recovers only part of dense-only's
lead, because rank-based fusion promotes weak hits regardless of list length.

**Test 5 — recall ceiling: is the missing recall a ranking or retrieval
problem?** Dense-only, retrieved to depth 200 instead of 10:

| depth | recall |
|---|---|
| @10 | 0.4891 |
| @20 | 0.6028 |
| @50 | 0.7632 |
| @100 | 0.8553 |
| @200 | **0.9395** |

**94% of relevant patents are retrieved somewhere in the top 200 — they are
simply ranked too low.** 30 of 105 relevant judgments sit at ranks 11–50,
just past the `patent_top_k = 10` cutoff. This reclassifies the recall
problem: it is **not** a representation failure (a bigger embedding model or
query expansion targets a problem that's largely absent), it **is** a ranking
failure — the most consequential finding in this evaluation, and the direct
motivation for the reranking experiment below.

**Test 6 — weighted RRF on the pooled ground truth (n=20), production arm.**
Re-running the weight sweep against `queries_pooled.jsonl` (fixing the CPC-anchor
bias that biased Tests 1–4 toward dense), at the shipped prefetch depth,
through `DenseRetriever(arm="weighted")` — the real code path behind
`RETRIEVAL_ARM=weighted`, not a harness re-implementation:

| dense : sparse | MRR | recall@5 | recall@10 | nDCG@10 | hit_rate@10 |
|---|---|---|---|---|---|
| native RRF *(shipped default)* | 0.9167 | 0.3666 | 0.5363 | 0.6578 | 1.0000 |
| 4 : 1 (80/20) | 0.9333 | 0.4050 | 0.5924 | 0.7330 | 1.0000 |
| **9 : 1 (90/10)** | 0.9417 | 0.4286 | **0.6235** | 0.7711 | 1.0000 |
| 19 : 1 (95/05) | 0.9417 | **0.4391** | 0.6254 | 0.7707 | 1.0000 |
| 1 : 0 (dense-only) | **0.9667** | 0.4318 | **0.6346** | **0.7845** | 1.0000 |

Every weighted row from 4:1 up beats the native default on all three
headline metrics. **90/10 is worth +0.025 MRR, +0.087 recall@10, and +0.113
nDCG@10** over what ships today, while keeping a live sparse arm for the
lexical register Test 2 protects. `RETRIEVAL_ARM=weighted` (with
`FUSION_DENSE_WEIGHT=0.9` / `FUSION_SPARSE_WEIGHT=0.1`) is shipped and
available, but **off by default**: every confidence interval in the table
overlaps every other, and enabling it costs an extra Qdrant round trip
(client-side fusion fetches each arm separately instead of one native
`FusionQuery` call).

### Conclusions

| Query register | Best arm | Evidence |
|---|---|---|
| Conceptual / semantic | dense-only, 90/10 close behind | Tests 1, 3, 4, 6 |
| Lexical / keyword (patent numbers, exact titles) | hybrid | Test 2 |

No single static configuration is optimal for both — the right amount of
BM25 depends on the query, not a constant. In priority order:

1. **Reranking is the highest-value lever** — Test 5's headroom (30 relevant
   patents at ranks 11–50) is reachable by a cross-encoder rerank in a way no
   amount of fusion tuning can reach. See [Reranking](#reranking) for why this
   was tried and shipped disabled anyway.
2. **Don't delete the sparse arm.** Test 2 shows it's the only path for
   exact-title/publication-number lookup, which dense retrieval cannot serve.
3. **Route on query shape** (unimplemented): send lexical-looking queries
   (publication-number tokens, quoted phrases, short keyword strings) through
   the hybrid path and prose queries through dense-only. Estimated +24% MRR,
   +20% recall@10 on this query set at the cost of one cheap classifier — but
   this improves ranking over the top 20, not recall past it; it doesn't
   substitute for reranking.
4. **`RETRIEVAL_ARM=weighted` (90/10) is a reasonable default to flip** once
   the overlapping confidence intervals are judged worth acting on.
5. **What NOT to spend effort on:** a larger/different embedding model or
   query expansion — Test 5 shows 94% of relevant patents are already
   retrieved within the top 200, so representation is not the bottleneck.

---

## Reranking

**Outcome: built, tested, wired end to end — and shipped disabled, because
measuring it showed it makes retrieval worse.**

Motivated directly by Test 5 above: 94% of relevant patents sit in the top
200 but only 49% in the top 10. A cross-encoder rerank over the top 50 was
the obvious way to reach that headroom. It didn't work.

**Results** (50 candidates reranked to 10, same queries/ground truth as the baseline):

| Model | MRR | recall@10 | nDCG@10 | hit_rate@10 | latency/query |
|---|---|---|---|---|---|
| none (dense-only) | 0.6077 | **0.4891** | **0.4489** | **1.0000** | — |
| `ms-marco-MiniLM-L-6-v2` | **0.6672** | 0.4404 | 0.4273 | 0.9474 | 2,386 ms |
| `ms-marco-MiniLM-L-12-v2` | 0.6632 | 0.4329 | 0.4244 | 0.9474 | 7,397 ms |
| `jina-reranker-v1-turbo-en` | 0.6496 | 0.3867 | 0.3892 | 0.9474 | 4,366 ms |

Every model shows the same trade, which is what makes it a property of the
task rather than a bad model choice: MRR improves (+0.04 to +0.06 — cross-
encoders are genuinely better at picking the single best result), but
`hit_rate@10` falls from a perfect 1.0 to 0.9474 (one query stops surfacing
anything relevant at all), recall@10 drops ~0.05, and latency is prohibitive
for interactive search (2.4–7.4s added per query).

**Why it fails here:** likely a domain/capacity mismatch. These rerankers
are trained on MS MARCO — short web queries against short web passages.
Patent abstracts are long, dense, and legalistic, and `text-embedding-3-small`
(the bi-encoder) is a larger, more modern model than a 22M-parameter
cross-encoder. The bi-encoder is simply stronger on this text, so letting a
weaker cross-encoder overrule its ordering loses more than it gains.

**Why disabled is the right default for this product**, specifically: every
query mode (`prior_art`, `infringement`, `landscape`) is coverage-oriented —
a missed prior-art reference or undetected infringement is worse than an
imperfectly-ordered top result. Losing `hit_rate@10 = 1.0` is the most
damaging part of the trade: the guarantee that every query surfaces
*something* relevant is worth more here than a better-ranked first result.
Were this a "jump me to the single best patent" tool instead, the MRR gain
would be the right trade.

**What's built and kept regardless of the default:**
`CrossEncoderReranker`/`RerankConfig`/`warm_reranker`
(`app/services/rerank/reranker.py`), the Stage-1b hook in
`HierarchicalRetriever`, `RERANK_ENABLED`/`RERANK_MODEL` in settings, startup
warm-up off the event loop, and a fallback to retrieval order on scoring
failure (never fatal). It's opt-in and reversible —
`RERANK_ENABLED=true` switches it on with no code change.

**What would actually capture the recall@50 headroom, if revisited:**
raising `patent_top_k` (crude but honest — recall@20 is already 0.60 vs
0.49 at @10, no model or latency cost); an LLM-based reranker (Gemini is
already a dependency, and far more capable on technical text, at the cost of
tokens/latency — best suited to `prior_art` mode where thoroughness beats
speed); or a patent-domain-tuned cross-encoder (a real project, not a config
change — nothing suitable exists off the shelf today).

---

## Known gaps and possible next steps

Still open as of this writing:

- **No citation-contract checker.** The orchestrator's citation contract
  (cite `[n]`, only `1..len(evidence)`, don't cite unused evidence) has no
  automated check. Proposed metrics: out-of-range citation rate, uncited-claim
  rate, unused-evidence rate, citation–patent agreement. All four are
  deterministic regex/set operations over a `QueryResponse` — cheap to add to
  `tests/` and the harness, and would validate the frontend's
  citation-to-evidence-card linking.
- **Answer generation is entirely unevaluated.** Only retrieval is measured
  here. A RAGAS-based harness was attempted previously and abandoned as
  broken beyond repair against the current API; an LLM-judge pass (Gemini is
  already a dependency) is a cheaper path than resurrecting it, but should
  come after the citation checker above.
- **Add a BM25 arm to `claims_hybrid`.** The strongest structural signal in
  [Per-query patterns](#per-query-patterns) — Stage 2 underperforms Stage 1 on
  every aggregate while running dense-only. Requires a collection recreate
  and re-index of 55,702 chunks.
- **Fixture set gaps:** no explicit `role: "control"` marking for `q19`/`q20`
  (they currently sit in the same aggregate as everything else, giving their
  underlying topics double weight); no abstention metric over the one
  zero-ground-truth query (`q18`) — the system's ability to correctly return
  "nothing relevant" is never scored; and no lexical/keyword queries in the
  graded set, which blocks validating the query-routing idea from
  [Ablation](#ablation--is-the-bm25-arm-earning-its-keep) conclusion 3.
- **No embedding cache or per-stage latency columns** in the harness output —
  every sweep re-embeds all 19–20 queries from scratch, and `EVAL_RERANKING.md`'s
  latency numbers above were hand-timed rather than harness-produced.

---

## Reproducing

```bash
cd apps/api
set -a && . ./.env && set +a
export PYTHONPATH="$(pwd)" PYTHONIOENCODING=utf-8

# baseline (shipped default, unweighted hybrid)
python -m evaluation.retrieval_eval \
  --queries evaluation/fixtures/queries.jsonl \
  --output-dir evaluation/results

# ablation: Test 1 (arm comparison)
for arm in hybrid dense bm25; do
  python -m evaluation.retrieval_eval --queries evaluation/fixtures/queries.jsonl \
    --output-dir evaluation/results --arm $arm --tag $arm
done

# ablation: Test 5 (recall ceiling)
python -m evaluation.retrieval_eval --queries evaluation/fixtures/queries.jsonl \
  --output-dir evaluation/results --arm dense --depth 200 \
  --patent-k 10,20,50,100,200 --tag ceiling

# ablation: Test 6 (weighted RRF sweep, pooled ground truth)
python -m evaluation.retrieval_eval --queries evaluation/fixtures/queries_pooled.jsonl \
  --output-dir evaluation/results --tag pooled
for w in "0.8 0.2 w8020" "0.9 0.1 w9010" "0.95 0.05 w9505" "1.0 0.0 wdense"; do
  set -- $w
  python -m evaluation.retrieval_eval --queries evaluation/fixtures/queries_pooled.jsonl \
    --output-dir evaluation/results --dense-weight $1 --sparse-weight $2 --tag $3
done

# reranking sweep
python -m evaluation.retrieval_eval --queries evaluation/fixtures/queries.jsonl \
  --output-dir evaluation/results --rerank --rerank-model Xenova/ms-marco-MiniLM-L-6-v2 --tag rerank
```

Requires a live Qdrant instance with the corpus indexed (`scripts/build_hupd_corpus.py`
→ `scripts/index_qdrant.py` / `scripts/populate_mongodb.py`) and costs one
embedding call per query. Fixture integrity is guarded offline by
`tests/test_eval_dataset.py::FixtureConsistencyTests`, which cross-checks
every `relevant_patent_ids`/`relevant_chunk_ids` entry against
`corpus_patents.jsonl`/`corpus_chunks.jsonl`.
