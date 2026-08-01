#!/usr/bin/env python3
"""
Build a G06 (computing/AI) corpus from the Harvard USPTO Patent Dataset.

Emits the two JSONL files the existing pipeline consumes:
    patents.jsonl -> scripts/index_qdrant.py patents --input
    chunks.jsonl  -> scripts/populate_mongodb.py --input

FREE-TIER CONSTRAINT -- DO NOT ADD HUPD FIELDS WITHOUT READING THIS.
A raw HUPD record is ~120 KB, dominated by `full_description`. Storing whole
records would put MongoDB at 6,000 x 120 KB = 720 MB, over Atlas M0's 512 MB
limit before any claims are counted. `full_description`, `background` and
`summary` are deliberately NOT ingested. Adding them back blows the tier
silently -- Mongo will simply start rejecting writes mid-run.

Dependencies: stdlib + huggingface_hub (already present as a transitive
dependency of fastembed; no requirements.txt change).

Usage:
    python scripts/build_hupd_corpus.py --discover
    python scripts/build_hupd_corpus.py --dry-run 50
    python scripts/build_hupd_corpus.py --target 6000 --out-dir data/hupd_g06 --verify
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import tarfile
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

REPO_ID = "HUPD/hupd"
REPO_TYPE = "dataset"
_ARCHIVE_RE = re.compile(r"^data/(\d{4})\.tar\.gz$")

# Claim boundaries run INLINE -- the HUPD claims blob contains no newlines.
# Using a \n-anchored pattern here (as app/services/patents/parse.py:93 does)
# silently returns the entire blob as one claim.
_CLAIM_START = re.compile(r"(?:^|(?<=\s))(\d{1,3})\.\s")

_NAN_SENTINELS = {"", "nan", "none", "null", "n/a", "<na>"}


# ----------------------------------------------------------------------
# Pure helpers (unit-tested in tests/test_build_hupd_corpus.py)
# ----------------------------------------------------------------------

def clean_str(value: Any) -> str:
    """Normalise a HUPD scalar, mapping sentinel junk to ''."""
    if value is None:
        return ""
    s = str(value).strip()
    return "" if s.lower() in _NAN_SENTINELS else s


def split_claims(blob: str, *, min_chars: int = 20) -> List[Tuple[int, str]]:
    """
    Split HUPD's single-line claims blob into (claim_no, text) pairs.

    The expected claim number is seeded from the FIRST candidate rather than
    hardcoded to 1: ~8% of records open with "1-20. (canceled) 21. A system..."
    where the first real claim is 21. Seeding recovers every one of those.
    """
    text = (blob or "").strip()
    if not text:
        return []

    candidates = list(_CLAIM_START.finditer(text))
    if not candidates:
        return [(1, text)]

    accepted = []
    expected = int(candidates[0].group(1))
    for m in candidates:
        if int(m.group(1)) == expected:
            accepted.append(m)
            expected += 1

    if not accepted:
        return [(1, text)]

    out: List[Tuple[int, str]] = []
    for i, m in enumerate(accepted):
        end = accepted[i + 1].start() if i + 1 < len(accepted) else len(text)
        body = text[m.end():end].strip()
        if len(body) >= min_chars:
            out.append((int(m.group(1)), body))

    return out or [(1, text)]


def normalise_patent_id(rec: Dict[str, Any]) -> str:
    """
    HUPD publication_number carries a date suffix: 'US20180184987A1-20180705'.
    Strip it. Fall back to the application number; '' means unusable.
    """
    pub = clean_str(rec.get("publication_number"))
    if pub:
        return pub.split("-", 1)[0]
    app = clean_str(rec.get("application_number"))
    return f"HUPD{app}" if app else ""


def cpc_labels(rec: Dict[str, Any], *, max_codes: int = 20) -> Tuple[str, List[str]]:
    """Return (main_label, deduped_all_labels). Codes are HUPD's raw slash-free form."""
    main = clean_str(rec.get("main_cpc_label"))

    raw = rec.get("cpc_labels") or []
    if isinstance(raw, str):
        raw = [raw]

    seen: set = set()
    all_codes: List[str] = []
    for candidate in ([main] + list(raw)):
        code = clean_str(candidate)
        if code and code not in seen:
            seen.add(code)
            all_codes.append(code)
            if len(all_codes) >= max_codes:
                break

    return main, all_codes


def cpc_prefixes_of(codes: Sequence[str]) -> List[str]:
    """4-char section+class+subclass prefixes, e.g. 'G06N'. The API filter target."""
    seen: set = set()
    out: List[str] = []
    for code in codes:
        prefix = code[:4]
        if len(prefix) == 4 and prefix not in seen:
            seen.add(prefix)
            out.append(prefix)
    return out


def matches_cpc_prefixes(
    main: str, all_codes: Sequence[str], wanted: Sequence[str], match_mode: str
) -> bool:
    pool = [main] if match_mode == "main" else all_codes
    return any(code.startswith(p) for code in pool if code for p in wanted)


def parse_filing_year(rec: Dict[str, Any]) -> Optional[int]:
    """HUPD filing_date is 'YYYYMMDD'. Tolerate separators; reject nonsense."""
    digits = re.sub(r"\D", "", clean_str(rec.get("filing_date")))
    if len(digits) >= 4:
        year = int(digits[:4])
        if 1900 <= year <= 2100:
            return year
    return None


def truncate_claims_to_budget(claims: Sequence[str], max_chars: int) -> List[str]:
    """
    Keep whole claims until the char budget is spent. Guards the 8192-token
    embedding limit: embed.py retries a deterministic 400 five times with
    backoff (~22s) before raising and killing the whole indexing run.
    """
    out: List[str] = []
    used = 0
    for claim in claims:
        if out and used + len(claim) > max_chars:
            break
        out.append(claim)
        used += len(claim) + 1
    return out


def chunk_doc_id(patent_id: str, section: str, chunk_id: int) -> str:
    """Byte-identical to populate_mongodb.py:54, so the two paths cannot diverge."""
    return f"{patent_id}::{section}::{chunk_id:04d}"


# ----------------------------------------------------------------------
# Record builders
# ----------------------------------------------------------------------

def build_patent_record(
    rec: Dict[str, Any],
    patent_id: str,
    kept_claims: Sequence[str],
    total_claims: int,
    codes: Sequence[str],
    prefixes: Sequence[str],
    main_cpc: str,
    year: int,
    max_patent_chars: int,
) -> Dict[str, Any]:
    """
    One line of patents.jsonl.

    No `text` key on purpose: _extract_patent_text_dense already reads
    title/abstract/claims, and adding `text` would duplicate content into
    both the dense and the BM25 arm.
    """
    budgeted = truncate_claims_to_budget(kept_claims, max_patent_chars)
    return {
        "id": patent_id,                       # popped to _id by _normalise_id
        "patent_id": patent_id,                # survives the pop; Stage 1 needs it
        "application_number": clean_str(rec.get("application_number")),
        "title": clean_str(rec.get("title")),
        "abstract": clean_str(rec.get("abstract")),
        "claims": " ".join(budgeted),
        "cpc": list(codes),                    # indexed KEYWORD
        "cpc_prefix": list(prefixes),          # indexed KEYWORD -- the API-01 target
        "main_cpc": main_cpc,
        "year": year,                          # indexed INTEGER; query.py:118 filters here
        "filing_year": year,                   # populate_mongodb / fixture parity
        "filing_date": clean_str(rec.get("filing_date")),
        "decision": clean_str(rec.get("decision")),
        "num_claims": total_claims,            # pre-cap, so "3" vs "50, kept 10" is visible
    }


def build_chunk_records(
    patent_id: str,
    title: str,
    abstract: str,
    claims: Sequence[Tuple[int, str]],
    codes: Sequence[str],
    prefixes: Sequence[str],
    year: int,
) -> List[Dict[str, Any]]:
    """
    Lines of chunks.jsonl: one abstract chunk (chunk_id 0) then claims (1..N).

    chunk_id is the ORDINAL and must stay an int -- populate_mongodb.py:54
    formats it with :04d. claim_no is the PRINTED number; the two diverge on
    canceled-claim patents (::claim::0001 carrying claim_no 21) and that is
    correct: ordinals stay dense, claim numbers stay truthful.
    """
    def base(section: str, chunk_id: int, claim_no: Optional[int], text: str) -> Dict[str, Any]:
        return {
            "id": chunk_doc_id(patent_id, section, chunk_id),
            "patent_id": patent_id,     # indexed KEYWORD; Stage 2's $in filter needs it
            "section": section,         # index_qdrant --sections reads this
            "chunk_id": chunk_id,
            "claim_no": claim_no,
            "title": title,
            "text": text,               # first key _extract_claim_text checks
            "cpc": list(codes),
            "cpc_prefix": list(prefixes),
            "filing_year": year,
            "year": year,
        }

    out = [base("abstract", 0, None, abstract)]
    for ordinal, (claim_no, body) in enumerate(claims, start=1):
        out.append(base("claim", ordinal, claim_no, body))
    return out


# ----------------------------------------------------------------------
# Acquisition
# ----------------------------------------------------------------------

def discover_years() -> Dict[int, float]:
    """Map year -> size in GB. Never hardcode a path that wasn't confirmed this run."""
    from huggingface_hub import HfApi

    api = HfApi()
    info = api.repo_info(REPO_ID, repo_type=REPO_TYPE, files_metadata=True)

    years: Dict[int, float] = {}
    for sibling in info.siblings:
        m = _ARCHIVE_RE.match(sibling.rfilename)
        if m:
            size = getattr(sibling, "size", None) or 0
            years[int(m.group(1))] = size / 1e9
    return years


def stream_year(year: int, cache: bool) -> Iterator[Dict[str, Any]]:
    """
    Yield one parsed JSON record per archive member.

    Members are matched on the .json suffix rather than a '<year>/<app>.json'
    template: only 2018's internal layout was inspected, and older archives
    may differ.
    """
    from huggingface_hub import hf_hub_download, hf_hub_url

    path = f"data/{year}.tar.gz"

    if cache:
        local = hf_hub_download(REPO_ID, path, repo_type=REPO_TYPE)
        tar = tarfile.open(local, mode="r:gz")
    else:
        url = hf_hub_url(REPO_ID, path, repo_type=REPO_TYPE)
        tar = tarfile.open(fileobj=urllib.request.urlopen(url), mode="r|gz")

    seen_json = 0
    try:
        for member in tar:
            if not member.isfile() or not member.name.endswith(".json"):
                continue
            handle = tar.extractfile(member)
            if handle is None:
                continue
            try:
                yield json.loads(handle.read())
                seen_json += 1
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
    finally:
        tar.close()

    if seen_json == 0:
        print(f"  [WARN] {year}: archive yielded no .json members -- layout may differ",
              file=sys.stderr)


# ----------------------------------------------------------------------
# Main pipeline
# ----------------------------------------------------------------------

class Stats:
    def __init__(self) -> None:
        self.scanned = 0
        self.cpc_match = 0
        self.emitted = 0
        self.rejected: Dict[str, int] = {}

    def reject(self, reason: str) -> None:
        self.rejected[reason] = self.rejected.get(reason, 0) + 1


def process_record(rec: Dict[str, Any], args, stats: Stats) -> Optional[Dict[str, Any]]:
    """Filter + parse one HUPD record into {'patent': ..., 'chunks': [...]} or None."""
    main, codes = cpc_labels(rec)
    if not matches_cpc_prefixes(main, codes, args.cpc_prefix, args.cpc_match):
        return None
    stats.cpc_match += 1

    patent_id = normalise_patent_id(rec)
    if not patent_id:
        stats.reject("no-usable-id")
        return None

    abstract = clean_str(rec.get("abstract"))
    if len(abstract) < args.min_abstract_chars:
        stats.reject("short-abstract")
        return None

    year = parse_filing_year(rec)
    if year is None:
        stats.reject("bad-filing-date")
        return None

    all_claims = split_claims(clean_str(rec.get("claims")))
    if not all_claims:
        stats.reject("empty-claims")
        return None

    kept = all_claims[:args.max_claims]
    prefixes = cpc_prefixes_of(codes)
    title = clean_str(rec.get("title"))

    return {
        "patent": build_patent_record(
            rec, patent_id, [body for _, body in kept], len(all_claims),
            codes, prefixes, main, year, args.max_patent_chars,
        ),
        "chunks": build_chunk_records(
            patent_id, title, abstract, kept, codes, prefixes, year,
        ),
    }


def run(args) -> int:
    available = discover_years()
    if not available:
        print("[ERROR] No data/<YYYY>.tar.gz found. HUPD layout has changed.",
              file=sys.stderr)
        return 2

    if args.discover:
        print(f"{REPO_ID} ({REPO_TYPE}) -- yearly archives:")
        for year in sorted(available):
            print(f"  {year}  {available[year]:5.2f} GB")
        print("\nNote: hupd_metadata_*.feather is UNUSED (needs pandas+pyarrow, absent).")
        return 0

    unknown = [y for y in args.years if y not in available]
    if unknown:
        print(f"[ERROR] Year(s) not in repo: {unknown}. Available: {sorted(available)}",
              file=sys.stderr)
        return 2

    stats = Stats()
    emitted_ids: set = set()
    patents: List[Dict[str, Any]] = []
    chunks: List[Dict[str, Any]] = []
    reservoir_n = 0
    rng = random.Random(args.sample_seed) if args.sample_seed is not None else None

    for year in args.years:
        print(f"[{year}] streaming...", flush=True)
        year_start = stats.emitted

        for rec in stream_year(year, args.cache_archives):
            stats.scanned += 1
            built = process_record(rec, args, stats)
            if built is None:
                continue

            pid = built["patent"]["patent_id"]
            if pid in emitted_ids:
                stats.reject("duplicate-id")
                continue

            if rng is None:
                emitted_ids.add(pid)
                patents.append(built["patent"])
                chunks.extend(built["chunks"])
                stats.emitted += 1
                if stats.emitted >= args.target:
                    break
            else:
                # Reservoir sampling (Algorithm R): unbiased, but requires
                # streaming every listed year in full.
                reservoir_n += 1
                if len(patents) < args.target:
                    emitted_ids.add(pid)
                    patents.append(built["patent"])
                    chunks.extend(built["chunks"])
                else:
                    j = rng.randrange(reservoir_n)
                    if j < args.target:
                        patents[j] = built["patent"]

        print(f"[{year}] scanned={stats.scanned} cpc-match={stats.cpc_match} "
              f"emitted={stats.emitted - year_start}", flush=True)

        if rng is None and stats.emitted >= args.target:
            print(f"  target {args.target} reached -- aborting remaining years")
            break

    if stats.emitted < args.target:
        print(f"\n[WARN] Exhausted all listed years with {stats.emitted} patents, "
              f"target was {args.target}. Add more --years or lower --target.",
              file=sys.stderr)

    if args.dry_run:
        print(f"\n--- DRY RUN: {len(patents)} patents, {len(chunks)} chunks, "
              f"nothing written ---")
        for record in (patents[:1] + chunks[:2]):
            print(json.dumps(record, indent=2)[:900])
        _print_summary(stats, patents, chunks)
        return 0

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    patents_path = Path(args.patents_out) if args.patents_out else out_dir / "patents.jsonl"
    chunks_path = Path(args.chunks_out) if args.chunks_out else out_dir / "chunks.jsonl"

    _write_jsonl(patents_path, patents)
    _write_jsonl(chunks_path, chunks)
    print(f"\nWrote {patents_path}  {len(patents)} patents")
    print(f"Wrote {chunks_path}  {len(chunks)} chunks")
    _print_summary(stats, patents, chunks)

    if args.verify:
        return _verify(patents, chunks)
    return 0


def _write_jsonl(path: Path, records: Sequence[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _print_summary(stats: Stats, patents, chunks) -> None:
    n_abstract = sum(1 for c in chunks if c["section"] == "abstract")
    print(f"\nChunks: {n_abstract} abstract + {len(chunks) - n_abstract} claim")
    if patents:
        print(f"Mean claims kept/patent: "
              f"{(len(chunks) - n_abstract) / len(patents):.2f}")
        hist: Dict[str, int] = {}
        for p in patents:
            for prefix in p["cpc_prefix"]:
                hist[prefix] = hist.get(prefix, 0) + 1
        top = sorted(hist.items(), key=lambda kv: -kv[1])[:8]
        print("CPC prefixes: " + ", ".join(f"{k}={v}" for k, v in top))
    if stats.rejected:
        print("Rejected: " + ", ".join(f"{k}={v}" for k, v in sorted(stats.rejected.items())))


def _verify(patents, chunks) -> int:
    """Assert every emitted record survives the REAL extractors embed.py will see."""
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from app.services.indexing.qdrant import QdrantHybridStore

    bad_p = [p["patent_id"] for p in patents
             if not QdrantHybridStore._extract_patent_text_dense(p).strip()]
    bad_c = [c["id"] for c in chunks
             if not QdrantHybridStore._extract_claim_text(c).strip()]

    if bad_p or bad_c:
        print(f"\n[VERIFY] FAILED: {len(bad_p)} patents and {len(bad_c)} chunks "
              f"produce empty extractor text.", file=sys.stderr)
        for item in (bad_p[:5] + bad_c[:5]):
            print(f"  {item}", file=sys.stderr)
        return 1

    print(f"\n[VERIFY] {len(patents)}/{len(patents)} patents and "
          f"{len(chunks)}/{len(chunks)} chunks produce non-empty extractor text.")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--discover", action="store_true",
                   help="List repo files and available years, then exit. No download.")
    p.add_argument("--dry-run", type=int, nargs="?", const=50, default=0,
                   metavar="N", help="Parse N patents, print samples, write nothing.")
    p.add_argument("--cpc-prefix", action="append", default=None, metavar="G06",
                   help="Repeatable CPC prefix filter (default: G06).")
    p.add_argument("--cpc-match", choices=["any", "main"], default="any",
                   help="Match all cpc_labels (14.3%% yield) or main only (9.4%%).")
    p.add_argument("--years", type=str, default="2018,2017,2016",
                   help="Comma-separated, processed in order. Default smallest-first.")
    p.add_argument("--target", type=int, default=6000)
    p.add_argument("--max-claims", type=int, default=10)
    p.add_argument("--min-abstract-chars", type=int, default=50)
    p.add_argument("--max-patent-chars", type=int, default=20000)
    p.add_argument("--sample-seed", type=int, default=None,
                   help="Reservoir-sample (unbiased) -- forces full download of all years.")
    p.add_argument("--cache-archives", action="store_true",
                   help="hf_hub_download (resumable) instead of in-memory stream.")
    p.add_argument("--out-dir", type=str, default="data/hupd_g06")
    p.add_argument("--patents-out", type=str, default=None)
    p.add_argument("--chunks-out", type=str, default=None)
    p.add_argument("--verify", action="store_true",
                   help="Re-check every record against the real Qdrant text extractors.")

    args = p.parse_args()
    args.cpc_prefix = args.cpc_prefix or ["G06"]
    if args.dry_run:
        args.target = args.dry_run

    try:
        args.years = [int(y.strip()) for y in args.years.split(",") if y.strip()]
    except ValueError:
        p.error("--years must be comma-separated 4-digit years")

    for name in ("target", "max_claims", "max_patent_chars"):
        if getattr(args, name) <= 0:
            p.error(f"--{name.replace('_', '-')} must be positive")

    return run(args)


if __name__ == "__main__":
    sys.exit(main())
