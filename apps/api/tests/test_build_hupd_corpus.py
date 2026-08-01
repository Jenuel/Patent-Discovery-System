"""Unit tests for the HUPD corpus builder.

These exercise pure parsing/record-building logic — no network, no tar
archives, no HuggingFace. The three contract tests at the bottom run the
emitted records through the *real* Qdrant text extractors and the *real*
populate_mongodb.py id formula, so a schema drift fails here rather than
6,000 records into a paid embedding run.

Run from apps/api: python -m unittest discover tests
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import build_hupd_corpus as bhc  # noqa: E402

from app.services.indexing.qdrant import QdrantHybridStore  # noqa: E402


# ----------------------------------------------------------------------
# Fixtures — abbreviated from real HUPD 2018 records
# ----------------------------------------------------------------------

def _hupd_record(**overrides: Any) -> Dict[str, Any]:
    """A normal G06 record. Claims run inline — HUPD uses no newlines."""
    rec = {
        "application_number": "15856905",
        "publication_number": "US20180184987A1-20180705",
        "patent_number": "nan",
        "title": "Machine Learning Accelerator With Systolic Array",
        "abstract": (
            "A hardware accelerator for neural network inference comprising a "
            "systolic array of processing elements and an on-chip weight buffer "
            "that reduces external memory bandwidth during convolution."
        ),
        "claims": (
            "1. A hardware accelerator comprising a systolic array of processing "
            "elements arranged in rows and columns, and an on-chip buffer coupled "
            "to the array. 2. The accelerator of claim 1, wherein the on-chip "
            "buffer stores quantized weights in an eight-bit format. 3. The "
            "accelerator of claim 2, further comprising a controller configured "
            "to schedule convolution operations across the rows."
        ),
        "main_cpc_label": "G06N3063",
        "cpc_labels": ["G06N3063", "G06F1580", "G06N304"],
        "filing_date": "20180228",
        "decision": "ACCEPTED",
        "full_description": "x" * 100_000,   # must never reach the output
    }
    rec.update(overrides)
    return rec


def _args(**overrides: Any):
    """Stand-in for the argparse namespace process_record consumes."""
    class A:
        cpc_prefix = ["G06"]
        cpc_match = "any"
        max_claims = 10
        min_abstract_chars = 50
        max_patent_chars = 20000

    a = A()
    for k, v in overrides.items():
        setattr(a, k, v)
    return a


# ----------------------------------------------------------------------
# split_claims — the part the dead parse.py gets wrong
# ----------------------------------------------------------------------

class SplitClaimsTests(unittest.TestCase):
    def test_splits_on_inline_separator(self):
        """HUPD claims have NO newlines; a \\n-anchored regex returns one chunk."""
        claims = bhc.split_claims(_hupd_record()["claims"])
        self.assertEqual([n for n, _ in claims], [1, 2, 3])
        self.assertTrue(claims[0][1].startswith("A hardware accelerator"))
        self.assertIn("eight-bit", claims[1][1])

    def test_strips_the_number_prefix_from_the_body(self):
        claims = bhc.split_claims("1. First claim body text here. 2. Second claim body.")
        self.assertFalse(claims[0][1].startswith("1."))

    def test_canceled_range_seeds_from_first_real_claim(self):
        """~8% of records open '1-20. (canceled) 21. ...' — seeding recovers them."""
        blob = (
            "1-20. (canceled) 21. A system for distributed training comprising a "
            "plurality of worker nodes coupled by a network fabric. 22. The system "
            "of claim 21, wherein each worker node comprises a gradient buffer."
        )
        claims = bhc.split_claims(blob)
        self.assertEqual([n for n, _ in claims], [21, 22])
        self.assertTrue(claims[0][1].startswith("A system for distributed"))

    def test_out_of_sequence_numbers_are_rejected(self):
        """A stray 'FIG. 7.' must not open a new claim when 7 isn't expected."""
        blob = ("1. A method as shown in FIG. 7. and described herein with "
                "reference to the drawings. 2. The method of claim 1 further "
                "comprising a calibration step.")
        self.assertEqual([n for n, _ in bhc.split_claims(blob)], [1, 2])

    def test_unnumbered_blob_falls_back_to_one_claim(self):
        blob = "A method of doing something useful without any claim numbering."
        self.assertEqual(bhc.split_claims(blob), [(1, blob)])

    def test_empty_blob_yields_nothing(self):
        self.assertEqual(bhc.split_claims(""), [])
        self.assertEqual(bhc.split_claims("   "), [])

    def test_no_segment_is_empty_or_whitespace(self):
        for _, body in bhc.split_claims(_hupd_record()["claims"]):
            self.assertTrue(body.strip())

    def test_segments_shorter_than_min_chars_are_dropped(self):
        blob = "1. Short. 2. A substantially longer claim body that clears the bar."
        claims = bhc.split_claims(blob, min_chars=20)
        self.assertEqual([n for n, _ in claims], [2])


# ----------------------------------------------------------------------
# Field extraction
# ----------------------------------------------------------------------

class PatentIdTests(unittest.TestCase):
    def test_strips_the_date_suffix(self):
        self.assertEqual(bhc.normalise_patent_id(_hupd_record()), "US20180184987A1")

    def test_falls_back_to_application_number(self):
        rec = _hupd_record(publication_number="")
        self.assertEqual(bhc.normalise_patent_id(rec), "HUPD15856905")

    def test_nan_sentinel_is_treated_as_missing(self):
        """HUPD writes the literal string 'nan' for absent numbers."""
        rec = _hupd_record(publication_number="nan")
        self.assertEqual(bhc.normalise_patent_id(rec), "HUPD15856905")

    def test_unusable_record_returns_empty(self):
        rec = _hupd_record(publication_number="nan", application_number="")
        self.assertEqual(bhc.normalise_patent_id(rec), "")


class CpcTests(unittest.TestCase):
    def test_main_label_leads_and_duplicates_are_dropped(self):
        main, codes = bhc.cpc_labels(_hupd_record())
        self.assertEqual(main, "G06N3063")
        self.assertEqual(codes, ["G06N3063", "G06F1580", "G06N304"])

    def test_prefixes_are_four_chars_and_deduped(self):
        _, codes = bhc.cpc_labels(_hupd_record())
        self.assertEqual(bhc.cpc_prefixes_of(codes), ["G06N", "G06F"])

    def test_code_list_is_capped(self):
        rec = _hupd_record(cpc_labels=[f"G06N{i:04d}" for i in range(50)])
        _, codes = bhc.cpc_labels(rec, max_codes=5)
        self.assertEqual(len(codes), 5)

    def test_string_valued_cpc_labels_are_tolerated(self):
        _, codes = bhc.cpc_labels(_hupd_record(cpc_labels="G06T0700"))
        self.assertIn("G06T0700", codes)

    def test_match_any_scans_every_label(self):
        rec = _hupd_record(main_cpc_label="H04L2906", cpc_labels=["H04L2906", "G06N3063"])
        main, codes = bhc.cpc_labels(rec)
        self.assertTrue(bhc.matches_cpc_prefixes(main, codes, ["G06"], "any"))

    def test_match_main_ignores_secondary_labels(self):
        rec = _hupd_record(main_cpc_label="H04L2906", cpc_labels=["H04L2906", "G06N3063"])
        main, codes = bhc.cpc_labels(rec)
        self.assertFalse(bhc.matches_cpc_prefixes(main, codes, ["G06"], "main"))

    def test_slash_free_codes_still_prefix_match(self):
        """Real HUPD codes are 'G06T220710081', not 'G06T2207/10081'."""
        self.assertTrue(bhc.matches_cpc_prefixes("G06T220710081", [], ["G06"], "main"))
        self.assertFalse(bhc.matches_cpc_prefixes("G06T220710081", [], ["G06N"], "main"))

    def test_empty_prefix_list_matches_nothing(self):
        self.assertFalse(bhc.matches_cpc_prefixes("G06N3063", [], [], "main"))


class FilingYearTests(unittest.TestCase):
    def test_parses_yyyymmdd(self):
        self.assertEqual(bhc.parse_filing_year(_hupd_record()), 2018)

    def test_tolerates_separators(self):
        self.assertEqual(bhc.parse_filing_year({"filing_date": "2018-02-28"}), 2018)

    def test_missing_or_garbage_returns_none(self):
        self.assertIsNone(bhc.parse_filing_year({"filing_date": ""}))
        self.assertIsNone(bhc.parse_filing_year({"filing_date": "nan"}))
        self.assertIsNone(bhc.parse_filing_year({}))

    def test_out_of_range_year_rejected(self):
        self.assertIsNone(bhc.parse_filing_year({"filing_date": "18000101"}))


class TruncateTests(unittest.TestCase):
    def test_never_exceeds_budget_and_cuts_at_a_claim_boundary(self):
        claims = ["a" * 100, "b" * 100, "c" * 100]
        kept = bhc.truncate_claims_to_budget(claims, 250)
        self.assertEqual(kept, claims[:2])
        self.assertTrue(all(len(set(c)) == 1 for c in kept))  # no mid-claim cut

    def test_first_claim_always_survives_even_if_oversized(self):
        """Dropping every claim would produce a text-less record embed.py rejects."""
        self.assertEqual(bhc.truncate_claims_to_budget(["x" * 5000], 100), ["x" * 5000])


# ----------------------------------------------------------------------
# Record builders
# ----------------------------------------------------------------------

class PatentRecordTests(unittest.TestCase):
    def _build(self) -> Dict[str, Any]:
        rec = _hupd_record()
        main, codes = bhc.cpc_labels(rec)
        claims = bhc.split_claims(rec["claims"])
        return bhc.build_patent_record(
            rec, "US20180184987A1", [b for _, b in claims], len(claims),
            codes, bhc.cpc_prefixes_of(codes), main, 2018, 20000,
        )

    def test_year_and_filing_year_agree(self):
        p = self._build()
        self.assertEqual(p["year"], p["filing_year"])
        self.assertIsInstance(p["year"], int)

    def test_no_text_key(self):
        """`text` would duplicate content into both the dense and BM25 arms."""
        self.assertNotIn("text", self._build())

    def test_full_description_is_never_carried_over(self):
        """The 120 KB field that would blow Atlas M0's 512 MB limit."""
        p = self._build()
        self.assertNotIn("full_description", p)
        self.assertNotIn("background", p)
        self.assertNotIn("summary", p)

    def test_id_and_patent_id_both_present(self):
        """_normalise_id pops `id`; `patent_id` must survive for Stage 1."""
        p = self._build()
        self.assertEqual(p["id"], p["patent_id"])

    def test_num_claims_is_the_precap_total(self):
        rec = _hupd_record()
        main, codes = bhc.cpc_labels(rec)
        claims = bhc.split_claims(rec["claims"])
        p = bhc.build_patent_record(
            rec, "X", [b for _, b in claims[:2]], len(claims),
            codes, [], main, 2018, 20000,
        )
        self.assertEqual(p["num_claims"], 3)


class ChunkRecordTests(unittest.TestCase):
    def _build(self) -> List[Dict[str, Any]]:
        rec = _hupd_record()
        _, codes = bhc.cpc_labels(rec)
        claims = bhc.split_claims(rec["claims"])
        return bhc.build_chunk_records(
            "US20180184987A1", rec["title"], rec["abstract"], claims,
            codes, bhc.cpc_prefixes_of(codes), 2018,
        )

    def test_one_abstract_chunk_then_claims(self):
        chunks = self._build()
        self.assertEqual(chunks[0]["section"], "abstract")
        self.assertEqual([c["section"] for c in chunks[1:]], ["claim"] * 3)
        self.assertEqual(len(chunks), 4)

    def test_chunk_id_is_an_int(self):
        """populate_mongodb.py:54 formats it with :04d — a str raises ValueError."""
        for c in self._build():
            self.assertIsInstance(c["chunk_id"], int)

    def test_chunk_ids_are_dense_ordinals(self):
        self.assertEqual([c["chunk_id"] for c in self._build()], [0, 1, 2, 3])

    def test_abstract_has_no_claim_no(self):
        self.assertIsNone(self._build()[0]["claim_no"])

    def test_ordinal_and_claim_no_diverge_on_canceled_ranges(self):
        """::claim::0001 legitimately carries claim_no 21."""
        claims = bhc.split_claims(
            "1-20. (canceled) 21. A system comprising a plurality of worker nodes "
            "coupled together. 22. The system of claim 21 with a gradient buffer."
        )
        chunks = bhc.build_chunk_records("P", "T", "A" * 60, claims, [], [], 2018)
        first_claim = chunks[1]
        self.assertEqual(first_claim["chunk_id"], 1)
        self.assertEqual(first_claim["claim_no"], 21)

    def test_every_chunk_carries_patent_id(self):
        """Stage 2 filters on {'patent_id': {'$in': [...]}} — a miss returns zero."""
        for c in self._build():
            self.assertEqual(c["patent_id"], "US20180184987A1")


# ----------------------------------------------------------------------
# process_record — the filter/reject path
# ----------------------------------------------------------------------

class ProcessRecordTests(unittest.TestCase):
    def test_accepts_a_normal_g06_record(self):
        out = bhc.process_record(_hupd_record(), _args(), bhc.Stats())
        self.assertIsNotNone(out)
        self.assertEqual(out["patent"]["patent_id"], "US20180184987A1")
        self.assertEqual(len(out["chunks"]), 4)

    def test_non_g06_is_filtered_without_a_reject_tally(self):
        """A CPC miss is not a data defect, so it must not inflate rejections."""
        stats = bhc.Stats()
        rec = _hupd_record(main_cpc_label="C02F1044", cpc_labels=["C02F1044"])
        self.assertIsNone(bhc.process_record(rec, _args(), stats))
        self.assertEqual(stats.rejected, {})
        self.assertEqual(stats.cpc_match, 0)

    def test_short_abstract_is_rejected_and_counted(self):
        stats = bhc.Stats()
        self.assertIsNone(bhc.process_record(_hupd_record(abstract="Too short."),
                                             _args(), stats))
        self.assertEqual(stats.rejected.get("short-abstract"), 1)

    def test_bad_filing_date_is_rejected(self):
        stats = bhc.Stats()
        self.assertIsNone(bhc.process_record(_hupd_record(filing_date="nan"),
                                             _args(), stats))
        self.assertEqual(stats.rejected.get("bad-filing-date"), 1)

    def test_unusable_id_is_rejected(self):
        stats = bhc.Stats()
        rec = _hupd_record(publication_number="nan", application_number="")
        self.assertIsNone(bhc.process_record(rec, _args(), stats))
        self.assertEqual(stats.rejected.get("no-usable-id"), 1)

    def test_max_claims_caps_the_chunk_count(self):
        out = bhc.process_record(_hupd_record(), _args(max_claims=2), bhc.Stats())
        self.assertEqual(len(out["chunks"]), 3)          # 1 abstract + 2 claims
        self.assertEqual(out["patent"]["num_claims"], 3)  # pre-cap total preserved


# ----------------------------------------------------------------------
# Contract tests against the REAL consumers
# ----------------------------------------------------------------------

class ConsumerContractTests(unittest.TestCase):
    """The highest-value tests: these import production code, not fakes."""

    def setUp(self):
        self.built = bhc.process_record(_hupd_record(), _args(), bhc.Stats())

    def test_patent_survives_the_real_dense_extractor(self):
        """embed.py:70 raises on ANY empty input and aborts the whole batch."""
        text = QdrantHybridStore._extract_patent_text_dense(self.built["patent"])
        self.assertTrue(text.strip())
        self.assertIn("Machine Learning Accelerator", text)
        self.assertIn("systolic array", text)
        self.assertIn("hardware accelerator", text)

    def test_dense_and_sparse_texts_differ(self):
        """Title doubling belongs to BM25 only; patent_id must stay out of dense."""
        p = self.built["patent"]
        dense = QdrantHybridStore._extract_patent_text_dense(p)
        sparse = QdrantHybridStore._extract_patent_text_sparse(p)
        self.assertNotEqual(dense, sparse)
        self.assertIn("US20180184987A1", sparse)
        self.assertNotIn("US20180184987A1", dense)

    def test_every_chunk_survives_the_real_claim_extractor(self):
        for c in self.built["chunks"]:
            self.assertTrue(
                QdrantHybridStore._extract_claim_text(c).strip(),
                f"chunk {c['id']} extracts to empty text",
            )

    def test_chunk_id_matches_populate_mongodb_formula(self):
        """
        Re-executes populate_mongodb.py:54 verbatim. Our explicit `id` and the
        script's synthesised fallback must be byte-identical, so the two id
        mechanisms can never diverge.
        """
        for c in self.built["chunks"]:
            expected = f"{c['patent_id']}::{c['section']}::{c['chunk_id']:04d}"
            self.assertEqual(c["id"], expected)

    def test_patent_payload_covers_every_qdrant_index(self):
        """Missing an indexed key means that filter silently matches nothing."""
        p = self.built["patent"]
        for field in QdrantHybridStore.PATENT_PAYLOAD_INDEXES:
            self.assertIn(field, p, f"patent record is missing indexed field '{field}'")

    def test_chunk_payload_covers_every_qdrant_index(self):
        for c in self.built["chunks"]:
            for field in QdrantHybridStore.CLAIM_PAYLOAD_INDEXES:
                self.assertIn(field, c, f"chunk is missing indexed field '{field}'")


if __name__ == "__main__":
    unittest.main()
